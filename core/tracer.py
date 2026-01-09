from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Final, TypedDict

from shared.models import JSONValue
from shared.sanitize import safe_json_loads, sanitize_record

TRACE_LOG = Path("logs/trace.log")
MAX_TRACE_LOG_BYTES: Final[int] = 2_000_000
MAX_TRACE_LOG_BACKUPS: Final[int] = 3
_WRITE_LOCK = threading.Lock()


class TraceRecord(TypedDict, total=False):
    timestamp: str
    event: str
    message: str
    meta: dict[str, JSONValue]


def _rotate_log_if_needed(
    path: Path,
    *,
    max_bytes: int,
    max_backups: int,
    logger: logging.Logger,
) -> None:
    if max_bytes <= 0 or max_backups <= 0:
        return
    if not path.exists():
        return
    try:
        size = path.stat().st_size
    except OSError as exc:
        logger.warning("Tracer: не удалось прочитать размер лога %s: %s", path, exc)
        return
    if size < max_bytes:
        return
    try:
        oldest = path.with_name(f"{path.name}.{max_backups}")
        if oldest.exists():
            oldest.unlink()
        for index in range(max_backups - 1, 0, -1):
            src = path.with_name(f"{path.name}.{index}")
            if src.exists():
                src.rename(path.with_name(f"{path.name}.{index + 1}"))
        path.rename(path.with_name(f"{path.name}.1"))
        logger.info("Tracer: лог %s превысил %d байт, выполнена ротация", path, max_bytes)
    except OSError as exc:
        logger.warning("Tracer: ошибка ротации лога %s: %s", path, exc)


class Tracer:
    """Менеджер трассировки reasoning-цепочек агента."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or TRACE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("SlavikAI.Tracer")

    def log(self, event_type: str, message: str, meta: dict[str, JSONValue] | None = None) -> None:
        record: TraceRecord = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_type,
            "message": message,
            "meta": meta or {},
        }
        sanitized_record = sanitize_record(
            {
                "timestamp": record["timestamp"],
                "event": record["event"],
                "message": record["message"],
                "meta": record["meta"],
            },
        )
        with _WRITE_LOCK:
            _rotate_log_if_needed(
                self.path,
                max_bytes=MAX_TRACE_LOG_BYTES,
                max_backups=MAX_TRACE_LOG_BACKUPS,
                logger=self._logger,
            )
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sanitized_record, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 50) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            lines = list(deque(handle, maxlen=limit))
        records: list[TraceRecord] = []
        bad_lines = 0
        required_keys = {"timestamp", "event", "message"}
        for line in lines:
            if not line.strip():
                bad_lines += 1
                continue
            data = safe_json_loads(line)
            if not isinstance(data, dict):
                bad_lines += 1
                continue
            if not required_keys.issubset(data):
                bad_lines += 1
                continue
            meta = data.get("meta")
            if meta is not None and not isinstance(meta, dict):
                bad_lines += 1
                continue
            record: TraceRecord = {
                "timestamp": str(data.get("timestamp")),
                "event": str(data.get("event")),
                "message": str(data.get("message")),
                "meta": meta or {},
            }
            records.append(record)
        if bad_lines:
            self._logger.warning(
                "Tracer: пропущено %d некорректных строк в %s",
                bad_lines,
                self.path,
            )
        return records
