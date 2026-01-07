from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Final

from shared.models import JSONValue, ToolCallRecord
from shared.sanitize import safe_json_loads, sanitize_record

DEFAULT_LOG_PATH = Path("logs/tool_calls.log")
MAX_TOOL_LOG_BYTES: Final[int] = 2_000_000
MAX_TOOL_LOG_BACKUPS: Final[int] = 3
_WRITE_LOCK = threading.Lock()


class ToolCallLogger:
    """Журнал вызовов инструментов с хранением на диске."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger("SlavikAI.ToolCallLogger")

    def _rotate_log_if_needed(self) -> None:
        if MAX_TOOL_LOG_BYTES <= 0 or MAX_TOOL_LOG_BACKUPS <= 0:
            return
        if not self.path.exists():
            return
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            self._logger.warning(
                "ToolCallLogger: не удалось прочитать размер лога %s: %s",
                self.path,
                exc,
            )
            return
        if size < MAX_TOOL_LOG_BYTES:
            return
        try:
            oldest = self.path.with_name(f"{self.path.name}.{MAX_TOOL_LOG_BACKUPS}")
            if oldest.exists():
                oldest.unlink()
            for index in range(MAX_TOOL_LOG_BACKUPS - 1, 0, -1):
                src = self.path.with_name(f"{self.path.name}.{index}")
                if src.exists():
                    src.rename(self.path.with_name(f"{self.path.name}.{index + 1}"))
            self.path.rename(self.path.with_name(f"{self.path.name}.1"))
            self._logger.info(
                "ToolCallLogger: лог %s превысил %d байт, выполнена ротация",
                self.path,
                MAX_TOOL_LOG_BYTES,
            )
        except OSError as exc:
            self._logger.warning(
                "ToolCallLogger: ошибка ротации лога %s: %s", self.path, exc
            )

    def log(
        self,
        tool: str,
        ok: bool,
        error: str | None = None,
        meta: dict[str, JSONValue] | None = None,
        args: dict[str, JSONValue] | None = None,
    ) -> None:
        record = ToolCallRecord(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            tool=tool,
            ok=ok,
            error=error,
            meta=meta,
            args=args,
        )
        payload = sanitize_record(record.__dict__)
        with _WRITE_LOCK:
            self._rotate_log_if_needed()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 100) -> list[ToolCallRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            lines = list(deque(handle, maxlen=limit))
        records: list[ToolCallRecord] = []
        bad_lines = 0
        for line in lines:
            if not line.strip():
                bad_lines += 1
                continue
            data = safe_json_loads(line)
            if not isinstance(data, dict):
                bad_lines += 1
                continue
            records.append(
                ToolCallRecord(
                    timestamp=str(data.get("timestamp")),
                    tool=str(data.get("tool")),
                    ok=bool(data.get("ok")),
                    error=data.get("error"),
                    meta=data.get("meta"),
                    args=data.get("args"),
                ),
            )
        if bad_lines:
            self._logger.warning(
                "ToolCallLogger: пропущено %d некорректных строк в %s",
                bad_lines,
                self.path,
            )
        return records
