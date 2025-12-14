from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

from shared.models import JSONValue

TRACE_LOG = Path("logs/trace.log")


class TraceRecord(TypedDict, total=False):
    timestamp: str
    event: str
    message: str
    meta: dict[str, JSONValue]


class Tracer:
    """Менеджер трассировки reasoning-цепочек агента."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or TRACE_LOG
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, message: str, meta: dict[str, JSONValue] | None = None) -> None:
        record: TraceRecord = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event_type,
            "message": message,
            "meta": meta or {},
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 50) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
        return [json.loads(line) for line in lines]
