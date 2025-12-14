from __future__ import annotations

import json
import time
from pathlib import Path

from shared.models import JSONValue, ToolCallRecord

DEFAULT_LOG_PATH = Path("logs/tool_calls.log")


class ToolCallLogger:
    """Журнал вызовов инструментов с хранением на диске."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or DEFAULT_LOG_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)

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
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 100) -> list[ToolCallRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            lines = handle.readlines()[-limit:]
        records: list[ToolCallRecord] = []
        for line in lines:
            data = json.loads(line)
            records.append(
                ToolCallRecord(
                    timestamp=str(data.get("timestamp")),
                    tool=str(data.get("tool")),
                    ok=bool(data.get("ok")),
                    error=data.get("error"),
                    meta=data.get("meta"),
                    args=data.get("args"),
                )
            )
        return records
