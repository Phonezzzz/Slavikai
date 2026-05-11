from __future__ import annotations

import time
from dataclasses import dataclass, field

from shared.models import JSONValue, ToolRequest, ToolResult

_TOOL_TO_EVENT: dict[str, str] = {
    "workspace_list": "file_read",
    "workspace_read": "file_read",
    "workspace_write": "file_written",
    "workspace_patch": "file_written",
    "workspace_terminal_run": "command_started",
    "workspace_diff": "git_diff_updated",
    "workspace_test": "test_result",
}


def _event_for_tool(tool_name: str) -> str:
    return _TOOL_TO_EVENT.get(tool_name, "tool_started")


@dataclass
class ComputerActivityLog:
    """Captures tool call activity via ToolGateway hooks.

    Pre-call records the start; post-call records the finish event.
    Entries are consumed via drain() — same pattern as auto_progress_events.
    """

    _events: list[dict[str, JSONValue]] = field(default_factory=list)

    def pre_call(self, request: ToolRequest) -> float:
        ts = time.time()
        self._events.append(
            {
                "kind": "tool_started",
                "tool": request.name,
                "ts": ts,
            }
        )
        return ts

    def post_call(
        self,
        request: ToolRequest,
        result: ToolResult,
        context: object | None,
    ) -> None:
        ts = time.time()
        start_ts = context if isinstance(context, float) else ts
        kind = _event_for_tool(request.name)
        entry: dict[str, JSONValue] = {
            "kind": kind,
            "tool": request.name,
            "ok": result.ok,
            "ts": ts,
            "duration_ms": round((ts - start_ts) * 1000),
        }
        raw_path = request.args.get("path")
        if isinstance(raw_path, str) and raw_path:
            entry["path"] = raw_path
        raw_command = request.args.get("command")
        if isinstance(raw_command, str) and raw_command:
            entry["command"] = raw_command
        if not result.ok and result.error:
            entry["error"] = result.error
        self._events.append(entry)

    def drain(self) -> list[dict[str, JSONValue]]:
        events = list(self._events)
        self._events.clear()
        return events
