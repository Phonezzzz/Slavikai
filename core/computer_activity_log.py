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


def build_computer_activity_summary(
    events: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    """Build a deterministic summary from raw ComputerActivityLog events.

    Pure function — no LLM, no I/O.  Called on the stored computer_events
    list so the session snapshot can expose it alongside the raw events.
    """
    tools_started = 0
    tools_finished = 0
    files_read = 0
    files_written = 0
    commands_run = 0
    errors_count = 0
    latest_error: str | None = None
    has_diff = False
    tests_seen = False

    for ev in events:
        kind = ev.get("kind")
        ok = ev.get("ok")

        if kind == "tool_started":
            tools_started += 1
            continue

        tools_finished += 1

        if kind == "file_read":
            files_read += 1
        elif kind == "file_written":
            files_written += 1
        elif kind == "command_started":
            commands_run += 1
        elif kind == "git_diff_updated":
            has_diff = True
        elif kind == "test_result":
            tests_seen = True

        if ok is False:
            errors_count += 1
            err = ev.get("error")
            if isinstance(err, str) and err:
                latest_error = err

    return {
        "total_events": len(events),
        "tools_started": tools_started,
        "tools_finished": tools_finished,
        "files_read": files_read,
        "files_written": files_written,
        "commands_run": commands_run,
        "errors_count": errors_count,
        "latest_error": latest_error,
        "has_diff": has_diff,
        "tests_seen": tests_seen,
    }
