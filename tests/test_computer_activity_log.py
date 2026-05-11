"""Tests for ComputerActivityLog — verifies mechanism, not just effect."""

from __future__ import annotations

import pytest

from core.computer_activity_log import ComputerActivityLog
from shared.models import ToolRequest, ToolResult


@pytest.fixture()
def log() -> ComputerActivityLog:
    return ComputerActivityLog()


def test_pre_call_appends_tool_started(log: ComputerActivityLog) -> None:
    log.pre_call(ToolRequest("workspace_read", {"path": "src/main.py"}))
    events = log.drain()
    assert len(events) == 1
    assert events[0]["kind"] == "tool_started"
    assert events[0]["tool"] == "workspace_read"


def test_post_call_appends_file_read_for_workspace_read(log: ComputerActivityLog) -> None:
    ctx = log.pre_call(ToolRequest("workspace_read", {"path": "a.py"}))
    log.post_call(ToolRequest("workspace_read", {"path": "a.py"}), ToolResult.success({}), ctx)
    events = log.drain()
    post_event = events[1]
    assert post_event["kind"] == "file_read"
    assert post_event["path"] == "a.py"
    assert post_event["ok"] is True


def test_post_call_appends_file_written_for_workspace_write(
    log: ComputerActivityLog,
) -> None:
    request = ToolRequest("workspace_write", {"path": "b.py", "content": "x"})
    ctx = log.pre_call(request)
    log.post_call(request, ToolResult.success({}), ctx)
    events = log.drain()
    assert events[1]["kind"] == "file_written"
    assert events[1]["path"] == "b.py"


def test_post_call_appends_command_started_for_terminal_run(
    log: ComputerActivityLog,
) -> None:
    request = ToolRequest("workspace_terminal_run", {"command": "pytest"})
    ctx = log.pre_call(request)
    log.post_call(request, ToolResult.success({"output": "ok"}), ctx)
    events = log.drain()
    assert events[1]["kind"] == "command_started"
    assert events[1]["command"] == "pytest"


def test_post_call_includes_error_on_failure(log: ComputerActivityLog) -> None:
    request = ToolRequest("workspace_write", {"path": "x.py", "content": ""})
    ctx = log.pre_call(request)
    log.post_call(request, ToolResult.failure("permission denied"), ctx)
    events = log.drain()
    assert events[1]["ok"] is False
    assert events[1]["error"] == "permission denied"


def test_drain_clears_events(log: ComputerActivityLog) -> None:
    log.pre_call(ToolRequest("workspace_list", {"path": ""}))
    log.drain()
    assert log.drain() == []


def test_post_call_records_duration_ms(log: ComputerActivityLog) -> None:
    ctx = log.pre_call(ToolRequest("workspace_read", {"path": "f.py"}))
    log.post_call(
        ToolRequest("workspace_read", {"path": "f.py"}),
        ToolResult.success({}),
        ctx,
    )
    events = log.drain()
    assert isinstance(events[1].get("duration_ms"), int)
    assert events[1]["duration_ms"] >= 0


def test_pre_call_returns_float_timestamp(log: ComputerActivityLog) -> None:
    result = log.pre_call(ToolRequest("workspace_list", {"path": ""}))
    assert isinstance(result, float)
    assert result > 0


def test_unknown_tool_maps_to_tool_started_kind(log: ComputerActivityLog) -> None:
    request = ToolRequest("some_unknown_tool", {})
    ctx = log.pre_call(request)
    log.post_call(request, ToolResult.success({}), ctx)
    events = log.drain()
    assert events[1]["kind"] == "tool_started"


def test_each_tool_call_appends_two_events(log: ComputerActivityLog) -> None:
    for name in ("workspace_list", "workspace_read", "workspace_write"):
        request = ToolRequest(name, {"path": "p"})
        ctx = log.pre_call(request)
        log.post_call(request, ToolResult.success({}), ctx)
    events = log.drain()
    assert len(events) == 6
