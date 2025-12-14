from __future__ import annotations

from shared.models import ToolRequest, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry


def test_tool_registry_logs_success(tmp_path) -> None:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("dummy", handler, enabled=True)
    result = registry.call(ToolRequest(name="dummy", args={"k": "v"}))
    assert result.ok

    records = registry.read_recent_calls()
    assert len(records) == 1
    assert records[0].tool == "dummy"
    assert records[0].ok
    assert records[0].args == {"k": "v"}


def test_tool_registry_logs_disabled(tmp_path) -> None:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({})

    registry.register("dummy", handler, enabled=False)
    result = registry.call(ToolRequest(name="dummy"))
    assert not result.ok

    records = registry.read_recent_calls()
    assert len(records) == 1
    assert records[0].tool == "dummy"
    assert not records[0].ok
