from __future__ import annotations

import pytest

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


def test_tool_registry_allows_read_in_ask_mode(tmp_path) -> None:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("dummy", handler, enabled=True, capability="read")
    registry.set_execution_policy(mode="ask")
    result = registry.call(ToolRequest(name="dummy"))
    assert result.ok


def test_tool_registry_blocks_write_in_plan_mode(tmp_path) -> None:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("workspace_write", handler, enabled=True, capability="write")
    registry.set_execution_policy(mode="plan")
    result = registry.call(ToolRequest(name="workspace_write"))
    assert not result.ok
    assert "PLAN_READ_ONLY_BLOCK" in (result.error or "")


def test_confirmed_decision_tool_is_server_only_and_does_not_open_other_writes(
    tmp_path,
) -> None:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tool_calls.log"))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"saved": True})

    registry.register(
        "memory_save_confirmed",
        handler,
        enabled=True,
        capability="write",
        model_exposed=False,
        confirmed_decision_only=True,
    )
    registry.register("workspace_write", handler, enabled=True, capability="write")
    registry.set_execution_policy(mode="ask")

    unconfirmed = registry.call(ToolRequest(name="memory_save_confirmed"))
    confirmed = registry.call(
        ToolRequest(name="memory_save_confirmed"),
        confirmed_decision=True,
    )
    unrelated_write = registry.call(
        ToolRequest(name="workspace_write"),
        confirmed_decision=True,
    )

    assert not unconfirmed.ok
    assert "CONFIRMED_DECISION_REQUIRED" in (unconfirmed.error or "")
    assert confirmed.ok
    assert not unrelated_write.ok
    assert "ASK_READ_ONLY_BLOCK" in (unrelated_write.error or "")
    assert "memory_save_confirmed" not in {spec.name for spec in registry.list_tool_specs()}


def test_confirmed_decision_tool_cannot_be_model_exposed() -> None:
    registry = ToolRegistry()

    with pytest.raises(ValueError, match="не может быть model_exposed"):
        registry.register(
            "unsafe",
            lambda _: ToolResult.success(),
            confirmed_decision_only=True,
        )


def test_tool_registry_plan_guard_blocks_tool_outside_step(tmp_path) -> None:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))

    def handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("workspace_write", handler, enabled=True, capability="write")
    registry.set_execution_policy(
        mode="act",
        active_plan={
            "plan_id": "plan-1",
            "plan_hash": "hash-1",
            "steps": [
                {
                    "step_id": "step-1",
                    "allowed_tool_kinds": ["workspace_read"],
                }
            ],
        },
        active_task={
            "task_id": "task-1",
            "plan_id": "plan-1",
            "plan_hash": "hash-1",
            "current_step_id": "step-1",
            "status": "running",
        },
        enforce_plan_guard=True,
    )
    result = registry.call(ToolRequest(name="workspace_write"))
    assert not result.ok
    assert "BLOCKED_OUTSIDE_PLAN" in (result.error or "")
