from __future__ import annotations

import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


def test_safe_mode_requires_approval_and_blocks_tool() -> None:
    registry = ToolRegistry()
    called = {"ok": False}

    def handler(_: ToolRequest) -> ToolResult:
        called["ok"] = True
        return ToolResult.success({"output": "ok"})

    registry.register("shell", handler, enabled=True)
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=True,
            session_id="s1",
            approved_categories=set(),
        ),
    )

    with pytest.raises(ApprovalRequired):
        gateway.call(ToolRequest(name="shell", args={"command": "rm -rf /"}))
    assert called["ok"] is False


def test_approved_category_allows_tool() -> None:
    registry = ToolRegistry()
    called = {"ok": False}

    def handler(_: ToolRequest) -> ToolResult:
        called["ok"] = True
        return ToolResult.success({"output": "ok"})

    registry.register("shell", handler, enabled=True)
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=True,
            session_id="s1",
            approved_categories={"SUDO", "EXEC_ARBITRARY"},
        ),
    )

    result = gateway.call(ToolRequest(name="shell", args={"command": "sudo echo hi"}))
    assert result.ok is True
    assert called["ok"] is True


def test_safe_mode_disabled_allows_tool() -> None:
    registry = ToolRegistry()
    called = {"ok": False}

    def handler(_: ToolRequest) -> ToolResult:
        called["ok"] = True
        return ToolResult.success({"output": "ok"})

    registry.register("shell", handler, enabled=True)
    gateway = ToolGateway(
        registry,
        approval_context=ApprovalContext(
            safe_mode=False,
            session_id="s1",
            approved_categories=set(),
        ),
    )

    result = gateway.call(ToolRequest(name="shell", args={"command": "rm -rf /"}))
    assert result.ok is True
    assert called["ok"] is True


def test_confirmed_server_tool_bypasses_only_desktop_action_policy() -> None:
    registry = ToolRegistry()
    called: list[str] = []

    def handler(request: ToolRequest) -> ToolResult:
        called.append(request.name)
        return ToolResult.success({"output": "ok"})

    registry.register(
        "memory_save_confirmed",
        handler,
        enabled=True,
        capability="write",
        model_exposed=False,
        confirmed_decision_only=True,
        execution_targets={"desktop"},
    )
    registry.register(
        "unrelated_write",
        handler,
        enabled=True,
        capability="write",
        execution_targets={"desktop"},
    )
    registry.set_execution_policy(mode="desktop")
    context = ApprovalContext(
        safe_mode=True,
        session_id="s1",
        approved_categories=set(),
        execution_target="desktop",
    )

    confirmed = ToolGateway(
        registry,
        approval_context=context,
        confirmed_decision=True,
    ).call(ToolRequest(name="memory_save_confirmed"))
    unrelated = ToolGateway(
        registry,
        approval_context=context,
        confirmed_decision=True,
    ).call(ToolRequest(name="unrelated_write"))

    assert confirmed.ok is True
    assert not unrelated.ok
    assert unrelated.error == "POLICY_DENY: desktop_tool_unknown"
    assert called == ["memory_save_confirmed"]
