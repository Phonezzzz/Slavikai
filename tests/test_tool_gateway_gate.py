"""E01-T04: ToolGateway is the single execution gate.

Acceptance (ROADMAP.md E01-T04): an instrumentation test confirms every
model-originated state-changing call passes the gateway.
Negative: a synthetic tool, called directly, fails the test gate.

The gate itself is deliberately small: the tests record every
ToolGateway.call — via the gateway pre_call hook where the test builds the
gateway, and via a monkeypatched ToolGateway.call where production code
builds the gateway internally (Agent.call_tool) — and assert the executed
tool names were observed there. A direct tool.handle() call never reaches
ToolGateway.call, so the same assertion fails for it; that is the negative
test. The audit behind this file (all production call sites, verified by
reading the code) is in ROADMAP.md under E01-T04.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.agent import Agent
from core.run_contract import RunPolicy
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig, ToolCall
from shared.models import LLMMessage, PlanStep, PlanStepStatus, ToolRequest, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import set_workspace_root


class StaticBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        del messages, config
        return LLMResult(text="ok")


class BatchBrain(Brain):
    supports_native_tools = True

    def __init__(self, batches):
        self.batches = iter(batches)

    def generate(self, messages, config=None, tools=None):
        return LLMResult(usage=LLMUsage(0, 0, 0), text="answer", tool_calls=next(self.batches, []))


def _calls(*names):
    return [ToolCall(id=f"{name}-{i}", name=name, arguments={}) for i, name in enumerate(names)]


def _registry(tmp_path: Path) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "calls.jsonl"))
    dispatched: list[str] = []

    def handler(request: ToolRequest) -> ToolResult:
        dispatched.append(request.name)
        return ToolResult.success({"ok": True})

    registry.register("workspace_write", handler, capability="write")
    registry.register("workspace_read", handler, capability="read")
    registry.register("synthetic_write", handler, capability="write")
    return registry, dispatched


def _build_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=StaticBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    return agent


def _assert_gateway_observed(observed: list[str], name: str) -> None:
    """The gate: a tool execution counts only if ToolGateway.call saw it."""
    if name not in observed:
        raise AssertionError(
            f"gateway gate: tool '{name}' executed without passing ToolGateway.call"
        )


def test_loop_dispatch_goes_through_gateway(tmp_path: Path) -> None:
    """Acceptance: the one AgentToolLoop dispatches model calls via the gateway.

    Covers the chat/auto/desktop loop sites — they all use this same class
    and the same _dispatch_model_tool_call helper.
    """
    registry, dispatched = _registry(tmp_path)
    observed: list[str] = []
    gateway = ToolGateway(
        registry,
        run_policy=RunPolicy(mode="act", execution_target="sandbox"),
        pre_call=lambda request: observed.append(request.name),
    )
    result = AgentToolLoop().run(
        brain=BatchBrain([_calls("workspace_write", "workspace_read"), []]),
        gateway=gateway,
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(policy=gateway.run_policy),
    )
    assert observed == ["workspace_write", "workspace_read"]
    assert dispatched == ["workspace_write", "workspace_read"]
    assert len(result.tool_calls) == 2
    assert all(call.result.ok for call in result.tool_calls)


def test_agent_call_tool_goes_through_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance: Agent.call_tool (plan audit, decision resume, UI writes,
    TTS/STT) reaches the tool only via a gateway built by the agent."""
    agent = _build_agent(tmp_path)
    observed: list[str] = []
    original_call = ToolGateway.call

    def recording(self: ToolGateway, request: ToolRequest) -> ToolResult:
        observed.append(request.name)
        return original_call(self, request)

    monkeypatch.setattr(ToolGateway, "call", recording)
    set_workspace_root(str(tmp_path))
    try:
        # Safe mode is on by default; a workspace write needs its category.
        agent.set_session_context("s", {"FS_DELETE_OVERWRITE"})
        write = agent.call_tool("workspace_write", {"path": "gate.txt", "content": "hi\n"})
        assert write.ok, write.error
        assert observed == ["workspace_write"]
        assert (tmp_path / "gate.txt").read_text(encoding="utf-8") == "hi\n"
        read = agent.call_tool("workspace_read", {"path": "gate.txt"})
        assert read.ok, read.error
        assert observed == ["workspace_write", "workspace_read"]
        # The workspace_run fence refuses BEFORE the gateway: no gateway call.
        refused = agent.call_tool("workspace_run", {"path": "gate.py"})
        assert not refused.ok
        assert observed == ["workspace_write", "workspace_read"]
    finally:
        set_workspace_root(None)


def test_mwv_plan_step_goes_through_gateway(tmp_path: Path) -> None:
    """Acceptance: the MWV packet runner executes plan steps via the gateway."""
    agent = _build_agent(tmp_path)
    registry, dispatched = _registry(tmp_path)
    observed: list[str] = []
    gateway = ToolGateway(
        registry,
        run_policy=RunPolicy(mode="act", execution_target="sandbox"),
        pre_call=lambda request: observed.append(request.name),
    )
    step = PlanStep(description="write the file", operation="workspace_write", tool_args={})
    done = agent._execute_mwv_plan_step(step, gateway)
    assert done.status == PlanStepStatus.DONE
    assert observed == ["workspace_write"]
    assert dispatched == ["workspace_write"]


def test_direct_tool_call_fails_gateway_gate(tmp_path: Path) -> None:
    """Negative: a synthetic tool, called directly, fails the test gate.

    The direct handle() call DOES execute the tool (dispatched proves it),
    but the gateway never sees it, so the gate trips. The control shows the
    same tool through gateway.call passes the gate.
    """
    registry, dispatched = _registry(tmp_path)
    observed: list[str] = []
    gateway = ToolGateway(
        registry,
        run_policy=RunPolicy(mode="act", execution_target="sandbox"),
        pre_call=lambda request: observed.append(request.name),
    )
    descriptor = registry.get_descriptor("synthetic_write")
    assert descriptor is not None
    direct = descriptor.handler(ToolRequest(name="synthetic_write", args={}))
    assert direct.ok
    assert dispatched == ["synthetic_write"]
    with pytest.raises(AssertionError, match="gateway gate"):
        _assert_gateway_observed(observed, "synthetic_write")
    via_gateway = gateway.call(ToolRequest(name="synthetic_write", args={}))
    assert via_gateway.ok
    _assert_gateway_observed(observed, "synthetic_write")
    assert dispatched == ["synthetic_write", "synthetic_write"]
