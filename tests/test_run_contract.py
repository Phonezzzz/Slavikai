"""E01-T02: RunSpec/RunPolicy — immutable run-level contract, snapshot isolation.

Acceptance criteria (ROADMAP.md E01-T02):
(a) two concurrent runs with different target/policy do not affect each other;
(b) mutating the global ToolRegistry mode mid-run does not change the run's
    effective policy.

On the old code both fail: the run path mutated the shared ToolRegistry
(set_execution_policy inside _build_tool_gateway) and ToolGateway.call()
read the global mode/target at call time, so a second run or a mid-run
global mutation changed the first run's effective policy.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from core.agent import Agent
from core.run_contract import (
    RunBudget,
    RunPolicy,
    build_run_spec,
    intent_for_mode,
    target_for_mode,
)
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig, ToolCall
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry


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

    def write_handler(request: ToolRequest) -> ToolResult:
        dispatched.append(request.name)
        return ToolResult.success({"written": True})

    def read_handler(request: ToolRequest) -> ToolResult:
        dispatched.append(request.name)
        return ToolResult.success({"read": True})

    registry.register("workspace_write", write_handler, capability="write")
    registry.register("workspace_read", read_handler, capability="read")
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


def _run_loop(registry: ToolRegistry, gateway: ToolGateway):
    return AgentToolLoop().run(
        brain=BatchBrain([_calls("workspace_write"), []]),
        gateway=gateway,
        messages=[LLMMessage(role="user", content="go")],
        tools=registry.list_tool_specs(policy=gateway.run_policy),
    )


# --- contract unit tests ---------------------------------------------------


def test_run_spec_is_frozen() -> None:
    spec = build_run_spec(mode="act")
    with pytest.raises(FrozenInstanceError):
        spec.intent = "plan"
    with pytest.raises(FrozenInstanceError):
        spec.policy.mode = "ask"
    with pytest.raises(FrozenInstanceError):
        spec.budget.max_iterations = 1


def test_run_policy_deep_copies_plan_and_task() -> None:
    plan = {"plan_id": "p1", "steps": [{"tool": "workspace_write"}]}
    task = {"task_id": "t1"}
    policy = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=task,
    )
    plan["steps"].append({"tool": "other"})
    plan["plan_id"] = "p2"
    task["task_id"] = "t2"
    assert policy.active_plan == {"plan_id": "p1", "steps": [{"tool": "workspace_write"}]}
    assert policy.active_task == {"task_id": "t1"}


def test_run_policy_nested_collections_immutable() -> None:
    """Codex review (PR #156, P2): nested plan/task data must be immutable too.

    frozen=True only blocks field reassignment, and the old deepcopy only
    protected against mutating the source dicts — the snapshot's own nested
    lists/dicts were still mutable through the policy, so a caller could
    change the gateway's effective policy mid-run.
    """
    plan = {
        "plan_id": "p1",
        "steps": [{"step_id": "s1", "allowed_tool_kinds": ["workspace_read"]}],
    }
    task = {"task_id": "t1"}
    policy = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=task,
    )
    snapshot: Any = policy.active_plan
    # The finding's exact repro: appending a tool kind to the snapshot.
    with pytest.raises(TypeError):
        snapshot["steps"][0]["allowed_tool_kinds"].append("workspace_write")
    with pytest.raises(TypeError):
        snapshot["steps"].append({"step_id": "s2"})
    with pytest.raises(TypeError):
        snapshot["steps"][0]["allowed_tool_kinds"][0] = "workspace_write"
    with pytest.raises(TypeError):
        snapshot["plan_id"] = "p2"
    task_snapshot: Any = policy.active_task
    with pytest.raises(TypeError):
        task_snapshot["task_id"] = "t2"
    # Reads still work and the snapshot is unchanged.
    assert policy.active_plan == {
        "plan_id": "p1",
        "steps": [{"step_id": "s1", "allowed_tool_kinds": ["workspace_read"]}],
    }
    assert policy.active_task == {"task_id": "t1"}


def test_snapshot_nested_mutation_cannot_change_gateway_policy(tmp_path: Path) -> None:
    """Gateway level: a failed nested mutation leaves the policy enforced.

    With the plan guard on, workspace_write is blocked for the current step;
    attempting the finding's mutation must raise and the gateway must keep
    blocking the write afterwards.
    """
    registry, dispatched = _registry(tmp_path)
    plan = {
        "plan_id": "p1",
        "plan_hash": "h1",
        "steps": [{"step_id": "s1", "allowed_tool_kinds": ["workspace_read"]}],
    }
    task = {"plan_id": "p1", "plan_hash": "h1", "current_step_id": "s1"}
    policy = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=task,
        enforce_plan_guard=True,
    )
    gateway = ToolGateway(registry, run_policy=policy)
    blocked = gateway.call(ToolRequest(name="workspace_write"))
    assert not blocked.ok
    assert "BLOCKED_OUTSIDE_PLAN" in (blocked.error or "")
    assert dispatched == []
    snapshot: Any = policy.active_plan
    with pytest.raises(TypeError):
        snapshot["steps"][0]["allowed_tool_kinds"].append("workspace_write")
    still_blocked = gateway.call(ToolRequest(name="workspace_write"))
    assert not still_blocked.ok
    assert "BLOCKED_OUTSIDE_PLAN" in (still_blocked.error or "")
    assert dispatched == []
    # The allowed read still goes through.
    assert gateway.call(ToolRequest(name="workspace_read")).ok
    assert dispatched == ["workspace_read"]


@pytest.mark.parametrize(
    ("mode", "intent", "target"),
    [
        ("ask", "answer", "sandbox"),
        ("plan", "plan", "sandbox"),
        ("act", "execute", "sandbox"),
        ("auto", "execute", "sandbox"),
        ("desktop", "execute", "desktop"),
    ],
)
def test_intent_and_target_derivation(mode, intent, target) -> None:
    assert intent_for_mode(mode) == intent
    assert target_for_mode(mode) == target
    spec = build_run_spec(mode=mode)
    assert spec.intent == intent
    assert spec.target == target
    assert spec.policy.mode == mode
    assert spec.policy.execution_target == target
    assert spec.budget == RunBudget()


def test_loop_kwargs_unbounded_by_default() -> None:
    assert RunBudget().loop_kwargs() == {
        "max_iterations": 8,
        "max_tool_calls": None,
        "max_model_tokens": None,
        "expires_at_monotonic": None,
    }


def test_loop_kwargs_expiry_from_started_monotonic() -> None:
    budget = RunBudget(
        max_iterations=3, max_tool_calls=5, max_model_tokens=100, max_runtime_seconds=10.0
    )
    kwargs = budget.loop_kwargs(started_monotonic=1000.0)
    assert kwargs["max_iterations"] == 3
    assert kwargs["max_tool_calls"] == 5
    assert kwargs["max_model_tokens"] == 100
    assert kwargs["expires_at_monotonic"] == 1010.0


# --- acceptance: snapshot isolation ---------------------------------------


def test_two_gateways_with_different_policies_are_independent(tmp_path: Path) -> None:
    """Positive acceptance: two runs, different policy, one registry, no interference."""
    registry, dispatched = _registry(tmp_path)
    ask_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="ask", execution_target="sandbox")
    )
    act_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="act", execution_target="sandbox")
    )
    blocked = ask_gateway.call(ToolRequest(name="workspace_write"))
    assert not blocked.ok
    assert "ASK_READ_ONLY_BLOCK" in (blocked.error or "")
    assert act_gateway.call(ToolRequest(name="workspace_write")).ok
    assert dispatched == ["workspace_write"]  # only the act run executed the write
    assert ask_gateway.call(ToolRequest(name="workspace_read")).ok
    assert dispatched == ["workspace_write", "workspace_read"]


def test_global_mode_mutation_does_not_change_run_policy(tmp_path: Path) -> None:
    """Negative acceptance: mutating the global registry mode mid-run is inert."""
    registry, dispatched = _registry(tmp_path)
    gateway = ToolGateway(registry, run_policy=RunPolicy(mode="act", execution_target="sandbox"))
    assert gateway.call(ToolRequest(name="workspace_write")).ok
    registry.set_execution_policy(mode="ask")  # global mutation mid-run
    assert gateway.call(ToolRequest(name="workspace_write")).ok
    assert dispatched == ["workspace_write", "workspace_write"]
    # The other direction: an ask snapshot stays blocked after the global flips to act.
    ask_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="ask", execution_target="sandbox")
    )
    registry.set_execution_policy(mode="act")
    blocked = ask_gateway.call(ToolRequest(name="workspace_write"))
    assert not blocked.ok
    assert "ASK_READ_ONLY_BLOCK" in (blocked.error or "")
    # A gateway without a snapshot (legacy path) still follows the global mode.
    legacy = ToolGateway(registry)
    registry.set_execution_policy(mode="ask")
    assert not legacy.call(ToolRequest(name="workspace_write")).ok
    registry.set_execution_policy(mode="act")
    assert legacy.call(ToolRequest(name="workspace_write")).ok


def test_list_tool_specs_uses_snapshot_target(tmp_path: Path) -> None:
    registry, _ = _registry(tmp_path)
    registry.register(
        "desktop_only",
        lambda request: ToolResult.success({}),
        capability="read",
        execution_targets=("desktop",),
    )
    registry.set_execution_policy(mode="act")  # global target is sandbox
    desktop_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="desktop", execution_target="desktop")
    )
    desktop_specs = {
        spec.name for spec in registry.list_tool_specs(policy=desktop_gateway.run_policy)
    }
    assert "desktop_only" in desktop_specs
    assert "desktop_only" not in {spec.name for spec in registry.list_tool_specs()}
    # The other direction: a sandbox snapshot hides desktop-only tools
    # even when the global target is desktop.
    registry.set_execution_policy(mode="desktop")
    sandbox_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="act", execution_target="sandbox")
    )
    assert "desktop_only" not in {
        spec.name for spec in registry.list_tool_specs(policy=sandbox_gateway.run_policy)
    }
    assert "desktop_only" in {spec.name for spec in registry.list_tool_specs()}


# --- loop level ------------------------------------------------------------


def test_loop_run_enforces_snapshot_policy(tmp_path: Path) -> None:
    """Loop-level positive acceptance: ask run blocks the write, act run executes it."""
    registry, dispatched = _registry(tmp_path)
    ask_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="ask", execution_target="sandbox")
    )
    act_gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="act", execution_target="sandbox")
    )
    ask_result = _run_loop(registry, ask_gateway)
    assert dispatched == []
    assert len(ask_result.tool_calls) == 1
    assert "ASK_READ_ONLY_BLOCK" in (ask_result.tool_calls[0].result.error or "")
    assert ask_result.error is None  # blocked call is recorded, not fatal to the loop
    act_result = _run_loop(registry, act_gateway)
    assert dispatched == ["workspace_write"]
    assert act_result.tool_calls[0].result.ok


def test_loop_run_ignores_global_mode_mutation(tmp_path: Path) -> None:
    """Loop-level negative acceptance: a global mode flip between runs changes nothing."""
    registry, dispatched = _registry(tmp_path)
    gateway = ToolGateway(registry, run_policy=RunPolicy(mode="act", execution_target="sandbox"))
    _run_loop(registry, gateway)
    assert dispatched == ["workspace_write"]
    registry.set_execution_policy(mode="ask")  # global mutation between runs
    _run_loop(registry, gateway)
    assert dispatched == ["workspace_write", "workspace_write"]


# --- Agent wiring ----------------------------------------------------------


def test_build_tool_gateway_snapshots_without_mutating_global_registry(
    tmp_path: Path,
) -> None:
    """Agent wiring: _build_tool_gateway attaches the snapshot, global state untouched."""
    agent = _build_agent(tmp_path)
    dispatched: list[str] = []
    agent.tool_registry.register(
        "e01t02_write_probe",
        lambda request: dispatched.append(request.name) or ToolResult.success({"ok": True}),
        capability="write",
    )
    agent.tool_registry.set_execution_policy(mode="ask")  # global is ask
    agent.runtime_mode = "act"
    gateway = agent._build_tool_gateway()
    assert gateway.run_policy is not None
    assert gateway.run_policy.mode == "act"
    assert gateway.run_policy.execution_target == "sandbox"
    # The snapshot wins over the global mode when the registry is called with it.
    assert agent.tool_registry.call(
        ToolRequest(name="e01t02_write_probe"), policy=gateway.run_policy
    ).ok
    # The global registry was NOT mutated by _build_tool_gateway (still ask).
    blocked = agent.tool_registry.call(ToolRequest(name="e01t02_write_probe"))
    assert not blocked.ok
    assert "ASK_READ_ONLY_BLOCK" in (blocked.error or "")
    # _run_spec derives the contract from the current runtime state.
    agent.runtime_mode = "ask"
    spec = agent._run_spec()
    assert spec.intent == "answer"
    assert spec.policy.mode == "ask"
    assert spec.target == "sandbox"
