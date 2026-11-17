"""E01-T03: ToolRegistry listing derives from the RunSpec.

Acceptance (ROADMAP.md E01-T03):
(a) an `answer` (ask) spec lists no write tools;
(b) an `execute` (act/auto/desktop) spec lists only tools qualified for its target;
(c) an unknown target fails closed (no tools listed, no silent fallback);
(d) negative: a tool cannot be manually invoked through the registry
    bypassing the target check.

The legacy no-policy path keeps working as the plan-drafting catalog
(sandbox tools, write included) but never follows the global mutable
target, so it cannot smuggle desktop tools into a listing or a call.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from core.run_contract import RunPolicy
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ExecutionTarget, ToolRegistry


def _registry(tmp_path: Path) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "calls.jsonl"))
    dispatched: list[str] = []

    def handler(request: ToolRequest) -> ToolResult:
        dispatched.append(request.name)
        return ToolResult.success({"ok": True})

    registry.register("workspace_write", handler, capability="write")
    registry.register("workspace_read", handler, capability="read")
    registry.register("workspace_exec", handler, capability="exec")
    registry.register("desktop_only", handler, capability="write", execution_targets={"desktop"})
    registry.register("desktop_read", handler, capability="read", execution_targets={"desktop"})
    registry.register(
        "both_targets",
        handler,
        capability="write",
        execution_targets={"sandbox", "desktop"},
    )
    return registry, dispatched


def _names(registry: ToolRegistry, policy: RunPolicy | None = None) -> set[str]:
    return {spec.name for spec in registry.list_tool_specs(policy=policy)}


def test_answer_spec_lists_no_write_tools(tmp_path: Path) -> None:
    """Acceptance (a): an answer (ask) run sees no write or exec tools."""
    registry, _ = _registry(tmp_path)
    names = _names(registry, RunPolicy(mode="ask", execution_target="sandbox"))
    assert "workspace_write" not in names
    assert "workspace_exec" not in names
    assert "workspace_read" in names
    # Plan mode is read-only at call time too, so the listing matches.
    plan_names = _names(registry, RunPolicy(mode="plan", execution_target="sandbox"))
    assert "workspace_write" not in plan_names
    assert "workspace_read" in plan_names


def test_execute_spec_lists_only_target_qualified_tools(tmp_path: Path) -> None:
    """Acceptance (b): execute lists only tools qualified for its target."""
    registry, _ = _registry(tmp_path)
    sandbox = _names(registry, RunPolicy(mode="act", execution_target="sandbox"))
    assert sandbox == {
        "workspace_write",
        "workspace_read",
        "workspace_exec",
        "both_targets",
    }
    desktop = _names(registry, RunPolicy(mode="desktop", execution_target="desktop"))
    assert desktop == {"desktop_only", "desktop_read", "both_targets"}
    # auto is an execute intent on the sandbox target.
    auto = _names(registry, RunPolicy(mode="auto", execution_target="sandbox"))
    assert auto == sandbox


def test_unknown_target_lists_nothing_and_call_is_blocked(tmp_path: Path) -> None:
    """Acceptance (c): an unknown target fails closed, never falls back."""
    registry, dispatched = _registry(tmp_path)
    bogus = RunPolicy(mode="act", execution_target=cast(ExecutionTarget, "bogus"))
    assert registry.list_tool_specs(policy=bogus) == []
    blocked = registry.call(ToolRequest(name="workspace_read"), policy=bogus)
    assert not blocked.ok
    assert "EXECUTION_TARGET_BLOCK" in (blocked.error or "")
    assert dispatched == []


def test_manual_call_without_policy_cannot_bypass_target(tmp_path: Path) -> None:
    """Acceptance (d) negative: a direct registry call cannot smuggle a desktop tool.

    The global target is set to desktop first (the smuggling setup: stale
    legacy state). A sandbox run's target still refuses the desktop tool,
    and so does a direct call without any policy.
    """
    registry, dispatched = _registry(tmp_path)
    registry.set_execution_policy(mode="desktop")  # stale global desktop mode
    sandbox_policy = RunPolicy(mode="act", execution_target="sandbox")
    desktop_policy = RunPolicy(mode="desktop", execution_target="desktop")

    # The legacy listing never follows the global target.
    assert "desktop_only" not in _names(registry)
    assert "workspace_write" in _names(registry)

    # Direct call without a policy: refused, the global desktop mode does
    # not authorize it.
    smuggled = registry.call(ToolRequest(name="desktop_only"))
    assert not smuggled.ok
    assert "EXECUTION_TARGET_BLOCK" in (smuggled.error or "")
    assert dispatched == []

    # Same through a sandbox policy, and through a sandbox gateway.
    assert not registry.call(ToolRequest(name="desktop_only"), policy=sandbox_policy).ok
    gateway = ToolGateway(registry, run_policy=sandbox_policy)
    assert not gateway.call(ToolRequest(name="desktop_only")).ok
    assert dispatched == []

    # Control: an explicit desktop policy still allows the desktop tool.
    assert registry.call(ToolRequest(name="desktop_only"), policy=desktop_policy).ok
    assert dispatched == ["desktop_only"]


def test_plan_guard_narrows_listing_to_current_step(tmp_path: Path) -> None:
    """The listing mirrors the call-time plan guard: only the current step's tools."""
    registry, _ = _registry(tmp_path)
    plan = {
        "plan_id": "p1",
        "plan_hash": "h1",
        "steps": [{"step_id": "s1", "allowed_tool_kinds": ["workspace_read"]}],
    }
    task = {"plan_id": "p1", "plan_hash": "h1", "current_step_id": "s1"}
    guarded = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=task,
        enforce_plan_guard=True,
    )
    names = _names(registry, guarded)
    assert "workspace_read" in names
    assert "workspace_write" not in names
    assert "workspace_exec" not in names

    # A broken plan (plan_id mismatch) fails the listing closed: the guard
    # blocks every call then, so the model sees no tools.
    broken_task = {"plan_id": "p2", "plan_hash": "h1", "current_step_id": "s1"}
    broken = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=broken_task,
        enforce_plan_guard=True,
    )
    assert registry.list_tool_specs(policy=broken) == []

    # Control: the same plan without the guard lists the write tool.
    unguarded = RunPolicy(
        mode="act",
        execution_target="sandbox",
        active_plan=plan,
        active_task=task,
        enforce_plan_guard=False,
    )
    assert "workspace_write" in _names(registry, unguarded)


def test_legacy_listing_is_sandbox_catalog(tmp_path: Path) -> None:
    """The legacy no-policy path keeps working for plan drafting.

    It lists the full sandbox catalog (write included: drafting is
    planning, not execution) and never desktop tools, whatever the
    global target is set to.
    """
    registry, _ = _registry(tmp_path)
    for mode in ("ask", "auto", "desktop"):
        registry.set_execution_policy(mode=mode)
        names = _names(registry)
        assert names == {
            "workspace_write",
            "workspace_read",
            "workspace_exec",
            "both_targets",
        }
        assert "desktop_only" not in names
        assert "desktop_read" not in names

    # And a legacy no-policy call for a sandbox tool still works when the
    # global mode allows it (the coding-skill path).
    registry.set_execution_policy(mode="auto")
    assert registry.call(ToolRequest(name="workspace_write")).ok
