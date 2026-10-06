import asyncio
from pathlib import Path

import pytest

from core.agent import Agent
from core.approval_policy import ApprovalRequired
from core.auto_runtime import AutoContinuationUnavailable, AutoOrchestrator
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall
from shared.models import ToolResult


class ApprovalBatchBrain(Brain):
    supports_native_tools = True

    def __init__(self):
        self.initial_requests = 0
        self.history = []

    def generate(self, messages, config=None, tools=None):
        self.history.append(list(messages))
        if messages[-1].role == "tool":
            return LLMResult(text="completed")
        self.initial_requests += 1
        return LLMResult(
            text="batch",
            tool_calls=[
                ToolCall(id="before-approval", name="before_approval", arguments={}),
                ToolCall(id="pending-original", name="pending_network", arguments={}),
            ],
        )


@pytest.mark.parametrize(
    "changed_scope",
    [
        None,
        "session_id",
        "user_id",
        "runtime_mode",
        "runtime_workspace_root",
        "brain",
        "main_config",
        "expired",
    ],
)
def test_auto_resume_does_not_repeat_completed_tool(tmp_path: Path, changed_scope: str | None):
    brain = ApprovalBatchBrain()
    agent = Agent(
        brain=brain,
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.set_session_context("auto-session", set())
    agent.set_runtime_state(
        mode="auto", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    agent.apply_runtime_workspace_root(str(tmp_path))
    (tmp_path / "Makefile").write_text("check:\n\t@true\n")
    executions = []
    for name, risks in [("before_approval", []), ("pending_network", ["network"])]:
        agent.tool_registry.register(
            name,
            lambda request: executions.append(request.name)
            or ToolResult.success({"output": request.name}),
            enabled=True,
            capability="read",
            risk_classes=risks,
            description="Diagnostic read",
            parameters_schema={"type": "object"},
        )
    orchestrator = AutoOrchestrator(agent, workspace_root=tmp_path)
    try:
        with pytest.raises(ApprovalRequired) as stopped:
            orchestrator.run_v1("read both")
        assert stopped.value.continuation is not None
        assert executions == ["before_approval"]
        run_id = agent.last_auto_state["run_id"]
        paused = orchestrator._paused_runs[run_id]
        original_frame = paused.frame
        original_plan = original_frame.state["plan"]
        cancelled = asyncio.Event()
        cancelled.set()
        with pytest.raises(AutoContinuationUnavailable, match="not_started"):
            orchestrator.resume(run_id, cancellation_token=cancelled)
        assert orchestrator._paused_runs[run_id] is paused
        assert executions == ["before_approval"]
        if changed_scope == "expired":
            original_frame.started_monotonic -= original_frame.budgets.max_runtime_seconds + 1
            stopped_outcome = orchestrator.resume(run_id)
            assert stopped_outcome.stop_reason_code.value == "BUDGET_EXHAUSTED"
            assert executions == ["before_approval"]
            assert brain.initial_requests == 1
            assert run_id not in orchestrator._paused_runs
            return
        if changed_scope is not None:
            previous = getattr(agent, changed_scope)
            setattr(agent, changed_scope, "foreign-scope")
            with pytest.raises(AutoContinuationUnavailable):
                orchestrator.resume(run_id)
            assert orchestrator._paused_runs[run_id] is paused
            assert executions == ["before_approval"]
            setattr(agent, changed_scope, previous)
        # Core scope test; exact once authorization проверяется отдельно через HTTP.
        agent.set_session_context("auto-session", {"NETWORK_RISK"})
        orchestrator.resume(run_id)
        assert agent.last_auto_state["run_id"] == run_id
        assert agent.last_auto_state["plan"] == original_plan
        assert agent.last_auto_state["started_at"] == original_frame.started_at
        with pytest.raises(AutoContinuationUnavailable):
            orchestrator.resume(run_id)
        assert executions == ["before_approval", "pending_network"]
        assert brain.initial_requests == 1
        assert brain.history[-1][-1].tool_call_id == "pending-original"
    finally:
        agent.close()
