from __future__ import annotations

from pathlib import Path

import pytest

import core.agent as agent_module
from core.agent import Agent
from core.mwv.manager import MWVRunResult
from core.mwv.models import (
    ChangeType,
    MWVMessage,
    RetryDecision,
    RetryPolicy,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from core.mwv.routing import RouteDecision
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, PlanStep, PlanStepStatus, TaskPlan, WorkspaceDiffEntry
from tools.workspace_tools import WORKSPACE_ROOT


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(brain=DummyBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))


def test_mwv_context_and_task_builder(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    agent.session_id = "session-1"
    agent.tools_enabled["safe_mode"] = True
    agent.approved_categories = {"FS_DELETE_OVERWRITE"}

    context = agent._build_mwv_context()
    assert context.session_id == "session-1"
    assert context.safe_mode is True
    assert context.workspace_root == str(WORKSPACE_ROOT)
    assert "FS_DELETE_OVERWRITE" in context.approved_categories

    decision = RouteDecision(route="mwv", reason="trigger:code_change", risk_flags=["code_change"])
    builder = agent._mwv_task_builder(decision)
    messages = [MWVMessage(role="user", content="fix bug")]
    task = builder(messages, context)
    assert task.goal == "fix bug"
    assert task.context["route_reason"] == "trigger:code_change"
    assert "code_change" in task.context["risk_flags"]


def test_mwv_goal_and_changes(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    task = TaskPacket(
        task_id="t",
        session_id="s",
        trace_id="trace",
        goal="base goal",
        constraints=["constraint A", "constraint B"],
    )
    goal = agent._build_mwv_goal(task)
    assert "Ограничения" in goal
    assert "constraint A" in goal

    diffs = [
        WorkspaceDiffEntry(path="a.txt", added=1, removed=0, diff=""),
        WorkspaceDiffEntry(path="b.txt", added=0, removed=2, diff=""),
        WorkspaceDiffEntry(path="c.txt", added=3, removed=1, diff=""),
    ]
    changes = agent._mwv_changes_from_diffs(diffs)
    assert changes == [
        WorkChange(path="a.txt", change_type=ChangeType.CREATE, summary="+1/-0"),
        WorkChange(path="b.txt", change_type=ChangeType.DELETE, summary="+0/-2"),
        WorkChange(path="c.txt", change_type=ChangeType.UPDATE, summary="+3/-1"),
    ]


def test_mwv_diagnostics_and_formatting(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    plan = TaskPlan(
        goal="g",
        steps=[
            PlanStep(description="ok", status=PlanStepStatus.DONE, result="done"),
            PlanStep(description="fail", status=PlanStepStatus.ERROR, result="boom"),
        ],
    )
    diagnostics = agent._build_mwv_diagnostics(plan)
    assert diagnostics["steps_total"] == 2
    assert len(diagnostics["step_errors"]) == 1

    task = TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")
    work_ok = WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="summary")
    verify_ok = VerificationResult(
        status=VerificationStatus.PASSED,
        command=["check"],
        exit_code=0,
        stdout="ok",
        stderr="",
        duration_seconds=0.1,
        error=None,
    )
    ok_result = MWVRunResult(
        task=task,
        work_result=work_ok,
        verification_result=verify_ok,
        attempt=1,
        max_attempts=3,
        retry_decision=None,
    )
    ok_response = agent._format_mwv_response(ok_result)
    assert "Итог: проверки пройдены" in ok_response
    assert "Попытка: 1/3" in ok_response
    assert "Verifier: PASS" in ok_response
    assert "Изменения:" in ok_response

    work_fail = WorkResult(
        task_id="t",
        status=WorkStatus.FAILURE,
        summary="",
        diagnostics={"step_errors": [{"description": "step", "result": "bad"}]},
    )
    fail_result = MWVRunResult(
        task=task,
        work_result=work_fail,
        verification_result=verify_ok,
        attempt=1,
        max_attempts=3,
        retry_decision=None,
    )
    fail_response = agent._format_mwv_response(fail_result)
    assert "Что случилось" in fail_response
    assert "ошибка выполнения" in fail_response.lower()
    assert "trace_id=trace" in fail_response

    verify_error = VerificationResult(
        status=VerificationStatus.ERROR,
        command=["check"],
        exit_code=None,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        error="boom",
    )
    error_result = MWVRunResult(
        task=task,
        work_result=work_ok,
        verification_result=verify_error,
        attempt=1,
        max_attempts=3,
        retry_decision=None,
    )
    error_response = agent._format_mwv_response(error_result)
    assert "Что случилось" in error_response
    assert "Ошибка проверки" in error_response
    assert "trace_id=trace" in error_response

    retry_decision = RetryDecision(
        policy=RetryPolicy.LIMITED,
        allow_retry=False,
        reason="retry_limit_reached",
        attempt=3,
        max_retries=2,
    )
    verify_failed = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )
    retry_result = MWVRunResult(
        task=task,
        work_result=work_ok,
        verification_result=verify_failed,
        attempt=3,
        max_attempts=3,
        retry_decision=retry_decision,
    )
    response = agent._format_mwv_response(retry_result)
    assert "Что случилось" in response
    assert "Проверки не прошли" in response
    assert "trace_id=trace" in response


def test_mwv_flow_runs_through_worker_and_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent = _make_agent(tmp_path)

    def _worker(task: TaskPacket, _context: RunContext) -> WorkResult:
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")

    class DummyVerifierRuntime:
        def run(self, _context: RunContext) -> VerificationResult:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                command=["check"],
                exit_code=0,
                stdout="ok",
                stderr="",
                duration_seconds=0.1,
                error=None,
            )

    monkeypatch.setattr(agent, "_mwv_worker_runner", _worker)
    monkeypatch.setattr(agent_module, "VerifierRuntime", DummyVerifierRuntime)

    decision = RouteDecision(route="mwv", reason="trigger:tools", risk_flags=["tools"])
    response = agent._run_mwv_flow(
        [LLMMessage(role="user", content="fix code")],
        raw_input="fix code",
        decision=decision,
        record_in_history=False,
    )
    assert "Итог: проверки пройдены" in response


def test_mwv_default_next_steps_on_verifier_fail(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    task = TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")
    work_ok = WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="summary")
    verify_failed = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )
    result = MWVRunResult(
        task=task,
        work_result=work_ok,
        verification_result=verify_failed,
        attempt=1,
        max_attempts=3,
        retry_decision=None,
    )
    response = agent._format_mwv_response(result)
    assert "Что делать дальше" in response
    assert "trace_id=trace" in response


def test_mwv_verifier_note_uses_exit_code(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    note = agent._mwv_verifier_note(
        VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=2,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error=None,
        )
    )
    assert note == "exit_code=2"
