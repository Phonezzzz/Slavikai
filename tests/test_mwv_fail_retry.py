from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.agent import Agent
from core.mwv.manager import ManagerRuntime
from core.mwv.models import (
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        _ = (messages, config)
        return LLMResult(text="ok")


def _build_task(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
    goal = messages[-1].content if messages else ""
    return TaskPacket(
        task_id="task-1",
        session_id=context.session_id,
        trace_id=context.trace_id,
        goal=goal,
        messages=list(messages),
    )


def _make_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def test_mwv_fail_retry_then_success() -> None:
    manager = ManagerRuntime(task_builder=_build_task)
    context = RunContext(
        session_id="s1",
        trace_id="trace-1",
        workspace_root="/tmp",
        safe_mode=True,
        max_retries=2,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="почини тесты")]
    seen_constraints: list[list[str]] = []

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        _ = run_context
        seen_constraints.append(list(task.constraints))
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")

    verifier_results = [
        VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=1,
            stdout="",
            stderr="tests failed",
            duration_seconds=0.1,
            error=None,
        ),
        VerificationResult(
            status=VerificationStatus.PASSED,
            command=["check"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            error=None,
        ),
    ]

    def _verifier(run_context: RunContext) -> VerificationResult:
        _ = run_context
        return verifier_results.pop(0)

    result = manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
    assert result.verification_result.status == VerificationStatus.PASSED
    assert result.attempt == 2
    assert len(seen_constraints) == 2
    assert any("минимально по логам" in item for item in seen_constraints[1])


def test_mwv_fail_after_max_retries_returns_stop_response(tmp_path: Path) -> None:
    manager = ManagerRuntime(task_builder=_build_task)
    context = RunContext(
        session_id="s1",
        trace_id="trace-stop",
        workspace_root="/tmp",
        safe_mode=True,
        max_retries=1,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="почини тесты")]

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        _ = run_context
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")

    def _verifier(run_context: RunContext) -> VerificationResult:
        _ = run_context
        return VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=1,
            stdout="",
            stderr="tests failed",
            duration_seconds=0.1,
            error=None,
        )

    result = manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
    assert result.retry_decision is not None
    assert result.retry_decision.reason == "retry_limit_reached"

    agent = _make_agent(tmp_path)
    response = agent._format_mwv_response(result)
    assert "Что случилось" in response
    assert "Проверки не прошли" in response
    report = extract_report_block(response)
    assert report["route"] == "mwv"
    assert report["stop_reason_code"] == "VERIFIER_FAILED"
    attempts = report["attempts"]
    assert isinstance(attempts, dict)
    assert attempts["current"] == 2
    assert attempts["max"] == 2
