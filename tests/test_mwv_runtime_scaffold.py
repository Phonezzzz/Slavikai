from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core.mwv.manager import (
    ManagerRuntime,
    build_retry_task,
    decide_retry,
    summarize_verifier_failure,
)
from core.mwv.models import (
    MWVMessage,
    RetryDecision,
    RetryPolicy,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from core.mwv.routing import RouteDecision
from core.mwv.verifier import VerifierRunner
from core.mwv.verifier_runtime import VerifierRuntime, _default_runner
from core.mwv.worker import WorkerRuntime
from shared.models import JSONValue


def test_manager_runtime_uses_builder() -> None:
    def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        return TaskPacket(
            task_id="t1",
            session_id=context.session_id,
            trace_id=context.trace_id,
            goal=messages[-1].content,
            messages=list(messages),
        )

    manager = ManagerRuntime(task_builder=_build)
    context = RunContext(
        session_id="s",
        trace_id="t",
        workspace_root="/tmp",
        safe_mode=True,
    )
    packet = manager.build_task_packet([MWVMessage(role="user", content="go")], context)
    assert packet.goal == "go"
    assert packet.session_id == "s"


def test_manager_runtime_uses_route_classifier() -> None:
    def _classifier(
        _messages: Sequence[MWVMessage],
        _input: str,
        _context: dict[str, JSONValue] | None,
    ) -> RouteDecision:
        return RouteDecision(route="mwv", reason="test", risk_flags=["tools"])

    def _builder(_messages: Sequence[MWVMessage], _context: RunContext) -> TaskPacket:
        return TaskPacket(task_id="t", session_id="s", trace_id="t", goal="g")

    manager = ManagerRuntime(task_builder=_builder, route_classifier=_classifier)
    decision = manager.decide_route([], "x", {})
    assert decision.route == "mwv"
    assert decision.risk_flags == ["tools"]


def test_worker_runtime_delegates() -> None:
    def _runner(task: TaskPacket, _context: RunContext) -> WorkResult:
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")

    worker = WorkerRuntime(runner=_runner)
    result = worker.run(
        TaskPacket(task_id="t", session_id="s", trace_id="t", goal="g"),
        RunContext(session_id="s", trace_id="t", workspace_root="/tmp", safe_mode=True),
    )
    assert result.status == WorkStatus.SUCCESS


@dataclass(frozen=True)
class DummyVerifierRunner:
    result: VerificationResult

    def run(self) -> VerificationResult:
        return self.result


def test_verifier_runtime_returns_runner_result() -> None:
    expected = VerificationResult(
        status=VerificationStatus.PASSED,
        command=["check"],
        exit_code=0,
        stdout="ok",
        stderr="",
        duration_seconds=0.1,
        error=None,
    )
    runtime = VerifierRuntime(runner=DummyVerifierRunner(expected))
    result = runtime.run(
        RunContext(session_id="s", trace_id="t", workspace_root="/tmp", safe_mode=True)
    )
    assert result == expected


def test_default_verifier_runner_factory() -> None:
    runner = _default_runner()
    assert isinstance(runner, VerifierRunner)


def test_retry_decision_contract() -> None:
    decision = RetryDecision(
        policy=RetryPolicy.LIMITED,
        allow_retry=True,
        reason="failed",
        attempt=1,
        max_retries=2,
    )
    assert decision.policy == RetryPolicy.LIMITED
    assert decision.allow_retry is True
    assert decision.llm_hint is None


def test_retry_decision_stops_on_limit() -> None:
    decision = decide_retry(
        work_result=WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="ok"),
        verification_result=VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=1,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error=None,
        ),
        attempt=2,
        max_attempts=2,
    )
    assert decision.allow_retry is False
    assert decision.reason == "retry_limit_reached"


def test_retry_decision_blocks_on_worker_failure() -> None:
    decision = decide_retry(
        work_result=WorkResult(task_id="t", status=WorkStatus.FAILURE, summary="fail"),
        verification_result=VerificationResult(
            status=VerificationStatus.PASSED,
            command=["check"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            error=None,
        ),
        attempt=1,
        max_attempts=3,
    )
    assert decision.allow_retry is False
    assert decision.reason == "worker_failed"


def test_retry_decision_blocks_on_verifier_error() -> None:
    decision = decide_retry(
        work_result=WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="ok"),
        verification_result=VerificationResult(
            status=VerificationStatus.ERROR,
            command=["check"],
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error="boom",
        ),
        attempt=1,
        max_attempts=3,
    )
    assert decision.allow_retry is False
    assert decision.reason == "verifier_error"


def test_build_retry_task_uses_stricter_constraint_on_second_retry() -> None:
    task = TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")
    decision = RetryDecision(
        policy=RetryPolicy.LIMITED,
        allow_retry=True,
        reason="verifier_failed",
        attempt=2,
        max_retries=2,
    )
    failure = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )
    updated = build_retry_task(task, failure, decision)
    assert any("максимально минимальный diff" in item for item in updated.constraints)


def test_build_retry_task_appends_llm_hint() -> None:
    task = TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")
    decision = RetryDecision(
        policy=RetryPolicy.LIMITED,
        allow_retry=True,
        reason="verifier_failed",
        attempt=1,
        max_retries=2,
        llm_hint="Проверь импорт pathlib.",
    )
    failure = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )
    updated = build_retry_task(task, failure, decision)
    assert any("LLM-уточнение: Проверь импорт pathlib." == item for item in updated.constraints)


def test_summarize_verifier_failure_handles_empty_output() -> None:
    result = summarize_verifier_failure(
        VerificationResult(
            status=VerificationStatus.FAILED,
            command=["check"],
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error=None,
        )
    )
    assert result == "Неизвестная ошибка проверки"


def test_summarize_verifier_failure_uses_error_message() -> None:
    result = summarize_verifier_failure(
        VerificationResult(
            status=VerificationStatus.ERROR,
            command=["check"],
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=0.1,
            error=None,
        )
    )
    assert result == "Ошибка верификации"


def test_summarize_verifier_failure_uses_exit_code() -> None:
    result = summarize_verifier_failure(
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
    assert result == "Код возврата: 2"


def test_run_flow_stops_on_worker_failure() -> None:
    def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        return TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")

    manager = ManagerRuntime(task_builder=_build)
    context = RunContext(
        session_id="s",
        trace_id="t",
        workspace_root="/tmp",
        safe_mode=True,
        max_retries=2,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="go")]

    def _worker(_task: TaskPacket, _context: RunContext) -> WorkResult:
        return WorkResult(task_id="t", status=WorkStatus.FAILURE, summary="fail")

    def _verifier(_context: RunContext) -> VerificationResult:
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=["check"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
            error=None,
        )

    result = manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
    assert result.work_result.status == WorkStatus.FAILURE
    assert result.retry_decision is not None
    assert result.retry_decision.allow_retry is False


def test_run_flow_retry_uses_retry_hint_generator() -> None:
    def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        return TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")

    manager = ManagerRuntime(
        task_builder=_build,
        retry_hint_generator=lambda task, result, decision: "Исправь импорт json",
    )
    context = RunContext(
        session_id="s",
        trace_id="t",
        workspace_root="/tmp",
        safe_mode=True,
        max_retries=2,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="go")]
    seen_constraints: list[list[str]] = []

    def _worker(task: TaskPacket, _context: RunContext) -> WorkResult:
        seen_constraints.append(list(task.constraints))
        return WorkResult(task_id="t", status=WorkStatus.SUCCESS, summary="ok")

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

    def _verifier(_context: RunContext) -> VerificationResult:
        return verifier_results.pop(0)

    result = manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
    assert result.verification_result.status == VerificationStatus.PASSED
    assert result.attempt == 2
    assert len(seen_constraints) == 2
    assert any("LLM-уточнение: Исправь импорт json" in item for item in seen_constraints[1])
