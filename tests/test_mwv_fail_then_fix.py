from __future__ import annotations

from collections.abc import Sequence

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


def test_mwv_fail_then_fix() -> None:
    def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        goal = messages[-1].content if messages else ""
        return TaskPacket(
            task_id="t1",
            session_id=context.session_id,
            trace_id=context.trace_id,
            goal=goal,
        )

    manager = ManagerRuntime(task_builder=_build)
    context = RunContext(
        session_id="s",
        trace_id="t",
        workspace_root="/tmp",
        safe_mode=True,
        max_retries=2,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="исправь тесты")]
    seen_tasks: list[TaskPacket] = []

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        seen_tasks.append(task)
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
        return verifier_results.pop(0)

    result = manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
    assert result.verification_result.status == VerificationStatus.PASSED
    assert result.attempt == 2
    assert len(seen_tasks) == 2
    assert any("минимальные изменения" in item for item in seen_tasks[1].constraints)
