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


def test_mwv_happy_path() -> None:
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

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        return WorkResult(task_id="t1", status=WorkStatus.SUCCESS, summary="ok")

    def _verifier(run_context: RunContext) -> VerificationResult:
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
    assert result.verification_result.status == VerificationStatus.PASSED
    assert result.work_result.status == WorkStatus.SUCCESS
    assert result.attempt == 1
