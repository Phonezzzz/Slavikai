from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.mwv.manager import ManagerRuntime
from core.mwv.models import (
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from core.mwv.verifier import VerifierRunner


def test_mwv_stops_on_verifier_failure(tmp_path: Path) -> None:
    script_path = tmp_path / "check.sh"
    script_path.write_text(
        '#!/usr/bin/env bash\necho "tests failed" 1>&2\nexit 1\n',
        encoding="utf-8",
    )

    def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
        goal = messages[-1].content if messages else ""
        return TaskPacket(
            task_id="t1",
            session_id=context.session_id,
            trace_id=context.trace_id,
            goal=goal,
        )

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        return WorkResult(task_id=task.task_id, status=WorkStatus.SUCCESS, summary="ok")

    runner = VerifierRunner(script_path=script_path)
    manager = ManagerRuntime(task_builder=_build)
    context = RunContext(
        session_id="s",
        trace_id="t",
        workspace_root=str(tmp_path),
        safe_mode=True,
        max_retries=0,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="исправь тесты")]

    result = manager.run_flow(messages, context, worker=_worker, verifier=lambda ctx: runner.run())

    assert result.verification_result.status == VerificationStatus.FAILED
    assert result.retry_decision is not None
    assert result.retry_decision.allow_retry is False
    assert "tests failed" in result.verification_result.stderr
