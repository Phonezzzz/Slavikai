from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from core.mwv.manager import ManagerRuntime
from core.mwv.models import (
    ChangeType,
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)


def _build_task(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
    goal = messages[-1].content if messages else ""
    return TaskPacket(
        task_id="task-1",
        session_id=context.session_id,
        trace_id=context.trace_id,
        goal=goal,
        messages=list(messages),
    )


def test_mwv_e2e_happy_path_with_workspace_change(tmp_path: Path) -> None:
    workspace = tmp_path / "sandbox" / "project"
    workspace.mkdir(parents=True)
    target = workspace / "notes.txt"
    target.write_text("start\n", encoding="utf-8")

    manager = ManagerRuntime(task_builder=_build_task)
    context = RunContext(
        session_id="s1",
        trace_id="trace-1",
        workspace_root=str(workspace),
        safe_mode=True,
        max_retries=2,
        attempt=1,
    )
    messages = [MWVMessage(role="user", content="добавь запись в файл notes.txt")]

    def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
        _ = (task, run_context)
        target.write_text("start\nmwv change\n", encoding="utf-8")
        return WorkResult(
            task_id="task-1",
            status=WorkStatus.SUCCESS,
            summary="updated workspace file",
            changes=[
                WorkChange(
                    path="notes.txt",
                    change_type=ChangeType.UPDATE,
                    summary="+1/-0",
                )
            ],
        )

    def _verifier(run_context: RunContext) -> VerificationResult:
        _ = run_context
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
    assert result.attempt == 1
    assert result.verification_result.status == VerificationStatus.PASSED
    assert result.work_result.status == WorkStatus.SUCCESS
    assert result.work_result.changes == [
        WorkChange(path="notes.txt", change_type=ChangeType.UPDATE, summary="+1/-0")
    ]
    assert "mwv change" in target.read_text(encoding="utf-8")
