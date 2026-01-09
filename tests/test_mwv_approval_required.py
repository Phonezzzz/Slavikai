from __future__ import annotations

from collections.abc import Sequence

import pytest

from core.approval_policy import ApprovalPrompt, ApprovalRequest, ApprovalRequired
from core.mwv.manager import ManagerRuntime
from core.mwv.models import (
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    WorkResult,
)


def test_mwv_approval_required_is_propagated() -> None:
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
    )
    messages = [MWVMessage(role="user", content="удали файл")]

    approval_request = ApprovalRequest(
        category="FS_DELETE_OVERWRITE",
        required_categories=["FS_DELETE_OVERWRITE"],
        prompt=ApprovalPrompt(
            what="Удалить файл",
            why="Запрос пользователя",
            risk="Можно потерять данные.",
            changes=["Удаление файла"],
        ),
        tool="fs",
        details={},
        session_id=context.session_id,
    )

    def _worker(_task: TaskPacket, _context: RunContext) -> WorkResult:
        raise ApprovalRequired(approval_request)

    def _verifier(_context: RunContext) -> VerificationResult:
        raise AssertionError("Verifier should not run when approval is required.")

    with pytest.raises(ApprovalRequired):
        manager.run_flow(messages, context, worker=_worker, verifier=_verifier)
