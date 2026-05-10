from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.mwv.coding_skill import build_workspace_gateway, work_change_from_request
from core.mwv.models import (
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from core.mwv.single_attempt import MWVSingleAttemptResult, MWVSingleAttemptRuntime
from core.mwv.verifier import VerifierRunner
from core.mwv.verifier_runtime import (
    VerifierRuntime,
    canonical_check_command,
    has_canonical_repo_verifier,
)
from shared.models import JSONValue, ToolRequest
from tools.workspace_tools import workspace_root_context

SingleAttemptToolRequestBuilder = Callable[[TaskPacket, RunContext], Sequence[ToolRequest]]


@dataclass(frozen=True)
class CodingTaskRuntime:
    workspace_root: Path
    verifier: VerifierRunner | None = None
    request_builder: SingleAttemptToolRequestBuilder | None = None

    def run(self, user_input: str) -> MWVSingleAttemptResult:
        messages = [MWVMessage(role="user", content=user_input)]
        context = RunContext(
            session_id="local",
            trace_id=str(uuid.uuid4()),
            workspace_root=str(self.workspace_root),
            safe_mode=True,
            max_retries=0,
            attempt=1,
        )

        runtime = MWVSingleAttemptRuntime(
            task_builder=self._task_builder(),
            worker=self._worker,
            verifier=self._verifier_runner(),
        )
        return runtime.run(messages, context)

    def _task_builder(
        self,
    ) -> Callable[[Sequence[MWVMessage], RunContext], TaskPacket]:
        def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
            verifier: dict[str, JSONValue] = {}
            if has_canonical_repo_verifier(self.workspace_root):
                verifier = {
                    "command": canonical_check_command(),
                    "cwd": ".",
                }
            return TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=messages[-1].content if messages else "",
                messages=list(messages),
                verifier=verifier,
                context={"workspace_root": str(self.workspace_root)},
            )

        return _build

    def _worker(self, task: TaskPacket, context: RunContext) -> WorkResult:
        if self.request_builder is None:
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary="no gateway tool requests configured",
            )

        requests = list(self.request_builder(task, context))
        if not requests:
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary="empty gateway tool request list",
            )

        gateway = build_workspace_gateway()
        changes: list[WorkChange] = []
        tool_summaries: list[str] = []
        with workspace_root_context(self.workspace_root):
            for request in requests:
                result = gateway.call(request)
                if not result.ok:
                    return WorkResult(
                        task_id=task.task_id,
                        status=WorkStatus.FAILURE,
                        summary=f"{request.name} failed: {result.error}",
                        changes=changes,
                        tool_summaries=tool_summaries,
                        tool_calls_used=len(tool_summaries) + 1,
                    )
                tool_summaries.append(f"{request.name}: ok")
                change = work_change_from_request(request)
                if change is not None:
                    changes.append(change)

        return WorkResult(
            task_id=task.task_id,
            status=WorkStatus.SUCCESS,
            summary=f"executed {len(requests)} gateway tool request(s)",
            changes=changes,
            tool_summaries=tool_summaries,
            tool_calls_used=len(requests),
        )

    def _verifier_runner(self) -> Callable[[TaskPacket, RunContext], VerificationResult]:
        script_path = self.workspace_root / "scripts" / "check.sh"
        runner = self.verifier or VerifierRunner(script_path=script_path)
        runtime = VerifierRuntime(runner=runner)

        def _run(task: TaskPacket, context: RunContext) -> VerificationResult:
            return runtime.run(task, context)

        return _run
