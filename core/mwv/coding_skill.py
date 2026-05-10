from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.mwv.manager import ManagerRuntime, MWVRunResult
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
from core.mwv.verifier import VerifierRunner
from core.mwv.verifier_summary import summarize_verifier_failure
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest
from tools.tool_descriptors import get_tool_metadata
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import (
    ApplyPatchTool,
    ReadFileTool,
    WriteFileTool,
    workspace_root_context,
)

ToolRequestBuilder = Callable[[TaskPacket, RunContext], Sequence[ToolRequest]]


@dataclass(frozen=True)
class CodingSkillResult:
    run_result: MWVRunResult
    report: str


@dataclass(frozen=True)
class CodingSkill:
    workspace_root: Path
    verifier: VerifierRunner | None = None
    max_retries: int = 2
    request_builder: ToolRequestBuilder | None = None

    def run(self, user_input: str) -> CodingSkillResult:
        messages = [MWVMessage(role="user", content=user_input)]
        context = RunContext(
            session_id="local",
            trace_id=str(uuid.uuid4()),
            workspace_root=str(self.workspace_root),
            safe_mode=True,
            max_retries=self.max_retries,
            attempt=1,
        )

        manager = ManagerRuntime(task_builder=self._task_builder())
        verifier = self.verifier or VerifierRunner()

        def _run_verifier(_: TaskPacket, __: RunContext) -> VerificationResult:
            return verifier.run()

        result = manager.run_flow(messages, context, worker=self._worker, verifier=_run_verifier)
        report = _build_report(result)
        return CodingSkillResult(run_result=result, report=report)

    def _task_builder(self) -> Callable[[Sequence[MWVMessage], RunContext], TaskPacket]:
        def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
            goal = messages[-1].content if messages else ""
            return TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=goal,
                messages=list(messages),
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
        with workspace_root_context(self.workspace_root):
            for request in requests:
                result = gateway.call(request)
                if not result.ok:
                    return WorkResult(
                        task_id=task.task_id,
                        status=WorkStatus.FAILURE,
                        summary=f"{request.name} failed: {result.error}",
                        changes=changes,
                    )
                change = work_change_from_request(request)
                if change is not None:
                    changes.append(change)

        return WorkResult(
            task_id=task.task_id,
            status=WorkStatus.SUCCESS,
            summary=f"executed {len(requests)} gateway tool request(s)",
            changes=changes,
        )


def build_workspace_gateway() -> ToolGateway:
    registry = ToolRegistry()
    read_metadata = get_tool_metadata("workspace_read")
    registry.register(
        "workspace_read",
        ReadFileTool(),
        enabled=True,
        capability="read",
        description=read_metadata.description,
        parameters_schema=read_metadata.parameters_schema,
    )
    write_metadata = get_tool_metadata("workspace_write")
    registry.register(
        "workspace_write",
        WriteFileTool(),
        enabled=True,
        capability="write",
        description=write_metadata.description,
        parameters_schema=write_metadata.parameters_schema,
    )
    patch_metadata = get_tool_metadata("workspace_patch")
    registry.register(
        "workspace_patch",
        ApplyPatchTool(),
        enabled=True,
        capability="write",
        description=patch_metadata.description,
        parameters_schema=patch_metadata.parameters_schema,
    )
    return ToolGateway(registry)


def work_change_from_request(request: ToolRequest) -> WorkChange | None:
    if request.name not in {"workspace_write", "workspace_patch"}:
        return None
    raw_path = request.args.get("path")
    path = str(raw_path) if isinstance(raw_path, str) and raw_path else request.name
    return WorkChange(
        path=path,
        change_type=ChangeType.UPDATE,
        summary=f"{request.name} via gateway",
    )


def _build_report(result: MWVRunResult) -> str:
    if result.work_result.status != WorkStatus.SUCCESS:
        return f"Worker failed: {result.work_result.summary}"
    if result.verification_result.status == VerificationStatus.PASSED:
        return "Verifier OK"
    summary = summarize_verifier_failure(result.verification_result)
    return f"Verifier failed: {summary}. See stdout/stderr for details."
