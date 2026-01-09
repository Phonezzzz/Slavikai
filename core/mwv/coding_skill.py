from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.mwv.manager import ManagerRuntime, MWVRunResult, summarize_verifier_failure
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

_TARGET_RE = re.compile(r"\b(?:file|файл)\b\s+([^\s]+)", re.IGNORECASE)

_COMMENT_PREFIXES = {
    ".py": "#",
    ".sh": "#",
    ".md": "<!--",
    ".txt": "#",
    ".toml": "#",
    ".yaml": "#",
    ".yml": "#",
}


@dataclass(frozen=True)
class CodingSkillResult:
    run_result: MWVRunResult
    report: str


@dataclass(frozen=True)
class CodingSkill:
    workspace_root: Path
    verifier: VerifierRunner | None = None
    max_retries: int = 2
    change_text: str = "mwv change"
    retry_text: str = "mwv min-diff change"

    def run(self, user_input: str) -> CodingSkillResult:
        target_path = _extract_target_path(user_input)
        if target_path is None:
            raise ValueError("Target file not found in request.")

        normalized = _resolve_workspace_path(self.workspace_root, target_path)
        messages = [MWVMessage(role="user", content=user_input)]
        context = RunContext(
            session_id="local",
            trace_id=str(uuid.uuid4()),
            workspace_root=str(self.workspace_root),
            safe_mode=True,
            max_retries=self.max_retries,
            attempt=1,
        )

        manager = ManagerRuntime(task_builder=self._task_builder(normalized))
        verifier = self.verifier or VerifierRunner()

        def _run_verifier(_: RunContext) -> VerificationResult:
            return verifier.run()

        result = manager.run_flow(messages, context, worker=self._worker, verifier=_run_verifier)
        report = _build_report(result)
        return CodingSkillResult(run_result=result, report=report)

    def _task_builder(
        self,
        path: Path,
    ) -> Callable[[Sequence[MWVMessage], RunContext], TaskPacket]:
        def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
            _ = messages
            return TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=f"Update {path.name}",
                context={"target_path": str(path)},
            )

        return _build

    def _worker(self, task: TaskPacket, context: RunContext) -> WorkResult:
        target_raw = task.context.get("target_path")
        if not isinstance(target_raw, str) or not target_raw:
            return WorkResult(task_id=task.task_id, status=WorkStatus.FAILURE, summary="no target")

        path = Path(target_raw)
        if not path.exists():
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary=f"file missing: {path.name}",
            )

        original = path.read_text(encoding="utf-8")
        updated = _apply_change(
            original,
            path,
            attempt=context.attempt,
            change_text=self.change_text,
            retry_text=self.retry_text,
        )
        if updated == original:
            status = WorkStatus.SUCCESS if context.attempt > 1 else WorkStatus.FAILURE
            return WorkResult(
                task_id=task.task_id,
                status=status,
                summary=f"no changes for {path.name}",
            )

        path.write_text(updated, encoding="utf-8")
        summary = f"updated {path.name}"
        changes = [
            WorkChange(path=str(path), change_type=ChangeType.UPDATE, summary=summary),
        ]
        return WorkResult(
            task_id=task.task_id,
            status=WorkStatus.SUCCESS,
            summary=summary,
            changes=changes,
        )


def _extract_target_path(text: str) -> str | None:
    match = _TARGET_RE.search(text)
    if match is None:
        return None
    candidate = match.group(1).strip().strip("\"'`").strip(".,;:")
    return candidate or None


def _resolve_workspace_path(root: Path, raw_path: str) -> Path:
    root = root.resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Path outside workspace.") from exc
    return candidate


def _comment_line(path: Path, text: str) -> str:
    prefix = _COMMENT_PREFIXES.get(path.suffix.lower(), "#")
    if prefix == "<!--":
        return f"<!-- {text} -->"
    return f"{prefix} {text}"


def _apply_change(
    content: str,
    path: Path,
    *,
    attempt: int,
    change_text: str,
    retry_text: str,
) -> str:
    base_line = _comment_line(path, change_text)
    retry_line = _comment_line(path, retry_text)
    if attempt <= 1:
        if base_line in content:
            return content
        return _append_line(content, base_line)

    if base_line in content:
        return content.replace(base_line, retry_line, 1)
    if retry_line in content:
        return content
    return _append_line(content, retry_line)


def _append_line(content: str, line: str) -> str:
    stripped = content.rstrip("\n")
    return f"{stripped}\n{line}\n"


def _build_report(result: MWVRunResult) -> str:
    if result.work_result.status != WorkStatus.SUCCESS:
        return f"Worker failed: {result.work_result.summary}"
    if result.verification_result.status == VerificationStatus.PASSED:
        return "Verifier OK"
    summary = summarize_verifier_failure(result.verification_result)
    return f"Verifier failed: {summary}. See stdout/stderr for details."
