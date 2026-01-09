from __future__ import annotations

import re
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.mwv.models import (
    ChangeType,
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
from core.mwv.verifier_runtime import VerifierRuntime

_TARGET_RE = re.compile(r"\b(?:file|\u0444\u0430\u0439\u043b)\b\s+([^\s]+)", re.IGNORECASE)

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
class CodingTaskRuntime:
    workspace_root: Path
    verifier: VerifierRunner | None = None
    change_text: str = "mwv change"

    def run(self, user_input: str) -> MWVSingleAttemptResult:
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
            max_retries=0,
            attempt=1,
        )

        runtime = MWVSingleAttemptRuntime(
            task_builder=self._task_builder(normalized),
            worker=self._worker,
            verifier=self._verifier_runner(),
        )
        return runtime.run(messages, context)

    def _task_builder(
        self,
        path: Path,
    ) -> Callable[[Sequence[MWVMessage], RunContext], TaskPacket]:
        def _build(_messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
            return TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=f"Update {path.name}",
                context={"target_path": str(path)},
            )

        return _build

    def _worker(self, task: TaskPacket, _context: RunContext) -> WorkResult:
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
        updated = _apply_change(original, path, self.change_text)
        if updated == original:
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
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

    def _verifier_runner(self) -> Callable[[RunContext], VerificationResult]:
        script_path = self.workspace_root / "scripts" / "check.sh"
        runner = self.verifier or VerifierRunner(script_path=script_path)
        runtime = VerifierRuntime(runner=runner)
        return runtime.run


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


def _apply_change(content: str, path: Path, change_text: str) -> str:
    line = _comment_line(path, change_text)
    if line in content:
        return content
    return _append_line(content, line)


def _append_line(content: str, line: str) -> str:
    stripped = content.rstrip("\n")
    return f"{stripped}\n{line}\n"
