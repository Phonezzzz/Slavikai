from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from core.mwv.models import (
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)

TaskBuilder = Callable[[Sequence[MWVMessage], RunContext], TaskPacket]
WorkRunner = Callable[[TaskPacket, RunContext], WorkResult]
VerifierRunner = Callable[[RunContext], VerificationResult]


@dataclass(frozen=True)
class MWVDiagnostics:
    summary: str
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str


@dataclass(frozen=True)
class MWVReport:
    changed_files: list[str]
    verifier_status: VerificationStatus
    verifier_duration_ms: int
    next_steps: list[str]
    diagnostics: MWVDiagnostics | None = None

    def render(self) -> str:
        lines = ["Changed files:"]
        if self.changed_files:
            lines.extend([f"- {path}" for path in self.changed_files])
        else:
            lines.append("- none")
        lines.append(
            f"Verifier: {_status_label(self.verifier_status)} ({self.verifier_duration_ms} ms)"
        )
        if self.diagnostics is not None:
            lines.append("Diagnostics:")
            lines.append(f"- Reason: {self.diagnostics.summary}")
            lines.append(f"- Command: {' '.join(self.diagnostics.command)}")
        lines.append("Next steps:")
        lines.extend([f"- {step}" for step in self.next_steps])
        return "\n".join(lines)


@dataclass(frozen=True)
class MWVSingleAttemptResult:
    task: TaskPacket
    work_result: WorkResult
    verification_result: VerificationResult
    report: MWVReport
    report_text: str
    attempt: int


@dataclass(frozen=True)
class MWVSingleAttemptRuntime:
    task_builder: TaskBuilder
    worker: WorkRunner
    verifier: VerifierRunner

    def run(
        self,
        messages: Sequence[MWVMessage],
        context: RunContext,
    ) -> MWVSingleAttemptResult:
        task = self.task_builder(messages, context)
        attempt_context = replace(context, attempt=1, max_retries=0)
        work_result = self.worker(task, attempt_context)
        verification_result = self.verifier(attempt_context)
        report = build_mwv_report(work_result, verification_result)
        return MWVSingleAttemptResult(
            task=task,
            work_result=work_result,
            verification_result=verification_result,
            report=report,
            report_text=report.render(),
            attempt=attempt_context.attempt,
        )


def build_mwv_report(
    work_result: WorkResult,
    verification_result: VerificationResult,
) -> MWVReport:
    changed_files = _collect_changed_files(work_result)
    diagnostics = None
    if verification_result.status != VerificationStatus.PASSED:
        diagnostics = _build_diagnostics(verification_result)
    return MWVReport(
        changed_files=changed_files,
        verifier_status=verification_result.status,
        verifier_duration_ms=verification_result.duration_ms,
        next_steps=_next_steps(work_result, verification_result),
        diagnostics=diagnostics,
    )


def _collect_changed_files(work_result: WorkResult) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for change in work_result.changes:
        if change.path in seen:
            continue
        seen.add(change.path)
        ordered.append(change.path)
    return ordered


def _build_diagnostics(result: VerificationResult) -> MWVDiagnostics:
    return MWVDiagnostics(
        summary=_summarize_verifier_failure(result),
        command=list(result.command),
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _summarize_verifier_failure(result: VerificationResult) -> str:
    if result.status == VerificationStatus.ERROR and result.error:
        return result.error
    text = (result.stderr or result.stdout or "").strip()
    if text:
        return text.splitlines()[0][:200]
    if result.exit_code is None:
        return "verifier_failed"
    return f"exit_code={result.exit_code}"


def _next_steps(
    work_result: WorkResult,
    verification_result: VerificationResult,
) -> list[str]:
    if verification_result.status == VerificationStatus.PASSED:
        if work_result.status == WorkStatus.SUCCESS:
            return ["Review the changes", "Continue with the next task"]
        return ["Review the task inputs", "Retry with clearer instructions"]
    return ["Inspect verifier output", "Fix failing checks and retry"]


def _status_label(status: VerificationStatus) -> str:
    if status == VerificationStatus.PASSED:
        return "PASS"
    if status == VerificationStatus.FAILED:
        return "FAIL"
    return "ERROR"
