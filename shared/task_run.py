from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class RunState(StrEnum):
    ADMITTED = "admitted"
    RUNNING = "running"
    RESULT_SUBMITTED = "result_submitted"
    RECOVERY_REQUIRED = "recovery_required"
    ABORTED = "aborted"


class TransitionAuthority(StrEnum):
    RUNTIME = "runtime"
    RECOVERY = "recovery"
    USER = "user"


@dataclass(frozen=True)
class RunScope:
    principal_id: str
    session_id: str

    def __post_init__(self) -> None:
        for value in (self.principal_id, self.session_id):
            if not value or value != value.strip():
                raise ValueError("Scope identities must be non-empty and normalized")


@dataclass(frozen=True)
class TaskRun:
    task_id: str
    task_revision: int
    task_run_id: str
    attempt_id: str
    scope: RunScope
    state: RunState
    version: int
    created_at: str
    terminal_at: str | None


@dataclass(frozen=True)
class RunTransition:
    version: int
    prior_state: RunState | None
    state: RunState
    authority: TransitionAuthority
    reason_code: str
    occurred_at: str
    submission_id: str | None = None


class ProducerKind(StrEnum):
    MODEL = "model"
    NATIVE_LOOP = "native_loop"
    AUTO = "auto"
    MWV = "mwv"
    DESKTOP = "desktop"
    LOCAL = "local"


@dataclass(frozen=True)
class ProducerReport:
    """Untrusted producer text, not canonical tool evidence or acceptance."""

    kind: ProducerKind
    text: str
    failure_code: str | None = None
    failure_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProducerKind) or not isinstance(self.text, str):
            raise ValueError("Typed producer report required")
        if (self.failure_code is None) != (self.failure_message is None):
            raise ValueError("Incomplete response failure")
        if self.failure_code is not None and (
            not isinstance(self.failure_code, str) or not isinstance(self.failure_message, str)
        ):
            raise ValueError("Invalid response failure")


@dataclass(frozen=True)
class RunSubmission:
    submission_id: str
    task_id: str
    task_revision: int
    task_run_id: str
    attempt_id: str
    scope: RunScope
    version: int
    created_at: str
    report: ProducerReport
    content_digest: str
