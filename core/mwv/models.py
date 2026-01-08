from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from shared.models import JSONValue


class WorkStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


class RetryPolicy(str, Enum):
    NONE = "none"
    LIMITED = "limited"


@dataclass(frozen=True)
class MWVMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class TaskPacket:
    task_id: str
    session_id: str
    trace_id: str
    goal: str
    messages: list[MWVMessage] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    context: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkChange:
    path: str
    change_type: ChangeType
    summary: str


@dataclass(frozen=True)
class WorkResult:
    task_id: str
    status: WorkStatus
    summary: str
    changes: list[WorkChange] = field(default_factory=list)
    tool_summaries: list[str] = field(default_factory=list)
    diagnostics: dict[str, JSONValue] = field(default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class RunContext:
    session_id: str
    trace_id: str
    workspace_root: str
    safe_mode: bool
    approved_categories: list[str] = field(default_factory=list)
    max_retries: int = 2
    attempt: int = 1


@dataclass(frozen=True)
class RetryDecision:
    policy: RetryPolicy
    allow_retry: bool
    reason: str
    attempt: int
    max_retries: int
