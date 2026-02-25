from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, Literal

from shared.models import JSONValue


class WorkStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class VerificationStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class ChangeType(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RENAME = "rename"


class RetryPolicy(StrEnum):
    NONE = "none"
    LIMITED = "limited"


class StopReasonCode(StrEnum):
    BLOCKED_SKILL_AMBIGUOUS = "BLOCKED_SKILL_AMBIGUOUS"
    BLOCKED_SKILL_DEPRECATED = "BLOCKED_SKILL_DEPRECATED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    VERIFIER_FAILED = "VERIFIER_FAILED"
    REPLAN_REQUIRED = "REPLAN_REQUIRED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    MWV_INTERNAL_ERROR = "MWV_INTERNAL_ERROR"
    COMMAND_LANE_NOTICE = "COMMAND_LANE_NOTICE"
    WORKER_FAILED = "WORKER_FAILED"


MWV_REPORT_PREFIX: Final[str] = "MWV_REPORT_JSON="


@dataclass(frozen=True)
class MWVMessage:
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class TaskStepContract:
    step_id: str
    title: str
    description: str
    allowed_tool_kinds: list[str] = field(default_factory=list)
    inputs: dict[str, JSONValue] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    acceptance_checks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TaskPacket:
    task_id: str
    session_id: str
    trace_id: str
    goal: str
    packet_revision: int = 1
    packet_hash: str = ""
    messages: list[MWVMessage] = field(default_factory=list)
    steps: list[TaskStepContract] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    policy: dict[str, JSONValue] = field(default_factory=dict)
    scope: dict[str, JSONValue] = field(default_factory=dict)
    budgets: dict[str, JSONValue] = field(default_factory=dict)
    approvals: dict[str, JSONValue] = field(default_factory=dict)
    verifier: dict[str, JSONValue] = field(default_factory=dict)
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

    @property
    def ok(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def duration_ms(self) -> int:
        return int(self.duration_seconds * 1000)


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
    llm_hint: str | None = None


def task_packet_hash_payload(packet: TaskPacket) -> str:
    payload: dict[str, JSONValue] = {
        "task_id": packet.task_id,
        "packet_revision": packet.packet_revision,
        "session_id": packet.session_id,
        "trace_id": packet.trace_id,
        "goal": packet.goal,
        "messages": [
            {"role": message.role, "content": message.content} for message in packet.messages
        ],
        "steps": [
            {
                "step_id": step.step_id,
                "title": step.title,
                "description": step.description,
                "allowed_tool_kinds": list(step.allowed_tool_kinds),
                "inputs": dict(step.inputs),
                "expected_outputs": list(step.expected_outputs),
                "acceptance_checks": list(step.acceptance_checks),
            }
            for step in packet.steps
        ],
        "constraints": list(packet.constraints),
        "policy": dict(packet.policy),
        "scope": dict(packet.scope),
        "budgets": dict(packet.budgets),
        "approvals": dict(packet.approvals),
        "verifier": dict(packet.verifier),
        "context": dict(packet.context),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def with_task_packet_hash(packet: TaskPacket) -> TaskPacket:
    packet_hash = task_packet_hash_payload(packet)
    if packet.packet_hash == packet_hash:
        return packet
    return TaskPacket(
        task_id=packet.task_id,
        session_id=packet.session_id,
        trace_id=packet.trace_id,
        goal=packet.goal,
        packet_revision=packet.packet_revision,
        packet_hash=packet_hash,
        messages=list(packet.messages),
        steps=list(packet.steps),
        constraints=list(packet.constraints),
        policy=dict(packet.policy),
        scope=dict(packet.scope),
        budgets=dict(packet.budgets),
        approvals=dict(packet.approvals),
        verifier=dict(packet.verifier),
        context=dict(packet.context),
    )


def is_task_packet_hash_valid(packet: TaskPacket) -> bool:
    if not packet.packet_hash:
        return False
    return packet.packet_hash == task_packet_hash_payload(packet)
