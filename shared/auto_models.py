from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from shared.models import JSONValue


class AutoRunStatus(StrEnum):
    IDLE = "idle"
    PLANNING = "planning"
    CODING = "coding"
    MERGING = "merging"
    VERIFYING = "verifying"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED_CONFLICT = "failed_conflict"
    FAILED_VERIFIER = "failed_verifier"
    FAILED_WORKER = "failed_worker"
    FAILED_INTERNAL = "failed_internal"
    CANCELLED = "cancelled"


AUTO_RUN_STATUSES: Final[set[str]] = {item.value for item in AutoRunStatus}
AUTO_RUN_ACTIVE_STATUSES: Final[set[str]] = {
    AutoRunStatus.PLANNING.value,
    AutoRunStatus.CODING.value,
    AutoRunStatus.MERGING.value,
    AutoRunStatus.VERIFYING.value,
    AutoRunStatus.WAITING_APPROVAL.value,
}
AUTO_RUN_TERMINAL_STATUSES: Final[set[str]] = {
    AutoRunStatus.IDLE.value,
    AutoRunStatus.COMPLETED.value,
    AutoRunStatus.FAILED_CONFLICT.value,
    AutoRunStatus.FAILED_VERIFIER.value,
    AutoRunStatus.FAILED_WORKER.value,
    AutoRunStatus.FAILED_INTERNAL.value,
    AutoRunStatus.CANCELLED.value,
}
AUTO_CODER_POOL_DEFAULT: Final[int] = 3
AUTO_CODER_POOL_MIN: Final[int] = 1
AUTO_CODER_POOL_MAX: Final[int] = 6


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                result.append(cleaned)
    return result


def _normalize_json(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    if isinstance(value, dict):
        payload: dict[str, JSONValue] = {}
        for key, item in value.items():
            payload[str(key)] = _normalize_json(item)
        return payload
    return str(value)


@dataclass(frozen=True)
class AutoShard:
    shard_id: str
    goal: str
    path_scope: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    acceptance_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "shard_id": self.shard_id,
            "goal": self.goal,
            "path_scope": list(self.path_scope),
            "depends_on": list(self.depends_on),
            "acceptance_checks": list(self.acceptance_checks),
        }


@dataclass(frozen=True)
class AutoPlan:
    plan_id: str
    goal: str
    shards: list[AutoShard] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "shards": [item.to_dict() for item in self.shards],
        }


@dataclass(frozen=True)
class AutoCoderState:
    coder_id: str
    shard_id: str
    status: str
    changed_paths: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "coder_id": self.coder_id,
            "shard_id": self.shard_id,
            "status": self.status,
            "changed_paths": list(self.changed_paths),
            "diagnostics": list(self.diagnostics),
        }


@dataclass(frozen=True)
class AutoVerifierState:
    status: str
    command: list[str] = field(default_factory=list)
    exit_code: int | None = None
    error: str | None = None
    duration_ms: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "status": self.status,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class AutoState:
    run_id: str
    status: AutoRunStatus
    goal: str
    root_path: str
    pool_size: int
    started_at: str
    updated_at: str
    planner: dict[str, JSONValue] = field(default_factory=dict)
    plan: dict[str, JSONValue] | None = None
    coders: list[dict[str, JSONValue]] = field(default_factory=list)
    merge: dict[str, JSONValue] = field(default_factory=dict)
    verifier: dict[str, JSONValue] | None = None
    approval: dict[str, JSONValue] | None = None
    error: str | None = None
    error_code: str | None = None
    missing_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "goal": self.goal,
            "root_path": self.root_path,
            "pool_size": self.pool_size,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "planner": dict(self.planner),
            "plan": dict(self.plan) if self.plan is not None else None,
            "coders": [dict(item) for item in self.coders],
            "merge": dict(self.merge),
            "verifier": dict(self.verifier) if self.verifier is not None else None,
            "approval": dict(self.approval) if self.approval is not None else None,
            "error": self.error,
            "error_code": self.error_code,
            "missing_paths": list(self.missing_paths),
        }


def normalize_auto_state(value: object) -> dict[str, JSONValue] | None:
    if not isinstance(value, dict):
        return None
    run_id_raw = value.get("run_id")
    status_raw = value.get("status")
    goal_raw = value.get("goal")
    started_at_raw = value.get("started_at")
    updated_at_raw = value.get("updated_at")
    if not isinstance(run_id_raw, str) or not run_id_raw.strip():
        return None
    if not isinstance(status_raw, str) or status_raw not in AUTO_RUN_STATUSES:
        return None
    now = utc_now_iso()
    normalized: dict[str, JSONValue] = {
        "run_id": run_id_raw.strip(),
        "status": status_raw,
        "goal": goal_raw if isinstance(goal_raw, str) else "",
        "root_path": (value.get("root_path") if isinstance(value.get("root_path"), str) else ""),
        "pool_size": _normalize_pool_size(value.get("pool_size")),
        "started_at": (
            started_at_raw if isinstance(started_at_raw, str) and started_at_raw.strip() else now
        ),
        "updated_at": (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        ),
        "planner": _normalize_object(value.get("planner")) or {},
        "plan": _normalize_object(value.get("plan")),
        "coders": _normalize_object_list(value.get("coders")),
        "merge": _normalize_object(value.get("merge")) or {},
        "verifier": _normalize_object(value.get("verifier")),
        "approval": _normalize_object(value.get("approval")),
        "error": value.get("error") if isinstance(value.get("error"), str) else None,
        "error_code": (
            value.get("error_code")
            if isinstance(value.get("error_code"), str) and value.get("error_code")
            else None
        ),
        "missing_paths": _normalize_string_list(value.get("missing_paths")),
    }
    return normalized


def _normalize_pool_size(value: object) -> int:
    if isinstance(value, int) and value > 0:
        return max(AUTO_CODER_POOL_MIN, min(AUTO_CODER_POOL_MAX, value))
    return AUTO_CODER_POOL_DEFAULT


def _normalize_object(value: object) -> dict[str, JSONValue] | None:
    if not isinstance(value, dict):
        return None
    payload: dict[str, JSONValue] = {}
    for key, item in value.items():
        if isinstance(item, list) and key in {"depends_on", "path_scope", "acceptance_checks"}:
            payload[str(key)] = _normalize_string_list(item)
        else:
            payload[str(key)] = _normalize_json(item)
    return payload


def _normalize_object_list(value: object) -> list[dict[str, JSONValue]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, JSONValue]] = []
    for item in value:
        normalized = _normalize_object(item)
        if normalized is not None:
            result.append(normalized)
    return result
