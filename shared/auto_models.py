from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from shared.models import JSONValue


class AutoRunStatus(StrEnum):
    IDLE = "idle"
    # The only non-terminal phase the preset emits while the
    # canonical loop runs. PLANNING/CODING/MERGING stay valid
    # status values for states already in flight, but the
    # preset never claims those phases.
    RUNNING = "running"
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
    AutoRunStatus.RUNNING.value,
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
    budgets: dict[str, JSONValue] = field(default_factory=dict)
    verifier: dict[str, JSONValue] | None = None
    skill: dict[str, JSONValue] | None = None
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
            "budgets": dict(self.budgets),
            "verifier": dict(self.verifier) if self.verifier is not None else None,
            "skill": dict(self.skill) if self.skill is not None else None,
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
        "budgets": _normalize_object(value.get("budgets")) or {},
        "verifier": _normalize_object(value.get("verifier")),
        "skill": _normalize_object(value.get("skill")),
        "approval": _normalize_object(value.get("approval")),
        "error": value.get("error") if isinstance(value.get("error"), str) else None,
        "error_code": (
            value.get("error_code")
            if isinstance(value.get("error_code"), str) and value.get("error_code")
            else None
        ),
        "missing_paths": _normalize_string_list(value.get("missing_paths")),
    }
    metrics = _normalize_execution_metrics(value.get("execution_metrics"))
    if metrics is not None:
        normalized["execution_metrics"] = metrics
    return normalized


def _normalize_execution_metrics(value: object) -> dict[str, JSONValue] | None:
    if not isinstance(value, dict):
        return None
    metrics = _normalize_model_usage_metrics(value) or {}
    if "files_touched" in value:
        count = value["files_touched"]
        # Invalid/unobserved file effects stay unknown, never coerced to zero.
        metrics["files_touched"] = count if type(count) is int and count >= 0 else None
    # Honest loop counters: preserved only as non-negative ints, dropped
    # (not zeroed) when absent or invalid.
    for key in ("tool_calls_used", "iterations"):
        if key in value:
            count = value[key]
            if type(count) is int and count >= 0:
                metrics[key] = count
    return metrics or None


def _normalize_model_usage_metrics(value: object) -> dict[str, JSONValue] | None:
    if not isinstance(value, dict):
        return None
    usage = value.get("tool_loop_model_usage")
    if not isinstance(usage, dict):
        return None
    counts: dict[str, int] = {}
    for key in ("started_calls", "reported_calls", "unknown_calls"):
        count = usage.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        counts[key] = count
    started, reported, unknown = (
        counts[key] for key in ("started_calls", "reported_calls", "unknown_calls")
    )
    if started != reported + unknown:
        return None
    raw_sum = usage.get("reported_sum")
    if not isinstance(raw_sum, dict):
        return None
    sums: list[int] = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        count = raw_sum.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        sums.append(count)
    if reported == 0 and any(sums):
        return None
    if "total_tokens" not in usage:
        return None
    total = usage.get("total_tokens")
    expected = sums[2] if started > 0 and unknown == 0 else None
    if type(total) is not type(expected) or total != expected:
        return None
    return {
        "tool_loop_model_usage": {
            "started_calls": started,
            "reported_calls": reported,
            "unknown_calls": unknown,
            "reported_sum": {
                "prompt_tokens": sums[0],
                "completion_tokens": sums[1],
                "total_tokens": sums[2],
            },
            "total_tokens": total,
        }
    }


# No coder pool exists anymore; pool_size is a legacy UI field.
# Keep a positive value as-is, default to a single worker.
def _normalize_pool_size(value: object) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return 1


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
