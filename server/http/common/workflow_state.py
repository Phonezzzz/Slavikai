from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from shared.models import JSONValue


def normalize_mode_value(value: object, *, default: str, session_modes: set[str]) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in session_modes:
            return normalized
    return default


def normalize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                normalized.append(cleaned)
    return normalized


def normalize_json_value(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        return [normalize_json_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized[str(key)] = normalize_json_value(item)
        return normalized
    return str(value)


def normalize_plan_step(
    step: object,
    *,
    plan_step_statuses: set[str],
) -> dict[str, JSONValue] | None:
    if not isinstance(step, dict):
        return None
    step_id_raw = step.get("step_id")
    title_raw = step.get("title")
    description_raw = step.get("description")
    if not isinstance(step_id_raw, str) or not step_id_raw.strip():
        return None
    if not isinstance(title_raw, str) or not title_raw.strip():
        return None
    if not isinstance(description_raw, str):
        return None
    status_raw = step.get("status")
    status = (
        status_raw if isinstance(status_raw, str) and status_raw in plan_step_statuses else "todo"
    )
    evidence_raw = step.get("evidence")
    evidence = normalize_json_value(evidence_raw) if isinstance(evidence_raw, dict) else None
    return {
        "step_id": step_id_raw.strip(),
        "title": title_raw.strip(),
        "description": description_raw,
        "allowed_tool_kinds": normalize_string_list(step.get("allowed_tool_kinds")),
        "acceptance_checks": normalize_string_list(step.get("acceptance_checks")),
        "status": status,
        "evidence": evidence,
    }


def plan_hash_payload(plan: dict[str, JSONValue]) -> str:
    steps_raw = plan.get("steps")
    step_items = steps_raw if isinstance(steps_raw, list) else []
    payload = {
        "goal": plan.get("goal"),
        "scope_in": plan.get("scope_in"),
        "scope_out": plan.get("scope_out"),
        "assumptions": plan.get("assumptions"),
        "inputs_needed": plan.get("inputs_needed"),
        "steps": [
            {
                "step_id": item.get("step_id"),
                "title": item.get("title"),
                "description": item.get("description"),
                "allowed_tool_kinds": item.get("allowed_tool_kinds"),
                "acceptance_checks": item.get("acceptance_checks"),
            }
            for item in step_items
            if isinstance(item, dict)
        ],
        "exit_criteria": plan.get("exit_criteria"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def normalize_plan_payload(
    raw: object,
    *,
    plan_statuses: set[str],
    plan_step_statuses: set[str],
    utc_now_iso: Callable[[], str],
    normalize_plan_step_fn: Callable[[object], dict[str, JSONValue] | None] | None = None,
    normalize_string_list_fn: Callable[[object], list[str]] | None = None,
    normalize_json_value_fn: Callable[[object], JSONValue] | None = None,
    plan_hash_payload_fn: Callable[[dict[str, JSONValue]], str] | None = None,
) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    plan_id_raw = raw.get("plan_id")
    if not isinstance(plan_id_raw, str) or not plan_id_raw.strip():
        return None
    status_raw = raw.get("status")
    status = status_raw if isinstance(status_raw, str) and status_raw in plan_statuses else "draft"
    goal_raw = raw.get("goal")
    goal = goal_raw if isinstance(goal_raw, str) else ""
    created_at_raw = raw.get("created_at")
    updated_at_raw = raw.get("updated_at")
    now = utc_now_iso()
    step_normalizer = normalize_plan_step_fn or (
        lambda item: normalize_plan_step(item, plan_step_statuses=plan_step_statuses)
    )
    string_list_normalizer = normalize_string_list_fn or normalize_string_list
    json_value_normalizer = normalize_json_value_fn or normalize_json_value
    hash_builder = plan_hash_payload_fn or plan_hash_payload
    steps_raw = raw.get("steps")
    steps: list[dict[str, JSONValue]] = []
    if isinstance(steps_raw, list):
        for item in steps_raw:
            normalized = step_normalizer(item)
            if normalized is not None:
                steps.append(normalized)
    plan_revision_raw = raw.get("plan_revision")
    plan_revision = (
        plan_revision_raw if isinstance(plan_revision_raw, int) and plan_revision_raw > 0 else 1
    )
    normalized_plan: dict[str, JSONValue] = {
        "plan_id": plan_id_raw.strip(),
        "plan_hash": "",
        "plan_revision": plan_revision,
        "status": status,
        "goal": goal,
        "scope_in": string_list_normalizer(raw.get("scope_in")),
        "scope_out": string_list_normalizer(raw.get("scope_out")),
        "assumptions": string_list_normalizer(raw.get("assumptions")),
        "inputs_needed": string_list_normalizer(raw.get("inputs_needed")),
        "audit_log": (
            [json_value_normalizer(item) for item in raw.get("audit_log", [])]
            if isinstance(raw.get("audit_log"), list)
            else []
        ),
        "steps": steps,
        "exit_criteria": string_list_normalizer(raw.get("exit_criteria")),
        "created_at": (
            created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
        ),
        "updated_at": (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        ),
        "approved_at": raw.get("approved_at") if isinstance(raw.get("approved_at"), str) else None,
        "approved_by": raw.get("approved_by") if isinstance(raw.get("approved_by"), str) else None,
    }
    normalized_plan["plan_hash"] = hash_builder(normalized_plan)
    return normalized_plan


def normalize_task_payload(
    raw: object,
    *,
    task_statuses: set[str],
    utc_now_iso: Callable[[], str],
) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    task_id_raw = raw.get("task_id")
    plan_id_raw = raw.get("plan_id")
    plan_hash_raw = raw.get("plan_hash")
    if not isinstance(task_id_raw, str) or not task_id_raw.strip():
        return None
    if not isinstance(plan_id_raw, str) or not plan_id_raw.strip():
        return None
    if not isinstance(plan_hash_raw, str) or not plan_hash_raw.strip():
        return None
    status_raw = raw.get("status")
    status = (
        status_raw if isinstance(status_raw, str) and status_raw in task_statuses else "running"
    )
    current_step_raw = raw.get("current_step_id")
    current_step = (
        current_step_raw.strip()
        if isinstance(current_step_raw, str) and current_step_raw.strip()
        else None
    )
    started_at_raw = raw.get("started_at")
    updated_at_raw = raw.get("updated_at")
    now = utc_now_iso()
    return {
        "task_id": task_id_raw.strip(),
        "plan_id": plan_id_raw.strip(),
        "plan_hash": plan_hash_raw.strip(),
        "current_step_id": current_step,
        "status": status,
        "started_at": (
            started_at_raw if isinstance(started_at_raw, str) and started_at_raw.strip() else now
        ),
        "updated_at": (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        ),
    }


def plan_revision_value(plan: dict[str, JSONValue]) -> int:
    raw = plan.get("plan_revision")
    if isinstance(raw, int) and raw > 0:
        return raw
    return 1


def increment_plan_revision(plan: dict[str, JSONValue]) -> int:
    next_value = plan_revision_value(plan) + 1
    plan["plan_revision"] = next_value
    return next_value


def decision_workflow_context(
    *,
    mode: str,
    active_plan: dict[str, JSONValue] | None,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    step_id: str | None = None
    task_id: str | None = None
    if isinstance(active_task, dict):
        step_raw = active_task.get("current_step_id")
        task_raw = active_task.get("task_id")
        if isinstance(step_raw, str) and step_raw.strip():
            step_id = step_raw.strip()
        if isinstance(task_raw, str) and task_raw.strip():
            task_id = task_raw.strip()
    plan_id: str | None = None
    plan_hash: str | None = None
    if isinstance(active_plan, dict):
        plan_id_raw = active_plan.get("plan_id")
        plan_hash_raw = active_plan.get("plan_hash")
        if isinstance(plan_id_raw, str) and plan_id_raw.strip():
            plan_id = plan_id_raw.strip()
        if isinstance(plan_hash_raw, str) and plan_hash_raw.strip():
            plan_hash = plan_hash_raw.strip()
    return {
        "mode": mode,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "task_id": task_id,
        "step_id": step_id,
    }
