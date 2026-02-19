from __future__ import annotations

import uuid
from collections.abc import Callable

from core.approval_policy import ApprovalCategory
from shared.models import JSONValue


def normalize_ui_decision_options(
    raw: object,
    *,
    normalize_json_value_fn: Callable[[object], JSONValue],
) -> list[dict[str, JSONValue]]:
    if not isinstance(raw, list):
        return []
    options: list[dict[str, JSONValue]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        option_id = item.get("id")
        title = item.get("title")
        action = item.get("action")
        if not isinstance(option_id, str) or not option_id.strip():
            continue
        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(action, str) or not action.strip():
            continue
        payload_raw = item.get("payload")
        risk_raw = item.get("risk")
        payload_normalized = (
            normalize_json_value_fn(payload_raw) if isinstance(payload_raw, dict) else {}
        )
        risk = risk_raw if isinstance(risk_raw, str) and risk_raw.strip() else "low"
        options.append(
            {
                "id": option_id.strip(),
                "title": title.strip(),
                "action": action.strip(),
                "payload": payload_normalized,
                "risk": risk,
            }
        )
    return options


def normalize_ui_decision(
    raw: object,
    *,
    session_id: str | None,
    trace_id: str | None,
    utc_now_iso: Callable[[], str],
    normalize_json_value_fn: Callable[[object], JSONValue],
    normalize_ui_decision_options_fn: Callable[[object], list[dict[str, JSONValue]]],
    ui_decision_kinds: set[str],
    ui_decision_statuses: set[str],
) -> dict[str, JSONValue] | None:
    if not isinstance(raw, dict):
        return None
    if not raw:
        return None

    decision_id_raw = raw.get("id")
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        return None
    decision_id = decision_id_raw.strip()
    now = utc_now_iso()

    kind_raw = raw.get("kind")
    decision_type_raw = raw.get("decision_type")
    status_raw = raw.get("status")
    reason_raw = raw.get("reason")
    summary_raw = raw.get("summary")
    context_raw = raw.get("context")
    options_raw = raw.get("options")
    default_option_id_raw = raw.get("default_option_id")
    created_at_raw = raw.get("created_at")
    updated_at_raw = raw.get("updated_at")
    resolved_at_raw = raw.get("resolved_at")

    context: dict[str, JSONValue] = {}
    if isinstance(context_raw, dict):
        for key, value in context_raw.items():
            context[str(key)] = normalize_json_value_fn(value)
    if session_id and "session_id" not in context:
        context["session_id"] = session_id
    if trace_id and "trace_id" not in context:
        context["trace_id"] = trace_id

    options = normalize_ui_decision_options_fn(options_raw)
    default_option_id: str | None = None
    if isinstance(default_option_id_raw, str) and default_option_id_raw.strip():
        default_option_id = default_option_id_raw.strip()
    elif options:
        first_id = options[0].get("id")
        default_option_id = first_id if isinstance(first_id, str) else None

    if (
        isinstance(kind_raw, str)
        and kind_raw in ui_decision_kinds
        and isinstance(status_raw, str)
        and status_raw in ui_decision_statuses
    ):
        kind = kind_raw
        status = status_raw
        blocking = raw.get("blocking") is True
        reason = reason_raw if isinstance(reason_raw, str) and reason_raw.strip() else "decision"
        summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else reason
        proposed_action_raw = raw.get("proposed_action")
        proposed_action = (
            normalize_json_value_fn(proposed_action_raw)
            if isinstance(proposed_action_raw, dict)
            else {}
        )
        created_at = (
            created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
        )
        updated_at = (
            updated_at_raw if isinstance(updated_at_raw, str) and updated_at_raw.strip() else now
        )
        resolved_at = (
            resolved_at_raw
            if isinstance(resolved_at_raw, str) and resolved_at_raw.strip()
            else None
        )
        return {
            "id": decision_id,
            "kind": kind,
            "decision_type": (
                decision_type_raw.strip()
                if isinstance(decision_type_raw, str) and decision_type_raw.strip()
                else None
            ),
            "status": status,
            "blocking": blocking,
            "reason": reason,
            "summary": summary,
            "proposed_action": proposed_action,
            "options": options,
            "default_option_id": default_option_id,
            "context": context,
            "created_at": created_at,
            "updated_at": updated_at,
            "resolved_at": resolved_at,
        }

    reason = reason_raw if isinstance(reason_raw, str) and reason_raw.strip() else "decision"
    summary = summary_raw if isinstance(summary_raw, str) and summary_raw.strip() else reason
    created_at = (
        created_at_raw if isinstance(created_at_raw, str) and created_at_raw.strip() else now
    )
    return {
        "id": decision_id,
        "kind": "decision",
        "decision_type": (
            decision_type_raw.strip()
            if isinstance(decision_type_raw, str) and decision_type_raw.strip()
            else None
        ),
        "status": "pending",
        "blocking": True,
        "reason": reason,
        "summary": summary,
        "proposed_action": {},
        "options": options,
        "default_option_id": default_option_id,
        "context": context,
        "created_at": created_at,
        "updated_at": now,
        "resolved_at": None,
    }


def build_ui_approval_decision(
    *,
    approval_request: dict[str, JSONValue],
    session_id: str,
    source_endpoint: str,
    resume_payload: dict[str, JSONValue],
    trace_id: str | None,
    workflow_context: dict[str, JSONValue] | None,
    all_categories: set[ApprovalCategory],
    normalize_json_value_fn: Callable[[object], JSONValue],
    utc_now_iso: Callable[[], str],
) -> dict[str, JSONValue]:
    prompt_raw = approval_request.get("prompt")
    prompt = prompt_raw if isinstance(prompt_raw, dict) else {}
    what_raw = prompt.get("what")
    why_raw = prompt.get("why")
    category_raw = approval_request.get("category")
    tool_raw = approval_request.get("tool")
    details_raw = approval_request.get("details")
    required_raw = approval_request.get("required_categories")
    required_categories: list[str] = []
    if isinstance(required_raw, list):
        for item in required_raw:
            if isinstance(item, str) and item in all_categories:
                required_categories.append(item)

    summary = (
        what_raw.strip()
        if isinstance(what_raw, str) and what_raw.strip()
        else "Требуется подтверждение действия."
    )
    reason = (
        why_raw.strip() if isinstance(why_raw, str) and why_raw.strip() else "approval_required"
    )
    proposed_action: dict[str, JSONValue] = {
        "category": category_raw if isinstance(category_raw, str) else "",
        "required_categories": required_categories,
        "tool": tool_raw if isinstance(tool_raw, str) else "",
        "details": normalize_json_value_fn(details_raw) if isinstance(details_raw, dict) else {},
    }
    now = utc_now_iso()
    context_payload: dict[str, JSONValue] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "source_endpoint": source_endpoint,
        "resume_payload": resume_payload,
    }
    if isinstance(workflow_context, dict):
        for key, value in workflow_context.items():
            context_payload[str(key)] = normalize_json_value_fn(value)
    return {
        "id": f"decision-{uuid.uuid4().hex}",
        "kind": "approval",
        "decision_type": "tool_approval",
        "status": "pending",
        "blocking": True,
        "reason": reason,
        "summary": summary,
        "proposed_action": proposed_action,
        "options": [
            {
                "id": "approve_once",
                "title": "Approve once",
                "action": "approve_once",
                "payload": {},
                "risk": "medium",
            },
            {
                "id": "approve_session",
                "title": "Approve session",
                "action": "approve_session",
                "payload": {},
                "risk": "high",
            },
            {
                "id": "edit_and_approve",
                "title": "Edit and approve",
                "action": "edit_and_approve",
                "payload": {},
                "risk": "medium",
            },
            {
                "id": "reject",
                "title": "Reject",
                "action": "reject",
                "payload": {},
                "risk": "low",
            },
        ],
        "default_option_id": "approve_once",
        "context": context_payload,
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }


def build_plan_execute_decision(
    *,
    session_id: str,
    plan: dict[str, JSONValue],
    mode: str,
    active_task: dict[str, JSONValue] | None,
    utc_now_iso: Callable[[], str],
    plan_revision_value_fn: Callable[[dict[str, JSONValue]], int],
    decision_workflow_context_fn: Callable[
        [str, dict[str, JSONValue], dict[str, JSONValue] | None],
        dict[str, JSONValue],
    ],
) -> dict[str, JSONValue]:
    now = utc_now_iso()
    plan_id_raw = plan.get("plan_id")
    plan_id = plan_id_raw if isinstance(plan_id_raw, str) else ""
    revision = plan_revision_value_fn(plan)
    return {
        "id": f"decision-{uuid.uuid4().hex}",
        "kind": "decision",
        "decision_type": "plan_execute",
        "status": "pending",
        "blocking": True,
        "reason": "switch_to_act_required",
        "summary": "Для выполнения плана нужно подтверждение перехода в Act.",
        "proposed_action": {
            "plan_id": plan_id,
            "plan_revision": revision,
            "mode": mode,
        },
        "options": [
            {
                "id": "approve_once",
                "title": "Switch to Act and run",
                "action": "approve_once",
                "payload": {},
                "risk": "medium",
            },
            {
                "id": "edit_plan",
                "title": "Edit plan",
                "action": "edit_plan",
                "payload": {},
                "risk": "low",
            },
            {
                "id": "reject",
                "title": "Reject",
                "action": "reject",
                "payload": {},
                "risk": "low",
            },
        ],
        "default_option_id": "approve_once",
        "context": {
            "session_id": session_id,
            "decision_type": "plan_execute",
            "source_endpoint": "plan.execute",
            "plan_id": plan_id,
            "plan_revision": revision,
            "resume_payload": {
                "plan_id": plan_id,
                "plan_revision": revision,
            },
            "workflow": decision_workflow_context_fn(mode, plan, active_task),
        },
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }


def decision_is_pending_blocking(decision: dict[str, JSONValue] | None) -> bool:
    if not isinstance(decision, dict):
        return False
    status = decision.get("status")
    blocking = decision.get("blocking")
    return status == "pending" and blocking is True


def decision_with_status(
    decision: dict[str, JSONValue],
    *,
    status: str,
    resolved: bool,
    utc_now_iso: Callable[[], str],
) -> dict[str, JSONValue]:
    updated = dict(decision)
    updated["status"] = status
    updated["updated_at"] = utc_now_iso()
    if resolved:
        updated["resolved_at"] = utc_now_iso()
        updated["blocking"] = False
    return updated


def decision_type_value(decision: dict[str, JSONValue]) -> str:
    decision_type_raw = decision.get("decision_type")
    if isinstance(decision_type_raw, str) and decision_type_raw.strip():
        return decision_type_raw.strip()
    kind_raw = decision.get("kind")
    if kind_raw == "approval":
        return "tool_approval"
    return "decision"


def decision_categories(
    decision: dict[str, JSONValue],
    *,
    category_map: dict[str, ApprovalCategory],
) -> set[ApprovalCategory]:
    proposed_raw = decision.get("proposed_action")
    proposed = proposed_raw if isinstance(proposed_raw, dict) else {}
    categories_raw = proposed.get("required_categories")
    if not isinstance(categories_raw, list):
        return set()
    normalized: set[ApprovalCategory] = set()
    for item in categories_raw:
        if not isinstance(item, str):
            continue
        category = category_map.get(item)
        if category is not None:
            normalized.add(category)
    return normalized


def decision_mismatch_details(
    *,
    expected_id: str,
    actual_decision: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    actual_id_raw = actual_decision.get("id") if isinstance(actual_decision, dict) else None
    actual_status_raw = actual_decision.get("status") if isinstance(actual_decision, dict) else None
    actual_id = actual_id_raw if isinstance(actual_id_raw, str) else None
    actual_status = actual_status_raw if isinstance(actual_status_raw, str) else "missing"
    return {
        "expected_id": expected_id,
        "actual_id": actual_id,
        "actual_status": actual_status,
    }
