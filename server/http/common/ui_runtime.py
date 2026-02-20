from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Final

from aiohttp import web

from core.approval_policy import ALL_CATEGORIES, ApprovalCategory, ApprovalRequest
from server.http.common import decision_flow, plan_edit, streaming, workflow_runtime, workflow_state
from server.ui_hub import UIHub
from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HttpAPI")

SESSION_MODES: Final[set[str]] = {"ask", "plan", "act"}
PLAN_STATUSES: Final[set[str]] = {
    "draft",
    "approved",
    "running",
    "completed",
    "failed",
    "cancelled",
}
PLAN_STEP_STATUSES: Final[set[str]] = {
    "todo",
    "doing",
    "waiting_approval",
    "blocked",
    "done",
    "failed",
}
TASK_STATUSES: Final[set[str]] = {"running", "completed", "failed", "cancelled"}
PLAN_MAX_STEPS: Final[int] = 50
PLAN_MAX_PAYLOAD_BYTES: Final[int] = 64 * 1024
PLAN_MAX_TEXT_FIELD_CHARS: Final[int] = 4000
_CATEGORY_MAP: Final[dict[str, ApprovalCategory]] = {item: item for item in ALL_CATEGORIES}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _serialize_approval_request(
    approval_request: ApprovalRequest | None,
) -> dict[str, JSONValue] | None:
    if approval_request is None:
        return None
    return {
        "category": approval_request.category,
        "required_categories": list(approval_request.required_categories),
        "tool": approval_request.tool,
        "details": dict(approval_request.details),
        "session_id": approval_request.session_id,
        "prompt": {
            "what": approval_request.prompt.what,
            "why": approval_request.prompt.why,
            "risk": approval_request.prompt.risk,
            "changes": list(approval_request.prompt.changes),
        },
    }


def _normalize_mode_value(value: object, *, default: str = "ask") -> str:
    return workflow_state.normalize_mode_value(
        value,
        default=default,
        session_modes=SESSION_MODES,
    )


def _normalize_string_list(value: object) -> list[str]:
    return workflow_state.normalize_string_list(value)


def _normalize_plan_step(step: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_plan_step(
        step,
        plan_step_statuses=PLAN_STEP_STATUSES,
    )


def _plan_hash_payload(plan: dict[str, JSONValue]) -> str:
    return workflow_state.plan_hash_payload(plan)


def _normalize_plan_payload(raw: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_plan_payload(
        raw,
        plan_statuses=PLAN_STATUSES,
        plan_step_statuses=PLAN_STEP_STATUSES,
        utc_now_iso=_utc_now_iso,
        normalize_plan_step_fn=_normalize_plan_step,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _normalize_task_payload(raw: object) -> dict[str, JSONValue] | None:
    return workflow_state.normalize_task_payload(
        raw,
        task_statuses=TASK_STATUSES,
        utc_now_iso=_utc_now_iso,
    )


def _plan_revision_value(plan: dict[str, JSONValue]) -> int:
    return workflow_state.plan_revision_value(plan)


def _increment_plan_revision(plan: dict[str, JSONValue]) -> int:
    return workflow_state.increment_plan_revision(plan)


def _decision_workflow_context(
    *,
    mode: str,
    active_plan: dict[str, JSONValue] | None,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    return workflow_state.decision_workflow_context(
        mode=mode,
        active_plan=active_plan,
        active_task=active_task,
    )


def _normalize_ui_decision_options(raw: object) -> list[dict[str, JSONValue]]:
    return decision_flow.normalize_ui_decision_options(
        raw,
        normalize_json_value_fn=_normalize_json_value,
    )


def _normalize_ui_decision(
    raw: object,
    *,
    session_id: str | None = None,
    trace_id: str | None = None,
) -> dict[str, JSONValue] | None:
    return decision_flow.normalize_ui_decision(
        raw,
        session_id=session_id,
        trace_id=trace_id,
        utc_now_iso=_utc_now_iso,
        normalize_json_value_fn=_normalize_json_value,
        normalize_ui_decision_options_fn=_normalize_ui_decision_options,
        ui_decision_kinds={"approval", "decision"},
        ui_decision_statuses={"pending", "approved", "rejected", "executing", "resolved"},
    )


def _build_ui_approval_decision(
    *,
    approval_request: dict[str, JSONValue],
    session_id: str,
    source_endpoint: str,
    resume_payload: dict[str, JSONValue],
    trace_id: str | None = None,
    workflow_context: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return decision_flow.build_ui_approval_decision(
        approval_request=approval_request,
        session_id=session_id,
        source_endpoint=source_endpoint,
        resume_payload=resume_payload,
        trace_id=trace_id,
        workflow_context=workflow_context,
        all_categories=ALL_CATEGORIES,
        normalize_json_value_fn=_normalize_json_value,
        utc_now_iso=_utc_now_iso,
    )


def _build_plan_execute_decision(
    *,
    session_id: str,
    plan: dict[str, JSONValue],
    mode: str,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    return decision_flow.build_plan_execute_decision(
        session_id=session_id,
        plan=plan,
        mode=mode,
        active_task=active_task,
        utc_now_iso=_utc_now_iso,
        plan_revision_value_fn=_plan_revision_value,
        decision_workflow_context_fn=lambda m, p, t: _decision_workflow_context(
            mode=m,
            active_plan=p,
            active_task=t,
        ),
    )


def _decision_is_pending_blocking(decision: dict[str, JSONValue] | None) -> bool:
    return decision_flow.decision_is_pending_blocking(decision)


def _decision_with_status(
    decision: dict[str, JSONValue],
    *,
    status: str,
    resolved: bool = False,
) -> dict[str, JSONValue]:
    return decision_flow.decision_with_status(
        decision,
        status=status,
        resolved=resolved,
        utc_now_iso=_utc_now_iso,
    )


def _decision_type_value(decision: dict[str, JSONValue]) -> str:
    return decision_flow.decision_type_value(decision)


def _decision_categories(decision: dict[str, JSONValue]) -> set[ApprovalCategory]:
    return decision_flow.decision_categories(decision, category_map=_CATEGORY_MAP)


def _decision_mismatch_details(
    *,
    expected_id: str,
    actual_decision: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    return decision_flow.decision_mismatch_details(
        expected_id=expected_id,
        actual_decision=actual_decision,
    )


def _normalize_json_value(value: object) -> JSONValue:
    return workflow_state.normalize_json_value(value)


_build_default_plan_steps = workflow_runtime.build_default_plan_steps


def _stream_preview_ready_for_chat(preview_text: str) -> bool:
    return streaming._stream_preview_ready_for_chat(
        preview_text,
        chat_stream_warmup_chars=streaming.CHAT_STREAM_WARMUP_CHARS,
    )


async def _publish_canvas_stream(
    hub: UIHub,
    *,
    session_id: str,
    artifact_id: str,
    content: str,
) -> None:
    await streaming._publish_canvas_stream(
        hub,
        session_id=session_id,
        artifact_id=artifact_id,
        content=content,
    )


def _build_plan_draft(
    *,
    goal: str,
    audit_log: list[dict[str, JSONValue]],
) -> dict[str, JSONValue]:
    return workflow_runtime.build_plan_draft(
        goal=goal,
        audit_log=audit_log,
        utc_now_iso_fn=_utc_now_iso,
        plan_hash_payload_fn=_plan_hash_payload,
        build_default_plan_steps_fn=_build_default_plan_steps,
    )


def _plan_with_status(
    plan: dict[str, JSONValue],
    *,
    status: str,
) -> dict[str, JSONValue]:
    return workflow_runtime.plan_with_status(
        plan,
        status=status,
        utc_now_iso_fn=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _task_with_status(
    task: dict[str, JSONValue],
    *,
    status: str,
    current_step_id: str | None = None,
) -> dict[str, JSONValue]:
    return workflow_runtime.task_with_status(
        task,
        status=status,
        current_step_id=current_step_id,
        utc_now_iso_fn=_utc_now_iso,
    )


def _plan_mark_step(
    plan: dict[str, JSONValue],
    *,
    step_id: str,
    status: str,
    evidence: dict[str, JSONValue] | None = None,
) -> dict[str, JSONValue]:
    return plan_edit.plan_mark_step(
        plan,
        step_id=step_id,
        status=status,
        evidence=evidence,
        plan_step_statuses=PLAN_STEP_STATUSES,
        utc_now_iso=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _validate_text_limit(value: str, *, field: str) -> None:
    plan_edit.validate_text_limit(
        value,
        field=field,
        max_chars=PLAN_MAX_TEXT_FIELD_CHARS,
    )


def _find_forbidden_plan_key(value: JSONValue) -> str | None:
    return plan_edit.find_forbidden_plan_key(value)


def _normalize_plan_step_insert(raw: object) -> dict[str, JSONValue]:
    return plan_edit.normalize_plan_step_insert(
        raw,
        plan_step_statuses=PLAN_STEP_STATUSES,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
    )


def _normalize_plan_step_changes(raw: object) -> dict[str, JSONValue]:
    return plan_edit.normalize_plan_step_changes(
        raw,
        plan_step_statuses=PLAN_STEP_STATUSES,
        normalize_string_list_fn=_normalize_string_list,
        normalize_json_value_fn=_normalize_json_value,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
    )


def _validate_plan_document(plan: dict[str, JSONValue]) -> None:
    plan_edit.validate_plan_document(
        plan,
        plan_max_payload_bytes=PLAN_MAX_PAYLOAD_BYTES,
        plan_max_steps=PLAN_MAX_STEPS,
        validate_text_limit_fn=lambda value, field: _validate_text_limit(value, field=field),
        find_forbidden_plan_key_fn=_find_forbidden_plan_key,
        normalize_plan_step_insert_fn=_normalize_plan_step_insert,
    )


def _plan_apply_edit_operation(
    *,
    plan: dict[str, JSONValue],
    operation: dict[str, JSONValue],
) -> dict[str, JSONValue]:
    return plan_edit.plan_apply_edit_operation(
        plan=plan,
        operation=operation,
        normalize_plan_step_insert_fn=_normalize_plan_step_insert,
        normalize_plan_step_changes_fn=_normalize_plan_step_changes,
        utc_now_iso=_utc_now_iso,
        increment_plan_revision_fn=_increment_plan_revision,
        validate_plan_document_fn=_validate_plan_document,
        plan_hash_payload_fn=_plan_hash_payload,
    )


def _find_next_todo_step(plan: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    return plan_edit.find_next_todo_step(plan)


async def _set_current_plan_step_status(
    *,
    hub: UIHub,
    session_id: str,
    status: str,
) -> None:
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "act":
        return
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    if active_plan is None or active_task is None:
        return
    step_id_raw = active_task.get("current_step_id")
    if not isinstance(step_id_raw, str) or not step_id_raw.strip():
        return
    updated_plan = _plan_mark_step(
        active_plan,
        step_id=step_id_raw.strip(),
        status=status,
    )
    await hub.set_session_workflow(
        session_id,
        mode="act",
        active_plan=updated_plan,
        active_task=active_task,
    )


async def _publish_agent_activity(
    hub: UIHub,
    *,
    session_id: str,
    phase: str,
    detail: str | None = None,
) -> None:
    payload: dict[str, JSONValue] = {"session_id": session_id, "phase": phase}
    if detail is not None and detail.strip():
        payload["detail"] = detail.strip()
    event: dict[str, JSONValue] = {
        "type": "agent.activity",
        "payload": payload,
    }
    try:
        await hub.publish(session_id, event)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to publish agent activity event", exc_info=True)


async def _emit_status(
    hub: UIHub,
    *,
    session_id: str,
    phase: str,
    text: str,
    transient: bool = True,
) -> None:
    status_event = await hub.get_session_status_event(session_id)
    payload_raw = status_event.get("payload")
    payload = payload_raw if isinstance(payload_raw, dict) else {}
    next_payload: dict[str, JSONValue] = {
        "session_id": session_id,
        "state": payload.get("state") if isinstance(payload.get("state"), str) else "ok",
        "ok": payload.get("ok") is not False,
        "phase": phase.strip() or "progress",
        "text": text.strip(),
        "transient": transient,
    }
    event: dict[str, JSONValue] = {"type": "status", "payload": next_payload}
    try:
        await hub.publish(session_id, event)
    except Exception:  # noqa: BLE001
        logger.debug("Failed to publish status event", exc_info=True)


async def _run_plan_runner(
    *,
    app: web.Application,
    session_id: str,
    plan_id: str,
    task_id: str,
) -> None:
    return await workflow_runtime.run_plan_runner(
        app=app,
        session_id=session_id,
        plan_id=plan_id,
        task_id=task_id,
        normalize_plan_payload_fn=_normalize_plan_payload,
        normalize_task_payload_fn=_normalize_task_payload,
        normalize_mode_value_fn=lambda value: _normalize_mode_value(value, default="ask"),
        find_next_todo_step_fn=_find_next_todo_step,
        plan_with_status_fn=lambda plan, status: _plan_with_status(plan, status=status),
        task_with_status_fn=lambda task, status, current_step_id: _task_with_status(
            task,
            status=status,
            current_step_id=current_step_id,
        ),
        plan_mark_step_fn=lambda plan, step_id, status, evidence: _plan_mark_step(
            plan,
            step_id=step_id,
            status=status,
            evidence=evidence,
        ),
        utc_now_iso_fn=_utc_now_iso,
    )
