from __future__ import annotations

import asyncio
import uuid

from aiohttp import web

from server.http.common.responses import error_response, json_response
from server.http_api import (
    PLAN_AUDIT_MAX_READ_FILES,
    PLAN_AUDIT_MAX_SEARCH_CALLS,
    PLAN_AUDIT_MAX_TOTAL_BYTES,
    UI_SESSION_HEADER,
    _build_plan_draft,
    _build_plan_execute_decision,
    _normalize_json_value,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_ui_decision,
    _plan_apply_edit_operation,
    _plan_revision_value,
    _plan_with_status,
    _resolve_ui_session_id_for_principal,
    _run_plan_readonly_audit,
    _run_plan_runner,
    _session_forbidden_response,
    _task_with_status,
    _utc_now_iso,
    _workspace_root_for_session,
)
from server.ui_hub import UIHub
from shared.models import JSONValue


async def handle_ui_plan_draft(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "plan":
        return error_response(
            status=409,
            message="Draft доступен только в plan-режиме.",
            error_type="invalid_request_error",
            code="mode_not_plan",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    goal_raw = payload.get("goal")
    if not isinstance(goal_raw, str) or not goal_raw.strip():
        return error_response(
            status=400,
            message="goal обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    root = await _workspace_root_for_session(hub, session_id)
    audit_log, usage = _run_plan_readonly_audit(root=root)
    if (
        usage["read_files"] >= PLAN_AUDIT_MAX_READ_FILES
        or usage["total_bytes"] >= PLAN_AUDIT_MAX_TOTAL_BYTES
        or usage["search_calls"] >= PLAN_AUDIT_MAX_SEARCH_CALLS
    ):
        return error_response(
            status=409,
            message="Достигнут лимит read-only аудита.",
            error_type="invalid_request_error",
            code="PLAN_AUDIT_LIMIT_REACHED",
        )

    draft = _build_plan_draft(goal=goal_raw.strip(), audit_log=audit_log)
    await hub.set_session_workflow(session_id, active_plan=draft, active_task=None)
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="plan"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
            "audit_usage": usage,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_approve(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "plan":
        return error_response(
            status=409,
            message="Approve доступен только в plan-режиме.",
            error_type="invalid_request_error",
            code="mode_not_plan",
        )
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    if plan is None:
        return error_response(
            status=404,
            message="Draft plan не найден.",
            error_type="invalid_request_error",
            code="plan_not_found",
        )
    if plan.get("status") != "draft":
        return error_response(
            status=409,
            message="План должен быть в статусе draft.",
            error_type="invalid_request_error",
            code="plan_not_draft",
        )
    approved = _plan_with_status(plan, status="approved")
    await hub.set_session_workflow(session_id, active_plan=approved)
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="plan"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_edit(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    if mode != "plan":
        return error_response(
            status=409,
            message="Plan edit доступен только в plan-режиме.",
            error_type="invalid_request_error",
            code="mode_not_plan",
        )
    current_plan = _normalize_plan_payload(workflow.get("active_plan"))
    if current_plan is None:
        return error_response(
            status=404,
            message="План не найден.",
            error_type="invalid_request_error",
            code="plan_not_found",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    revision_raw = payload.get("plan_revision")
    if not isinstance(revision_raw, int) or revision_raw <= 0:
        return error_response(
            status=400,
            message="plan_revision обязателен и должен быть положительным int.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    current_revision = _plan_revision_value(current_plan)
    if revision_raw != current_revision:
        return error_response(
            status=409,
            message="plan_revision mismatch.",
            error_type="invalid_request_error",
            code="plan_revision_mismatch",
            details={
                "expected_revision": current_revision,
                "actual_revision": revision_raw,
            },
        )
    operation_raw = payload.get("operation")
    if not isinstance(operation_raw, dict):
        return error_response(
            status=400,
            message="operation обязателен и должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    operation = {str(key): _normalize_json_value(value) for key, value in operation_raw.items()}
    try:
        updated_plan = _plan_apply_edit_operation(plan=current_plan, operation=operation)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    await hub.set_session_workflow(
        session_id,
        mode="plan",
        active_plan=updated_plan,
        active_task=None,
    )
    await hub.set_session_decision(session_id, None)
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="plan"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_execute(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    if plan is None:
        return error_response(
            status=404,
            message="План не найден.",
            error_type="invalid_request_error",
            code="plan_not_found",
        )
    if plan.get("status") != "approved":
        return error_response(
            status=409,
            message="Сначала approve план.",
            error_type="invalid_request_error",
            code="plan_not_approved",
        )
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    expected_revision = payload.get("plan_revision")
    actual_revision = _plan_revision_value(plan)
    if (
        isinstance(expected_revision, int)
        and expected_revision > 0
        and expected_revision != actual_revision
    ):
        return error_response(
            status=409,
            message="plan_revision mismatch.",
            error_type="invalid_request_error",
            code="plan_revision_mismatch",
            details={"expected_revision": expected_revision, "actual_revision": actual_revision},
        )

    if mode == "plan":
        decision = _build_plan_execute_decision(
            session_id=session_id,
            plan=plan,
            mode=mode,
            active_task=active_task,
        )
        await hub.set_session_decision(session_id, decision)
        response = json_response(
            {
                "error": {
                    "message": "Switch to Act required before execution.",
                    "type": "invalid_request_error",
                    "code": "switch_to_act_required",
                    "trace_id": None,
                    "details": {
                        "session_id": session_id,
                        "plan_id": plan.get("plan_id"),
                        "plan_revision": actual_revision,
                    },
                },
                "session_id": session_id,
                "decision": _normalize_ui_decision(decision, session_id=session_id),
                "mode": mode,
                "active_plan": plan,
                "active_task": active_task,
            },
            status=409,
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    if mode != "act":
        return error_response(
            status=409,
            message="Выполнение возможно только в act-режиме.",
            error_type="invalid_request_error",
            code="mode_not_act",
        )

    task: dict[str, JSONValue] = {
        "task_id": f"task-{uuid.uuid4().hex}",
        "plan_id": plan.get("plan_id"),
        "plan_hash": plan.get("plan_hash"),
        "current_step_id": None,
        "status": "running",
        "started_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    running_plan = _plan_with_status(plan, status="running")
    await hub.set_session_decision(session_id, None)
    await hub.set_session_workflow(
        session_id,
        mode="act",
        active_plan=running_plan,
        active_task=task,
    )
    plan_id_raw = task.get("plan_id")
    task_id_raw = task.get("task_id")
    if isinstance(plan_id_raw, str) and isinstance(task_id_raw, str):
        asyncio.create_task(
            _run_plan_runner(
                app=request.app,
                session_id=session_id,
                plan_id=plan_id_raw,
                task_id=task_id_raw,
            )
        )
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="act"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_plan_cancel(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    plan = _normalize_plan_payload(workflow.get("active_plan"))
    task = _normalize_task_payload(workflow.get("active_task"))
    if plan is not None:
        plan = _plan_with_status(plan, status="cancelled")
    if task is not None:
        task = _task_with_status(task, status="cancelled", current_step_id=None)
    await hub.set_session_workflow(session_id, active_plan=plan, active_task=task)
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response
