from __future__ import annotations

from aiohttp import web

from server.http.common.responses import error_response, json_response
from server.http_api import (
    SESSION_MODES,
    UI_SESSION_HEADER,
    _load_effective_session_security,
    _normalize_auto_state,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_ui_decision,
    _resolve_ui_session_id_for_principal,
    _session_forbidden_response,
    _workspace_root_for_session,
)
from server.ui_hub import UIHub
from shared.models import JSONValue


async def handle_ui_status(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workspace_root = await _workspace_root_for_session(hub, session_id)
    effective_tools, policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    workflow = await hub.get_session_workflow(session_id)
    decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    selected_model = await hub.get_session_model(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "decision": decision,
            "selected_model": selected_model,
            "workspace_root": str(workspace_root),
            "policy": policy,
            "tools_state": effective_tools,
            "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
            "active_task": _normalize_task_payload(workflow.get("active_task")),
            "auto_state": _normalize_auto_state(workflow.get("auto_state")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_state(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    payload: dict[str, JSONValue] = {
        "ok": True,
        "session_id": session_id,
        "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
        "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
        "active_task": _normalize_task_payload(workflow.get("active_task")),
        "auto_state": _normalize_auto_state(workflow.get("auto_state")),
        "pending_decision": decision,
    }
    response = json_response(payload)
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_mode(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
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
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    mode_raw = payload.get("mode")
    if not isinstance(mode_raw, str) or mode_raw.strip().lower() not in SESSION_MODES:
        return error_response(
            status=400,
            message="mode должен быть ask|plan|act|auto.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    next_mode = _normalize_mode_value(mode_raw, default="ask")
    workflow = await hub.get_session_workflow(session_id)
    current_mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    auto_state = _normalize_auto_state(workflow.get("auto_state"))
    auto_status_raw = auto_state.get("status") if isinstance(auto_state, dict) else None
    auto_status = auto_status_raw if isinstance(auto_status_raw, str) else "idle"
    auto_run_active = auto_status in {
        "planning",
        "coding",
        "merging",
        "verifying",
        "waiting_approval",
    }
    if next_mode == "act" and current_mode != "plan":
        return error_response(
            status=409,
            message="В act можно перейти только из plan-режима.",
            error_type="invalid_request_error",
            code="mode_transition_not_allowed",
        )
    if current_mode == "plan" and next_mode == "act":
        confirm = payload.get("confirm") is True
        if not confirm:
            return error_response(
                status=409,
                message="Для перехода plan->act нужен confirm=true.",
                error_type="invalid_request_error",
                code="mode_confirm_required",
            )
        if active_plan is None or active_plan.get("status") != "approved":
            return error_response(
                status=409,
                message="Нужен approved план для перехода в act.",
                error_type="invalid_request_error",
                code="plan_not_approved",
            )
    if next_mode == "auto":
        if current_mode in {"plan", "act"} and (active_plan is not None or active_task is not None):
            return error_response(
                status=409,
                message="Нельзя перейти в auto при активном plan/act workflow.",
                error_type="invalid_request_error",
                code="mode_transition_not_allowed",
            )
    if current_mode == "auto" and next_mode == "ask" and auto_run_active:
        return error_response(
            status=409,
            message="Нельзя выйти из auto: auto-run ещё активен.",
            error_type="invalid_request_error",
            code="auto_run_active",
        )
    if current_mode == "auto" and next_mode in {"plan", "act"}:
        return error_response(
            status=409,
            message="Переход auto->plan|act запрещён до завершения auto-run.",
            error_type="invalid_request_error",
            code="mode_transition_not_allowed",
        )
    if next_mode == "ask":
        await hub.set_session_workflow(
            session_id,
            mode="ask",
            active_task=None,
        )
    elif next_mode == "auto":
        await hub.set_session_workflow(
            session_id,
            mode="auto",
            active_plan=None,
            active_task=None,
        )
    else:
        await hub.set_session_workflow(session_id, mode=next_mode)
    updated = await hub.get_session_workflow(session_id)
    response = json_response(
        {
            "ok": True,
            "session_id": session_id,
            "mode": _normalize_mode_value(updated.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(updated.get("active_plan")),
            "active_task": _normalize_task_payload(updated.get("active_task")),
            "auto_state": _normalize_auto_state(updated.get("auto_state")),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response
