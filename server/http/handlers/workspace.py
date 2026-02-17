from __future__ import annotations

from pathlib import Path

from aiohttp import web

from config.tools_config import DEFAULT_TOOLS_STATE
from server.http.common.responses import error_response, json_response
from server.http_api import (
    POLICY_PROFILES,
    UI_SESSION_HEADER,
    _build_ui_approval_decision,
    _decision_workflow_context,
    _index_workspace_root,
    _load_effective_session_security,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_policy_profile,
    _normalize_task_payload,
    _normalize_tools_state_payload,
    _normalize_ui_decision,
    _resolve_ui_session_id_for_principal,
    _resolve_workspace_root_candidate,
    _session_forbidden_response,
    _set_current_plan_step_status,
    _utc_now_iso,
    _workspace_git_diff,
    _workspace_root_for_session,
)
from server.ui_hub import UIHub
from shared.models import JSONValue


async def handle_ui_redirect(request: web.Request) -> web.StreamResponse:
    raise web.HTTPFound("/ui/")


async def handle_ui_index(request: web.Request) -> web.FileResponse:
    dist_path: Path = request.app["ui_dist_path"]
    index_path = dist_path / "index.html"
    return web.FileResponse(path=index_path)


async def handle_workspace_index(request: web.Request) -> web.FileResponse:
    return await handle_ui_index(request)


async def handle_ui_workspace_root_select(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_store = request.app["session_store"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    if _normalize_mode_value(workflow.get("mode"), default="ask") == "plan":
        return error_response(
            status=409,
            message="PLAN_READ_ONLY_BLOCK: plan-режим допускает только read-only действия.",
            error_type="invalid_request_error",
            code="PLAN_READ_ONLY_BLOCK",
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
    path_raw = payload.get("root_path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        return error_response(
            status=400,
            message="root_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    _, effective_policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    profile_raw = effective_policy.get("profile")
    profile = _normalize_policy_profile(profile_raw)
    try:
        target_root = _resolve_workspace_root_candidate(path_raw.strip(), policy_profile=profile)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    current_root = await _workspace_root_for_session(hub, session_id)
    if current_root == target_root:
        response = json_response(
            {
                "session_id": session_id,
                "root_path": str(current_root),
                "applied": True,
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    require_approval = profile == "yolo"
    required_category = "FS_OUTSIDE_WORKSPACE"
    approved = await session_store.get_categories(session_id)
    if not require_approval or required_category in approved:
        await hub.set_workspace_root(session_id, str(target_root))
        response = json_response(
            {
                "session_id": session_id,
                "root_path": str(target_root),
                "applied": True,
            }
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response

    approval_request: dict[str, JSONValue] = {
        "category": required_category,
        "required_categories": [required_category],
        "tool": "workspace_root_select",
        "prompt": {
            "what": "Сменить Workspace Root",
            "why": "Требуется подтверждение смены рабочей директории.",
            "risk": "Доступ к другой директории проекта.",
            "changes": [str(target_root)],
        },
        "details": {
            "root_path": str(target_root),
            "policy_profile": profile,
        },
    }
    workflow = await hub.get_session_workflow(session_id)
    mode = _normalize_mode_value(workflow.get("mode"), default="ask")
    active_plan = _normalize_plan_payload(workflow.get("active_plan"))
    active_task = _normalize_task_payload(workflow.get("active_task"))
    decision = _build_ui_approval_decision(
        approval_request=approval_request,
        session_id=session_id,
        source_endpoint="workspace.root_select",
        resume_payload={
            "root_path": str(target_root),
            "session_id": session_id,
        },
        workflow_context=_decision_workflow_context(
            mode=mode,
            active_plan=active_plan,
            active_task=active_task,
        ),
    )
    await _set_current_plan_step_status(
        hub=hub,
        session_id=session_id,
        status="waiting_approval",
    )
    await hub.set_session_decision(session_id, decision)
    response = json_response(
        {
            "session_id": session_id,
            "decision": _normalize_ui_decision(decision, session_id=session_id),
        },
        status=202,
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_index_run(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workflow = await hub.get_session_workflow(session_id)
    if _normalize_mode_value(workflow.get("mode"), default="ask") == "plan":
        return error_response(
            status=409,
            message="PLAN_READ_ONLY_BLOCK: plan-режим допускает только read-only действия.",
            error_type="invalid_request_error",
            code="PLAN_READ_ONLY_BLOCK",
        )
    root_path = await _workspace_root_for_session(hub, session_id)
    try:
        stats = _index_workspace_root(root_path)
    except ValueError as exc:
        return error_response(
            status=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="workspace_index_failed",
        )
    except RuntimeError as exc:
        return error_response(
            status=500,
            message=str(exc),
            error_type="internal_error",
            code="workspace_index_failed",
        )
    response = json_response(
        {
            "session_id": session_id,
            **stats,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_git_diff(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    root_path = await _workspace_root_for_session(hub, session_id)
    diff, error = _workspace_git_diff(root_path)
    response = json_response(
        {
            "session_id": session_id,
            "root_path": str(root_path),
            "diff": diff,
            "error": error,
            "ok": error is None,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_workspace_root_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    root_path = await _workspace_root_for_session(hub, session_id)
    _, policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    response = json_response(
        {
            "session_id": session_id,
            "root_path": str(root_path),
            "policy": policy,
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_security_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workspace_root = await _workspace_root_for_session(hub, session_id)
    effective_tools, effective_policy = await _load_effective_session_security(
        hub=hub,
        session_id=session_id,
    )
    response = json_response(
        {
            "session_id": session_id,
            "tools_state": effective_tools,
            "policy": effective_policy,
            "workspace_root": str(workspace_root),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_session_security_post(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
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
    invalid_top_keys = sorted(
        {str(key).strip() for key in payload if str(key).strip() not in {"tools", "policy"}}
    )
    if invalid_top_keys:
        return error_response(
            status=400,
            message=f"Неподдерживаемые поля: {', '.join(invalid_top_keys)}.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    if "policy" in payload:
        policy_raw = payload.get("policy")
        if not isinstance(policy_raw, dict):
            return error_response(
                status=400,
                message="policy должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        profile: str | None = None
        yolo_armed: bool | None = None
        yolo_armed_at: str | None = None

        if "profile" in policy_raw:
            profile_raw = policy_raw.get("profile")
            if (
                not isinstance(profile_raw, str)
                or profile_raw.strip().lower() not in POLICY_PROFILES
            ):
                return error_response(
                    status=400,
                    message="policy.profile должен быть sandbox|index|yolo.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            profile = profile_raw.strip().lower()

        if "yolo_armed" in policy_raw:
            yolo_armed_raw = policy_raw.get("yolo_armed")
            if not isinstance(yolo_armed_raw, bool):
                return error_response(
                    status=400,
                    message="policy.yolo_armed должен быть bool.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            yolo_armed = yolo_armed_raw

        if profile == "yolo" or yolo_armed is True:
            confirm_raw = policy_raw.get("yolo_confirm")
            confirm_text_raw = policy_raw.get("yolo_confirm_text")
            confirm_ok = (
                confirm_raw is True
                and isinstance(confirm_text_raw, str)
                and confirm_text_raw.strip().upper() == "YOLO"
            )
            if not confirm_ok:
                return error_response(
                    status=400,
                    message=(
                        "Для включения YOLO требуется подтверждение "
                        "(yolo_confirm=true, yolo_confirm_text='YOLO')."
                    ),
                    error_type="invalid_request_error",
                    code="yolo_confirmation_required",
                )
            if yolo_armed is None:
                yolo_armed = True

        workspace_root = await _workspace_root_for_session(hub, session_id)
        current_policy = await hub.get_session_policy(session_id)
        current_profile = _normalize_policy_profile(current_policy.get("profile"))
        target_profile = profile or current_profile
        try:
            _resolve_workspace_root_candidate(
                str(workspace_root),
                policy_profile=target_profile,
            )
        except ValueError as exc:
            return error_response(
                status=409,
                message=str(exc),
                error_type="invalid_request_error",
                code="policy_root_incompatible",
                details={
                    "session_id": session_id,
                    "workspace_root": str(workspace_root),
                    "policy_profile": target_profile,
                },
            )

        if yolo_armed is True:
            yolo_armed_at = _utc_now_iso()
        if yolo_armed is False:
            yolo_armed_at = None
        await hub.set_session_policy(
            session_id,
            profile=profile,
            yolo_armed=yolo_armed,
            yolo_armed_at=yolo_armed_at,
        )

    if "tools" in payload:
        tools_raw = payload.get("tools")
        if not isinstance(tools_raw, dict):
            return error_response(
                status=400,
                message="tools должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        state_raw = tools_raw.get("state")
        if not isinstance(state_raw, dict):
            return error_response(
                status=400,
                message="tools.state должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        unknown_tools = sorted(
            {str(key).strip() for key in state_raw if str(key).strip() not in DEFAULT_TOOLS_STATE}
        )
        if unknown_tools:
            return error_response(
                status=400,
                message=f"Неизвестные tools ключи: {', '.join(unknown_tools)}.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        normalized_tools = _normalize_tools_state_payload(state_raw)
        if len(normalized_tools) != len(state_raw):
            return error_response(
                status=400,
                message="tools.state должен быть bool map.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        await hub.set_session_tools_state(
            session_id,
            tools_state=normalized_tools,
            merge=True,
        )

    effective_tools, effective_policy = await _load_effective_session_security(
        hub=hub,
        session_id=session_id,
    )
    workspace_root = await _workspace_root_for_session(hub, session_id)
    response = json_response(
        {
            "session_id": session_id,
            "tools_state": effective_tools,
            "policy": effective_policy,
            "workspace_root": str(workspace_root),
        }
    )
    response.headers[UI_SESSION_HEADER] = session_id
    return response
