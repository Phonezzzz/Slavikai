from __future__ import annotations

from typing import cast

from aiohttp import web
from multidict import CIMultiDict

from server import http_api as api
from server.http.common.responses import error_response, json_response
from server.http.handlers import workspace as legacy_workspace
from server.http_api import (
    UI_SESSION_HEADER,
    _ensure_chat_owned,
    _ensure_workspace_owned,
    _load_effective_session_security,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_tools_state_payload,
    _normalize_ui_decision,
    _resolve_transport_token,
)
from server.ui_hub import UIHub
from shared.models import JSONValue


def _transport_headers(request: web.Request) -> dict[str, str]:
    return {UI_SESSION_HEADER: _resolve_transport_token(request)}


def _workspace_id(request: web.Request) -> str:
    return request.match_info.get("workspace_id", "").strip()


def _chat_id(request: web.Request) -> str:
    return request.match_info.get("chat_id", "").strip()


def _workspace_request(request: web.Request) -> web.Request:
    workspace_id = _workspace_id(request)
    headers = CIMultiDict(request.headers)
    headers[UI_SESSION_HEADER] = workspace_id
    return request.clone(headers=headers)


async def handle_ui_workspaces_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = request.get("principal_id")
    if not isinstance(principal_id, str) or not principal_id.strip():
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    workspaces = await hub.list_workspaces(principal_id)
    response = json_response(
        {
            "ok": True,
            "workspaces": [cast("dict[str, JSONValue]", dict(item)) for item in workspaces],
        },
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_models(request: web.Request) -> web.Response:
    provider_query = request.query.get("provider", "").strip().lower()
    if provider_query:
        providers = [provider_query]
    else:
        providers = sorted(api.SUPPORTED_MODEL_PROVIDERS)
    payload_items: list[dict[str, JSONValue]] = []
    for provider in providers:
        models, error_text = api._fetch_provider_models(provider)
        payload_items.append({"provider": provider, "models": models, "error": error_text})
    response = json_response({"providers": payload_items})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspaces_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    principal_id = request.get("principal_id")
    if not isinstance(principal_id, str) or not principal_id.strip():
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    payload: dict[str, object] = {}
    if request.can_read_body:
        try:
            payload_raw = await request.json()
        except Exception:
            payload_raw = {}
        if isinstance(payload_raw, dict):
            payload = payload_raw
    title_raw = payload.get("title")
    workspace_id = await hub.create_workspace(
        principal_id,
        title=title_raw if isinstance(title_raw, str) else None,
    )
    workspace = await hub.get_workspace(workspace_id)
    response = json_response({"ok": True, "workspace": workspace}, status=201)
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    workspace = await hub.get_workspace(workspace_id)
    if workspace is None:
        return error_response(
            status=404,
            message="Workspace not found.",
            error_type="invalid_request_error",
            code="workspace_not_found",
        )
    response = json_response({"ok": True, "workspace": workspace})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_delete(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    deleted = await hub.delete_workspace(workspace_id)
    if not deleted:
        return error_response(
            status=404,
            message="Workspace not found.",
            error_type="invalid_request_error",
            code="workspace_not_found",
        )
    response = json_response({"ok": True, "workspace_id": workspace_id, "deleted": True})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_root_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    root_path = await hub.get_workspace_root(workspace_id)
    tools_state, policy = await _load_effective_session_security(hub=hub, session_id=workspace_id)
    response = json_response(
        {
            "ok": True,
            "workspace_id": workspace_id,
            "root_path": root_path,
            "policy": policy,
            "tools_state": tools_state,
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_root_select(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
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
    root_raw = payload.get("root_path")
    if not isinstance(root_raw, str) or not root_raw.strip():
        return error_response(
            status=400,
            message="root_path обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    await hub.set_workspace_root(workspace_id, root_raw.strip())
    await hub.append_workspace_activity(
        workspace_id,
        kind="root_select",
        summary=f"Workspace root set to {root_raw.strip()}",
        payload={"root_path": root_raw.strip()},
    )
    response = json_response(
        {
            "ok": True,
            "workspace_id": workspace_id,
            "root_path": root_raw.strip(),
            "applied": True,
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_activity_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    activity = await hub.get_workspace_activity(workspace_id)
    response = json_response({"ok": True, "workspace_id": workspace_id, "activity": activity})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_security_get(request: web.Request) -> web.Response:
    return await handle_ui_workspace_root_get(request)


async def handle_ui_workspace_security_post(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
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
    policy_raw = payload.get("policy")
    if isinstance(policy_raw, dict):
        profile_raw = policy_raw.get("profile")
        yolo_armed_raw = policy_raw.get("yolo_armed")
        await hub.set_session_policy(
            workspace_id,
            profile=profile_raw if isinstance(profile_raw, str) else None,
            yolo_armed=yolo_armed_raw if isinstance(yolo_armed_raw, bool) else None,
            yolo_armed_at=api._utc_now_iso() if yolo_armed_raw is True else None,
        )
        await hub.append_workspace_activity(
            workspace_id,
            kind="security_change",
            summary="Workspace policy updated",
            payload={"policy": policy_raw},
        )
    tools_raw = payload.get("tools")
    if isinstance(tools_raw, dict):
        state_raw = tools_raw.get("state")
        normalized_tools = _normalize_tools_state_payload(state_raw)
        await hub.set_session_tools_state(workspace_id, normalized_tools)
        await hub.append_workspace_activity(
            workspace_id,
            kind="security_change",
            summary="Workspace tools state updated",
            payload={"tools": normalized_tools},
        )
    effective_tools, effective_policy = await _load_effective_session_security(
        hub=hub,
        session_id=workspace_id,
    )
    response = json_response(
        {
            "ok": True,
            "workspace_id": workspace_id,
            "tools_state": effective_tools,
            "policy": effective_policy,
            "root_path": await hub.get_workspace_root(workspace_id),
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_chats_list(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    chats = await hub.list_chats(workspace_id)
    response = json_response(
        {
            "ok": True,
            "workspace_id": workspace_id,
            "chats": [cast("dict[str, JSONValue]", dict(item)) for item in chats],
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_workspace_chats_create(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    workspace_id = _workspace_id(request)
    ownership_error = await _ensure_workspace_owned(request, hub, workspace_id)
    if ownership_error is not None:
        return ownership_error
    payload: dict[str, object] = {}
    if request.can_read_body:
        try:
            payload_raw = await request.json()
        except Exception:
            payload_raw = {}
        if isinstance(payload_raw, dict):
            payload = payload_raw
    title_raw = payload.get("title")
    chat_id = await hub.create_chat(
        workspace_id,
        title=title_raw if isinstance(title_raw, str) else None,
    )
    chat = await hub.get_chat(chat_id)
    response = json_response({"ok": True, "chat": chat}, status=201)
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    chat_id = _chat_id(request)
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        return ownership_error
    chat = await hub.get_chat(chat_id)
    if chat is None:
        return error_response(
            status=404,
            message="Chat not found.",
            error_type="invalid_request_error",
            code="chat_not_found",
        )
    workflow = await hub.get_session_workflow(chat_id)
    response = json_response(
        {
            "ok": True,
            "chat": chat,
            "state": {
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
                "auto_state": workflow.get("auto_state"),
                "decision": _normalize_ui_decision(
                    await hub.get_session_decision(chat_id),
                    session_id=chat_id,
                ),
            },
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_delete(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    chat_id = _chat_id(request)
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        return ownership_error
    deleted = await hub.delete_chat(chat_id)
    if not deleted:
        return error_response(
            status=404,
            message="Chat not found.",
            error_type="invalid_request_error",
            code="chat_not_found",
        )
    response = json_response({"ok": True, "chat_id": chat_id, "deleted": True})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_title_patch(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    chat_id = _chat_id(request)
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        return ownership_error
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
    title_raw = payload.get("title")
    if not isinstance(title_raw, str) or not title_raw.strip():
        return error_response(
            status=400,
            message="title обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    result = await hub.set_chat_title(chat_id, title_raw)
    response = json_response({"ok": True, **result})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_messages_get(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    chat_id = _chat_id(request)
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        return ownership_error
    messages = await hub.get_chat_messages(chat_id)
    response = json_response({"ok": True, "chat_id": chat_id, "messages": messages})
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_model_post(request: web.Request) -> web.Response:
    hub: UIHub = request.app["ui_hub"]
    chat_id = _chat_id(request)
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        return ownership_error
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
    provider = payload.get("provider")
    model = payload.get("model")
    if (
        not isinstance(provider, str)
        or not provider.strip()
        or not isinstance(model, str)
        or not model.strip()
    ):
        return error_response(
            status=400,
            message="provider и model обязательны.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    await hub.set_session_model(chat_id, provider.strip(), model.strip())
    response = json_response(
        {
            "ok": True,
            "chat_id": chat_id,
            "selected_model": await hub.get_session_model(chat_id),
        }
    )
    response.headers.update(_transport_headers(request))
    return response


async def handle_ui_chat_state_get(request: web.Request) -> web.Response:
    from server.http.handlers import workflow

    return await workflow.handle_ui_state(request)


async def handle_ui_chat_mode_post(request: web.Request) -> web.Response:
    from server.http.handlers import workflow

    return await workflow.handle_ui_mode(request)


async def handle_ui_chat_runtime_init(request: web.Request) -> web.Response:
    from server.http.handlers import workflow

    return await workflow.handle_ui_runtime_init(request)


async def handle_ui_chat_send(request: web.Request) -> web.Response:
    from server.http.handlers import ui_chat

    return await ui_chat.handle_ui_chat_send(request)


async def handle_ui_chat_decision_respond(request: web.Request) -> web.Response:
    from server.http.handlers import decision

    return await decision.handle_ui_decision_respond(request)


async def handle_ui_workspace_tree(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_tree(_workspace_request(request))


async def handle_ui_workspace_file_get(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_get(_workspace_request(request))


async def handle_ui_workspace_file_put(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_put(_workspace_request(request))


async def handle_ui_workspace_file_create(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_create(_workspace_request(request))


async def handle_ui_workspace_file_rename(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_rename(_workspace_request(request))


async def handle_ui_workspace_file_move(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_move(_workspace_request(request))


async def handle_ui_workspace_file_delete(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_file_delete(_workspace_request(request))


async def handle_ui_workspace_patch(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_patch(_workspace_request(request))


async def handle_ui_workspace_run(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_run(_workspace_request(request))


async def handle_ui_workspace_terminal_run(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_terminal_run(_workspace_request(request))


async def handle_ui_workspace_index_run(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_index_run(_workspace_request(request))


async def handle_ui_workspace_git_diff(request: web.Request) -> web.Response:
    return await legacy_workspace.handle_ui_workspace_git_diff(_workspace_request(request))


async def handle_ui_chat_plan_draft(request: web.Request) -> web.Response:
    from server.http.handlers import plan

    return await plan.handle_ui_plan_draft(request)


async def handle_ui_chat_plan_approve(request: web.Request) -> web.Response:
    from server.http.handlers import plan

    return await plan.handle_ui_plan_approve(request)


async def handle_ui_chat_plan_edit(request: web.Request) -> web.Response:
    from server.http.handlers import plan

    return await plan.handle_ui_plan_edit(request)


async def handle_ui_chat_plan_execute(request: web.Request) -> web.Response:
    from server.http.handlers import plan

    return await plan.handle_ui_plan_execute(request)


async def handle_ui_chat_plan_cancel(request: web.Request) -> web.Response:
    from server.http.handlers import plan

    return await plan.handle_ui_plan_cancel(request)
