from __future__ import annotations

import logging

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from server import http_api as api
from server.http.common.responses import error_response, json_response
from server.http_api import (
    UI_GITHUB_REQUIRED_CATEGORIES,
    UI_PROJECT_COMMANDS,
    UI_SESSION_HEADER,
    _apply_agent_runtime_state,
    _build_github_import_approval_request,
    _build_model_config,
    _build_ui_approval_decision,
    _decision_is_pending_blocking,
    _decision_workflow_context,
    _extract_decision_payload,
    _model_not_allowed_response,
    _model_not_selected_response,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_trace_id,
    _normalize_ui_decision,
    _parse_github_import_args,
    _publish_agent_activity,
    _resolve_agent,
    _resolve_provider_api_key,
    _resolve_ui_session_id_for_principal,
    _serialize_approval_request,
    _session_forbidden_response,
    _set_current_plan_step_status,
    _split_response_and_report,
    _ui_messages_to_llm,
)
from server.ui_hub import UIHub
from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HttpAPI")


async def handle_ui_project_command(request: web.Request) -> web.Response:
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]
    hub: UIHub = request.app["ui_hub"]

    session_id: str | None = None
    status_opened = False
    error = False
    try:
        try:
            agent = await _resolve_agent(request)
        except ModelNotAllowedError as exc:
            return _model_not_allowed_response(exc.model_id)
        if agent is None:
            return _model_not_selected_response()

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

        command = str(payload.get("command") or "").strip().lower()
        args_raw = str(payload.get("args") or "").strip()
        if command not in UI_PROJECT_COMMANDS:
            return error_response(
                status=400,
                message="Поддерживаются project команды: find, index, github_import.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        user_command = f"/project {command}"
        if args_raw:
            user_command = f"{user_command} {args_raw}"

        resolved_session_id, session_error = await _resolve_ui_session_id_for_principal(
            request,
            hub,
        )
        if session_error is not None:
            return session_error
        if resolved_session_id is None:
            return _session_forbidden_response()
        session_id = resolved_session_id
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            return _model_not_selected_response()
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        active_plan = _normalize_plan_payload(workflow.get("active_plan"))
        active_task = _normalize_task_payload(workflow.get("active_task"))
        await hub.set_session_status(session_id, "busy")
        status_opened = True
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="request.received",
            detail="project",
        )

        approved_categories = await session_store.get_categories(session_id)
        user_message = hub.create_message(role="user", content=user_command)
        await hub.append_message(session_id, user_message)
        user_message_id_raw = user_message.get("message_id")
        user_message_id = (
            user_message_id_raw
            if isinstance(user_message_id_raw, str) and user_message_id_raw
            else None
        )
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="context.prepared",
            detail="project",
        )

        if command == "github_import":
            try:
                repo_url, branch = _parse_github_import_args(args_raw)
            except ValueError as exc:
                return error_response(
                    status=400,
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            missing_categories = [
                category
                for category in UI_GITHUB_REQUIRED_CATEGORIES
                if category not in approved_categories
            ]
            if missing_categories:
                approval_request_obj = _build_github_import_approval_request(
                    session_id=session_id,
                    repo_url=repo_url,
                    branch=branch,
                    required_categories=missing_categories,
                )
                approval_payload = _serialize_approval_request(approval_request_obj)
                approval_decision = _build_ui_approval_decision(
                    approval_request=approval_payload or {},
                    session_id=session_id,
                    source_endpoint="project.command",
                    resume_payload={
                        "source_request": {
                            "command": command,
                            "args": args_raw,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
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
                await hub.set_session_decision(session_id, approval_decision)
                messages = await hub.get_messages(session_id)
                current_decision = await hub.get_session_decision(session_id)
                current_model = await hub.get_session_model(session_id)
                current_workflow = await hub.get_session_workflow(session_id)
                response = json_response(
                    {
                        "session_id": session_id,
                        "messages": messages,
                        "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                        "selected_model": current_model,
                        "trace_id": None,
                        "approval_request": approval_payload,
                        "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                        "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                        "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                    },
                )
                response.headers[UI_SESSION_HEADER] = session_id
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="approval.required",
                    detail="project/github_import",
                )
                return response

            try:
                target_path, relative_target = api._resolve_github_target(repo_url)
            except ValueError as exc:
                return error_response(
                    status=400,
                    message=str(exc),
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="github.clone.start",
                detail=repo_url,
            )
            cloned, clone_result = await api._clone_github_repository(
                repo_url=repo_url,
                branch=branch,
                target_path=target_path,
            )
            if not cloned:
                response_text = f"Командный режим (без MWV)\n{clone_result}"
            else:
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="github.index.start",
                    detail=relative_target,
                )
                indexed, index_result = api._index_imported_project(relative_target)
                await _publish_agent_activity(
                    hub,
                    session_id=session_id,
                    phase="github.index.end",
                    detail=index_result,
                )
                if indexed:
                    response_text = (
                        "Командный режим (без MWV)\n"
                        f"GitHub import completed: {repo_url}\n"
                        f"Path: {relative_target}\n"
                        f"Index: {index_result}"
                    )
                else:
                    response_text = (
                        "Командный режим (без MWV)\n"
                        f"GitHub import completed with indexing errors: {repo_url}\n"
                        f"Path: {relative_target}\n"
                        f"Index error: {index_result}"
                    )
            await hub.append_message(
                session_id,
                hub.create_message(
                    role="assistant",
                    content=response_text,
                    parent_user_message_id=user_message_id,
                ),
            )
            await hub.set_session_decision(session_id, None)
            messages = await hub.get_messages(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": None,
                    "approval_request": None,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                },
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="project/github_import",
            )
            return response

        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))

        mwv_report: dict[str, JSONValue] | None = None
        ui_decision: dict[str, JSONValue] | None = None
        async with agent_lock:
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                api_key = _resolve_provider_api_key(selected_model["provider"])
                agent.reconfigure_models(model_config, main_api_key=api_key, persist=False)
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    status=400,
                    message=f"Не удалось применить модель сессии: {exc}",
                    error_type="configuration_error",
                    code="model_config_invalid",
                )
            try:
                await _apply_agent_runtime_state(agent=agent, hub=hub, session_id=session_id)
                agent.set_session_context(session_id, approved_categories)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to set session context for ui project command",
                    exc_info=True,
                    extra={
                        "session_id": session_id,
                        "approved_categories": sorted(approved_categories),
                        "error": str(exc),
                    },
                )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.start",
                detail="project",
            )
            response_raw = agent.respond(llm_messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.end",
                detail="project",
            )
            trace_id = getattr(agent, "last_chat_interaction_id", None)
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )
            decision_raw = _extract_decision_payload(response_text)
            ui_decision = _normalize_ui_decision(
                decision_raw,
                session_id=session_id,
                trace_id=_normalize_trace_id(trace_id),
            )
            if approval_request is not None:
                ui_decision = _build_ui_approval_decision(
                    approval_request=approval_request,
                    session_id=session_id,
                    source_endpoint="project.command",
                    resume_payload={
                        "source_request": {
                            "command": command,
                            "args": args_raw,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
                    },
                    trace_id=_normalize_trace_id(trace_id),
                    workflow_context=_decision_workflow_context(
                        mode=mode,
                        active_plan=active_plan,
                        active_task=active_task,
                    ),
                )

        if _decision_is_pending_blocking(ui_decision):
            await _set_current_plan_step_status(
                hub=hub,
                session_id=session_id,
                status="waiting_approval",
            )
            await hub.set_session_decision(session_id, ui_decision)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="approval.required",
                detail="project",
            )
            messages = await hub.get_messages(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                    "selected_model": current_model,
                    "trace_id": trace_id,
                    "approval_request": approval_request,
                    "mwv_report": mwv_report,
                    "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                    "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                    "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="project",
            )
            return response

        await hub.append_message(
            session_id,
            hub.create_message(
                role="assistant",
                content=response_text,
                trace_id=trace_id if isinstance(trace_id, str) else None,
                parent_user_message_id=user_message_id,
            ),
        )
        await hub.set_session_decision(session_id, ui_decision)
        messages = await hub.get_messages(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        current_workflow = await hub.get_session_workflow(session_id)
        response = json_response(
            {
                "session_id": session_id,
                "messages": messages,
                "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
                "mwv_report": mwv_report,
                "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                "active_task": _normalize_task_payload(current_workflow.get("active_task")),
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="response.ready",
            detail="project",
        )
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="error",
                detail=f"project: {exc}",
            )
            await hub.set_session_status(session_id, "error")
        return error_response(
            status=500,
            message=f"Project command error: {exc}",
            error_type="internal_error",
            code="agent_error",
        )
    finally:
        if status_opened and session_id is not None and not error:
            await hub.set_session_status(session_id, "ok")
