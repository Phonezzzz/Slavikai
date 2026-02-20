from __future__ import annotations

from typing import Final

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from server.http.common.responses import error_response, json_response
from server.http_api import (
    _apply_agent_runtime_state,
    _build_ui_approval_decision,
    _decision_is_pending_blocking,
    _decision_mismatch_response,
    _decision_type_value,
    _decision_with_status,
    _decision_workflow_context,
    _emit_status,
    _ensure_session_owned,
    _extract_decision_payload,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_trace_id,
    _normalize_ui_decision,
    _resolve_agent,
    _serialize_approval_request,
    _split_response_and_report,
    _ui_messages_to_llm,
    _workspace_root_for_session,
)
from shared.models import JSONValue
from tools.workspace_tools import set_workspace_root as set_runtime_workspace_root

RUNTIME_DECISION_ACTIONS: Final[set[str]] = {"ask_user", "proceed_safe", "retry", "abort"}


def _last_user_message(messages: list[dict[str, JSONValue]]) -> dict[str, JSONValue] | None:
    for item in reversed(messages):
        role_raw = item.get("role")
        role = role_raw if isinstance(role_raw, str) else ""
        if role == "user":
            return item
    return None


def _runtime_input_text(
    *,
    decision: dict[str, JSONValue],
    messages: list[dict[str, JSONValue]],
) -> str | None:
    context_raw = decision.get("context")
    context = context_raw if isinstance(context_raw, dict) else {}
    user_input_raw = context.get("user_input")
    if isinstance(user_input_raw, str) and user_input_raw.strip():
        return user_input_raw.strip()
    latest_user = _last_user_message(messages)
    if latest_user is None:
        return None
    content_raw = latest_user.get("content")
    if isinstance(content_raw, str) and content_raw.strip():
        return content_raw.strip()
    return None


def _runtime_resume_text(action: str) -> str:
    if action == "ask_user":
        return "Ожидаю уточнение от пользователя."
    if action == "abort":
        return "Выполнение остановлено."
    if action == "proceed_safe":
        return "Повторяю запрос в безопасном режиме."
    return "Повторяю запрос."


async def handle_ui_runtime_decision_respond(request: web.Request) -> web.Response:
    hub = request.app["ui_hub"]
    session_store = request.app["session_store"]
    agent_lock = request.app["agent_lock"]

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
    allowed_keys = {"session_id", "decision_id", "action", "payload"}
    forbidden_keys = sorted(
        {str(key).strip() for key in payload if str(key).strip() not in allowed_keys}
    )
    if forbidden_keys:
        return error_response(
            status=400,
            message=(
                "Допустимы session_id, decision_id, action, payload. "
                f"Запрещённые поля: {', '.join(forbidden_keys)}."
            ),
            error_type="invalid_request_error",
            code="invalid_request_error",
            details={"forbidden_fields": forbidden_keys},
        )

    session_id_raw = payload.get("session_id")
    decision_id_raw = payload.get("decision_id")
    action_raw = payload.get("action")
    action_payload_raw = payload.get("payload")
    if not isinstance(session_id_raw, str) or not session_id_raw.strip():
        return error_response(
            status=400,
            message="session_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(decision_id_raw, str) or not decision_id_raw.strip():
        return error_response(
            status=400,
            message="decision_id обязателен.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(action_raw, str) or action_raw not in RUNTIME_DECISION_ACTIONS:
        return error_response(
            status=400,
            message="action должен быть ask_user|proceed_safe|retry|abort.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    if action_payload_raw is not None and not isinstance(action_payload_raw, dict):
        return error_response(
            status=400,
            message="payload должен быть объектом или null.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    session_id = session_id_raw.strip()
    decision_id = decision_id_raw.strip()
    action = action_raw
    ownership_error = await _ensure_session_owned(request, hub, session_id)
    if ownership_error is not None:
        return ownership_error

    current_decision = _normalize_ui_decision(
        await hub.get_session_decision(session_id),
        session_id=session_id,
    )
    if current_decision is None:
        return error_response(
            status=404,
            message="Pending decision not found.",
            error_type="invalid_request_error",
            code="decision_not_found",
        )
    current_id = current_decision.get("id")
    if not isinstance(current_id, str) or current_id != decision_id:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=current_decision,
        )
    current_status = current_decision.get("status")
    if current_status != "pending":
        status_value = current_status if isinstance(current_status, str) else "unknown"
        return json_response(
            {
                "ok": True,
                "decision": current_decision,
                "status": status_value,
                "resume_started": False,
                "already_resolved": status_value in {"resolved", "rejected"},
                "resume": None,
            }
        )

    decision_type = _decision_type_value(current_decision)
    if decision_type != "runtime_packet":
        return error_response(
            status=409,
            message=f"Unsupported decision_type: {decision_type}.",
            error_type="invalid_request_error",
            code="decision_type_not_supported",
            details={"decision_type": decision_type},
        )
    options_raw = current_decision.get("options")
    options = options_raw if isinstance(options_raw, list) else []
    allowed_actions: set[str] = set()
    for option in options:
        if not isinstance(option, dict):
            continue
        option_action_raw = option.get("action")
        if isinstance(option_action_raw, str) and option_action_raw.strip():
            allowed_actions.add(option_action_raw.strip())
    if action not in allowed_actions:
        return error_response(
            status=400,
            message=f"action '{action}' недоступен для текущего decision.",
            error_type="invalid_request_error",
            code="runtime_action_not_allowed",
            details={"allowed_actions": sorted(allowed_actions)},
        )

    if action in {"ask_user", "abort"}:
        target_status = "resolved" if action == "ask_user" else "rejected"
        resolved_decision = _decision_with_status(
            current_decision,
            status=target_status,
            resolved=True,
        )
        updated, latest = await hub.transition_session_decision(
            session_id,
            expected_id=decision_id,
            expected_status="pending",
            next_decision=resolved_decision,
        )
        normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
        if not updated:
            return _decision_mismatch_response(
                expected_id=decision_id,
                actual_decision=normalized_latest,
            )
        workflow = await hub.get_session_workflow(session_id)
        return json_response(
            {
                "ok": True,
                "decision": normalized_latest,
                "status": target_status,
                "resume_started": False,
                "already_resolved": target_status in {"resolved", "rejected"},
                "resume": {
                    "ok": True,
                    "action": action,
                    "source_endpoint": "runtime.packet",
                    "hint": _runtime_resume_text(action),
                },
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
            }
        )

    executing = _decision_with_status(current_decision, status="executing")
    updated, latest = await hub.transition_session_decision(
        session_id,
        expected_id=decision_id,
        expected_status="pending",
        next_decision=executing,
    )
    normalized_latest = _normalize_ui_decision(latest, session_id=session_id)
    if not updated:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=normalized_latest,
        )

    status_opened = False
    status_error = False
    next_decision: dict[str, JSONValue] | None = None
    resume: dict[str, JSONValue] = {
        "ok": False,
        "action": action,
        "source_endpoint": "runtime.packet",
    }
    try:
        messages = await hub.get_messages(session_id)
        rerun_input = _runtime_input_text(decision=current_decision, messages=messages)
        if rerun_input is None:
            resume = {"ok": False, "action": action, "error": "decision_resume_input_missing"}
        else:
            approved_categories = await session_store.get_categories(session_id)
            llm_messages = _ui_messages_to_llm(messages)
            try:
                agent = await _resolve_agent(request)
            except ModelNotAllowedError as exc:
                resume = {
                    "ok": False,
                    "action": action,
                    "error": f"model_not_allowed: {exc.model_id}",
                }
                agent = None
            if agent is None and "error" not in resume:
                resume = {"ok": False, "action": action, "error": "model_not_selected"}
            if agent is not None:
                await hub.set_session_status(session_id, "busy")
                status_opened = True
                await _emit_status(
                    hub,
                    session_id=session_id,
                    phase="runtime.decision",
                    text=_runtime_resume_text(action),
                )
                await _emit_status(
                    hub,
                    session_id=session_id,
                    phase="agent.respond.start",
                    text="Готовлю ответ…",
                )
                runtime_root = await _workspace_root_for_session(hub, session_id)
                original_tools = dict(getattr(agent, "tools_enabled", {}))
                runtime_setter = getattr(agent, "apply_runtime_tools_enabled", None)
                async with agent_lock:
                    set_runtime_workspace_root(runtime_root)
                    try:
                        await _apply_agent_runtime_state(
                            agent=agent,
                            hub=hub,
                            session_id=session_id,
                        )
                        agent.set_session_context(session_id, approved_categories)
                        if action == "proceed_safe" and callable(runtime_setter):
                            safe_state = dict(original_tools)
                            safe_state["safe_mode"] = True
                            runtime_setter(safe_state)
                        response_raw = agent.respond(llm_messages)
                    finally:
                        if action == "proceed_safe" and callable(runtime_setter):
                            runtime_setter(original_tools)
                        set_runtime_workspace_root(None)

                response_text, _ = _split_response_and_report(response_raw)
                trace_id = _normalize_trace_id(getattr(agent, "last_chat_interaction_id", None))
                approval_request = _serialize_approval_request(
                    getattr(agent, "last_approval_request", None),
                )
                decision_payload = _extract_decision_payload(response_text)
                next_decision = _normalize_ui_decision(
                    decision_payload,
                    session_id=session_id,
                    trace_id=trace_id,
                )
                if approval_request is not None:
                    workflow = await hub.get_session_workflow(session_id)
                    next_decision = _build_ui_approval_decision(
                        approval_request=approval_request,
                        session_id=session_id,
                        source_endpoint="runtime.packet",
                        resume_payload={"source_request": {"content": rerun_input}},
                        trace_id=trace_id,
                        workflow_context=_decision_workflow_context(
                            mode=_normalize_mode_value(workflow.get("mode"), default="ask"),
                            active_plan=_normalize_plan_payload(workflow.get("active_plan")),
                            active_task=_normalize_task_payload(workflow.get("active_task")),
                        ),
                    )
                await hub.set_session_output(session_id, response_text)
                latest_user = _last_user_message(messages)
                parent_user_message_id_raw = (
                    latest_user.get("message_id") if isinstance(latest_user, dict) else None
                )
                parent_user_message_id = (
                    parent_user_message_id_raw
                    if isinstance(parent_user_message_id_raw, str) and parent_user_message_id_raw
                    else None
                )
                assistant_message = hub.create_message(
                    role="assistant",
                    content=response_text,
                    trace_id=trace_id,
                    parent_user_message_id=parent_user_message_id,
                )
                await hub.append_message(session_id, assistant_message)
                resume = {
                    "ok": True,
                    "action": action,
                    "source_endpoint": "runtime.packet",
                    "trace_id": trace_id,
                    "pending_decision": bool(_decision_is_pending_blocking(next_decision)),
                }
                await _emit_status(
                    hub,
                    session_id=session_id,
                    phase="response.ready",
                    text="",
                )
    except Exception as exc:  # noqa: BLE001
        status_error = True
        resume = {"ok": False, "action": action, "error": str(exc)}
    finally:
        if status_opened:
            await hub.set_session_status(session_id, "error" if status_error else "ok")
            if status_error:
                await _emit_status(
                    hub,
                    session_id=session_id,
                    phase="error",
                    text="Ошибка при повторном выполнении.",
                )

    resolved = _decision_with_status(executing, status="resolved", resolved=True)
    updated_resolved, latest_resolved = await hub.transition_session_decision(
        session_id,
        expected_id=decision_id,
        expected_status="executing",
        next_decision=resolved,
    )
    normalized_resolved = _normalize_ui_decision(latest_resolved, session_id=session_id)
    if not updated_resolved:
        return _decision_mismatch_response(
            expected_id=decision_id,
            actual_decision=normalized_resolved,
        )
    if next_decision is not None:
        await hub.set_session_decision(session_id, next_decision)
    workflow = await hub.get_session_workflow(session_id)
    messages = await hub.get_messages(session_id)
    output_payload = await hub.get_session_output(session_id)
    files_payload = await hub.get_session_files(session_id)
    artifacts_payload = await hub.get_session_artifacts(session_id)
    current_decision = await hub.get_session_decision(session_id)
    return json_response(
        {
            "ok": True,
            "decision": _normalize_ui_decision(current_decision, session_id=session_id),
            "status": "resolved",
            "resume_started": True,
            "already_resolved": True,
            "resume": resume,
            "messages": messages,
            "output": output_payload,
            "files": files_payload or [],
            "artifacts": artifacts_payload or [],
            "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
            "active_task": _normalize_task_payload(workflow.get("active_task")),
        }
    )
