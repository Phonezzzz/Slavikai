from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import Literal

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from core.mwv.routing import classify_request
from core.skills.index import SkillIndex
from llm.types import LLMStreamChunk, ModelConfig
from server.http.common.chat_payload import (
    _extract_decision_payload,
    _normalize_trace_id,
    _parse_ui_chat_attachments,
    _request_likely_web_intent,
    _split_response_and_report,
    _ui_messages_to_llm,
)
from server.http.common.idempotency import (
    IdempotencyStore,
    fingerprint_json_payload,
    normalize_idempotency_key,
)
from server.http.common.responses import error_response, json_response
from server.http_api import (
    CANVAS_STATUS_CHARS_STEP,
    MAX_CONTENT_CHARS,
    MAX_TOTAL_PAYLOAD_CHARS,
    UI_SESSION_HEADER,
    _apply_agent_runtime_state,
    _build_canvas_chat_summary,
    _build_model_config,
    _build_output_artifacts,
    _build_ui_approval_decision,
    _decision_is_pending_blocking,
    _decision_workflow_context,
    _extract_files_from_tool_calls,
    _extract_named_files_from_output,
    _model_not_allowed_response,
    _model_not_selected_response,
    _normalize_auto_state,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _normalize_ui_decision,
    _publish_agent_activity,
    _publish_canvas_stream,
    _publish_chat_stream_delta,
    _publish_chat_stream_done,
    _publish_chat_stream_from_text,
    _publish_chat_stream_start,
    _request_likely_canvas,
    _resolve_agent,
    _resolve_provider_api_key,
    _resolve_ui_session_id_for_principal,
    _serialize_approval_request,
    _session_forbidden_response,
    _set_current_plan_step_status,
    _should_render_result_in_canvas,
    _split_chat_stream_chunks,
    _stream_preview_indicates_canvas,
    _stream_preview_ready_for_chat,
    _tool_calls_for_trace_id,
    _utc_now_iso,
    _workspace_root_for_session,
)
from server.ui_hub import UIHub
from shared.models import JSONValue, LLMMessage
from tools.workspace_tools import set_workspace_root as set_runtime_workspace_root

logger = logging.getLogger("SlavikAI.HttpAPI")

MessageLane = Literal["chat", "workspace"]


def _normalize_message_lane(value: object, *, default: MessageLane = "chat") -> MessageLane:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "workspace":
            return "workspace"
        if normalized == "chat":
            return "chat"
    return default


def _inception_runtime_config_for_lane(config: ModelConfig, lane: MessageLane) -> ModelConfig:
    if config.provider != "inception":
        return config
    return replace(
        config,
        reasoning_effort="instant",
        reasoning_summary=True,
        reasoning_summary_wait=False,
        diffusing=(lane == "workspace"),
    )


def _request_requires_root_gate(
    *,
    mode: str,
    content: str,
    llm_messages: list[LLMMessage],
    safe_mode: bool,
    skill_index: SkillIndex | None,
) -> bool:
    if content.strip().startswith("/"):
        return False
    if mode == "ask":
        return False
    route = classify_request(
        llm_messages,
        content,
        context={"safe_mode": safe_mode},
        skill_index=skill_index,
    )
    return route.route == "mwv"


def _auto_missing_paths(auto_state: dict[str, JSONValue] | None) -> list[str]:
    if not isinstance(auto_state, dict):
        return []
    status_raw = auto_state.get("status")
    error_code_raw = auto_state.get("error_code")
    if status_raw != "failed_worker" or error_code_raw != "missing_file":
        return []
    missing_raw = auto_state.get("missing_paths")
    if not isinstance(missing_raw, list):
        return []
    paths: list[str] = []
    for item in missing_raw:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                paths.append(cleaned)
    return paths


def _auto_missing_target_path_guidance(
    auto_state: dict[str, JSONValue] | None,
) -> str | None:
    if not isinstance(auto_state, dict):
        return None
    status_raw = auto_state.get("status")
    error_code_raw = auto_state.get("error_code")
    if status_raw != "failed_worker" or error_code_raw != "missing_target_path":
        return None
    return (
        "AUTO не может продолжить: для записи нужен явный путь к файлу в workspace. "
        "Укажи конкретный файл (например: `создай docs/raspberry-setup.md` "
        "или `обнови src/main.py`) и запусти ещё раз."
    )


def _build_missing_file_decision(
    *,
    session_id: str,
    missing_paths: list[str],
    root_path: str,
    mode: str,
    active_plan: dict[str, JSONValue] | None,
    active_task: dict[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    now = _utc_now_iso()
    return {
        "id": f"decision-{uuid.uuid4().hex}",
        "kind": "decision",
        "decision_type": "tool_approval",
        "status": "pending",
        "blocking": True,
        "reason": "missing_file",
        "summary": "Auto-run остановлен: не найдены нужные файлы в текущем root.",
        "proposed_action": {
            "error_code": "missing_file",
            "missing_paths": list(missing_paths),
            "root_path": root_path,
        },
        "options": [
            {
                "id": "approve_once",
                "title": "Понял",
                "action": "approve_once",
                "payload": {},
                "risk": "low",
            },
            {
                "id": "reject",
                "title": "Закрыть",
                "action": "reject",
                "payload": {},
                "risk": "low",
            },
        ],
        "default_option_id": "approve_once",
        "context": {
            "session_id": session_id,
            "source_endpoint": "chat.run_missing_file",
            "resume_payload": {
                "missing_paths": list(missing_paths),
                "root_path": root_path,
            },
            **_decision_workflow_context(
                mode=mode,
                active_plan=active_plan,
                active_task=active_task,
            ),
        },
        "created_at": now,
        "updated_at": now,
        "resolved_at": None,
    }


def _normalize_agent_decision(
    decision: dict[str, JSONValue],
    *,
    content: str,
    force_canvas: bool,
    lane: MessageLane,
    attachments: list[dict[str, str]],
    user_message_id: str | None,
    selected_model: dict[str, str],
) -> dict[str, JSONValue]:
    normalized = dict(decision)
    decision_type_raw = normalized.get("decision_type")
    decision_type = decision_type_raw.strip() if isinstance(decision_type_raw, str) else ""
    if decision_type in {"tool_approval", "plan_execute"}:
        return normalized
    normalized["decision_type"] = "agent_decision"

    context_raw = normalized.get("context")
    context = dict(context_raw) if isinstance(context_raw, dict) else {}
    source_endpoint_raw = context.get("source_endpoint")
    if not isinstance(source_endpoint_raw, str) or not source_endpoint_raw.strip():
        context["source_endpoint"] = "chat.agent_decision"

    resume_payload_raw = context.get("resume_payload")
    resume_payload = dict(resume_payload_raw) if isinstance(resume_payload_raw, dict) else {}
    source_request_raw = resume_payload.get("source_request")
    source_request = dict(source_request_raw) if isinstance(source_request_raw, dict) else {}
    source_request["content"] = content
    source_request["force_canvas"] = force_canvas
    source_request["lane"] = lane
    source_request["attachments"] = attachments
    resume_payload["source_request"] = source_request
    resume_payload["user_message_id"] = user_message_id
    resume_payload["selected_model_snapshot"] = {
        "provider": selected_model["provider"],
        "model": selected_model["model"],
    }
    context["resume_payload"] = resume_payload
    normalized["context"] = context
    return normalized


async def handle_ui_chat_send(
    request: web.Request,
    *,
    payload_override: dict[str, JSONValue] | None = None,
    bypass_root_gate: bool = False,
) -> web.Response:
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]
    hub: UIHub = request.app["ui_hub"]
    idempotency_store: IdempotencyStore = request.app["idempotency_store"]

    agent = None
    session_id: str | None = None
    status_opened = False
    error = False
    trace_id: str | None = None
    idempotency_key: str | None = None
    idempotency_fingerprint: str | None = None
    idempotency_active = False

    async def _complete_idempotency(payload: dict[str, JSONValue], *, status: int) -> None:
        if (
            not idempotency_active
            or idempotency_key is None
            or idempotency_fingerprint is None
            or session_id is None
        ):
            return
        await idempotency_store.complete(
            endpoint="ui.chat.send",
            session_id=session_id,
            key=idempotency_key,
            fingerprint=idempotency_fingerprint,
            status=status,
            payload=payload,
        )

    async def _abort_idempotency() -> None:
        if (
            not idempotency_active
            or idempotency_key is None
            or idempotency_fingerprint is None
            or session_id is None
        ):
            return
        await idempotency_store.abort(
            endpoint="ui.chat.send",
            session_id=session_id,
            key=idempotency_key,
            fingerprint=idempotency_fingerprint,
        )

    async def _complete_idempotency_error(
        *,
        status: int,
        message: str,
        error_type: str,
        code: str,
        trace: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        error_payload: dict[str, JSONValue] = {
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
                "trace_id": trace,
                "details": details or {},
            }
        }
        await _complete_idempotency(error_payload, status=status)

    try:
        try:
            agent = await _resolve_agent(request)
        except ModelNotAllowedError as exc:
            return _model_not_allowed_response(exc.model_id)
        if agent is None:
            return _model_not_selected_response()

        if payload_override is None:
            try:
                payload = await request.json()
            except Exception as exc:  # noqa: BLE001
                return error_response(
                    status=400,
                    message=f"Некорректный JSON: {exc}",
                    error_type="invalid_request_error",
                    code="invalid_json",
                )
        else:
            payload = dict(payload_override)
        if not isinstance(payload, dict):
            return error_response(
                status=400,
                message="JSON должен быть объектом.",
                error_type="invalid_request_error",
                code="invalid_json",
            )

        content_raw = payload.get("content")
        if not isinstance(content_raw, str):
            return error_response(
                status=400,
                message="content должен быть строкой.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        if len(content_raw) > MAX_CONTENT_CHARS:
            return error_response(
                status=413,
                message="content слишком длинный.",
                error_type="invalid_request_error",
                code="payload_too_large",
            )
        force_canvas_raw = payload.get("force_canvas")
        if force_canvas_raw is None:
            force_canvas = False
        elif isinstance(force_canvas_raw, bool):
            force_canvas = force_canvas_raw
        else:
            return error_response(
                status=400,
                message="force_canvas должен быть boolean.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        lane_raw = payload.get("lane")
        if lane_raw is None:
            lane: MessageLane = "chat"
        elif isinstance(lane_raw, str):
            lane = _normalize_message_lane(lane_raw, default="chat")
            if lane_raw.strip().lower() not in {"chat", "workspace"}:
                return error_response(
                    status=400,
                    message="lane должен быть chat|workspace.",
                    error_type="invalid_request_error",
                    code="invalid_request_error",
                )
        else:
            return error_response(
                status=400,
                message="lane должен быть chat|workspace.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        if lane == "workspace":
            force_canvas = False
        attachments_raw = payload.get("attachments")
        try:
            attachments, attachments_chars = _parse_ui_chat_attachments(attachments_raw)
        except ValueError as exc:
            return error_response(
                status=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        except OverflowError as exc:
            return error_response(
                status=413,
                message=str(exc),
                error_type="invalid_request_error",
                code="payload_too_large",
            )
        if not content_raw.strip() and not attachments:
            return error_response(
                status=400,
                message="Нужно передать content или attachments.",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )
        total_payload_chars = len(content_raw) + attachments_chars
        if total_payload_chars > MAX_TOTAL_PAYLOAD_CHARS:
            return error_response(
                status=413,
                message="Запрос слишком большой.",
                error_type="invalid_request_error",
                code="payload_too_large",
            )

        resolved_session_id, session_error = await _resolve_ui_session_id_for_principal(
            request,
            hub,
        )
        if session_error is not None:
            return session_error
        if resolved_session_id is None:
            return _session_forbidden_response()
        session_id = resolved_session_id
        if payload_override is None:
            idempotency_key = normalize_idempotency_key(request.headers.get("Idempotency-Key"))
        if idempotency_key is not None:
            idempotency_fingerprint = fingerprint_json_payload(
                {
                    "session_id": session_id,
                    "content": content_raw,
                    "force_canvas": force_canvas,
                    "lane": lane,
                    "attachments": attachments,
                    "bypass_root_gate": bypass_root_gate,
                }
            )
            idempotency_state, replay = await idempotency_store.begin(
                endpoint="ui.chat.send",
                session_id=session_id,
                key=idempotency_key,
                fingerprint=idempotency_fingerprint,
            )
            if idempotency_state == "replay" and replay is not None:
                response = json_response(replay.payload, status=replay.status)
                response.headers[UI_SESSION_HEADER] = session_id
                return response
            if idempotency_state == "conflict":
                return error_response(
                    status=409,
                    message="Idempotency-Key уже использован с другим payload.",
                    error_type="invalid_request_error",
                    code="idempotency_key_reused",
                )
            if idempotency_state == "in_progress":
                return error_response(
                    status=409,
                    message="Запрос с таким Idempotency-Key уже выполняется.",
                    error_type="invalid_request_error",
                    code="idempotency_in_progress",
                )
            idempotency_active = True
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            await _abort_idempotency()
            return _model_not_selected_response()
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        active_plan = _normalize_plan_payload(workflow.get("active_plan"))
        active_task = _normalize_task_payload(workflow.get("active_task"))
        auto_state = _normalize_auto_state(workflow.get("auto_state"))
        session_root = await _workspace_root_for_session(hub, session_id)
        session_messages = await hub.get_messages(session_id, lane=lane)
        chat_messages_payload = await hub.get_messages(session_id, lane="chat")
        workspace_messages_payload = await hub.get_messages(session_id, lane="workspace")
        preflight_messages = _ui_messages_to_llm(session_messages)

        requires_root_gate = _request_requires_root_gate(
            mode=mode,
            content=content_raw,
            llm_messages=preflight_messages,
            safe_mode=bool(getattr(agent, "tools_enabled", {}).get("safe_mode", False)),
            skill_index=getattr(agent, "skill_index", None),
        )
        if requires_root_gate and not bypass_root_gate:
            existing_decision = _normalize_ui_decision(
                await hub.get_session_decision(session_id),
                session_id=session_id,
            )
            if _decision_is_pending_blocking(existing_decision):
                await _abort_idempotency()
                return error_response(
                    status=409,
                    message="Pending decision already exists for this session.",
                    error_type="invalid_request_error",
                    code="decision_pending",
                )
            root_gate_approval_request: dict[str, JSONValue] = {
                "category": "EXEC_ARBITRARY",
                "required_categories": ["EXEC_ARBITRARY"],
                "tool": "chat_run_root",
                "prompt": {
                    "what": "Подтвердить запуск из выбранного Workspace Root",
                    "why": "Для auto/mwv запусков root подтверждается на каждый запуск.",
                    "risk": "Запуск может изменить файлы и/или выполнить команды в проекте.",
                    "changes": [str(session_root)],
                },
                "details": {
                    "root_path": str(session_root),
                    "mode": mode,
                },
            }
            root_gate_decision = _build_ui_approval_decision(
                approval_request=root_gate_approval_request,
                session_id=session_id,
                source_endpoint="chat.run_root",
                resume_payload={
                    "root_path": str(session_root),
                    "mode": mode,
                    "source_request": {
                        "content": content_raw,
                        "force_canvas": force_canvas,
                        "lane": lane,
                        "attachments": attachments,
                    },
                    "user_message_id": None,
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
            options_raw = root_gate_decision.get("options")
            if isinstance(options_raw, list):
                root_gate_decision["options"] = [
                    item
                    for item in options_raw
                    if isinstance(item, dict) and item.get("action") in {"approve_once", "reject"}
                ]
                root_gate_decision["default_option_id"] = "approve_once"
            await _set_current_plan_step_status(
                hub=hub,
                session_id=session_id,
                status="waiting_approval",
            )
            await hub.set_session_decision(session_id, root_gate_decision)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="approval.required",
                detail="chat.run_root",
            )
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = (
                await hub.get_session_artifacts(session_id) if lane == "chat" else []
            )
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response_payload_root_gate: dict[str, JSONValue] = {
                "session_id": session_id,
                "lane": lane,
                "messages": chat_messages_payload,
                "workspace_messages": workspace_messages_payload,
                "output": output_payload,
                "files": files_payload or [],
                "artifacts": artifacts_payload or [],
                "display": {
                    "target": "chat",
                    "artifact_id": None,
                    "forced": force_canvas,
                },
                "decision": _normalize_ui_decision(root_gate_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": None,
                "approval_request": None,
                "mwv_report": None,
                "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                "auto_state": _normalize_auto_state(current_workflow.get("auto_state")),
            }
            await _complete_idempotency(response_payload_root_gate, status=202)
            response = json_response(response_payload_root_gate, status=202)
            response.headers[UI_SESSION_HEADER] = session_id
            return response

        await hub.set_session_status(session_id, "busy")
        status_opened = True
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="request.received",
            detail="chat",
        )

        approved_categories = await session_store.get_categories(session_id)
        user_message = hub.create_message(
            role="user",
            content=content_raw.strip(),
            lane=lane,
            attachments=attachments,
        )
        await hub.append_message(session_id, user_message, lane=lane)
        user_message_id_raw = user_message.get("message_id")
        user_message_id = (
            user_message_id_raw
            if isinstance(user_message_id_raw, str) and user_message_id_raw
            else None
        )

        if not attachments and _request_likely_web_intent(content_raw):
            guidance_text = (
                "Для веб-поиска используй команду `/web <запрос>` в чате. "
                "После этого подтвердите approval, если он потребуется."
            )
            chat_stream_id = uuid.uuid4().hex
            await _publish_chat_stream_from_text(
                hub,
                session_id=session_id,
                stream_id=chat_stream_id,
                content=guidance_text,
                lane=lane,
            )
            if lane == "chat":
                await hub.set_session_output(session_id, guidance_text)
            assistant_message = hub.create_message(
                role="assistant",
                content=guidance_text,
                lane=lane,
                trace_id=None,
                parent_user_message_id=user_message_id,
            )
            await hub.append_message(session_id, assistant_message, lane=lane)
            messages = await hub.get_messages(session_id, lane="chat")
            workspace_messages = await hub.get_messages(session_id, lane="workspace")
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = (
                await hub.get_session_artifacts(session_id) if lane == "chat" else []
            )
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response_payload_guidance: dict[str, JSONValue] = {
                "session_id": session_id,
                "lane": lane,
                "messages": messages,
                "workspace_messages": workspace_messages,
                "output": output_payload,
                "files": files_payload or [],
                "artifacts": artifacts_payload or [],
                "display": {
                    "target": "chat",
                    "artifact_id": None,
                    "forced": force_canvas,
                },
                "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": None,
                "approval_request": None,
                "mwv_report": None,
                "mode": _normalize_mode_value(workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(workflow.get("active_plan")),
                "active_task": _normalize_task_payload(workflow.get("active_task")),
                "auto_state": _normalize_auto_state(workflow.get("auto_state")),
            }
            await _complete_idempotency(response_payload_guidance, status=200)
            response = json_response(response_payload_guidance)
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id, lane=lane))
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="context.prepared",
            detail="chat",
        )

        mwv_report: dict[str, JSONValue] | None = None
        approval_request: dict[str, JSONValue] | None = None
        ui_decision: dict[str, JSONValue] | None = None
        latest_auto_state: dict[str, JSONValue] | None = auto_state
        auto_progress_states: list[dict[str, JSONValue]] = []
        chat_stream_id = uuid.uuid4().hex
        live_stream_sent = False
        set_runtime_workspace_root(session_root)
        async with agent_lock:
            previous_trace_id = _normalize_trace_id(
                getattr(agent, "last_chat_interaction_id", None)
            )
            try:
                model_config = _build_model_config(
                    selected_model["provider"],
                    selected_model["model"],
                )
                model_config = _inception_runtime_config_for_lane(model_config, lane)
                api_key = _resolve_provider_api_key(selected_model["provider"])
                agent.reconfigure_models(model_config, main_api_key=api_key, persist=False)
            except Exception as exc:  # noqa: BLE001
                error_payload: dict[str, JSONValue] = {
                    "error": {
                        "message": f"Не удалось применить модель сессии: {exc}",
                        "type": "configuration_error",
                        "code": "model_config_invalid",
                        "trace_id": None,
                        "details": {},
                    }
                }
                await _complete_idempotency(error_payload, status=400)
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
                    "Failed to set session context for ui",
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
                detail="chat",
            )
            request_prefers_canvas = lane == "chat" and (
                force_canvas or _request_likely_canvas(content_raw)
            )
            response_raw: str
            respond_stream_method = getattr(agent, "respond_stream", None)
            if callable(respond_stream_method):
                stream_text = ""
                pending_chat_chunks: list[str] = []
                chat_stream_mode: Literal["pending", "chat", "canvas"] = (
                    "canvas" if request_prefers_canvas else "pending"
                )
                chat_content_stream_open = False
                canvas_status_stream_open = False
                canvas_status_chars = 0
                next_canvas_status_at = CANVAS_STATUS_CHARS_STEP
                try:
                    for stream_item in respond_stream_method(llm_messages):
                        delta = ""
                        delta_mode: Literal["append", "replace"] = "append"
                        if isinstance(stream_item, str):
                            delta = stream_item
                        elif isinstance(stream_item, LLMStreamChunk):
                            delta = stream_item.text
                            if stream_item.mode == "replace":
                                delta_mode = "replace"
                        elif isinstance(stream_item, dict):
                            delta_raw = stream_item.get("text")
                            if isinstance(delta_raw, str):
                                delta = delta_raw
                            mode_raw = stream_item.get("mode")
                            if mode_raw == "replace":
                                delta_mode = "replace"
                        if not delta:
                            continue
                        if delta_mode == "replace":
                            stream_text = delta
                        else:
                            stream_text = f"{stream_text}{delta}"
                        if chat_stream_mode == "canvas":
                            if delta_mode == "replace":
                                canvas_status_chars = len(delta)
                            else:
                                canvas_status_chars += len(delta)
                            if not canvas_status_stream_open:
                                await _publish_chat_stream_start(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    lane=lane,
                                )
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta="Статус: формирую результат в Canvas...",
                                    lane=lane,
                                )
                                live_stream_sent = True
                                canvas_status_stream_open = True
                                continue
                            if canvas_status_chars >= next_canvas_status_at:
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta="\nСтатус: обновляю содержимое Canvas...",
                                    lane=lane,
                                )
                                live_stream_sent = True
                                next_canvas_status_at += CANVAS_STATUS_CHARS_STEP
                            continue
                        if chat_stream_mode == "chat":
                            if delta_mode == "replace":
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta=delta,
                                    mode="replace",
                                    lane=lane,
                                )
                                live_stream_sent = True
                            else:
                                for chunk in _split_chat_stream_chunks(delta):
                                    await _publish_chat_stream_delta(
                                        hub,
                                        session_id=session_id,
                                        stream_id=chat_stream_id,
                                        delta=chunk,
                                        mode="append",
                                        lane=lane,
                                    )
                                    live_stream_sent = True
                                    await asyncio.sleep(0.005)
                            continue
                        if lane == "workspace" and delta_mode == "replace":
                            chat_stream_mode = "chat"
                            await _publish_chat_stream_start(
                                hub,
                                session_id=session_id,
                                stream_id=chat_stream_id,
                                lane=lane,
                            )
                            chat_content_stream_open = True
                            await _publish_chat_stream_delta(
                                hub,
                                session_id=session_id,
                                stream_id=chat_stream_id,
                                delta=delta,
                                mode="replace",
                                lane=lane,
                            )
                            live_stream_sent = True
                            continue
                        if delta_mode == "replace":
                            pending_chat_chunks = [delta]
                            pending_preview = delta
                        else:
                            pending_chat_chunks.append(delta)
                            pending_preview = "".join(pending_chat_chunks)
                        if lane == "chat" and _stream_preview_indicates_canvas(pending_preview):
                            chat_stream_mode = "canvas"
                            continue
                        if not _stream_preview_ready_for_chat(pending_preview):
                            continue
                        chat_stream_mode = "chat"
                        await _publish_chat_stream_start(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                            lane=lane,
                        )
                        chat_content_stream_open = True
                        if delta_mode == "replace":
                            await _publish_chat_stream_delta(
                                hub,
                                session_id=session_id,
                                stream_id=chat_stream_id,
                                delta=pending_preview,
                                mode="replace",
                                lane=lane,
                            )
                            live_stream_sent = True
                        else:
                            for chunk in _split_chat_stream_chunks(pending_preview):
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta=chunk,
                                    mode="append",
                                    lane=lane,
                                )
                                live_stream_sent = True
                                await asyncio.sleep(0.005)
                        pending_chat_chunks = []
                    if chat_content_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                            lane=lane,
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                            lane=lane,
                        )
                        canvas_status_stream_open = False
                    response_raw_candidate = getattr(agent, "last_stream_response_raw", None)
                    if isinstance(response_raw_candidate, str) and response_raw_candidate.strip():
                        response_raw = response_raw_candidate
                    else:
                        response_raw = stream_text
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Live stream failed; fallback to regular respond",
                        exc_info=True,
                        extra={"session_id": session_id, "error": str(exc)},
                    )
                    if chat_content_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                            lane=lane,
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                            lane=lane,
                        )
                        canvas_status_stream_open = False
                    response_raw = agent.respond(llm_messages)
            else:
                response_raw = agent.respond(llm_messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="agent.respond.end",
                detail="chat",
            )
            decision = _extract_decision_payload(response_text)
            if isinstance(decision, dict):
                decision = _normalize_agent_decision(
                    decision,
                    content=content_raw,
                    force_canvas=force_canvas,
                    lane=lane,
                    attachments=attachments,
                    user_message_id=user_message_id,
                    selected_model=selected_model,
                )
            trace_id = _normalize_trace_id(getattr(agent, "last_chat_interaction_id", None))
            if trace_id == previous_trace_id:
                # Не используем trace предыдущего ответа для вывода файлов текущего ответа.
                trace_id = None
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )
            latest_auto_state = _normalize_auto_state(getattr(agent, "last_auto_state", None))
            drain_auto_progress = getattr(agent, "drain_auto_progress_events", None)
            if callable(drain_auto_progress):
                drained = drain_auto_progress()
                if isinstance(drained, list):
                    for item in drained:
                        normalized_progress = _normalize_auto_state(item)
                        if normalized_progress is not None:
                            auto_progress_states.append(normalized_progress)
            ui_decision = _normalize_ui_decision(
                decision,
                session_id=session_id,
                trace_id=trace_id,
            )
            if approval_request is not None:
                approval_source_endpoint_raw = getattr(agent, "last_approval_source_endpoint", None)
                approval_source_endpoint = (
                    approval_source_endpoint_raw.strip()
                    if isinstance(approval_source_endpoint_raw, str)
                    and approval_source_endpoint_raw.strip()
                    else "chat.send"
                )
                approval_resume_payload_raw = getattr(agent, "last_approval_resume_payload", None)
                approval_resume_payload = (
                    dict(approval_resume_payload_raw)
                    if isinstance(approval_resume_payload_raw, dict)
                    else {}
                )
                if not approval_resume_payload:
                    approval_resume_payload = {
                        "source_request": {
                            "content": content_raw,
                            "force_canvas": force_canvas,
                            "lane": lane,
                            "attachments": attachments,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
                    }
                source_request_raw = approval_resume_payload.get("source_request")
                source_request = (
                    dict(source_request_raw) if isinstance(source_request_raw, dict) else {}
                )
                source_request["content"] = content_raw
                source_request["force_canvas"] = force_canvas
                source_request["lane"] = lane
                source_request["attachments"] = attachments
                approval_resume_payload["source_request"] = source_request
                ui_decision = _build_ui_approval_decision(
                    approval_request=approval_request,
                    session_id=session_id,
                    source_endpoint=approval_source_endpoint,
                    resume_payload=approval_resume_payload,
                    trace_id=trace_id,
                    workflow_context=_decision_workflow_context(
                        mode=mode,
                        active_plan=active_plan,
                        active_task=active_task,
                    ),
                )
        set_runtime_workspace_root(None)

        if latest_auto_state is not None:
            await hub.set_session_workflow(session_id, auto_state=latest_auto_state)
        for progress_state in auto_progress_states:
            await hub.publish(
                session_id,
                {
                    "type": "auto.progress",
                    "payload": {
                        "session_id": session_id,
                        "auto_state": progress_state,
                    },
                },
            )

        missing_paths = _auto_missing_paths(latest_auto_state)
        if ui_decision is None and missing_paths:
            ui_decision = _build_missing_file_decision(
                session_id=session_id,
                missing_paths=missing_paths,
                root_path=str(session_root),
                mode=mode,
                active_plan=active_plan,
                active_task=active_task,
            )
        guidance = _auto_missing_target_path_guidance(latest_auto_state)
        if guidance:
            trimmed = response_text.strip()
            response_text = f"{trimmed}\n\n{guidance}" if trimmed else guidance

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
                detail="chat",
            )
            messages = await hub.get_messages(session_id, lane="chat")
            workspace_messages = await hub.get_messages(session_id, lane="workspace")
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = (
                await hub.get_session_artifacts(session_id) if lane == "chat" else []
            )
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response_payload_blocking_decision: dict[str, JSONValue] = {
                "session_id": session_id,
                "lane": lane,
                "messages": messages,
                "workspace_messages": workspace_messages,
                "output": output_payload,
                "files": files_payload or [],
                "artifacts": artifacts_payload or [],
                "display": {
                    "target": "chat",
                    "artifact_id": None,
                    "forced": force_canvas,
                },
                "decision": _normalize_ui_decision(current_decision, session_id=session_id),
                "selected_model": current_model,
                "trace_id": trace_id,
                "approval_request": approval_request,
                "mwv_report": mwv_report,
                "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
                "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
                "active_task": _normalize_task_payload(current_workflow.get("active_task")),
                "auto_state": _normalize_auto_state(current_workflow.get("auto_state")),
            }
            await _complete_idempotency(response_payload_blocking_decision, status=200)
            response = json_response(response_payload_blocking_decision)
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        if lane == "chat":
            await hub.set_session_output(session_id, response_text)
        files_from_tools: list[str] = []
        if lane == "chat" and trace_id:
            tool_calls = _tool_calls_for_trace_id(trace_id) or []
            files_from_tools = _extract_files_from_tool_calls(tool_calls)
            if files_from_tools:
                await hub.merge_session_files(session_id, files_from_tools)
        named_files_count = (
            len(_extract_named_files_from_output(response_text)) if lane == "chat" else 0
        )
        display_target: Literal["chat", "canvas"]
        if lane != "chat" or approval_request is not None:
            display_target = "chat"
        else:
            display_target = (
                "canvas"
                if _should_render_result_in_canvas(
                    response_text=response_text,
                    files_from_tools=files_from_tools,
                    named_files_count=named_files_count,
                    force_canvas=force_canvas,
                )
                else "chat"
            )
        artifact_payloads: list[dict[str, JSONValue]] = []
        if lane == "chat":
            artifact_payloads = _build_output_artifacts(
                response_text=response_text,
                display_target=display_target,
                files_from_tools=files_from_tools,
            )
            for artifact in artifact_payloads:
                await hub.append_session_artifact(session_id, artifact)
        first_artifact = artifact_payloads[0] if artifact_payloads else None
        artifact_id_raw = first_artifact.get("id") if first_artifact is not None else None
        artifact_id = artifact_id_raw if isinstance(artifact_id_raw, str) else None
        if lane == "chat" and display_target == "canvas" and artifact_id is not None:
            stream_source_raw = (
                first_artifact.get("content") if first_artifact is not None else response_text
            )
            stream_source = (
                stream_source_raw if isinstance(stream_source_raw, str) else response_text
            )
            asyncio.create_task(
                _publish_canvas_stream(
                    hub,
                    session_id=session_id,
                    artifact_id=artifact_id,
                    content=stream_source,
                )
            )
        if display_target == "chat" and not live_stream_sent:
            await _publish_chat_stream_from_text(
                hub,
                session_id=session_id,
                stream_id=chat_stream_id,
                content=response_text,
                lane=lane,
            )
        if lane == "chat":
            chat_summary = (
                response_text
                if approval_request is not None or display_target == "chat" or force_canvas
                else _build_canvas_chat_summary(
                    artifact_title=None,
                    content_preview=None,
                )
            )
        else:
            chat_summary = response_text
        assistant_message = hub.create_message(
            role="assistant",
            content=chat_summary,
            lane=lane,
            trace_id=trace_id,
            parent_user_message_id=user_message_id,
        )
        await hub.append_message(session_id, assistant_message, lane=lane)
        await hub.set_session_decision(session_id, ui_decision)
        messages = await hub.get_messages(session_id, lane="chat")
        workspace_messages = await hub.get_messages(session_id, lane="workspace")
        output_payload = await hub.get_session_output(session_id)
        files_payload = await hub.get_session_files(session_id)
        artifacts_payload = await hub.get_session_artifacts(session_id) if lane == "chat" else []
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        current_workflow = await hub.get_session_workflow(session_id)
        response_payload_final: dict[str, JSONValue] = {
            "session_id": session_id,
            "lane": lane,
            "messages": messages,
            "workspace_messages": workspace_messages,
            "output": output_payload,
            "files": files_payload or [],
            "artifacts": artifacts_payload or [],
            "display": {
                "target": display_target,
                "artifact_id": artifact_id,
                "forced": force_canvas,
            },
            "decision": _normalize_ui_decision(current_decision, session_id=session_id),
            "selected_model": current_model,
            "trace_id": trace_id,
            "approval_request": approval_request,
            "mwv_report": mwv_report,
            "mode": _normalize_mode_value(current_workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(current_workflow.get("active_plan")),
            "active_task": _normalize_task_payload(current_workflow.get("active_task")),
            "auto_state": _normalize_auto_state(current_workflow.get("auto_state")),
        }
        await _complete_idempotency(response_payload_final, status=200)
        response = json_response(
            response_payload_final,
        )
        response.headers[UI_SESSION_HEADER] = session_id
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="response.ready",
            detail="chat",
        )
        return response
    except Exception:  # noqa: BLE001
        error = True
        resolved_trace_id = trace_id or _normalize_trace_id(
            getattr(agent, "last_chat_interaction_id", None)
        )
        logger.exception(
            "UI chat send failed",
            extra={
                "session_id": session_id,
                "trace_id": resolved_trace_id,
            },
        )
        if status_opened and session_id is not None:
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="error",
                detail="chat: internal_error",
            )
            await hub.set_session_status(session_id, "error")
        await _complete_idempotency_error(
            status=500,
            message="Внутренняя ошибка агента. См. trace_id.",
            error_type="internal_error",
            code="agent_error",
            trace=resolved_trace_id,
        )
        return error_response(
            status=500,
            message="Внутренняя ошибка агента. См. trace_id.",
            error_type="internal_error",
            code="agent_error",
            trace_id=resolved_trace_id,
        )
    finally:
        set_runtime_workspace_root(None)
        if status_opened and session_id is not None and not error:
            await hub.set_session_status(session_id, "ok")
