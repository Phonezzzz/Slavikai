from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Literal

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from server.http.common.chat_payload import (
    _extract_decision_payload,
    _normalize_trace_id,
    _parse_ui_chat_attachments,
    _request_likely_web_intent,
    _split_response_and_report,
    _ui_messages_to_llm,
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
    _canvas_summary_title_from_artifact,
    _decision_is_pending_blocking,
    _decision_workflow_context,
    _emit_status,
    _extract_files_from_tool_calls,
    _extract_named_files_from_output,
    _model_not_allowed_response,
    _model_not_selected_response,
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
)
from server.ui_hub import UIHub
from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HttpAPI")


async def handle_ui_chat_send(request: web.Request) -> web.Response:
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
        selected_model = await hub.get_session_model(session_id)
        if selected_model is None:
            return _model_not_selected_response()
        workflow = await hub.get_session_workflow(session_id)
        mode = _normalize_mode_value(workflow.get("mode"), default="ask")
        active_plan = _normalize_plan_payload(workflow.get("active_plan"))
        active_task = _normalize_task_payload(workflow.get("active_task"))

        await hub.set_session_status(session_id, "busy")
        status_opened = True
        await _emit_status(
            hub,
            session_id=session_id,
            phase="request.received",
            text="Запрос получен…",
        )
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
            attachments=attachments,
        )
        await hub.append_message(session_id, user_message)
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
            )
            await hub.set_session_output(session_id, guidance_text)
            assistant_message = hub.create_message(
                role="assistant",
                content=guidance_text,
                trace_id=None,
                parent_user_message_id=user_message_id,
            )
            await hub.append_message(session_id, assistant_message)
            messages = await hub.get_messages(session_id)
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = await hub.get_session_artifacts(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
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
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        llm_messages = _ui_messages_to_llm(await hub.get_messages(session_id))
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="context.prepared",
            detail="chat",
        )
        await _emit_status(
            hub,
            session_id=session_id,
            phase="context.prepared",
            text="Собираю контекст…",
        )

        mwv_report: dict[str, JSONValue] | None = None
        trace_id: str | None = None
        approval_request: dict[str, JSONValue] | None = None
        ui_decision: dict[str, JSONValue] | None = None
        chat_stream_id = uuid.uuid4().hex
        live_stream_sent = False
        async with agent_lock:
            previous_trace_id = _normalize_trace_id(
                getattr(agent, "last_chat_interaction_id", None)
            )
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
            await _emit_status(
                hub,
                session_id=session_id,
                phase="agent.respond.start",
                text="Готовлю ответ…",
            )
            request_prefers_canvas = force_canvas or _request_likely_canvas(content_raw)
            response_raw: str
            respond_stream_method = getattr(agent, "respond_stream", None)
            if callable(respond_stream_method):
                stream_chunks: list[str] = []
                pending_chat_chunks: list[str] = []
                chat_stream_mode: Literal["pending", "chat", "canvas"] = (
                    "canvas" if request_prefers_canvas else "pending"
                )
                chat_content_stream_open = False
                canvas_status_stream_open = False
                canvas_status_chars = 0
                next_canvas_status_at = CANVAS_STATUS_CHARS_STEP
                try:
                    for delta in respond_stream_method(llm_messages):
                        if not isinstance(delta, str) or not delta:
                            continue
                        stream_chunks.append(delta)
                        if chat_stream_mode == "canvas":
                            canvas_status_chars += len(delta)
                            if not canvas_status_stream_open:
                                await _publish_chat_stream_start(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                )
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta="Статус: формирую результат в Canvas...",
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
                                )
                                live_stream_sent = True
                                next_canvas_status_at += CANVAS_STATUS_CHARS_STEP
                            continue
                        if chat_stream_mode == "chat":
                            for chunk in _split_chat_stream_chunks(delta):
                                await _publish_chat_stream_delta(
                                    hub,
                                    session_id=session_id,
                                    stream_id=chat_stream_id,
                                    delta=chunk,
                                )
                                live_stream_sent = True
                                await asyncio.sleep(0.005)
                            continue
                        pending_chat_chunks.append(delta)
                        pending_preview = "".join(pending_chat_chunks)
                        if _stream_preview_indicates_canvas(pending_preview):
                            chat_stream_mode = "canvas"
                            continue
                        if not _stream_preview_ready_for_chat(pending_preview):
                            continue
                        chat_stream_mode = "chat"
                        await _publish_chat_stream_start(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        chat_content_stream_open = True
                        for chunk in _split_chat_stream_chunks(pending_preview):
                            await _publish_chat_stream_delta(
                                hub,
                                session_id=session_id,
                                stream_id=chat_stream_id,
                                delta=chunk,
                            )
                            live_stream_sent = True
                            await asyncio.sleep(0.005)
                        pending_chat_chunks = []
                    if chat_content_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
                        )
                        canvas_status_stream_open = False
                    response_raw_candidate = getattr(agent, "last_stream_response_raw", None)
                    if isinstance(response_raw_candidate, str) and response_raw_candidate.strip():
                        response_raw = response_raw_candidate
                    else:
                        response_raw = "".join(stream_chunks)
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
                        )
                        chat_content_stream_open = False
                    if canvas_status_stream_open:
                        await _publish_chat_stream_done(
                            hub,
                            session_id=session_id,
                            stream_id=chat_stream_id,
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
            trace_id = _normalize_trace_id(getattr(agent, "last_chat_interaction_id", None))
            if trace_id == previous_trace_id:
                # Не используем trace предыдущего ответа для вывода файлов текущего ответа.
                trace_id = None
            approval_request = _serialize_approval_request(
                getattr(agent, "last_approval_request", None),
            )
            ui_decision = _normalize_ui_decision(
                decision,
                session_id=session_id,
                trace_id=trace_id,
            )
            if approval_request is not None:
                ui_decision = _build_ui_approval_decision(
                    approval_request=approval_request,
                    session_id=session_id,
                    source_endpoint="chat.send",
                    resume_payload={
                        "source_request": {
                            "content": content_raw,
                            "force_canvas": force_canvas,
                            "attachments": attachments,
                        },
                        "user_message_id": user_message_id,
                        "selected_model_snapshot": {
                            "provider": selected_model["provider"],
                            "model": selected_model["model"],
                        },
                    },
                    trace_id=trace_id,
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
            await _emit_status(
                hub,
                session_id=session_id,
                phase="approval.required",
                text="Требуется подтверждение действия.",
            )
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="approval.required",
                detail="chat",
            )
            messages = await hub.get_messages(session_id)
            output_payload = await hub.get_session_output(session_id)
            files_payload = await hub.get_session_files(session_id)
            artifacts_payload = await hub.get_session_artifacts(session_id)
            current_decision = await hub.get_session_decision(session_id)
            current_model = await hub.get_session_model(session_id)
            current_workflow = await hub.get_session_workflow(session_id)
            response = json_response(
                {
                    "session_id": session_id,
                    "messages": messages,
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
                }
            )
            response.headers[UI_SESSION_HEADER] = session_id
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="response.ready",
                detail="chat",
            )
            return response

        await hub.set_session_output(session_id, response_text)
        files_from_tools: list[str] = []
        if trace_id:
            tool_calls = _tool_calls_for_trace_id(trace_id) or []
            files_from_tools = _extract_files_from_tool_calls(tool_calls)
            if files_from_tools:
                await hub.merge_session_files(session_id, files_from_tools)
        named_files_count = len(_extract_named_files_from_output(response_text))
        display_target: Literal["chat", "canvas"]
        if approval_request is not None:
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
        artifact_title = _canvas_summary_title_from_artifact(first_artifact)
        if display_target == "canvas" and artifact_id is not None:
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
            )
        chat_summary = (
            response_text
            if approval_request is not None or display_target == "chat"
            else _build_canvas_chat_summary(artifact_title=artifact_title)
        )
        assistant_message = hub.create_message(
            role="assistant",
            content=chat_summary,
            trace_id=trace_id,
            parent_user_message_id=user_message_id,
        )
        await hub.append_message(session_id, assistant_message)
        await hub.set_session_decision(session_id, ui_decision)
        messages = await hub.get_messages(session_id)
        output_payload = await hub.get_session_output(session_id)
        files_payload = await hub.get_session_files(session_id)
        artifacts_payload = await hub.get_session_artifacts(session_id)
        current_decision = await hub.get_session_decision(session_id)
        current_model = await hub.get_session_model(session_id)
        current_workflow = await hub.get_session_workflow(session_id)
        response = json_response(
            {
                "session_id": session_id,
                "messages": messages,
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
            },
        )
        response.headers[UI_SESSION_HEADER] = session_id
        await _publish_agent_activity(
            hub,
            session_id=session_id,
            phase="response.ready",
            detail="chat",
        )
        await _emit_status(
            hub,
            session_id=session_id,
            phase="response.ready",
            text="",
        )
        return response
    except Exception as exc:  # noqa: BLE001
        error = True
        if status_opened and session_id is not None:
            await _publish_agent_activity(
                hub,
                session_id=session_id,
                phase="error",
                detail=f"chat: {exc}",
            )
            await _emit_status(
                hub,
                session_id=session_id,
                phase="error",
                text="Ошибка выполнения запроса.",
            )
            await hub.set_session_status(session_id, "error")
        return error_response(
            status=500,
            message=f"Agent error: {exc}",
            error_type="internal_error",
            code="agent_error",
        )
    finally:
        if status_opened and session_id is not None and not error:
            await hub.set_session_status(session_id, "ok")
