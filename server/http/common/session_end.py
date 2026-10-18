from __future__ import annotations

import asyncio
import copy
import uuid
from typing import cast

from aiohttp import web

from core.decision.memory_save import build_memory_save_packet
from core.decision.session_end import build_session_end_packet, is_session_end_decision
from server.agent_provider import ScopedAgentProvider
from server.http.common.chat_cancellation import (
    ChatCancellationRegistry,
    ChatGenerationAlreadyActive,
)
from server.http.common.chat_payload import _ui_messages_to_llm
from server.http.common.mode_transitions import _auto_run_active
from server.http.common.runtime_contract import (
    AgentProtocol,
    _agent_lock_for_request,
    _agent_scope,
    _resolve_agent,
)
from shared.models import JSONValue


def pending_session_end(decision: dict[str, JSONValue] | None) -> bool:
    return (
        is_session_end_decision(decision)
        and decision is not None
        and decision.get("status") in {"pending", "executing"}
    )


async def _busy(request: web.Request, session_id: str) -> bool:
    workflow = await request.app["ui_hub"].get_session_workflow(session_id)
    auto = workflow.get("auto_state")
    task = workflow.get("active_task")
    return _auto_run_active(auto if isinstance(auto, dict) else None) or (
        isinstance(task, dict) and task.get("status") in {"running", "queued", "waiting_approval"}
    )


async def begin_session_end(
    request: web.Request, session_id: str, content: str
) -> dict[str, JSONValue]:
    hub = request.app["ui_hub"]
    registry = cast(ChatCancellationRegistry, request.app["chat_cancellation_registry"])
    try:
        generation = await registry.start(session_id=session_id, stream_id=uuid.uuid4().hex)
    except ChatGenerationAlreadyActive:
        return {"error": "chat_generation_in_progress"}
    try:
        if await _busy(request, session_id):
            return {"error": "task_already_running"}
        async with _agent_lock_for_request(request, session_id):
            decision = await hub.get_session_decision(session_id)
            if decision and decision.get("status") in {"pending", "executing"}:
                if pending_session_end(decision):
                    return {"decision": decision}
                return {"error": "decision_pending"}
            provider = cast(ScopedAgentProvider[AgentProtocol], request.app["agent_provider"])
            agent = await provider.get_existing_for_current_task(_agent_scope(request, session_id))
            if await _busy(request, session_id) or (
                agent is not None and agent.session_context_busy()
            ):
                return {"error": "task_already_running"}
            previous = decision or {}
            decision = build_session_end_packet().to_dict()
            message = hub.create_message(role="user", content=content, lane="chat")
            # Исключаем только эту команду из transcript, не переписывая историю.
            decision["context"] = {
                "source_endpoint": "session.end",
                "command_message_id": message["message_id"],
            }
            previous_id = previous.get("id")
            previous_status = previous.get("status")
            started, _ = await hub.transition_session_decision(
                session_id,
                expected_id=previous_id if isinstance(previous_id, str) else None,
                expected_status=previous_status if isinstance(previous_status, str) else None,
                next_decision=decision,
                cancellation_tokens=(generation.token,),
            )
            if not started:
                return {
                    "error": "session_end_cancelled"
                    if generation.token.is_set()
                    else "decision_pending"
                }
            await hub.append_message(session_id, message, lane="chat")
            return {"decision": decision}
    finally:
        await registry.finish(generation)


async def _publish_close_notice(
    request: web.Request, session_id: str, text: str
) -> list[JSONValue]:
    hub = request.app["ui_hub"]
    try:
        await hub.append_message(
            session_id, hub.create_message(role="assistant", content=text, lane="chat"), lane="chat"
        )
    except Exception:  # noqa: BLE001
        request.app["http_api_logger"].exception("Session closed; notice publication failed")
        return [
            {
                "code": "session_notice_failed",
                "message": "Завершение выполнено, но служебное сообщение не удалось сохранить.",
            }
        ]
    return []


async def resolve_session_end(
    request: web.Request,
    session_id: str,
    decision: dict[str, JSONValue],
    choice: str,
    edited_action: dict[str, JSONValue] | None,
    cancellation: asyncio.Event,
) -> dict[str, JSONValue]:
    hub = request.app["ui_hub"]
    registry = cast(ChatCancellationRegistry, request.app["chat_cancellation_registry"])
    result: dict[str, JSONValue] = {
        "ok": False,
        "source_endpoint": "session.end",
        "resume_started": False,
    }
    try:
        generation = await registry.start(session_id=session_id, stream_id=uuid.uuid4().hex)
    except ChatGenerationAlreadyActive:
        return {**result, "error": "chat_generation_in_progress"}
    try:
        if await _busy(request, session_id):
            return {**result, "error": "task_already_running"}
        # No/Cancel не создают Agent и не требуют настроенной модели.
        provider = cast(ScopedAgentProvider[AgentProtocol], request.app["agent_provider"])
        is_preview = decision.get("decision_type") == "memory_save"
        needs_model = choice in {"confirm", "edit_and_confirm"}
        try:
            agent = (
                await _resolve_agent(request, session_id)
                if needs_model
                else await provider.get_existing_for_current_task(_agent_scope(request, session_id))
            )
        except PermissionError:
            return {**result, "error": "model_access_denied"}
        except Exception:  # noqa: BLE001
            request.app["http_api_logger"].exception("Session model resolution failed")
            return {**result, "error": "session_summary_failed"}
        async with _agent_lock_for_request(request, session_id):
            if await _busy(request, session_id) or (
                agent is not None and agent.session_context_busy()
            ):
                return {**result, "error": "task_already_running"}
            if cancellation.is_set() or generation.token.is_set():
                return {**result, "error": "session_end_cancelled"}
            latest = await hub.get_session_decision(session_id)
            if (
                not latest
                or latest.get("id") != decision.get("id")
                or latest.get("status") != "executing"
            ):
                return {**result, "error": "session_end_state_changed"}
            if choice == "abort" or (is_preview and choice == "reject"):
                return {**result, "ok": True, "data": {"closed": False, "saved": False}}
            if not is_preview and choice == "reject":
                if agent is not None:
                    agent.close_session_context()
                warnings = await _publish_close_notice(
                    request,
                    session_id,
                    "Сессия завершена без сохранения в Memory. История чата сохранена.",
                )
                data: dict[str, JSONValue] = {"closed": True, "saved": False}
                if warnings:
                    data["warnings"] = warnings
                return {**result, "ok": True, "data": data}

            if agent is None:
                return {**result, "error": "model_not_selected"}
            categories = await request.app["session_store"].get_categories(
                _agent_scope(request, session_id)
            )
            agent.set_session_context(session_id, categories)
            if not is_preview:
                context = decision.get("context")
                command_id = (
                    context.get("command_message_id") if isinstance(context, dict) else None
                )
                history = await hub.get_messages(session_id, lane="chat")
                history = [
                    message for message in history if message.get("message_id") != command_id
                ]
                try:
                    preview = await asyncio.to_thread(
                        agent.prepare_session_summary,
                        _ui_messages_to_llm(history),
                        cancellation_token=generation.token,
                    )
                except Exception:  # noqa: BLE001
                    request.app["http_api_logger"].exception("Session summary preparation failed")
                    return {**result, "error": "session_summary_failed"}
                if cancellation.is_set() or generation.token.is_set():
                    return {**result, "error": "session_end_cancelled"}
                if preview is None:
                    return {**result, "error": "session_summary_empty"}
                packet = build_memory_save_packet(preview).to_dict()
                packet["context"] = {"source_endpoint": "session.end"}
                packet["summary"] = (
                    "Проверьте данные для Memory. Сохранение завершит сессию; "
                    "«Не сохранять» вернёт к разговору. История чата останется."
                )
                published, latest = await hub.transition_session_decision(
                    session_id,
                    expected_id=str(decision["id"]),
                    expected_status="executing",
                    next_decision=packet,
                    cancellation_tokens=(cancellation, generation.token),
                )
                if not published:
                    code = (
                        "session_end_cancelled"
                        if cancellation.is_set() or generation.token.is_set()
                        else "session_end_state_changed"
                    )
                    return {**result, "error": code}
                return {**result, "ok": True, "published_decision": latest}
            proposed = decision.get("proposed_action")
            preview = copy.deepcopy(proposed) if isinstance(proposed, dict) else {}
            if choice == "edit_and_confirm":
                text = edited_action.get("text") if edited_action else None
                claims = preview.get("claims")
                if not isinstance(text, str) or not text.strip():
                    return {**result, "error": "invalid_memory_edit"}
                if (
                    not isinstance(claims, list)
                    or len(claims) != 1
                    or not isinstance(claims[0], dict)
                ):
                    return {**result, "error": "memory_preview_invalid"}
                preview["text"] = text
                claims[0]["value_json"] = {"text": text}
            latest = await hub.get_session_decision(session_id)
            if (
                not latest
                or latest.get("id") != decision.get("id")
                or latest.get("status") != "executing"
            ):
                return {**result, "error": "session_end_state_changed"}
            if cancellation.is_set() or generation.token.is_set():
                return {**result, "error": "session_end_cancelled"}
            try:
                saved = agent.apply_confirmed_memory_save(
                    preview, source_kind="session.summary", source_id=session_id
                )
            except Exception:  # noqa: BLE001
                request.app["http_api_logger"].exception("Session Memory confirmation failed")
                return {**result, "error": "memory_write_failed"}
            if not saved.ok:
                return {**result, "error": "memory_write_failed"}
            agent.close_session_context()
            warnings = await _publish_close_notice(
                request,
                session_id,
                "Данные сохранены в Memory. Сессия завершена. История чата сохранена.",
            )
            data = {**saved.data, "closed": True, "saved": True}
            if warnings:
                data["warnings"] = warnings
            return {**result, "ok": True, "data": data}

    finally:
        await registry.finish(generation)
