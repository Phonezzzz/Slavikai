from __future__ import annotations

import logging
import time
import uuid

from aiohttp import web

from config.model_whitelist import ModelNotAllowedError
from server.http.common.responses import error_response, json_response
from server.http_api import (
    TOOL_PIPELINE_ENABLED,
    _extract_session_id,
    _model_not_allowed_response,
    _model_not_selected_response,
    _parse_chat_request,
    _resolve_agent,
    _split_response_and_report,
)
from shared.models import JSONValue

logger = logging.getLogger("SlavikAI.HttpAPI")


async def handle_chat_completions(request: web.Request) -> web.Response:
    try:
        agent = await _resolve_agent(request)
    except ModelNotAllowedError as exc:
        return _model_not_allowed_response(exc.model_id)
    if agent is None:
        return _model_not_selected_response()
    agent_lock = request.app["agent_lock"]
    session_store = request.app["session_store"]

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

    parsed, error_text = _parse_chat_request(payload)
    if parsed is None:
        return error_response(
            status=400,
            message=error_text,
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    session_id = _extract_session_id(request, payload) or parsed.session_id
    if session_id is None:
        session_id = str(uuid.uuid4())

    approved_categories = await session_store.get_categories(session_id)
    try:
        agent.set_session_context(session_id, approved_categories)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to set session context",
            exc_info=True,
            extra={
                "session_id": session_id,
                "approved_categories": sorted(approved_categories),
                "error": str(exc),
            },
        )

    if parsed.stream:
        return error_response(
            status=400,
            message="Streaming is not supported.",
            error_type="not_supported",
            code="streaming_not_supported",
        )

    if parsed.tool_calling_present and not TOOL_PIPELINE_ENABLED:
        return error_response(
            status=400,
            message="Tool calling is not supported.",
            error_type="not_supported",
            code="tool_calling_not_supported",
        )

    trace_id: str | None = None
    mwv_report: dict[str, JSONValue] | None = None
    try:
        async with agent_lock:
            response_raw = agent.respond(parsed.messages)
            response_text, mwv_report = _split_response_and_report(response_raw)
            trace_id = agent.last_chat_interaction_id
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=500,
            message=f"Agent error: {exc}",
            error_type="internal_error",
            code="agent_error",
            trace_id=trace_id,
        )

    if parsed.sampling_warnings:
        agent.tracer.log(
            "request_warning",
            "Ignored unknown sampling params",
            {"fields": parsed.sampling_warnings},
        )

    approval_request = getattr(agent, "last_approval_request", None)
    if approval_request is not None:
        if trace_id is None:
            trace_id = agent.last_chat_interaction_id
        return error_response(
            status=400,
            message="Approval required.",
            error_type="tool_error",
            code="approval_required",
            trace_id=trace_id,
            details={
                "category": approval_request.category,
                "required_categories": approval_request.required_categories,
                "session_id": approval_request.session_id,
                "prompt": {
                    "what": approval_request.prompt.what,
                    "why": approval_request.prompt.why,
                    "risk": approval_request.prompt.risk,
                    "changes": approval_request.prompt.changes,
                },
                "blocked_reason": "approval_required",
            },
        )

    if trace_id is None:
        return error_response(
            status=500,
            message="Trace ID was not generated.",
            error_type="internal_error",
            code="trace_id_missing",
        )

    session_approved = bool(approved_categories)
    safe_mode = bool(agent.tools_enabled.get("safe_mode", False))

    slavik_meta: dict[str, JSONValue] = {
        "trace_id": trace_id,
        "session_id": session_id,
        "session_approved": session_approved,
        "safe_mode": safe_mode,
    }
    if mwv_report is not None:
        slavik_meta["mwv_report"] = mwv_report

    response_payload: dict[str, JSONValue] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": parsed.model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": response_text},
            }
        ],
        "slavik_meta": slavik_meta,
    }
    return json_response(response_payload)
