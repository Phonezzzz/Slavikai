from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from aiohttp import web

from server.http_api import (
    UI_SESSION_HEADER,
    _ensure_chat_owned,
    _extract_ui_session_id,
    _resolve_transport_token,
    _session_forbidden_response,
)
from server.ui_hub import UIHub
from shared.models import JSONValue


def _encode_sse_event(event: Mapping[str, JSONValue]) -> bytes:
    payload = json.dumps(event, ensure_ascii=False)
    event_id_raw = event.get("id")
    event_id = event_id_raw.strip() if isinstance(event_id_raw, str) else ""
    if event_id:
        return f"id: {event_id}\ndata: {payload}\n\n".encode()
    return f"data: {payload}\n\n".encode()


async def handle_ui_events_stream(request: web.Request) -> web.StreamResponse:
    hub: UIHub = request.app["ui_hub"]
    chat_id = request.match_info.get("chat_id", "").strip()
    if not chat_id:
        legacy_id = _extract_ui_session_id(request)
        chat_id = legacy_id.strip() if isinstance(legacy_id, str) and legacy_id.strip() else ""
    if not chat_id:
        return web.json_response(
            {"error": {"message": "chat_id обязателен.", "code": "invalid_request_error"}},
            status=400,
        )
    ownership_error = await _ensure_chat_owned(request, hub, chat_id)
    if ownership_error is not None:
        if not request.match_info.get("chat_id", "").strip() and ownership_error.status == 403:
            return _session_forbidden_response()
        return ownership_error
    last_event_id_raw = request.headers.get("Last-Event-ID")
    last_event_id = (
        last_event_id_raw.strip()
        if isinstance(last_event_id_raw, str) and last_event_id_raw.strip()
        else None
    )
    queue = await hub.subscribe(chat_id)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            UI_SESSION_HEADER: _resolve_transport_token(request),
        },
    )
    await response.prepare(request)
    initial_status_event = await hub.get_session_status_event(chat_id)
    await response.write(_encode_sse_event(initial_status_event))
    initial_workflow_event = await hub.get_session_workflow_event(chat_id)
    await response.write(_encode_sse_event(initial_workflow_event))
    replay_events = await hub.get_events_since(chat_id, after_event_id=last_event_id)
    for replay_event in replay_events:
        await response.write(_encode_sse_event(replay_event))

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await response.write(_encode_sse_event(event))
            except asyncio.TimeoutError:  # noqa: UP041
                await response.write(b": keep-alive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        await hub.unsubscribe(chat_id, queue)
    return response
