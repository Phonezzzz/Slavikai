from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

from aiohttp import web

from server.http_api import (
    UI_SESSION_HEADER,
    _resolve_ui_session_id_for_principal,
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
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    last_event_id_raw = request.headers.get("Last-Event-ID")
    last_event_id = (
        last_event_id_raw.strip()
        if isinstance(last_event_id_raw, str) and last_event_id_raw.strip()
        else None
    )
    queue = await hub.subscribe(session_id)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            UI_SESSION_HEADER: session_id,
        },
    )
    await response.prepare(request)
    initial_status_event = await hub.get_session_status_event(session_id)
    await response.write(_encode_sse_event(initial_status_event))
    initial_workflow_event = await hub.get_session_workflow_event(session_id)
    await response.write(_encode_sse_event(initial_workflow_event))
    replay_events, stale_last_event_id = await hub.get_events_since(
        session_id,
        last_event_id=last_event_id,
    )
    for replay_event in replay_events:
        await response.write(_encode_sse_event(replay_event))
    if stale_last_event_id and last_event_id is not None:
        resync_event = hub.build_resync_required_event(
            session_id=session_id,
            reason="last_event_id_stale",
            resync_only=True,
        )
        await response.write(_encode_sse_event(resync_event))

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
        await hub.unsubscribe(session_id, queue)
    return response
