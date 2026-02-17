from __future__ import annotations

import asyncio
import json

from aiohttp import web

from server.http_api import (
    UI_SESSION_HEADER,
    _normalize_mode_value,
    _normalize_plan_payload,
    _normalize_task_payload,
    _resolve_ui_session_id_for_principal,
    _session_forbidden_response,
)
from server.ui_hub import UIHub


async def handle_ui_events_stream(request: web.Request) -> web.StreamResponse:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
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
    initial_status_payload = json.dumps(initial_status_event, ensure_ascii=False)
    await response.write(f"data: {initial_status_payload}\n\n".encode())
    initial_workflow = await hub.get_session_workflow(session_id)
    initial_workflow_event = {
        "type": "session.workflow",
        "payload": {
            "session_id": session_id,
            "mode": _normalize_mode_value(initial_workflow.get("mode"), default="ask"),
            "active_plan": _normalize_plan_payload(initial_workflow.get("active_plan")),
            "active_task": _normalize_task_payload(initial_workflow.get("active_task")),
        },
    }
    initial_workflow_payload = json.dumps(initial_workflow_event, ensure_ascii=False)
    await response.write(f"data: {initial_workflow_payload}\n\n".encode())

    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                payload = json.dumps(event, ensure_ascii=False)
                await response.write(f"data: {payload}\n\n".encode())
            except asyncio.TimeoutError:  # noqa: UP041
                await response.write(b": keep-alive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        await hub.unsubscribe(session_id, queue)
    return response
