from __future__ import annotations

import asyncio
import json
from typing import cast

from aiohttp import web

from server.http.common.responses import error_response, json_response
from server.http_api import (
    UI_SESSION_HEADER,
    _load_effective_session_security,
    _resolve_ui_session_id_for_principal,
    _session_forbidden_response,
    _workspace_root_for_session,
)
from server.terminal_manager import TerminalManager
from server.ui_hub import UIHub
from shared.models import JSONValue


def _encode_sse_event(event: dict[str, JSONValue]) -> bytes:
    payload = json.dumps(event, ensure_ascii=False)
    event_id_raw = event.get("id")
    event_id = event_id_raw.strip() if isinstance(event_id_raw, str) else ""
    if event_id:
        return f"id: {event_id}\ndata: {payload}\n\n".encode()
    return f"data: {payload}\n\n".encode()


async def _resolve_terminal_session_id(
    request: web.Request,
) -> tuple[str | None, web.Response | None]:
    hub: UIHub = request.app["ui_hub"]
    session_id, session_error = await _resolve_ui_session_id_for_principal(request, hub)
    if session_error is not None:
        return None, session_error
    if session_id is None:
        return None, _session_forbidden_response()
    return session_id, None


def _terminal_manager(request: web.Request) -> TerminalManager:
    return cast(TerminalManager, request.app["terminal_manager"])


async def _require_terminal_yolo(request: web.Request, session_id: str) -> web.Response | None:
    hub: UIHub = request.app["ui_hub"]
    _, effective_policy = await _load_effective_session_security(hub=hub, session_id=session_id)
    profile_raw = effective_policy.get("profile")
    profile = profile_raw.strip().lower() if isinstance(profile_raw, str) else "sandbox"
    if profile == "yolo":
        return None
    return error_response(
        status=403,
        message="Real terminal requires policy.profile == 'yolo'.",
        error_type="forbidden",
        code="terminal_yolo_required",
    )


def _terminal_absent_payload(*, session_id: str, workspace_root: str) -> dict[str, JSONValue]:
    return {
        "session_id": session_id,
        "terminal": {
            "terminal_id": None,
            "status": "not_started",
            "workspace_root": workspace_root,
            "spawn_cwd": None,
            "rows": None,
            "cols": None,
            "created_at": None,
            "updated_at": None,
            "closed_at": None,
            "exit_code": None,
            "output": "",
        },
    }


def _normalize_dimension(value: object, *, default: int) -> int:
    if not isinstance(value, int):
        return default
    if value < 1:
        return default
    if value > 500:
        return 500
    return value


async def handle_ui_terminal_create(request: web.Request) -> web.Response:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    yolo_error = await _require_terminal_yolo(request, session_id)
    if yolo_error is not None:
        return yolo_error
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        return error_response(
            status=400,
            message="JSON должен быть объектом.",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    workspace_root = str(await _workspace_root_for_session(request.app["ui_hub"], session_id))
    rows = _normalize_dimension(payload.get("rows"), default=24)
    cols = _normalize_dimension(payload.get("cols"), default=80)
    snapshot = await _terminal_manager(request).create_or_get(
        session_id,
        workspace_root=workspace_root,
        rows=rows,
        cols=cols,
    )
    response = json_response({"session_id": session_id, "terminal": snapshot})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_terminal_get(request: web.Request) -> web.Response:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workspace_root = str(await _workspace_root_for_session(request.app["ui_hub"], session_id))
    snapshot = await _terminal_manager(request).get_snapshot(session_id)
    payload: dict[str, JSONValue]
    if snapshot is not None:
        payload = {"session_id": session_id, "terminal": snapshot}
    else:
        payload = _terminal_absent_payload(session_id=session_id, workspace_root=workspace_root)
    response = json_response(payload)
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_terminal_input(request: web.Request) -> web.Response:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    yolo_error = await _require_terminal_yolo(request, session_id)
    if yolo_error is not None:
        return yolo_error
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
    input_raw = payload.get("input")
    if not isinstance(input_raw, str):
        return error_response(
            status=400,
            message="input должен быть строкой.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    try:
        snapshot = await _terminal_manager(request).write_input(session_id, input_raw)
    except KeyError:
        return error_response(
            status=404,
            message="Terminal not started.",
            error_type="invalid_request_error",
            code="terminal_not_started",
        )
    except RuntimeError as exc:
        return error_response(
            status=409,
            message=str(exc),
            error_type="invalid_request_error",
            code="terminal_not_running",
        )
    response = json_response({"session_id": session_id, "terminal": snapshot})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_terminal_resize(request: web.Request) -> web.Response:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    yolo_error = await _require_terminal_yolo(request, session_id)
    if yolo_error is not None:
        return yolo_error
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
    rows_raw = payload.get("rows")
    cols_raw = payload.get("cols")
    if not isinstance(rows_raw, int) or not isinstance(cols_raw, int):
        return error_response(
            status=400,
            message="rows и cols должны быть int.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        )
    rows = _normalize_dimension(rows_raw, default=24)
    cols = _normalize_dimension(cols_raw, default=80)
    try:
        snapshot = await _terminal_manager(request).resize(session_id, rows=rows, cols=cols)
    except KeyError:
        return error_response(
            status=404,
            message="Terminal not started.",
            error_type="invalid_request_error",
            code="terminal_not_started",
        )
    except RuntimeError as exc:
        return error_response(
            status=409,
            message=str(exc),
            error_type="invalid_request_error",
            code="terminal_not_running",
        )
    response = json_response({"session_id": session_id, "terminal": snapshot})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_terminal_close(request: web.Request) -> web.Response:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    workspace_root = str(await _workspace_root_for_session(request.app["ui_hub"], session_id))
    try:
        snapshot = await _terminal_manager(request).close(session_id)
    except KeyError:
        response = json_response(
            _terminal_absent_payload(session_id=session_id, workspace_root=workspace_root)
        )
        response.headers[UI_SESSION_HEADER] = session_id
        return response
    response = json_response({"session_id": session_id, "terminal": snapshot})
    response.headers[UI_SESSION_HEADER] = session_id
    return response


async def handle_ui_terminal_stream(request: web.Request) -> web.StreamResponse:
    session_id, session_error = await _resolve_terminal_session_id(request)
    if session_error is not None:
        return session_error
    if session_id is None:
        return _session_forbidden_response()
    manager = _terminal_manager(request)
    snapshot = await manager.get_snapshot(session_id)
    if snapshot is None:
        return error_response(
            status=404,
            message="Terminal not started.",
            error_type="invalid_request_error",
            code="terminal_not_started",
        )
    last_event_id_raw = request.headers.get("Last-Event-ID")
    last_event_id = (
        last_event_id_raw.strip()
        if isinstance(last_event_id_raw, str) and last_event_id_raw.strip()
        else None
    )
    queue = await manager.subscribe(session_id)
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
    replay_events, _ = await manager.get_events_since(session_id, last_event_id=last_event_id)
    for replay_event in replay_events:
        await response.write(_encode_sse_event(replay_event))
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
                await response.write(_encode_sse_event(event))
            except TimeoutError:
                await response.write(b": keep-alive\n\n")
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        await manager.unsubscribe(session_id, queue)
    return response
