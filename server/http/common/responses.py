from __future__ import annotations

from aiohttp import web

from shared.models import JSONValue


def json_response(payload: dict[str, JSONValue], *, status: int = 200) -> web.Response:
    return web.json_response(payload, status=status)


def error_response(
    *,
    status: int,
    message: str,
    error_type: str,
    code: str,
    trace_id: str | None = None,
    details: dict[str, JSONValue] | None = None,
) -> web.Response:
    error_payload: dict[str, JSONValue] = {
        "message": message,
        "type": error_type,
        "code": code,
        "trace_id": trace_id,
        "details": details or {},
    }
    return json_response({"error": error_payload}, status=status)
