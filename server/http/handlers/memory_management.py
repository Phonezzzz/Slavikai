from __future__ import annotations

import asyncio
import sqlite3
from typing import Literal, cast

from aiohttp import web

from config.http_server_config import HttpAuthConfig
from memory.management import MemoryManagement
from server import http_api as api
from server.http.common.auth import _owner_principal_id, _request_principal_id
from server.http.common.responses import error_response, json_response
from server.principal_storage import principal_storage_paths
from shared.models import JSONValue

_STATUS = {
    "memory_item_not_found": 404,
    "memory_revision_conflict": 409,
    "memory_delete_not_found": 404,
    "memory_delete_conflict": 409,
    "memory_delete_expired": 410,
    "memory_storage_unavailable": 503,
}


def _error(code: str, status: int = 400) -> web.Response:
    return error_response(
        status=_STATUS.get(code, status),
        message="Операция Memory не выполнена.",
        error_type="invalid_request_error",
        code=code,
    )


async def _handle(
    request: web.Request, action: Literal["list", "preview", "confirm"]
) -> web.Response:
    principal = _request_principal_id(request)
    if principal is None:
        return _error("unauthorized", 401)
    auth = cast(HttpAuthConfig, request.app["auth_config"])
    paths = principal_storage_paths(
        principal_id=principal,
        owner_principal_id=_owner_principal_id(auth),
        memory_root=api.PROJECT_ROOT / "memory",
    )
    payload: object = None
    if action != "list":
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return _error("memory_cross_site_forbidden", 403)
        if request.content_type != "application/json":
            return _error("memory_json_required", 415)
        try:
            payload = await request.json()
        except (ValueError, UnicodeError):
            return _error("memory_payload_invalid")
    if action == "preview":
        if (
            not isinstance(payload, dict)
            or set(payload) != {"atom_id", "revision"}
            or not all(isinstance(payload[k], str) and payload[k] for k in ("atom_id", "revision"))
        ):
            return _error("memory_payload_invalid")
    elif action == "confirm":
        if (
            not isinstance(payload, dict)
            or set(payload) != {"operation_id", "confirm"}
            or payload["confirm"] is not True
            or not isinstance(payload["operation_id"], str)
            or not payload["operation_id"]
            or len(payload["operation_id"]) > 128
        ):
            return _error("memory_confirmation_required")
    try:
        limit = int(request.query.get("limit", "50"))
    except ValueError:
        return _error("memory_page_limit_invalid")
    cursor = request.query.get("cursor")

    def run() -> dict[str, JSONValue]:
        service = MemoryManagement(
            canonical_path=paths.canonical_atoms_db,
            vectors_path=paths.vectors_db,
            principal_id=principal,
        )
        try:
            if action == "list":
                return service.list_items(cursor=cursor, limit=limit)
            if not isinstance(payload, dict):
                raise ValueError("memory_payload_invalid")
            if action == "preview":
                atom_id, revision = payload["atom_id"], payload["revision"]
                if not isinstance(atom_id, str) or not isinstance(revision, str):
                    raise ValueError("memory_payload_invalid")
                return service.preview(atom_id, revision)
            operation_id = payload["operation_id"]
            if not isinstance(operation_id, str):
                raise ValueError("memory_payload_invalid")
            result = service.confirm(operation_id)
            status = result.data.get("status")
            if result.ok and status == "deleted":
                return {"status": "deleted", "subject": service.subject}
            if isinstance(status, str) and status in {"not_found", "conflict", "expired"}:
                raise ValueError("memory_delete_" + status)
            raise ValueError("memory_storage_unavailable")
        finally:
            service.close()

    try:
        data = await asyncio.to_thread(run)
    except ValueError as exc:
        code = str(exc)
        return _error(
            code if code.startswith("memory_") else "memory_storage_unavailable",
            400 if code.startswith("memory_") else 503,
        )
    except (sqlite3.Error, OSError):
        return _error("memory_storage_unavailable", 503)
    return json_response(data)


async def _private_response(
    request: web.Request, action: Literal["list", "preview", "confirm"]
) -> web.Response:
    response = await _handle(request, action)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Vary"] = "Cf-Access-Jwt-Assertion, Cookie"
    return response


async def handle_memory_items(request: web.Request) -> web.Response:
    return await _private_response(request, "list")


async def handle_memory_delete_preview(request: web.Request) -> web.Response:
    return await _private_response(request, "preview")


async def handle_memory_delete_confirm(request: web.Request) -> web.Response:
    return await _private_response(request, "confirm")
