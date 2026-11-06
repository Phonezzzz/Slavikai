from __future__ import annotations

import asyncio
from typing import cast
from uuid import UUID

from aiohttp import web

from core.task_run_storage import TaskRunStore
from server.http.common.auth import _ensure_session_owned, _request_principal_id
from server.http.common.responses import error_response, json_response
from server.ui_hub import UIHub
from shared.task_run import RunScope


async def handle_recovery_evidence(request: web.Request) -> web.Response:
    session_id = request.match_info["session_id"]
    denied = await _ensure_session_owned(request, cast(UIHub, request.app["ui_hub"]), session_id)
    if denied is not None:
        return denied
    principal = _request_principal_id(request)
    assert principal is not None  # The session ownership check requires a verified principal.

    run_id = request.match_info["run_id"]
    after = request.query.get("after", "")
    try:
        if str(UUID(run_id)) != run_id or (after and str(UUID(after)) != after):
            raise ValueError("Non-canonical UUID")
    except ValueError:
        return error_response(
            status=400,
            message="Некорректный идентификатор запуска или курсор.",
            error_type="invalid_request_error",
            code="invalid_recovery_evidence_id",
        )
    store = cast(TaskRunStore, request.app["task_run_store"])
    try:
        evidence = await asyncio.to_thread(
            store.recovery_evidence, RunScope(principal, session_id), run_id, after=after
        )
    except LookupError:
        return error_response(
            status=404,
            message="Свидетельства для сверки недоступны.",
            error_type="invalid_request_error",
            code="recovery_evidence_unavailable",
        )
    response = json_response(evidence)
    response.headers["Cache-Control"] = "no-store"
    return response
