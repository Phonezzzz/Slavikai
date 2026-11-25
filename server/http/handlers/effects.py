"""Read the bridge effect journal (E03-T08).

``GET /ui/api/bridge/effects`` — oldest-first page of the
caller's own effect records, each with its attached observations.
Read-only: no mutating endpoints exist in v1.

Scoping (rev. 2 §8.2): the caller sees only records with their
own ``principal_id``; asking for another principal's records is
403. There is no owner override in v1.

Gap markers (owner condition 7): gap detection is computed over
the WHOLE daemon effect sequence in the store, and is exposed
only to the owner, only on unfiltered reads (no ``run_id`` /
``outcome`` / ``principal_id`` filter). A filtered read cannot
see the whole sequence, so its gaps would be meaningless.
"""

from __future__ import annotations

import asyncio
from typing import cast

from aiohttp import web

from core.bridge_effect_store import BridgeEffectStore
from server.http.common.auth import _request_identity, _request_principal_id
from server.http.common.responses import error_response, json_response
from shared.models import JSONValue

_DEFAULT_LIMIT = 100
_MAX_LIMIT = 500
_OUTCOMES = ("effect", "no_effect", "unknown")


def _parse_limit(raw: str | None) -> int | None:
    if raw is None:
        return _DEFAULT_LIMIT
    try:
        value = int(raw, 10)
    except ValueError:
        return None
    if not 1 <= value <= _MAX_LIMIT:
        return None
    return value


def _parse_cursor(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


async def handle_bridge_effects(request: web.Request) -> web.Response:
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    identity = _request_identity(request)
    is_owner = identity is not None and identity.role == "owner"
    query = request.query
    requested_principal = query.get("principal_id")
    if requested_principal is not None and requested_principal != principal_id:
        return error_response(
            status=403,
            message="Records of another principal are not visible.",
            error_type="invalid_request_error",
            code="forbidden",
        )
    run_id = query.get("run_id")
    outcome = query.get("outcome")
    if outcome is not None and outcome not in _OUTCOMES:
        return error_response(
            status=400,
            message="outcome must be one of effect, no_effect, unknown.",
            error_type="invalid_request_error",
            code="invalid_outcome",
        )
    limit = _parse_limit(query.get("limit"))
    if limit is None:
        return error_response(
            status=400,
            message="limit must be an integer between 1 and 500.",
            error_type="invalid_request_error",
            code="invalid_limit",
        )
    cursor = _parse_cursor(query.get("cursor"))
    if query.get("cursor") is not None and cursor is None:
        return error_response(
            status=400,
            message="cursor must be a non-negative integer server seq.",
            error_type="invalid_request_error",
            code="invalid_cursor",
        )
    # Gap detection only in a context with access to the whole
    # corresponding daemon-effects sequence (owner condition 7):
    # the owner, on an unfiltered read. Anything else would
    # mistake a filtered view for a hole in the sequence.
    include_gaps = query.get("include_gaps", "false").lower() == "true"
    gaps_requested = (
        include_gaps
        and is_owner
        and run_id is None
        and outcome is None
        and requested_principal is None
    )
    store = cast(BridgeEffectStore, request.app["bridge_effect_store"])
    page = await asyncio.to_thread(
        store.list_effects,
        principal_id=principal_id,
        run_id=run_id,
        outcome=outcome,
        cursor=cursor,
        limit=limit,
        include_gaps=gaps_requested,
    )
    payload: dict[str, JSONValue] = cast(
        dict[str, JSONValue],
        {
            "records": page["records"],
            "next_cursor": page["next_cursor"],
            **({"gaps": page["gaps"]} if gaps_requested else {}),
        },
    )
    return json_response(payload)
