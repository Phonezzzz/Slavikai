from __future__ import annotations

import hashlib
import hmac
import logging
import os
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiohttp import web

from config.http_server_config import HttpAuthConfig
from server.http.common.responses import error_response

if TYPE_CHECKING:
    from server.ui_hub import UIHub

AUTH_PROTECTED_PREFIXES: tuple[str, ...] = ("/ui/api/", "/v1/", "/slavik/")


def _extract_ui_session_id(request: web.Request) -> str | None:
    header_value = request.headers.get("X-Slavik-Session", "").strip()
    if header_value:
        return header_value
    query_value = request.query.get("session_id", "").strip()
    if query_value:
        return query_value
    return None


def _extract_bearer_token(request: web.Request) -> str | None:
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header:
        return None
    parts = auth_header.split(" ", 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    normalized = token.strip()
    return normalized or None


def _is_auth_protected_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in AUTH_PROTECTED_PREFIXES)


def _principal_id_from_token(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"principal_{digest[:16]}"


def _resolve_request_principal_id(
    request: web.Request,
    auth_config: HttpAuthConfig,
) -> str | None:
    if auth_config.allow_unauth_local:
        return "local_unauth"
    presented_token = _extract_bearer_token(request)
    if presented_token is None:
        return None
    candidate_tokens: list[str] = []
    if auth_config.api_token:
        candidate_tokens.append(auth_config.api_token)
    admin_token = os.environ.get("SLAVIK_ADMIN_TOKEN", "").strip()
    if admin_token:
        candidate_tokens.append(admin_token)
    if any(hmac.compare_digest(presented_token, token) for token in candidate_tokens):
        return _principal_id_from_token(presented_token)
    return None


@web.middleware
async def auth_gate_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    if not _is_auth_protected_path(request.path):
        return await handler(request)
    auth_config: HttpAuthConfig = request.app["auth_config"]
    principal_id = _resolve_request_principal_id(request, auth_config)
    if principal_id is not None:
        request["principal_id"] = principal_id
        return await handler(request)
    if request.path == "/ui/api/settings":
        logger_raw = request.app.get("http_api_logger")
        logger = (
            logger_raw
            if isinstance(logger_raw, logging.Logger)
            else logging.getLogger("SlavikAI.HttpAPI")
        )
        snapshot_builder = request.app.get("settings_snapshot_builder")
        if callable(snapshot_builder):
            try:
                snapshot = snapshot_builder()
                logger.warning(
                    "Unauthorized settings access denied",
                    extra={
                        "reason": "unauthorized",
                        "path": request.path,
                        "remote": request.remote,
                        "settings_snapshot": snapshot,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unauthorized settings access denied (snapshot unavailable)",
                    extra={
                        "reason": "unauthorized",
                        "path": request.path,
                        "remote": request.remote,
                        "settings_snapshot_error": str(exc),
                    },
                )
    return error_response(
        status=401,
        message="Unauthorized.",
        error_type="invalid_request_error",
        code="unauthorized",
    )


def _request_principal_id(request: web.Request) -> str | None:
    principal_raw = request.get("principal_id")
    if not isinstance(principal_raw, str):
        return None
    normalized = principal_raw.strip()
    return normalized or None


def _require_admin_bearer(request: web.Request) -> web.Response | None:
    admin_token = os.environ.get("SLAVIK_ADMIN_TOKEN", "").strip()
    if not admin_token:
        return error_response(
            status=503,
            message="Admin token is not configured.",
            error_type="configuration_error",
            code="admin_token_not_configured",
        )
    presented_token = _extract_bearer_token(request)
    if presented_token is None or not hmac.compare_digest(presented_token, admin_token):
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    return None


def _session_forbidden_response() -> web.Response:
    return error_response(
        status=403,
        message="Session access forbidden.",
        error_type="invalid_request_error",
        code="session_forbidden",
    )


async def _resolve_ui_session_id_for_principal(
    request: web.Request,
    hub: UIHub,
) -> tuple[str | None, web.Response | None]:
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return (
            None,
            error_response(
                status=401,
                message="Unauthorized.",
                error_type="invalid_request_error",
                code="unauthorized",
            ),
        )
    try:
        session_id = await hub.get_or_create_session(
            _extract_ui_session_id(request),
            principal_id=principal_id,
        )
    except PermissionError:
        return None, _session_forbidden_response()
    return session_id, None


async def _ensure_session_owned(
    request: web.Request,
    hub: UIHub,
    session_id: str,
) -> web.Response | None:
    principal_id = _request_principal_id(request)
    if principal_id is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    access = await hub.get_session_access(session_id, principal_id)
    if access == "owned":
        return None
    if access == "missing":
        return error_response(
            status=404,
            message="Session not found.",
            error_type="invalid_request_error",
            code="session_not_found",
        )
    return _session_forbidden_response()
