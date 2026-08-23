from __future__ import annotations

from aiohttp import web

from config.http_server_config import HttpAuthConfig
from server.http.common.auth import (
    UI_AUTH_COOKIE,
    _principal_id_for_presented_token,
    _request_identity,
    _resolve_request_principal_id,
    _ui_auth_cookie_value,
)
from server.http.common.responses import error_response, json_response

_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30


def _secure_cookie_required(request: web.Request) -> bool:
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "").split(",", 1)[0].strip()
    return request.secure or forwarded_proto.lower() == "https"


async def handle_ui_auth_status(request: web.Request) -> web.Response:
    auth_config: HttpAuthConfig = request.app["auth_config"]
    if auth_config.browser_auth_mode == "cloudflare":
        identity = _request_identity(request)
        return json_response(
            {
                "ok": True,
                "authenticated": identity is not None,
                "auth_required": True,
                "auth_method": "cloudflare_access",
                "principal_id": identity.principal_id if identity is not None else None,
                "email": identity.email if identity is not None else None,
                "role": identity.role if identity is not None else None,
            }
        )
    return json_response(
        {
            "ok": True,
            "authenticated": _resolve_request_principal_id(request, auth_config) is not None,
            "auth_required": not auth_config.allow_unauth_local,
            "auth_method": "token",
        }
    )


async def handle_ui_auth_login(request: web.Request) -> web.Response:
    auth_config: HttpAuthConfig = request.app["auth_config"]
    if auth_config.browser_auth_mode == "cloudflare":
        return error_response(
            status=409,
            message="Browser authentication is managed by Cloudflare Access.",
            error_type="invalid_request_error",
            code="cloudflare_access_managed",
        )
    try:
        payload = await request.json()
    except Exception as exc:  # noqa: BLE001
        return error_response(
            status=400,
            message=f"Некорректный JSON: {exc}",
            error_type="invalid_request_error",
            code="invalid_json",
        )
    token_raw = payload.get("token") if isinstance(payload, dict) else None
    token = token_raw.strip() if isinstance(token_raw, str) else ""
    if not token or _principal_id_for_presented_token(token, auth_config) is None:
        return error_response(
            status=401,
            message="Unauthorized.",
            error_type="invalid_request_error",
            code="unauthorized",
        )
    response = json_response({"ok": True, "authenticated": True})
    response.set_cookie(
        UI_AUTH_COOKIE,
        _ui_auth_cookie_value(token),
        httponly=True,
        secure=_secure_cookie_required(request),
        samesite="Strict",
        max_age=_COOKIE_MAX_AGE_SECONDS,
        path="/",
    )
    return response


async def handle_ui_auth_logout(request: web.Request) -> web.Response:
    auth_config: HttpAuthConfig = request.app["auth_config"]
    if auth_config.browser_auth_mode == "cloudflare":
        return error_response(
            status=409,
            message="Browser authentication is managed by Cloudflare Access.",
            error_type="invalid_request_error",
            code="cloudflare_access_managed",
        )
    response = json_response({"ok": True, "authenticated": False})
    response.del_cookie(UI_AUTH_COOKIE, path="/")
    return response
