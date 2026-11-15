from __future__ import annotations

import asyncio
from typing import cast

from aiohttp import web
from yarl import URL

from llm.chatgpt_plan import ChatGPTPlan, ChatGPTPlanError
from server.http.common.auth import _request_identity
from server.http.common.responses import error_response, json_response
from shared.models import JSONValue


def chatgpt_plan_context(request: web.Request) -> tuple[ChatGPTPlan, str] | None:
    identity = _request_identity(request)
    if identity is None or identity.auth_method != "cloudflare_access" or identity.role != "owner":
        return None
    plan = cast(ChatGPTPlan | None, request.app.get("chatgpt_plan"))
    return (plan, identity.principal_id) if plan is not None else None


def require_chatgpt_plan(request: web.Request) -> tuple[ChatGPTPlan, str]:
    identity = _request_identity(request)
    if identity is None or identity.auth_method != "cloudflare_access" or identity.role != "owner":
        raise PermissionError("ChatGPT subscription requires the verified browser owner.")
    context = chatgpt_plan_context(request)
    if context is None:
        raise ChatGPTPlanError(
            "chatgpt_plan_unavailable",
            "Подключение подписки ChatGPT недоступно в этом развёртывании.",
            status=503,
        )
    return context


def chatgpt_plan_error(error: ChatGPTPlanError) -> web.Response:
    return _private(
        error_response(
            status=error.status,
            message=str(error),
            error_type="provider_error",
            code=error.code,
        ),
    )


def chatgpt_plan_forbidden() -> web.Response:
    return _private(
        error_response(
            status=403,
            message="Подключение подписки ChatGPT доступно владельцу через Cloudflare Access.",
            error_type="invalid_request_error",
            code="owner_required",
        ),
    )


def _private(response: web.Response) -> web.Response:
    response.headers["Cache-Control"] = "no-store"
    return response


def check_chatgpt_plan_origin(request: web.Request) -> web.Response | None:
    origin_raw = request.headers.get("Origin", "")
    try:
        origin = URL(origin_raw)
        valid = (
            origin.raw_authority == request.host
            and str(origin.origin()) == origin_raw
            and (
                origin.scheme == "https"
                or (origin.scheme == "http" and origin.host in {"127.0.0.1", "localhost", "::1"})
            )
        )
    except ValueError:
        valid = False
    if valid and request.headers.get("Sec-Fetch-Site") in {None, "same-origin"}:
        return None
    return _private(
        error_response(
            status=403,
            message="Подтвердите действие в открытой вкладке SlavikAI.",
            error_type="invalid_request_error",
            code="invalid_origin",
        ),
    )


async def handle_chatgpt_plan_status(request: web.Request) -> web.Response:
    try:
        plan, principal = require_chatgpt_plan(request)
        accounts = await asyncio.to_thread(plan.accounts, principal)
        authorization_status = await asyncio.to_thread(plan.authorization_status, principal)
    except PermissionError:
        return chatgpt_plan_forbidden()
    except ChatGPTPlanError as exc:
        if exc.code == "chatgpt_plan_unavailable":
            return _private(json_response({"status": "unavailable", "accounts": []}))
        return chatgpt_plan_error(exc)
    return _private(
        json_response(
            {
                "status": "enabled",
                "accounts": list(accounts),
                "authorization_status": authorization_status,
            }
        )
    )


async def _action_payload(request: web.Request) -> dict[str, JSONValue] | web.Response:
    origin_error = check_chatgpt_plan_origin(request)
    if origin_error is not None:
        return origin_error
    try:
        if request.content_type == "application/json":
            data = await request.json()
            if isinstance(data, dict):
                return cast(dict[str, JSONValue], data)
    except (ValueError, LookupError):
        pass
    return _invalid_payload()


def _invalid_payload() -> web.Response:
    return _private(
        error_response(
            status=400,
            message="Некорректные параметры подключения ChatGPT.",
            error_type="invalid_request_error",
            code="invalid_request_error",
        ),
    )


async def handle_chatgpt_plan_authorize(request: web.Request) -> web.Response:
    try:
        plan, principal = require_chatgpt_plan(request)
        payload = await _action_payload(request)
        if isinstance(payload, web.Response):
            return payload
        account = payload.get("account_id")
        reconsent = payload.get("reconsent", False)
        if (account is not None and not isinstance(account, str)) or not isinstance(
            reconsent, bool
        ):
            return _invalid_payload()
        url = await asyncio.to_thread(plan.begin, principal, account, reconsent=reconsent)
    except PermissionError:
        return chatgpt_plan_forbidden()
    except ChatGPTPlanError as exc:
        return chatgpt_plan_error(exc)
    return _private(json_response({"authorization_url": url}))


async def handle_chatgpt_plan_disconnect(request: web.Request) -> web.Response:
    try:
        plan, principal = require_chatgpt_plan(request)
        payload = await _action_payload(request)
        if isinstance(payload, web.Response):
            return payload
        account = payload.get("account_id")
        if not isinstance(account, str) or not account:
            return _invalid_payload()
        revoked = await asyncio.to_thread(plan.disconnect, principal, account)
    except PermissionError:
        return chatgpt_plan_forbidden()
    except ChatGPTPlanError as exc:
        return chatgpt_plan_error(exc)
    return _private(json_response({"revoked": revoked}))
