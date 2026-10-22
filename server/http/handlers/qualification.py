from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast

import requests
from aiohttp import web

from config.api_keys import ApiKeyStoreError
from llm.cancellation import GenerationCancelled, cancel_generation
from llm.capabilities import NativeToolQualification
from llm.local_http_brain import LocalHttpBrain
from llm.qualification import ProtocolQualificationError, run_inert_qualification
from llm.retry import ProviderRequestError
from llm.types import ModelConfig, is_custom_provider_id
from server import http_api as api
from server.agent_provider import ScopedAgentProvider
from server.http.common.auth import _ensure_session_owned, _require_owner
from server.http.common.responses import error_response, json_response
from server.http.common.runtime_contract import (
    AgentProtocol,
    _agent_scope,
    _require_custom_provider_owner,
    _resolve_runtime_main_for_session,
)
from server.http.common.ui_settings import _load_provider_instance
from server.ui_hub import UIHub

QUALIFICATION_TIMEOUT_SECONDS = 45.0


def _revision(path: Path) -> tuple[int, int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def _context_validator(config: ModelConfig, key: str) -> Callable[[], bool]:
    # Замыкание не удерживает Request/Agent или raw secret. Проверка ревизий
    # не позволяет rotation away/back оживить старое proof.
    provider_id, endpoint = config.provider, config.base_url
    paths = (api.API_KEYS_PATH, api.UI_SETTINGS_PATH)
    revisions = tuple(_revision(path) for path in paths)
    digest = hashlib.sha256(key.encode()).digest()

    def current() -> bool:
        try:
            instance = _load_provider_instance(provider_id)
            current_key = api._resolve_provider_api_key(provider_id)
            return (
                tuple(_revision(path) for path in paths) == revisions
                and instance is not None
                and endpoint == f"{instance['base_url']}/chat/completions"
                and current_key is not None
                and hashlib.sha256(current_key.encode()).digest() == digest
                and tuple(_revision(path) for path in paths) == revisions
            )
        except (OSError, ValueError, ApiKeyStoreError):
            return False

    return current


async def _watch_disconnect(request: web.Request, token: asyncio.Event) -> None:
    while not token.is_set():
        if request.transport is None or request.transport.is_closing():
            cancel_generation(token)
            return
        await asyncio.sleep(0.1)


async def handle_model_capabilities(request: web.Request) -> web.Response:
    denied = _require_owner(request)
    if denied is not None:
        return denied
    session_id = request.match_info["session_id"]
    hub = cast(UIHub, request.app["ui_hub"])
    denied = await _ensure_session_owned(request, hub, session_id)
    if denied is not None:
        return denied
    config = await _resolve_runtime_main_for_session(request, session_id)
    if config is None or not is_custom_provider_id(config.provider):
        return json_response({"state": "unknown", "can_check": False})
    _require_custom_provider_owner(request, config.provider, config)
    provider = cast(ScopedAgentProvider[AgentProtocol], request.app["agent_provider"])
    scope = _agent_scope(request, session_id)
    lock = provider.lock_for(scope)
    agent = await provider.get_existing_for_current_task(scope)
    state = "unknown"
    source = "not_observed"
    if agent is not None and isinstance(agent.brain, LocalHttpBrain) and not lock.locked():
        evidence = agent.brain.native_tool_capability_for(config).model
        state, source = evidence.state, evidence.source
    return json_response(
        {
            "state": state,
            "source": source,
            "can_check": (
                not provider.is_static
                and not lock.locked()
                and bool(api._resolve_provider_api_key(config.provider))
                and (agent is None or agent.last_approval_request is None)
            ),
        }
    )


async def handle_model_qualify(request: web.Request) -> web.Response:
    denied = _require_owner(request)
    if denied is not None:
        return denied
    session_id = request.match_info["session_id"]
    denied = await _ensure_session_owned(request, cast(UIHub, request.app["ui_hub"]), session_id)
    if denied is not None:
        return denied
    try:
        body = await request.json()
    except (ValueError, TypeError):
        body = None
    if not isinstance(body, dict) or set(body) != {"confirm"} or body["confirm"] is not True:
        return error_response(
            status=400,
            message="Нужен явный confirm=true.",
            error_type="invalid_request_error",
            code="confirmation_required",
        )
    provider = cast(ScopedAgentProvider[AgentProtocol], request.app["agent_provider"])
    lock = provider.lock_for(_agent_scope(request, session_id))
    if provider.is_static or lock.locked():
        return error_response(
            status=409,
            message="Модель занята или scope недоступен.",
            error_type="conflict_error",
            code="qualification_unavailable",
        )
    config = await _resolve_runtime_main_for_session(request, session_id)
    if config is None or not is_custom_provider_id(config.provider):
        return error_response(
            status=400,
            message="Нужна выбранная custom model.",
            error_type="invalid_request_error",
            code="custom_model_required",
        )
    _require_custom_provider_owner(request, config.provider, config)
    agent = await provider.get_for_current_task(_agent_scope(request, session_id), config)
    if lock.locked() or agent.last_approval_request is not None:
        return error_response(
            status=409,
            message="Сначала завершите выполнение или approval.",
            error_type="conflict_error",
            code="qualification_unavailable",
        )
    async with lock:
        if agent.last_approval_request is not None:
            return error_response(
                status=409,
                message="Сначала завершите approval.",
                error_type="conflict_error",
                code="qualification_unavailable",
            )
        agent.set_native_tool_qualification(None)
        if await _resolve_runtime_main_for_session(request, session_id) != config:
            return error_response(
                status=409,
                message="Модель изменилась.",
                error_type="conflict_error",
                code="qualification_context_changed",
            )
        key = api._resolve_provider_api_key(config.provider)
        if not key:
            return error_response(
                status=400,
                message="Нет собственного API key выбранного route.",
                error_type="invalid_request_error",
                code="provider_key_missing",
            )
        agent.reconfigure_models(config, main_api_key=key, persist=False)
        current = _context_validator(config, key)
        probe = LocalHttpBrain(config, api_key=key)
        if not isinstance(agent.brain, LocalHttpBrain) or (
            agent.brain.request_binding(config) != probe.request_binding(config)
        ):
            return error_response(
                status=409,
                message="Runtime credential context отличается от saved route.",
                error_type="conflict_error",
                code="qualification_context_changed",
            )
        token = asyncio.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(run_inert_qualification, probe, config, token, current)
        )
        monitor = asyncio.create_task(_watch_disconnect(request, token))
        state: Literal["supported", "unsupported", "unknown"] = "supported"
        message = "Sync, stream и role=tool protocol проверены для текущей session."
        timed_out = False
        try:
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker), timeout=QUALIFICATION_TIMEOUT_SECONDS
                )
            except ProtocolQualificationError as exc:
                state, message = "unsupported", str(exc)
            except TimeoutError:
                timed_out = True
                cancel_generation(token)
                state, message = "unknown", "Проверка превысила время; protocol не подтверждён."
            except GenerationCancelled:
                state, message = "unknown", "Проверка отменена."
            except (ProviderRequestError, requests.RequestException):
                state, message = "unknown", "Provider недоступен; protocol не подтверждён."
            if (
                (token.is_set() and not timed_out)
                or request.transport is None
                or request.transport.is_closing()
                or not current()
                or (await _resolve_runtime_main_for_session(request, session_id) != config)
            ):
                return error_response(
                    status=409,
                    message="Контекст проверки изменился или отменён.",
                    error_type="conflict_error",
                    code="qualification_context_changed",
                )
            if state != "unknown":
                agent.set_native_tool_qualification(
                    NativeToolQualification(
                        binding=probe.request_binding(config),
                        is_current=current,
                        state=state,
                    )
                )
            return json_response(
                {
                    "state": state,
                    "source": "inert_protocol_v1" if state != "unknown" else "not_observed",
                    "message": message,
                    "can_check": True,
                }
            )
        finally:
            cancel_generation(token)
            await asyncio.gather(worker, return_exceptions=True)
            monitor.cancel()
            with suppress(asyncio.CancelledError):
                await monitor
