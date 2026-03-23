from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable
from typing import Protocol, cast

import requests
from aiohttp import web

from config.model_whitelist import ensure_model_allowed
from core.approval_policy import ApprovalCategory, ApprovalRequest
from core.mwv.manager import MWVRunResult
from core.mwv.models import RunContext, TaskPacket
from llm.types import ModelConfig
from server.http.common.responses import error_response as _error_response
from server.http.common.ui_settings import _build_model_config
from server.lazy_agent import LazyAgentProvider
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import JSONValue, LLMMessage, ToolResult


class SessionApprovalStore:
    def __init__(self) -> None:
        self._approved: dict[str, set[ApprovalCategory]] = {}
        self._lock = asyncio.Lock()

    async def approve(
        self, session_id: str, categories: set[ApprovalCategory]
    ) -> set[ApprovalCategory]:
        async with self._lock:
            existing = self._approved.get(session_id, set())
            existing.update(categories)
            self._approved[session_id] = existing
            return set(existing)

    async def is_approved(self, session_id: str) -> bool:
        async with self._lock:
            return bool(self._approved.get(session_id))

    async def get_categories(self, session_id: str) -> set[ApprovalCategory]:
        async with self._lock:
            return set(self._approved.get(session_id, set()))


class TracerProtocol(Protocol):
    def log(
        self,
        event_type: str,
        message: str,
        meta: dict[str, JSONValue] | None = None,
    ) -> None: ...


class AgentProtocol(Protocol):
    brain: object
    tools_enabled: dict[str, bool]
    last_approval_request: ApprovalRequest | None
    last_chat_interaction_id: str | None
    tracer: TracerProtocol

    def set_session_context(
        self,
        session_id: str | None,
        approved_categories: set[ApprovalCategory],
    ) -> None: ...

    def reconfigure_models(
        self,
        main_config: ModelConfig,
        main_api_key: str | None = None,
        *,
        persist: bool = True,
    ) -> None: ...

    def respond(self, messages: list[LLMMessage]) -> str: ...

    def update_tools_enabled(self, state: dict[str, bool]) -> None: ...
    def apply_runtime_tools_enabled(self, state: dict[str, bool]) -> None: ...
    def apply_runtime_workspace_root(self, workspace_root: str | None) -> None: ...
    def run_task_packet(self, packet: TaskPacket, context: RunContext) -> MWVRunResult: ...

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult: ...

    def record_feedback_event(
        self,
        *,
        interaction_id: str,
        rating: FeedbackRating,
        labels: list[FeedbackLabel],
        free_text: str | None,
    ) -> None: ...


class RequestsModuleProtocol(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = ...,
        data: dict[str, str] | None = ...,
        files: dict[str, tuple[str, bytes, str]] | None = ...,
        timeout: int | float | None = ...,
    ) -> requests.Response: ...


class RuntimeModelStateProtocol(Protocol):
    async def get_global_main(self) -> ModelConfig | None: ...

    async def set_global_main(self, main: ModelConfig | None) -> None: ...

    async def get_session_override(self, session_id: str) -> ModelConfig | None: ...

    async def set_session_override(self, session_id: str, main: ModelConfig | None) -> None: ...

    async def clear_session_override(self, session_id: str) -> None: ...


class RuntimeModelResolverProtocol(Protocol):
    async def resolve_main(self, session_id: str | None) -> ModelConfig | None: ...


def _load_agent_factory() -> Callable[..., AgentProtocol]:
    module = importlib.import_module("core.agent")
    agent_factory = getattr(module, "Agent", None)
    if not callable(agent_factory):
        raise RuntimeError("Agent class not found in core.agent")
    return cast(Callable[..., AgentProtocol], agent_factory)


def _model_not_selected_response() -> web.Response:
    return _error_response(
        status=409,
        message="Не выбрана модель. Выберите модель в UI и повторите.",
        error_type="configuration_error",
        code="model_not_selected",
    )


def _model_not_allowed_response(model_id: str) -> web.Response:
    return _error_response(
        status=409,
        message=f"Модель '{model_id}' не входит в whitelist.",
        error_type="configuration_error",
        code="model_not_allowed",
        details={"model": model_id},
    )


async def _resolve_agent(request: web.Request) -> AgentProtocol | None:
    provider: LazyAgentProvider[AgentProtocol] = request.app["agent_provider"]
    try:
        return await provider.get()
    except RuntimeError as exc:
        if "Не выбрана модель" in str(exc):
            return None
        raise


async def _resolve_agent_for_base_http(request: web.Request) -> AgentProtocol | None:
    provider: LazyAgentProvider[AgentProtocol] = request.app["agent_provider"]
    if request.app.get("agent") is not None:
        try:
            return await provider.get()
        except RuntimeError as exc:
            if "Не выбрана модель" in str(exc):
                return None
            raise

    resolver = cast(RuntimeModelResolverProtocol, request.app["runtime_model_resolver"])
    main_config = await resolver.resolve_main(None)
    if main_config is None:
        return None
    ensure_model_allowed(main_config.model, main_config.provider)
    agent_factory = _load_agent_factory()
    try:
        return await provider.ensure(lambda: agent_factory(main_config=main_config))
    except RuntimeError as exc:
        if "Не выбрана модель" in str(exc):
            return None
        raise


def _selected_model_snapshot(main_config: ModelConfig) -> dict[str, str]:
    return {"provider": main_config.provider, "model": main_config.model}


async def _sync_session_runtime_override(
    request: web.Request,
    session_id: str,
    selected_model: dict[str, str] | None,
) -> ModelConfig | None:
    if not isinstance(selected_model, dict):
        return None
    provider = selected_model.get("provider")
    model_id = selected_model.get("model")
    if not isinstance(provider, str) or not isinstance(model_id, str):
        return None
    main_config = _build_model_config(provider, model_id)
    store = cast(RuntimeModelStateProtocol, request.app["runtime_model_state"])
    await store.set_session_override(session_id, main_config)
    return main_config


async def _resolve_runtime_main_for_session(
    request: web.Request,
    session_id: str,
) -> ModelConfig | None:
    resolver = cast(RuntimeModelResolverProtocol, request.app["runtime_model_resolver"])
    return await resolver.resolve_main(session_id)


async def _resolve_agent_for_ui_session(
    request: web.Request,
    session_id: str,
) -> tuple[AgentProtocol | None, ModelConfig | None]:
    main_config = await _resolve_runtime_main_for_session(request, session_id)
    if main_config is None:
        return None, None
    provider: LazyAgentProvider[AgentProtocol] = request.app["agent_provider"]
    if request.app.get("agent") is not None:
        try:
            return await provider.get(), main_config
        except RuntimeError as exc:
            if "Не выбрана модель" in str(exc):
                return None, None
            raise
    agent_factory = _load_agent_factory()
    try:
        return await provider.ensure(lambda: agent_factory(main_config=main_config)), main_config
    except RuntimeError as exc:
        if "Не выбрана модель" in str(exc):
            return None, None
        raise
