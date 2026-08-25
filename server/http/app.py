from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

from aiohttp import web

from config.http_server_config import (
    DEFAULT_MAX_REQUEST_BYTES,
    HttpAuthConfig,
    HttpServerConfig,
    ensure_http_auth_boot_config,
    resolve_http_auth_config,
    resolve_http_server_config,
)
from config.model_store import load_model_configs
from core.desktop_policy import DesktopPolicyStore
from core.desktop_runtime import DesktopRunCoordinator
from llm.types import ModelConfig
from server import http_api as api
from server.agent_provider import AgentScope, ScopedAgentProvider
from server.embedding_download import EmbeddingDownloadManager
from server.http.common.auth import _legacy_owner_principal_aliases, _owner_principal_id
from server.http.common.chat_cancellation import ChatCancellationRegistry
from server.http.common.idempotency import IdempotencyStore
from server.http.common.request_identity import (
    CloudflareAccessJWTVerifier,
    CloudflareAccessVerifier,
)
from server.http.common.runtime_contract import AgentProtocol, SessionApprovalStore
from server.http.common.runtime_model_state import (
    RuntimeModelResolver,
    build_runtime_model_state_from_persisted,
)
from server.principal_storage import principal_storage_paths
from server.ui_hub import UIHub
from server.ui_session_storage import SQLiteUISessionStorage, UISessionStorage
from tools.terminal_tool import TerminalTool

_load_dotenv: Callable[..., bool] | None
try:
    from dotenv import load_dotenv as _load_dotenv
except Exception:  # noqa: BLE001
    _load_dotenv = None

_DOTENV_LOAD_ATTEMPTED = False

logger = logging.getLogger("SlavikAI.HttpAPI")


def _load_project_dotenv() -> None:
    global _DOTENV_LOAD_ATTEMPTED
    if _DOTENV_LOAD_ATTEMPTED:
        return
    _DOTENV_LOAD_ATTEMPTED = True
    if _load_dotenv is None:
        logger.debug("python-dotenv is unavailable; skipping .env loading")
        return
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / ".env"
    loaded = _load_dotenv(env_path, override=False)
    if loaded:
        logger.info("Loaded environment from %s", env_path)
    else:
        logger.debug("No .env file loaded from %s", env_path)


async def _close_terminal_manager(app: web.Application) -> None:
    manager: TerminalTool = app["terminal_manager"]
    await manager.shutdown()


async def _close_chat_generations(app: web.Application) -> None:
    registry: ChatCancellationRegistry = app["chat_cancellation_registry"]
    errors = await registry.shutdown()
    for error in errors:
        logger.warning("Chat cancellation cleanup failed: %s", error)


async def _close_embedding_downloads(app: web.Application) -> None:
    manager: EmbeddingDownloadManager = app["embedding_download_manager"]
    await manager.shutdown()


async def _close_agent_provider(app: web.Application) -> None:
    provider = cast(ScopedAgentProvider[AgentProtocol], app["agent_provider"])
    await provider.close()


def create_app(
    *,
    agent: AgentProtocol | None = None,
    max_request_bytes: int | None = None,
    ui_storage: UISessionStorage | None = None,
    auth_config: HttpAuthConfig | None = None,
    cloudflare_access_verifier: CloudflareAccessVerifier | None = None,
    desktop_policy_store: DesktopPolicyStore | None = None,
) -> web.Application:
    _load_project_dotenv()
    config_max_bytes = max_request_bytes or DEFAULT_MAX_REQUEST_BYTES
    resolved_auth_config = auth_config or resolve_http_auth_config()
    app = web.Application(
        client_max_size=config_max_bytes,
        middlewares=[api.auth_gate_middleware],
    )
    app["auth_config"] = resolved_auth_config
    if (
        resolved_auth_config.browser_auth_mode == "cloudflare"
        and not resolved_auth_config.allow_unauth_local
    ):
        app["cloudflare_access_verifier"] = (
            cloudflare_access_verifier
            or CloudflareAccessJWTVerifier(
                issuer=resolved_auth_config.cloudflare_access_issuer,
                audience=resolved_auth_config.cloudflare_access_aud,
            )
        )
    app["http_api_logger"] = logger
    app["settings_snapshot_builder"] = api._build_settings_payload
    runtime_model_state = build_runtime_model_state_from_persisted(
        load_model_configs_fn=load_model_configs
    )
    app["runtime_model_state"] = runtime_model_state
    app["runtime_model_resolver"] = RuntimeModelResolver(runtime_model_state)
    owner_principal_id = _owner_principal_id(resolved_auth_config)
    desktop_run_coordinator = DesktopRunCoordinator()
    app["desktop_run_coordinator"] = desktop_run_coordinator
    resolved_desktop_policy_store = desktop_policy_store or DesktopPolicyStore(
        api.PROJECT_ROOT / ".run" / "desktop_approvals.json",
        legacy_subject_principal_id=owner_principal_id,
    )
    app["desktop_policy_store"] = resolved_desktop_policy_store
    if agent is None:

        def _factory(scope: AgentScope, main_config: ModelConfig | None) -> AgentProtocol:
            module = importlib.import_module("core.agent")
            agent_factory = getattr(module, "Agent", None)
            if not callable(agent_factory):
                raise RuntimeError("Agent class not found in core.agent")
            if main_config is None:
                raise RuntimeError("Не выбрана модель. Укажите model id в настройках.")
            embeddings_settings = api._load_embeddings_settings()
            storage = principal_storage_paths(
                principal_id=scope.principal_id,
                owner_principal_id=owner_principal_id,
                memory_root=api.PROJECT_ROOT / "memory",
            )
            return cast(
                "AgentProtocol",
                agent_factory(
                    main_config=main_config,
                    main_api_key=api._resolve_provider_api_key(main_config.provider),
                    user_id=scope.principal_id,
                    memory_db_path=str(storage.memory_db),
                    vectors_db_path=str(storage.vectors_db),
                    memory_companion_db_path=str(storage.memory_companion_db),
                    memory_inbox_db_path=str(storage.memory_categories_db),
                    canonical_atoms_db_path=str(storage.canonical_atoms_db),
                    embeddings_provider=embeddings_settings.provider,
                    embeddings_local_model=embeddings_settings.local_model,
                    embeddings_openai_model=embeddings_settings.openai_model,
                    embeddings_openai_api_key=api._resolve_provider_api_key("openai"),
                    desktop_policy_store=resolved_desktop_policy_store,
                    desktop_run_coordinator=desktop_run_coordinator,
                ),
            )

        app["agent_provider"] = ScopedAgentProvider(factory=_factory)
    else:
        app["agent_provider"] = ScopedAgentProvider.from_instance(agent)
    app["session_store"] = SessionApprovalStore()
    app["idempotency_store"] = IdempotencyStore()
    app["chat_cancellation_registry"] = ChatCancellationRegistry()
    app["terminal_manager"] = TerminalTool()
    app["embedding_download_manager"] = EmbeddingDownloadManager()
    resolved_ui_storage = ui_storage or SQLiteUISessionStorage(
        api.PROJECT_ROOT / ".run" / "ui_sessions.db",
    )
    provider = cast(ScopedAgentProvider[AgentProtocol], app["agent_provider"])
    app["ui_hub"] = UIHub(
        storage=resolved_ui_storage,
        legacy_principal_id=owner_principal_id,
        legacy_principal_aliases=_legacy_owner_principal_aliases(),
        on_session_pruned=lambda principal_id, session_id: provider.schedule_release(
            AgentScope(principal_id=principal_id, session_id=session_id)
        ),
    )
    app.on_cleanup.append(_close_chat_generations)
    app.on_cleanup.append(_close_terminal_manager)
    app.on_cleanup.append(_close_embedding_downloads)
    app.on_cleanup.append(_close_agent_provider)
    dist_path = api.PROJECT_ROOT / "ui" / "dist"
    app["ui_dist_path"] = dist_path
    from server.http.routes import register_routes

    register_routes(app)
    assets_path = dist_path / "assets"
    if assets_path.exists():
        app.router.add_static("/ui/assets/", assets_path)
    else:
        logger.warning("UI assets directory missing at %s; skipping static assets.", assets_path)
    return app


def run_server(config: HttpServerConfig) -> None:
    _load_project_dotenv()
    auth_config = ensure_http_auth_boot_config()
    app = create_app(
        max_request_bytes=config.max_request_bytes,
        auth_config=auth_config,
    )
    web.run_app(
        app,
        host=config.host,
        port=config.port,
        shutdown_timeout=8.0,
    )


def main() -> None:
    _load_project_dotenv()
    config = resolve_http_server_config()
    run_server(config)
