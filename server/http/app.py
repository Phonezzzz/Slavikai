from __future__ import annotations

import asyncio
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
from server import http_api as api
from server.http.common.idempotency import IdempotencyStore
from server.http.common.runtime_contract import AgentProtocol, SessionApprovalStore
from server.http.common.runtime_model_state import (
    RuntimeModelResolver,
    build_runtime_model_state_from_persisted,
)
from server.lazy_agent import LazyAgentProvider
from server.terminal_manager import TerminalManager
from server.ui_hub import UIHub
from server.ui_session_storage import SQLiteUISessionStorage, UISessionStorage

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
    manager: TerminalManager = app["terminal_manager"]
    await manager.shutdown()


def create_app(
    *,
    agent: AgentProtocol | None = None,
    max_request_bytes: int | None = None,
    ui_storage: UISessionStorage | None = None,
    auth_config: HttpAuthConfig | None = None,
) -> web.Application:
    _load_project_dotenv()
    config_max_bytes = max_request_bytes or DEFAULT_MAX_REQUEST_BYTES
    resolved_auth_config = auth_config or resolve_http_auth_config()
    app = web.Application(
        client_max_size=config_max_bytes,
        middlewares=[api.auth_gate_middleware],
    )
    app["auth_config"] = resolved_auth_config
    app["http_api_logger"] = logger
    app["settings_snapshot_builder"] = api._build_settings_payload
    runtime_model_state = build_runtime_model_state_from_persisted(
        load_model_configs_fn=load_model_configs
    )
    app["runtime_model_state"] = runtime_model_state
    app["runtime_model_resolver"] = RuntimeModelResolver(runtime_model_state)
    if agent is None:

        def _factory() -> AgentProtocol:
            module = importlib.import_module("core.agent")
            agent_factory = getattr(module, "Agent", None)
            if not callable(agent_factory):
                raise RuntimeError("Agent class not found in core.agent")
            return cast(
                "AgentProtocol",
                agent_factory(main_config=runtime_model_state.peek_global_main()),
            )

        app["agent"] = None
        app["agent_provider"] = LazyAgentProvider(factory=_factory)
    else:
        app["agent"] = agent
        app["agent_provider"] = LazyAgentProvider.from_instance(agent)
    app["agent_lock"] = asyncio.Lock()
    app["session_store"] = SessionApprovalStore()
    app["idempotency_store"] = IdempotencyStore()
    app["terminal_manager"] = TerminalManager()
    resolved_ui_storage = ui_storage or SQLiteUISessionStorage(
        api.PROJECT_ROOT / ".run" / "ui_sessions.db",
    )
    app["ui_hub"] = UIHub(storage=resolved_ui_storage)
    app.on_cleanup.append(_close_terminal_manager)
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
    ensure_http_auth_boot_config()
    app = create_app(max_request_bytes=config.max_request_bytes)
    web.run_app(app, host=config.host, port=config.port)


def main() -> None:
    config = resolve_http_server_config()
    run_server(config)
