from __future__ import annotations

import asyncio

from config.http_server_config import HttpAuthConfig, HttpServerConfig
from config.ui_embeddings_settings import UIEmbeddingsSettings
from llm.types import ModelConfig
from server.http import app as http_app
from server.ui_session_storage import InMemoryUISessionStorage

TEST_API_TOKEN = "test-http-app-token"


class StubAgent:
    pass


class CaptureAgentFactory:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> StubAgent:
        self.kwargs = dict(kwargs)
        return StubAgent()


def test_create_app_invokes_dotenv_loader(monkeypatch) -> None:
    calls: list[str] = []

    def _mark_dotenv_load() -> None:
        calls.append("loaded")

    monkeypatch.setattr(http_app, "_load_project_dotenv", _mark_dotenv_load)
    _ = http_app.create_app(
        agent=StubAgent(),
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )
    assert calls == ["loaded"]


def test_run_server_loads_dotenv_before_auth_validation(monkeypatch) -> None:
    calls: list[str] = []
    auth_config = HttpAuthConfig(api_token="from-dotenv", allow_unauth_local=False)

    monkeypatch.setattr(http_app, "_load_project_dotenv", lambda: calls.append("dotenv"))

    def _ensure_auth() -> HttpAuthConfig:
        calls.append("auth")
        return auth_config

    def _create_app(**kwargs):  # noqa: ANN003, ANN202
        calls.append("app")
        assert kwargs["auth_config"] == auth_config
        return object()

    monkeypatch.setattr(http_app, "ensure_http_auth_boot_config", _ensure_auth)
    monkeypatch.setattr(http_app, "create_app", _create_app)
    monkeypatch.setattr(http_app.web, "run_app", lambda *args, **kwargs: None)

    http_app.run_server(HttpServerConfig())

    assert calls == ["dotenv", "auth", "app"]


def test_load_project_dotenv_skips_when_dependency_missing(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "_load_dotenv", None)
    http_app._load_project_dotenv()


def test_create_app_attaches_runtime_model_state_and_resolver(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "_load_project_dotenv", lambda: None)
    monkeypatch.setattr(http_app, "load_model_configs", lambda: None)

    app = http_app.create_app(
        agent=StubAgent(),
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )

    assert "runtime_model_state" in app
    assert "runtime_model_resolver" in app


def test_create_app_hydrates_runtime_model_state_from_persisted_config(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "_load_project_dotenv", lambda: None)
    expected = ModelConfig(provider="xai", model="grok-test")
    monkeypatch.setattr(http_app, "load_model_configs", lambda: expected)

    app = http_app.create_app(
        agent=StubAgent(),
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )

    runtime_state = app["runtime_model_state"]
    resolver = app["runtime_model_resolver"]

    assert asyncio.run(runtime_state.get_global_main()) == expected
    assert asyncio.run(resolver.resolve_main(None)) == expected


def test_create_app_leaves_runtime_model_state_empty_when_persisted_missing(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "_load_project_dotenv", lambda: None)
    monkeypatch.setattr(http_app, "load_model_configs", lambda: None)

    app = http_app.create_app(
        agent=StubAgent(),
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )

    runtime_state = app["runtime_model_state"]
    resolver = app["runtime_model_resolver"]

    assert asyncio.run(runtime_state.get_global_main()) is None
    assert asyncio.run(resolver.resolve_main(None)) is None


def test_create_app_passes_ui_embeddings_settings_to_lazy_agent(monkeypatch) -> None:
    monkeypatch.setattr(http_app, "_load_project_dotenv", lambda: None)
    monkeypatch.setattr(http_app, "load_model_configs", lambda: None)
    factory = CaptureAgentFactory()

    class ModuleStub:
        Agent = factory

    monkeypatch.setattr(http_app.importlib, "import_module", lambda _name: ModuleStub)
    monkeypatch.setattr(
        http_app.api,
        "_load_embeddings_settings",
        lambda: UIEmbeddingsSettings(
            provider="openai",
            local_model="all-MiniLM-L6-v2",
            openai_model="text-embedding-3-small",
        ),
    )
    monkeypatch.setattr(
        http_app.api,
        "_resolve_provider_api_key",
        lambda provider: "openai-test-key" if provider == "openai" else None,
    )

    app = http_app.create_app(
        max_request_bytes=1_000_000,
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )
    _ = asyncio.run(app["agent_provider"].get())

    assert factory.kwargs is not None
    assert factory.kwargs["embeddings_provider"] == "openai"
    assert factory.kwargs["embeddings_openai_model"] == "text-embedding-3-small"
    assert factory.kwargs["embeddings_openai_api_key"] == "openai-test-key"
