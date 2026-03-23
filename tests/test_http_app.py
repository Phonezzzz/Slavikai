from __future__ import annotations

import asyncio

from config.http_server_config import HttpAuthConfig
from llm.types import ModelConfig
from server.http import app as http_app
from server.ui_session_storage import InMemoryUISessionStorage

TEST_API_TOKEN = "test-http-app-token"


class StubAgent:
    pass


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
