from __future__ import annotations

from config.http_server_config import HttpAuthConfig
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
