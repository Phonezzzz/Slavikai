from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import json
import stat
from pathlib import Path

import pytest
import requests

from config.api_keys import load_api_keys, save_api_keys

from .fakes import *


def _isolate_settings_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    api_keys_path = tmp_path / "config" / "api_keys.json"
    monkeypatch.setattr("server.http_api.API_KEYS_PATH", api_keys_path)
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", tmp_path / "ui_settings.json")
    for env_name in (
        "XAI_API_KEY",
        "OPENROUTER_API_KEY",
        "LOCAL_LLM_API_KEY",
        "INCEPTION_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    return api_keys_path


def _provider_settings(payload: object, provider_name: str) -> dict[str, object]:
    assert isinstance(payload, dict)
    settings = payload.get("settings")
    assert isinstance(settings, dict)
    providers = settings.get("providers")
    assert isinstance(providers, list)
    for provider in providers:
        if isinstance(provider, dict) and provider.get("provider") == provider_name:
            return provider
    raise AssertionError(f"Provider settings not found: {provider_name}")


def test_api_key_save_and_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    api_keys_path = _isolate_settings_paths(monkeypatch, tmp_path)
    checked: list[tuple[str, str]] = []

    def validate(provider: str, api_key: str) -> str | None:
        checked.append((provider, api_key))
        return None

    monkeypatch.setattr("server.http_api._validate_provider_api_key", validate)

    async def run() -> None:
        first_client = await _create_client(DummyAgent())
        try:
            response = await first_client.post(
                "/ui/api/settings",
                json={"providers": {"openrouter": {"api_key": "test-openrouter-key"}}},
            )
            assert response.status == 200
            payload = await response.json()
            provider = _provider_settings(payload, "openrouter")
            assert provider.get("api_key_set") is True
            assert provider.get("api_key_stored") is True
            assert provider.get("api_key_source") == "file"
            assert "test-openrouter-key" not in json.dumps(payload)
        finally:
            await first_client.close()

        reloaded_client = await _create_client(DummyAgent())
        try:
            response = await reloaded_client.get("/ui/api/settings")
            assert response.status == 200
            provider = _provider_settings(await response.json(), "openrouter")
            assert provider.get("api_key_set") is True
            assert provider.get("api_key_stored") is True
            assert provider.get("api_key_source") == "file"
        finally:
            await reloaded_client.close()

    asyncio.run(run())

    assert checked == [("openrouter", "test-openrouter-key")]
    assert load_api_keys(path=api_keys_path) == {"openrouter": "test-openrouter-key"}
    assert stat.S_IMODE(api_keys_path.stat().st_mode) == 0o600


def test_api_key_used_in_next_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "server.http_api._validate_provider_api_key",
        lambda provider, api_key: None,
    )
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["review-model"], None),
    )

    async def run() -> None:
        agent = CaptureConfigAgent()
        client = await _create_client(agent)
        try:
            save_response = await client.post(
                "/ui/api/settings",
                json={"providers": {"openrouter": {"api_key": "next-request-key"}}},
            )
            assert save_response.status == 200

            create_response = await client.post("/ui/api/sessions")
            session = (await create_response.json()).get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)

            select_response = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": "openrouter", "model": "review-model"},
            )
            assert select_response.status == 200

            send_response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "use saved credentials"},
            )
            assert send_response.status == 200
            assert agent.last_api_key == "next-request-key"
        finally:
            await client.close()

    asyncio.run(run())


def test_invalid_api_key_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_keys_path = _isolate_settings_paths(monkeypatch, tmp_path)
    save_api_keys({"openrouter": "previous-valid-key"}, path=api_keys_path)
    monkeypatch.setattr(
        "server.http_api._validate_provider_api_key",
        lambda provider, api_key: "API-ключ для openrouter отклонён провайдером (HTTP 401).",
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={"providers": {"openrouter": {"api_key": "rejected-key"}}},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_api_key"
            assert "отклонён" in str(error.get("message"))
            assert "rejected-key" not in json.dumps(payload)
        finally:
            await client.close()

    asyncio.run(run())

    assert load_api_keys(path=api_keys_path) == {"openrouter": "previous-valid-key"}


def test_provider_api_key_validation_uses_authenticated_models_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class ValidResponse:
        def raise_for_status(self) -> None:
            return None

    def get_valid(
        url: str,
        *,
        headers: dict[str, str],
        timeout: int,
    ) -> ValidResponse:
        observed.update(url=url, headers=headers, timeout=timeout)
        return ValidResponse()

    monkeypatch.setattr("server.http.common.ui_settings.requests.get", get_valid)

    from server.http.common import ui_settings

    assert ui_settings._validate_provider_api_key("openai", "  valid-key  ") is None
    assert observed == {
        "url": ui_settings.OPENAI_MODELS_ENDPOINT,
        "headers": {"Authorization": "Bearer valid-key"},
        "timeout": ui_settings.MODEL_FETCH_TIMEOUT,
    }

    class RejectedResponse:
        def raise_for_status(self) -> None:
            response = requests.Response()
            response.status_code = 401
            raise requests.HTTPError(response=response)

    monkeypatch.setattr(
        "server.http.common.ui_settings.requests.get",
        lambda *args, **kwargs: RejectedResponse(),
    )

    error = ui_settings._validate_provider_api_key("openrouter", "rejected-key")
    assert error == "API-ключ для openrouter отклонён провайдером (HTTP 401)."
    assert "rejected-key" not in error


def test_model_selection_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _isolate_settings_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["model-a", "model-b"], None),
    )

    async def run() -> None:
        agent = CaptureConfigAgent()
        client = await _create_client(agent)
        try:
            create_response = await client.post("/ui/api/sessions")
            session = (await create_response.json()).get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)

            select_response = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": "local", "model": "model-b"},
            )
            assert select_response.status == 200

            send_response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "use the selected model"},
            )
            assert send_response.status == 200
            assert agent.last_provider == "local"
            assert agent.last_model == "model-b"
        finally:
            await client.close()

    asyncio.run(run())


def test_environment_api_key_takes_priority_over_saved_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_keys_path = _isolate_settings_paths(monkeypatch, tmp_path)
    save_api_keys({"openrouter": "saved-file-key"}, path=api_keys_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-key")

    from server import http_api

    assert http_api._resolve_provider_api_key("openrouter") == "environment-key"
    assert http_api._provider_api_key_source("openrouter") == "env"
