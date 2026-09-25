from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

import pytest
import requests

from config.api_keys import load_api_keys
from llm.brain_factory import create_brain
from server.http.common import ui_settings

from .fakes import CaptureConfigAgent, DummyAgent, _create_client


class FakeResponse:
    def __init__(self, status: int, payload: object) -> None:
        self.status_code = status
        self.payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            raise requests.HTTPError(response=response)

    def json(self) -> object:
        return self.payload


class CompletionAgent(CaptureConfigAgent):
    def __init__(self) -> None:
        super().__init__()
        self.brain = None

    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True) -> None:
        super().reconfigure_models(main_config, main_api_key=main_api_key, persist=persist)
        self.brain = create_brain(main_config, api_key=main_api_key)

    def respond(self, messages) -> str:
        assert self.brain is not None
        return self.brain.generate(messages).text


def isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    settings_path = tmp_path / "ui_settings.json"
    key_path = tmp_path / "config" / "api_keys.json"
    monkeypatch.setattr(ui_settings, "UI_SETTINGS_PATH", settings_path)
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", settings_path)
    monkeypatch.setattr("server.http_api.API_KEYS_PATH", key_path)
    monkeypatch.setattr("server.http_api.MODEL_CONFIG_PATH", tmp_path / "model_config.json")
    return key_path


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://example.test", "https://example.test/v1"),
        ("https://example.test/", "https://example.test/v1"),
        ("https://example.test/v1", "https://example.test/v1"),
        ("https://example.test/v1/", "https://example.test/v1"),
        ("https://example.test/custom/api/", "https://example.test/custom/api"),
    ],
)
def test_base_url_normalization(base: str, expected: str) -> None:
    assert ui_settings._normalize_openai_base_url(base) == expected


def test_discovery_persistence_selection_and_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_path = isolate(monkeypatch, tmp_path)
    observed: list[tuple[str, str, str]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: int) -> FakeResponse:
        assert timeout == ui_settings.MODEL_FETCH_TIMEOUT
        observed.append(("models", url, headers["Authorization"]))
        return FakeResponse(200, {"data": [{"id": "unknown/vendor.model:latest"}]})

    def fake_post(
        url: str, *, json: dict[str, object], headers: dict[str, str], timeout: int
    ) -> FakeResponse:
        observed.append(("completion", url, headers["Authorization"]))
        assert json["model"] == "unknown/vendor.model:latest"
        assert timeout > 0
        return FakeResponse(200, {"choices": [{"message": {"content": "dynamic-ok"}}]})

    monkeypatch.setattr(ui_settings.requests, "get", fake_get)
    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)

    async def run() -> str:
        agent = CompletionAgent()
        client = await _create_client(agent)
        try:
            probe = await client.post(
                "/ui/api/provider-instances/probe",
                json={"base_url": "https://example.test/v1/", "api_key": "test-secret"},
            )
            assert probe.status == 200
            probe_body = await probe.json()
            assert probe_body["base_url"] == "https://example.test/v1"
            assert probe_body["models"] == ["unknown/vendor.model:latest"]
            create = await client.post(
                "/ui/api/provider-instances",
                json={
                    "display_name": "Any provider",
                    "base_url": "https://example.test/v1/",
                    "api_key": "test-secret",
                    "model": "unknown/vendor.model:latest",
                },
            )
            assert create.status == 200
            created = await create.json()
            provider = created["provider"]
            assert provider.startswith("custom-")
            assert "test-secret" not in json.dumps(created)

            settings = await client.get("/ui/api/settings")
            assert settings.status == 200
            settings_body = await settings.json()
            assert settings_body["settings"]["model"] is None
            assert "test-secret" not in json.dumps(settings_body)
            saved = next(
                item
                for item in settings_body["settings"]["providers"]
                if item["provider"] == provider
            )
            assert saved["display_name"] == "Any provider"
            assert saved["api_key_stored"] is True

            summary = await client.get("/ui/api/models?summary=1")
            listed = (await summary.json())["providers"]
            assert any(item["provider"] == provider for item in listed)

            session = await client.post("/ui/api/sessions")
            assert (await session.json())["session"]["selected_model"] is None
            session_id = (await session.json())["session"]["session_id"]
            selected = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": provider, "model": "unknown/vendor.model:latest"},
            )
            assert selected.status == 200
            send = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "hi"},
            )
            assert send.status == 200
            assert "dynamic-ok" in await send.text()
            assert agent.last_provider == provider
            assert agent.last_api_key == "test-secret"
            return provider
        finally:
            await client.close()

    provider = asyncio.run(run())
    assert load_api_keys(path=key_path)[provider] == "test-secret"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert "test-secret" not in (tmp_path / "ui_settings.json").read_text()
    assert (
        "completion",
        "https://example.test/v1/chat/completions",
        "Bearer test-secret",
    ) in observed
    assert all("/v1/v1/" not in url for _, url, _ in observed)


@pytest.mark.parametrize(
    ("http_status", "expected_status"),
    [
        (403, "models_unavailable"),
        (404, "models_unavailable"),
        (503, "models_unavailable"),
        (None, "provider_unavailable"),
    ],
)
def test_manual_model_when_models_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    http_status: int | None,
    expected_status: str,
) -> None:
    isolate(monkeypatch, tmp_path)

    def unavailable(*args: object, **kwargs: object) -> FakeResponse:
        if http_status is None:
            raise requests.ConnectionError("provider unavailable")
        return FakeResponse(http_status, {})

    monkeypatch.setattr(ui_settings.requests, "get", unavailable)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            probe = await client.post(
                "/ui/api/provider-instances/probe",
                json={"base_url": "https://example.test", "api_key": "manual-secret"},
            )
            assert (await probe.json())["status"] == expected_status
            created = await client.post(
                "/ui/api/provider-instances",
                json={
                    "display_name": "Unavailable catalog",
                    "base_url": "https://example.test",
                    "api_key": "manual-secret",
                    "model": "opaque-manual-id",
                },
            )
            assert created.status == 200
            provider = (await created.json())["provider"]
            session = await client.post("/ui/api/sessions")
            session_id = (await session.json())["session"]["session_id"]
            selected = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": provider, "model": "opaque-manual-id"},
            )
            assert selected.status == 200
        finally:
            await client.close()

    asyncio.run(run())


def test_invalid_api_key_never_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    key_path = isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings.requests,
        "get",
        lambda *args, **kwargs: FakeResponse(401, {"error": "rejected-secret"}),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/provider-instances",
                json={
                    "display_name": "Rejected",
                    "base_url": "https://example.test",
                    "api_key": "rejected-secret",
                    "model": "unknown-model",
                },
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "invalid_api_key"
            assert "rejected-secret" not in await response.text()
        finally:
            await client.close()

    asyncio.run(run())
    assert not key_path.exists()
    assert ui_settings._load_provider_instances() == {}
