from __future__ import annotations

import asyncio
import json
import stat
import threading
from pathlib import Path

import pytest
import requests
from aiohttp.test_utils import TestClient, TestServer

from config.api_keys import load_api_keys
from config.http_server_config import HttpAuthConfig
from llm.brain_factory import create_brain
from llm.types import ModelConfig
from server.http.app import create_app
from server.http.common import ui_settings
from server.http.common.request_identity import CloudflareAccessTokenError, VerifiedAccessClaims
from server.ui_session_storage import InMemoryUISessionStorage
from shared.models import LLMMessage

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
        ("http://localhost:1234", "http://localhost:1234/v1"),
        ("http://127.4.5.6/v1/", "http://127.4.5.6/v1"),
        ("http://[::1]:1234/custom/", "http://[::1]:1234/custom"),
    ],
)
def test_base_url_normalization(base: str, expected: str) -> None:
    assert ui_settings._normalize_openai_base_url(base) == expected


@pytest.mark.parametrize(
    "base", ["http://example.test", "http://192.168.1.2", "http://evil.localhost.example"]
)
def test_remote_http_rejected(base: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        ui_settings._normalize_openai_base_url(base)


def test_discovery_persistence_selection_and_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_path = isolate(monkeypatch, tmp_path)
    observed: list[tuple[str, str, str]] = []

    def fake_get(
        url: str, *, headers: dict[str, str], timeout: int, allow_redirects: bool
    ) -> FakeResponse:
        assert timeout == ui_settings.MODEL_FETCH_TIMEOUT
        assert allow_redirects is False
        observed.append(("models", url, headers["Authorization"]))
        return FakeResponse(200, {"data": [{"id": "unknown/vendor.model:latest"}]})

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: int,
        allow_redirects: bool,
    ) -> FakeResponse:
        assert allow_redirects is False
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
                },
            )
            assert create.status == 200
            created = await create.json()
            provider = created["provider"]
            assert provider.startswith("custom-")
            assert "model" not in created
            assert "test-secret" not in json.dumps(created)
            assert ui_settings._load_provider_instance(provider) == {
                "display_name": "Any provider",
                "base_url": "https://example.test/v1",
            }

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
            assert any(item["provider"] == provider and item["models"] == [] for item in listed)

            discovered = await client.get(f"/ui/api/models?provider={provider}&strict=1")
            assert (await discovered.json())["providers"][0]["models"] == [
                "unknown/vendor.model:latest"
            ]

            forbidden_manual = await client.post(
                "/ui/api/session-model",
                json={"provider": provider, "model": "not-in-catalog"},
            )
            assert forbidden_manual.status == 404

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
                },
            )
            assert created.status == 200
            provider = (await created.json())["provider"]
            models_response = await client.get(f"/ui/api/models?provider={provider}&strict=1")
            models_payload = (await models_response.json())["providers"][0]
            assert models_payload["models"] == []
            assert models_payload["status"] == expected_status
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


def test_rejected_saved_key_cannot_use_manual_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_path = isolate(monkeypatch, tmp_path)
    provider = ui_settings._save_provider_instance(
        display_name="Rejected later", base_url="https://example.test/v1"
    )
    from config.api_keys import save_api_keys

    save_api_keys({provider: "rejected-secret"}, path=key_path)
    monkeypatch.setattr(ui_settings.requests, "get", lambda *args, **kwargs: FakeResponse(401, {}))

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            session = await client.post("/ui/api/sessions")
            session_id = (await session.json())["session"]["session_id"]
            selected = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": provider, "model": "manual-id"},
            )
            assert selected.status == 400
            assert (await selected.json())["error"]["code"] == "invalid_api_key"
        finally:
            await client.close()

    asyncio.run(run())


def test_create_rejects_model_field(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    isolate(monkeypatch, tmp_path)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/provider-instances",
                json={
                    "display_name": "Example",
                    "base_url": "https://example.test",
                    "api_key": "secret",
                    "model": "wrong-place",
                },
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "model_selection_in_chat"
        finally:
            await client.close()

    asyncio.run(run())
    assert ui_settings._load_provider_instances() == {}


def test_custom_default_model_update_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate(monkeypatch, tmp_path)
    provider = ui_settings._save_provider_instance(
        display_name="Example", base_url="https://example.test/v1"
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={"model": {"provider": provider, "model": "opaque"}},
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "custom_model_session_only"
        finally:
            await client.close()

    asyncio.run(run())


def test_dynamic_provider_has_no_native_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    brain = create_brain(
        ModelConfig(
            provider="custom-0123456789abcdef0123456789abcdef",
            model="opaque",
            base_url="https://example.test/v1/chat/completions",
        ),
        api_key="test-secret",
    )
    local = create_brain(ModelConfig(provider="local", model="local-model"))
    assert brain.supports_native_tools is False
    assert brain.supports_streaming_tools is False
    assert local.supports_native_tools is True
    assert local.supports_streaming_tools is True
    monkeypatch.setattr(
        "llm.local_http_brain.requests.post", lambda *args, **kwargs: pytest.fail("tools sent")
    )
    from llm.types import ToolSpec

    with pytest.raises(RuntimeError, match="native_tools_required"):
        brain.generate(
            [LLMMessage(role="user", content="hello")],
            tools=[ToolSpec(name="test", description="test")],
        )


@pytest.mark.parametrize("path", ["/ui/api/provider-instances/probe", "/ui/api/provider-instances"])
def test_provider_probe_does_not_block_aiohttp_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, path: str
) -> None:
    isolate(monkeypatch, tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_get(*args: object, **kwargs: object) -> FakeResponse:
        started.set()
        assert release.wait(2)
        return FakeResponse(200, {"data": [{"id": "opaque"}]})

    monkeypatch.setattr(ui_settings.requests, "get", slow_get)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            body = {"base_url": "https://example.test", "api_key": "secret"}
            if path.endswith("provider-instances"):
                body["display_name"] = "Example"
            slow_request = asyncio.create_task(client.post(path, json=body))
            try:
                assert await asyncio.to_thread(started.wait, 1)
                health = await asyncio.wait_for(client.get("/ui/api/auth/status"), timeout=0.5)
                assert health.status == 200
            finally:
                release.set()
            assert (await slow_request).status == 200
        finally:
            release.set()
            await client.close()

    asyncio.run(run())


def test_member_cannot_use_owner_custom_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_path = isolate(monkeypatch, tmp_path)
    provider = ui_settings._save_provider_instance(
        display_name="Owner provider", base_url="https://example.test/v1"
    )
    from config.api_keys import save_api_keys

    save_api_keys({provider: "owner-secret"}, path=key_path)
    network_calls: list[str] = []

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        network_calls.append("models")
        return FakeResponse(200, {"data": [{"id": "opaque"}]})

    monkeypatch.setattr(ui_settings.requests, "get", fake_get)

    def fake_post(*args: object, **kwargs: object) -> FakeResponse:
        network_calls.append("completion")
        return FakeResponse(200, {"choices": [{"message": {"content": "owner-ok"}}]})

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)

    class StubVerifier:
        async def verify(self, token: str) -> VerifiedAccessClaims:
            if token == "owner-token":
                return VerifiedAccessClaims(email="owner@example.com")
            if token == "member-token":
                return VerifiedAccessClaims(email="member@example.com")
            raise CloudflareAccessTokenError("invalid")

    async def run() -> None:
        app = create_app(
            agent=CompletionAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(
                api_token="automation-token",
                browser_auth_mode="cloudflare",
                cloudflare_access_issuer="https://example.cloudflareaccess.com",
                cloudflare_access_aud="aud",
                owner_email="owner@example.com",
            ),
            cloudflare_access_verifier=StubVerifier(),
        )
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            owner = {"Cf-Access-Jwt-Assertion": "owner-token"}
            member = {"Cf-Access-Jwt-Assertion": "member-token"}
            summary = await client.get("/ui/api/models?summary=1", headers=member)
            assert provider not in json.dumps(await summary.json())
            settings = await client.get("/ui/api/settings", headers=member)
            assert provider not in json.dumps(await settings.json())
            direct = await client.get(f"/ui/api/models?provider={provider}", headers=member)
            assert direct.status == 403
            probe = await client.post(
                "/ui/api/provider-instances/probe",
                headers=member,
                json={"base_url": "https://example.test", "api_key": "member-key"},
            )
            assert probe.status == 403
            created = await client.post(
                "/ui/api/provider-instances",
                headers=member,
                json={
                    "display_name": "x",
                    "base_url": "https://example.test",
                    "api_key": "member-key",
                },
            )
            assert created.status == 403
            session = await client.post("/ui/api/sessions", headers=member)
            session_id = (await session.json())["session"]["session_id"]
            select = await client.post(
                "/ui/api/session-model",
                headers={**member, "X-Slavik-Session": session_id},
                json={"provider": provider, "model": "opaque"},
            )
            assert select.status == 403
            await app["ui_hub"].set_session_model(session_id, provider, "opaque")
            send = await client.post(
                "/ui/api/chat/send",
                headers={**member, "X-Slavik-Session": session_id},
                json={"content": "private"},
            )
            assert send.status == 403
            await app["runtime_model_state"].set_global_main(
                ModelConfig(
                    provider=provider,
                    model="opaque",
                    base_url="https://example.test/v1/chat/completions",
                )
            )
            automation = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": "Bearer automation-token"},
                json={"model": "slavik", "messages": [{"role": "user", "content": "private"}]},
            )
            assert automation.status == 403
            assert network_calls == []
            owner_models = await client.get(f"/ui/api/models?provider={provider}", headers=owner)
            assert (await owner_models.json())["providers"][0]["models"] == ["opaque"]
            assert network_calls == ["models"]
            owner_session = await client.post("/ui/api/sessions", headers=owner)
            owner_session_id = (await owner_session.json())["session"]["session_id"]
            owner_select = await client.post(
                "/ui/api/session-model",
                headers={**owner, "X-Slavik-Session": owner_session_id},
                json={"provider": provider, "model": "opaque"},
            )
            assert owner_select.status == 200
            owner_send = await client.post(
                "/ui/api/chat/send",
                headers={**owner, "X-Slavik-Session": owner_session_id},
                json={"content": "hello"},
            )
            assert owner_send.status == 200
            assert "owner-ok" in await owner_send.text()
            assert network_calls == ["models", "models", "completion"]
        finally:
            await client.close()

    asyncio.run(run())
