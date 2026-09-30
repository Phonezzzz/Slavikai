"""Loopback custom providers must bypass the system proxy.

With requests' default trust_env=True, even http://127.0.0.1 goes through
HTTP_PROXY, which would receive the Authorization API key in cleartext.
These tests use stubs only (no real sockets, per DevRules.md): they capture
the ``proxies`` kwarg our code passes and verify through requests'
``resolve_proxies`` that the environment proxy is actually bypassed.
"""

from __future__ import annotations

import pytest
import requests
from requests.utils import resolve_proxies

from llm.local_http_brain import LocalHttpBrain, proxies_for_provider_url
from llm.types import ModelConfig
from server.http.common import ui_settings
from server.http.common.ui_settings import _probe_openai_models
from shared.models import LLMMessage

_API_KEY = "secret-proxy-test-key"
_ENV_PROXY = "http://127.0.0.1:9999"


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


@pytest.fixture
def proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", _ENV_PROXY)
    monkeypatch.setenv("http_proxy", _ENV_PROXY)
    # Empty NO_PROXY: without the fix, requests would use the proxy.
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.delenv("ALL_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)


def _resolved_proxy(url: str, proxies: object) -> str | None:
    """What requests would actually use for this URL with trust_env=True."""
    prepared = requests.Request("GET", url).prepare()
    resolved = resolve_proxies(prepared, proxies, trust_env=True)  # type: ignore[arg-type]
    return resolved.get("http")


def test_proxies_for_provider_url() -> None:
    assert proxies_for_provider_url("http://127.0.0.1:8080/v1") == {
        "http": None,
        "https": None,
    }
    assert proxies_for_provider_url("http://localhost:8080/v1") == {
        "http": None,
        "https": None,
    }
    assert proxies_for_provider_url("http://[::1]:8080/v1") == {
        "http": None,
        "https": None,
    }
    assert proxies_for_provider_url("https://example.test/v1") is None
    assert proxies_for_provider_url("not a url") is None


def test_probe_bypasses_system_proxy_for_loopback(
    monkeypatch: pytest.MonkeyPatch, proxy_env: None
) -> None:
    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        assert kwargs["headers"]["Authorization"] == f"Bearer {_API_KEY}"  # type: ignore[index]
        return FakeResponse(200, {"data": [{"id": "m1"}]})

    monkeypatch.setattr(ui_settings.requests, "get", fake_get)
    models, status, _ = _probe_openai_models("http://127.0.0.1:1234/v1", _API_KEY)
    assert status == "ready"
    assert models == ["m1"]
    # The captured proxies mapping must defeat the environment proxy.
    assert _resolved_proxy("http://127.0.0.1:1234/v1/models", captured["proxies"]) is None
    # Negative control: without our mapping the env proxy would be used.
    assert _resolved_proxy("http://127.0.0.1:1234/v1/models", None) == _ENV_PROXY


def _make_brain() -> LocalHttpBrain:
    return LocalHttpBrain(
        default_config=ModelConfig(
            provider="custom-0123456789abcdef0123456789abcdef",
            model="opaque/model",
            base_url="http://127.0.0.1:1234/v1/chat/completions",
            api_key=_API_KEY,
        ),
        native_tools=False,
    )


def test_brain_plain_completion_bypasses_system_proxy(
    monkeypatch: pytest.MonkeyPatch, proxy_env: None
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        return FakeResponse(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr("llm.local_http_brain.requests.post", fake_post)
    result = _make_brain().generate([LLMMessage(role="user", content="hi")])
    assert result.text == "hi"
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", captured["proxies"]) is None
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", None) == _ENV_PROXY


def test_brain_cancellable_completion_bypasses_system_proxy(
    monkeypatch: pytest.MonkeyPatch, proxy_env: None
) -> None:
    captured: dict[str, object] = {}

    def fake_post(self: object, url: str, **kwargs: object) -> FakeResponse:
        captured.update(kwargs)
        assert kwargs.get("stream") is True
        return FakeResponse(200, {"choices": [{"message": {"content": "hi"}}]})

    monkeypatch.setattr(requests.Session, "post", fake_post)
    import asyncio

    events = list(
        _make_brain().generate_stream_events(
            [LLMMessage(role="user", content="hi")],
            cancellation_token=asyncio.Event(),
        )
    )
    assert events, "expected stream events"
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", captured["proxies"]) is None
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", None) == _ENV_PROXY
