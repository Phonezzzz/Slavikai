"""Loopback custom providers must bypass the system proxy.

With requests' default trust_env=True, even http://127.0.0.1 goes through
HTTP_PROXY, which would receive the Authorization API key in cleartext.
These tests use stubs only (no real sockets, per DevRules.md): they capture
the ``proxies`` kwarg our code passes and verify through requests'
``resolve_proxies`` that the environment proxy is actually bypassed.
"""

from __future__ import annotations

import aiohttp
import pytest
import requests
from requests.utils import resolve_proxies, select_proxy

from llm.local_http_brain import LocalHttpBrain
from llm.provider_http import proxies_for_provider_url
from llm.types import ModelConfig
from server.http.common import ui_settings
from server.http.common.ui_settings import _probe_openai_models
from shared.models import LLMMessage


@pytest.mark.parametrize(
    "proxy_name",
    ["HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"],
)
@pytest.mark.parametrize(
    "host", ["http://127.0.0.1:1234", "http://localhost:1234", "https://[::1]:1234"]
)
@pytest.mark.parametrize("path", ["probe", "completion", "custom_stream", "native_stream"])
def test_credential_request_reaches_direct_transport(monkeypatch, proxy_name, host, path):
    import asyncio
    import json

    for name in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(proxy_name, "http://proxy.invalid:8888")
    calls = []

    def send(session, request, **kwargs):
        proxy = select_proxy(request.url, kwargs.get("proxies", {}))
        calls.append(proxy)
        assert not proxy, "credential-bearing loopback request selected an inherited proxy"
        assert request.headers["Authorization"] == "Bearer synthetic-key"
        response = requests.Response()
        response.status_code = 200
        body = json.loads(request.body) if request.body else {}
        response._content = (
            b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\ndata: [DONE]\n\n'
            if body.get("stream")
            else json.dumps(
                {"data": [{"id": "m"}], "choices": [{"message": {"content": "ok"}}]}
            ).encode()
        )
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.Session, "send", send)

    async def async_send(session, method, url, **kwargs):
        from tests.fake_provider_http import FakeHttpResponse

        assert session.trust_env is False
        assert kwargs.get("proxy") is None
        assert kwargs["headers"]["Authorization"] == "Bearer synthetic-key"
        calls.append(None)
        return FakeHttpResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "_request", async_send)
    if path == "probe":
        assert _probe_openai_models(host + "/v1", "synthetic-key")[1] == "ready"
    else:
        brain = LocalHttpBrain(
            ModelConfig(
                provider="custom-" + "0" * 32,
                model="m",
                base_url=host + "/v1/chat/completions",
                api_key="synthetic-key",
            ),
            native_tools=path == "native_stream",
        )
        messages = [LLMMessage(role="user", content="hi")]
        if path == "completion":
            assert brain.generate(messages).text == "ok"
        else:
            assert list(brain.generate_stream_events(messages, cancellation_token=asyncio.Event()))
    assert len(calls) == 1


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
        "http": "",
        "https": "",
        "all": "",
    }
    assert proxies_for_provider_url("http://localhost:8080/v1") == {
        "http": "",
        "https": "",
        "all": "",
    }
    assert proxies_for_provider_url("http://[::1]:8080/v1") == {
        "http": "",
        "https": "",
        "all": "",
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
    assert _resolved_proxy("http://127.0.0.1:1234/v1/models", captured["proxies"]) in (None, "")
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
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", captured["proxies"]) in (
        None,
        "",
    )
    assert _resolved_proxy("http://127.0.0.1:1234/v1/chat/completions", None) == _ENV_PROXY


def test_brain_cancellable_completion_bypasses_system_proxy(monkeypatch, proxy_env):
    import asyncio

    from tests.fake_provider_http import FakeHttpResponse

    calls = []

    async def fake_post(session, method, url, **kwargs):
        assert session.trust_env is False
        assert kwargs.get("proxy") is None
        calls.append(kwargs["headers"]["Authorization"])
        return FakeHttpResponse()

    monkeypatch.setattr(aiohttp.ClientSession, "_request", fake_post)
    assert list(
        _make_brain().generate_stream_events(
            [LLMMessage(role="user", content="hi")], cancellation_token=asyncio.Event()
        )
    )
    assert calls == [f"Bearer {_API_KEY}"]
