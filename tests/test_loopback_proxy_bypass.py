"""Loopback custom providers must bypass the system proxy.

With requests' default trust_env=True, even http://127.0.0.1 goes through
HTTP_PROXY, which would receive the Authorization API key in cleartext.
"""

from __future__ import annotations

import asyncio
import http.server
import socketserver
import threading

import pytest

from llm.local_http_brain import LocalHttpBrain
from llm.types import ModelConfig
from server.http.common.ui_settings import _probe_openai_models
from shared.models import LLMMessage

_API_KEY = "secret-proxy-test-key"


class _RequestRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests: list[dict[str, str | None]] = []

    def record(self, handler: http.server.BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length", 0) or 0)
        if length:
            handler.rfile.read(length)
        with self._lock:
            self._requests.append(
                {
                    "path": handler.path,
                    "authorization": handler.headers.get("Authorization"),
                }
            )

    def saw_key(self, key: str) -> bool:
        with self._lock:
            return any(item["authorization"] == f"Bearer {key}" for item in self._requests)

    def count(self) -> int:
        with self._lock:
            return len(self._requests)


def _make_handler(recorder: _RequestRecorder, body: bytes | None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _handle(self) -> None:
            recorder.record(self)
            payload = body if body is not None else b"{}"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self) -> None:  # noqa: N802
            self._handle()

        def do_POST(self) -> None:  # noqa: N802
            self._handle()

        def log_message(self, *args) -> None:  # noqa: ANN001, ANN202
            pass

    return Handler


class _ProxyEnv:
    """A loopback provider plus a fake proxy, with proxy env forced on."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, provider_body: bytes) -> None:
        self.provider_recorder = _RequestRecorder()
        self.proxy_recorder = _RequestRecorder()
        socketserver.TCPServer.allow_reuse_address = True
        self._provider = socketserver.TCPServer(
            ("127.0.0.1", 0), _make_handler(self.provider_recorder, provider_body)
        )
        self._proxy = socketserver.TCPServer(
            ("127.0.0.1", 0), _make_handler(self.proxy_recorder, None)
        )
        self.provider_port = self._provider.server_address[1]
        proxy_port = self._proxy.server_address[1]
        self._threads = [
            threading.Thread(target=self._provider.serve_forever, daemon=True),
            threading.Thread(target=self._proxy.serve_forever, daemon=True),
        ]
        for thread in self._threads:
            thread.start()
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.setenv("http_proxy", proxy_url)
        # Empty NO_PROXY: without the fix, requests would use the proxy.
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")
        monkeypatch.delenv("ALL_PROXY", raising=False)
        monkeypatch.delenv("all_proxy", raising=False)

    @property
    def provider_base(self) -> str:
        return f"http://127.0.0.1:{self.provider_port}/v1"

    def close(self) -> None:
        self._provider.shutdown()
        self._proxy.shutdown()
        self._provider.server_close()
        self._proxy.server_close()


def _make_brain(env: _ProxyEnv) -> LocalHttpBrain:
    return LocalHttpBrain(
        default_config=ModelConfig(
            provider="custom-0123456789abcdef0123456789abcdef",
            model="opaque/model",
            base_url=f"{env.provider_base}/chat/completions",
            api_key=_API_KEY,
        ),
        native_tools=False,
    )


def test_probe_bypasses_system_proxy_for_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _ProxyEnv(monkeypatch, b'{"data":[{"id":"m1"}]}')
    try:
        models, status, _ = _probe_openai_models(env.provider_base, _API_KEY)
        assert status == "ready"
        assert models == ["m1"]
        assert env.provider_recorder.saw_key(_API_KEY)
        assert not env.proxy_recorder.saw_key(_API_KEY), "API key leaked to the proxy"
        assert env.proxy_recorder.count() == 0
    finally:
        env.close()


def test_brain_plain_completion_bypasses_system_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _ProxyEnv(monkeypatch, b'{"choices":[{"message":{"content":"hi"}}]}')
    try:
        result = _make_brain(env).generate([LLMMessage(role="user", content="hi")])
        assert result.text == "hi"
        assert env.provider_recorder.saw_key(_API_KEY)
        assert not env.proxy_recorder.saw_key(_API_KEY), "API key leaked to the proxy"
        assert env.proxy_recorder.count() == 0
    finally:
        env.close()


def test_brain_cancellable_completion_bypasses_system_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _ProxyEnv(monkeypatch, b'{"choices":[{"message":{"content":"hi"}}]}')
    try:
        token: asyncio.Event = asyncio.Event()
        events = list(
            _make_brain(env).generate_stream_events(
                [LLMMessage(role="user", content="hi")],
                cancellation_token=token,
            )
        )
        assert events, "expected stream events"
        assert env.provider_recorder.saw_key(_API_KEY)
        assert not env.proxy_recorder.saw_key(_API_KEY), "API key leaked to the proxy"
        assert env.proxy_recorder.count() == 0
    finally:
        env.close()
