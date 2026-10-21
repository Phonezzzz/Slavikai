from __future__ import annotations

import asyncio
import threading

import pytest

from llm.cancellation import cancel_generation
from llm.local_http_brain import LocalHttpBrain
from llm.stream_model import Done
from llm.types import ModelConfig
from shared.models import LLMMessage
from tests.fake_provider_http import HoldingProvider


def test_native_sync_disables_redirects(monkeypatch):
    def post(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        raise RuntimeError("observed")

    monkeypatch.setattr("requests.post", post)
    brain = LocalHttpBrain(ModelConfig(provider="local", model="fixture"))
    with pytest.raises(RuntimeError, match="observed"):
        brain.generate([LLMMessage(role="user", content="inert")])


@pytest.mark.parametrize("stage", ["dns", "connect", "tls", "headers", "body"])
def test_native_stream_cancels_before_and_after_headers(monkeypatch, stage):
    upstream = HoldingProvider(monkeypatch, stage, streaming=True)

    # Старый requests path не должен обращаться к настоящей сети даже в red run.
    def reject_old_transport(*args, **kwargs):
        raise AssertionError("native stream bypassed scoped transport")

    monkeypatch.setattr("requests.post", reject_old_transport)
    scheme = "https" if stage == "tls" else "http"
    brain = LocalHttpBrain(
        ModelConfig(
            provider="local",
            model="fixture",
            base_url=f"{scheme}://provider.test/v1/chat/completions",
        )
    )
    token = asyncio.Event()
    events, errors = [], []

    def run():
        try:
            events.extend(
                brain.generate_stream_events(
                    [LLMMessage(role="user", content="inert")],
                    cancellation_token=token,
                )
            )
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert upstream.started.wait(1)
        cancel_generation(token)
        worker.join(1)
        assert not worker.is_alive()
        assert not errors
        assert isinstance(events[-1], Done)
        assert events[-1].finish_reason == "cancelled"
        assert upstream.active == 0
        assert all(resolver.closed for resolver in upstream.resolvers)
        assert all(response.closed for response in upstream.responses)
        assert upstream.calls == 1
    finally:
        cancel_generation(token)
        worker.join(2)


def _stream_brain():
    return LocalHttpBrain(
        ModelConfig(
            provider="local",
            model="fixture",
            base_url="https://provider.test/v1/chat/completions",
        ),
        api_key="fixture-only",
    )


@pytest.mark.parametrize("text", ["Привет", "x" * 200_000], ids=["utf8", "large-delta"])
def test_native_stream_is_incremental_and_closes_on_consumer_stop(monkeypatch, text):
    import json
    from types import SimpleNamespace

    import aiohttp

    from llm.stream_model import TextDelta
    from tests.fake_provider_http import FakeHttpResponse

    responses, sessions, resolvers = [], [], []

    class SSE(FakeHttpResponse):
        @property
        def content(self):
            return self.reader

    async def request(session, method, url, **kwargs):
        sessions.append(session)
        resolvers.append(session.connector._resolver)
        assert kwargs["allow_redirects"] is False
        assert kwargs["json"]["stream"] is True
        assert session.timeout.total is None
        assert session.timeout.sock_read == 30
        response = SSE()
        response.reader = aiohttp.StreamReader(
            SimpleNamespace(
                _reading_paused=False,
                pause_reading=lambda: None,
                resume_reading=lambda: None,
            ),
            limit=65536,
        )
        line = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": text}}]}, ensure_ascii=False)
            + "\n\n"
        ).encode()
        # Настоящий StreamReader собирает UTF-8, разрезанный в середине codepoint.
        if text == "Привет":
            chunks = [bytes([byte]) for byte in line]
        else:
            chunks = [line[index : index + 8192] for index in range(0, len(line), 8192)]
        for chunk in chunks:
            response.reader.feed_data(chunk)
        # EOF не отправлен: первый delta должен выйти, не дожидаясь остального body.
        responses.append(response)
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    events = _stream_brain().generate_stream_events([LLMMessage(role="user", content="inert")])
    try:
        first = next(events)
        assert isinstance(first, TextDelta)
        assert first.text == text
        assert not responses[0].closed
    finally:
        events.close()
    assert responses[0].closed
    assert all(session.closed for session in sessions)
    assert all(resolver._resolver is None for resolver in resolvers)


@pytest.mark.parametrize("status", [307, 401, 429, 500])
def test_native_stream_errors_close_without_read_or_replay(monkeypatch, status):
    import aiohttp
    import requests

    from tests.fake_provider_http import FakeHttpResponse

    responses, sessions = [], []

    async def reject_read():
        raise AssertionError("Error body must not be read")

    async def request(session, method, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        sessions.append(session)
        response = FakeHttpResponse(status=status, read=reject_read)
        responses.append(response)
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    with pytest.raises(requests.HTTPError):
        list(_stream_brain().generate_stream_events([LLMMessage(role="user", content="inert")]))
    assert len(responses) == 1
    assert responses[0].closed
    assert sessions[0].closed


def test_native_pre_cancel_dispatches_nothing(monkeypatch):
    import aiohttp

    async def reject(*args, **kwargs):
        raise AssertionError("Cancelled request dispatched")

    monkeypatch.setattr(aiohttp.ClientSession, "_request", reject)
    token = asyncio.Event()
    token.set()
    events = list(
        _stream_brain().generate_stream_events(
            [LLMMessage(role="user", content="inert")],
            cancellation_token=token,
        )
    )
    assert len(events) == 1
    assert events[0].finish_reason == "cancelled"


def test_native_stream_preserves_tool_protocol_and_credentials_policy(monkeypatch):
    import json
    import ssl

    import aiohttp

    from llm.stream_model import ToolCallCompleted
    from llm.types import ToolSpec
    from tests.fake_provider_http import FakeHttpResponse

    calls = []
    response = None
    context = ssl.create_default_context()
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/fixture/ca.pem")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.test:8888")
    monkeypatch.setenv("all_proxy", "http://proxy.test:8888")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    def ssl_context(*args, **kwargs):
        assert kwargs["cafile"] == "/fixture/ca.pem"
        return context

    monkeypatch.setattr(ssl, "create_default_context", ssl_context)

    class SSE(FakeHttpResponse):
        async def _lines(self):
            for index, call_id, fragment in [
                (1, "b", '{"n":'),
                (0, "a", '{"n":1}'),
                (1, None, "2}"),
            ]:
                call = {"index": index, "function": {"arguments": fragment}}
                if call_id:
                    call.update(id=call_id)
                    call["function"]["name"] = "inert_fixture"
                yield (
                    "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]}) + "\n"
                ).encode()
            yield b"data: [DONE]\n"

    async def request(session, method, url, **kwargs):
        nonlocal response
        calls.append(kwargs["json"])
        assert session.trust_env is False
        assert kwargs["ssl"] is context
        assert kwargs["proxy"] == "http://proxy.test:8888"
        assert kwargs["headers"]["Authorization"] == "Bearer fixture-only"
        response = SSE()
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    previous = {
        "id": "previous",
        "type": "function",
        "function": {"name": "inert_fixture", "arguments": "{}"},
    }
    messages = [
        LLMMessage(role="user", content="inert"),
        LLMMessage(role="assistant", content="", tool_calls=[previous]),
        LLMMessage(role="tool", content='{"ok":true}', tool_call_id="previous"),
    ]
    tool = ToolSpec(
        name="inert_fixture", description="Inert fixture", parameters_schema={"type": "object"}
    )
    events = list(_stream_brain().generate_stream_events(messages, tools=[tool]))
    assert calls[0]["messages"] == [message.to_provider_dict() for message in messages]
    assert calls[0]["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert calls[0]["tool_choice"] == "auto"
    completed = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [(call.id, call.arguments) for call in completed] == [("a", {"n": 1}), ("b", {"n": 2})]
    assert response.closed


def test_native_cancel_between_deltas_closes_and_emits_no_completed_call(monkeypatch):
    import aiohttp

    from llm.stream_model import ToolCallCompleted, ToolCallStarted
    from tests.fake_provider_http import FakeHttpResponse

    response = None

    class SSE(FakeHttpResponse):
        async def _lines(self):
            yield (
                b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"x",'
                b'"function":{"name":"inert_fixture","arguments":"{"}}]}}]}\n'
            )
            raise AssertionError("Cancellation must prevent next body read")

    async def request(*args, **kwargs):
        nonlocal response
        response = SSE()
        return response

    monkeypatch.setattr(aiohttp.ClientSession, "_request", request)
    token = asyncio.Event()
    events = _stream_brain().generate_stream_events(
        [LLMMessage(role="user", content="inert")],
        cancellation_token=token,
    )
    assert isinstance(next(events), ToolCallStarted)
    cancel_generation(token)
    remaining = list(events)
    assert not any(isinstance(event, ToolCallCompleted) for event in remaining)
    assert remaining[-1].finish_reason == "cancelled"
    assert response.closed


def test_native_sync_rejects_redirect_without_retry(monkeypatch):
    import requests

    from llm.retry import ProviderRequestError

    calls = []

    def request(*args, **kwargs):
        assert kwargs["allow_redirects"] is False
        calls.append(kwargs)
        response = requests.Response()
        response.status_code = 307
        response._content = b'{"choices":[{"message":{"content":"must not accept"}}]}'
        response._content_consumed = True
        return response

    monkeypatch.setattr("requests.post", request)
    with pytest.raises(ProviderRequestError) as error:
        _stream_brain().generate([LLMMessage(role="user", content="inert")])
    assert error.value.status_code == 307
    assert error.value.attempts == 1
    assert len(calls) == 1
