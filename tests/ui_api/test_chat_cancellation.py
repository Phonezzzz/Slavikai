from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import threading
import time

import pytest

from core.agent_response import AgentResponse, ResponseProduced
from llm.cancellation import bind_cancellation_resource
from llm.local_http_brain import LocalHttpBrain
from llm.stream_model import Done, TextDelta
from llm.types import ModelConfig
from server.agent_provider import AgentScope

from .fakes import *


class _FakeProviderResponse:
    def __init__(self) -> None:
        self.closed = threading.Event()
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.closed.set()


class _CancellableStreamingAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.cleaned_up = threading.Event()
        self.provider_response = _FakeProviderResponse()
        self.yield_count = 0
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages, cancellation_token=None):  # noqa: ANN001
        del messages
        assert cancellation_token is not None
        try:
            with bind_cancellation_resource(cancellation_token, self.provider_response):
                self.started.set()
                while not cancellation_token.is_set():
                    time.sleep(0.01)
                    if cancellation_token.is_set():
                        break
                    self.yield_count += 1
                    yield TextDelta(text=f"partial-{self.yield_count} ")
                yield Done(finish_reason="cancelled")
        finally:
            self.cleaned_up.set()


async def _start_cancellable_send(
    client: TestClient,
    agent: _CancellableStreamingAgent,
) -> tuple[str, asyncio.Task]:
    status_response = await client.get("/ui/api/status")
    status_payload = await status_response.json()
    session_id = status_payload.get("session_id")
    assert isinstance(session_id, str) and session_id
    await _select_local_model(client, session_id)
    send_task = asyncio.create_task(
        client.post(
            "/ui/api/chat/send",
            json={"content": "write a deliberately long response"},
            headers={"X-Slavik-Session": session_id},
        )
    )
    started = await asyncio.to_thread(agent.started.wait, 2)
    assert started is True
    return session_id, send_task


async def _cancel_generation(client: TestClient, session_id: str):  # noqa: ANN202
    return await client.post(
        "/ui/api/chat/cancel",
        headers={"X-Slavik-Session": session_id},
    )


def test_chat_cancel_stops_generation() -> None:
    async def run() -> None:
        agent = _CancellableStreamingAgent()
        client = await _create_client(agent)
        try:
            session_id, send_task = await _start_cancellable_send(client, agent)
            cancel_response = await _cancel_generation(client, session_id)
            assert cancel_response.status == 200
            cancel_payload = await cancel_response.json()
            assert cancel_payload["cancelled"] is True

            chunks_at_confirmation = agent.yield_count
            await asyncio.sleep(0.05)
            assert agent.yield_count == chunks_at_confirmation

            send_response = await asyncio.wait_for(send_task, timeout=2)
            assert send_response.status == 200
            assert (await send_response.json())["cancelled"] is True
        finally:
            await client.close()

    asyncio.run(run())


def test_chat_cancel_cleans_up_resources() -> None:
    async def run() -> None:
        agent = _CancellableStreamingAgent()
        client = await _create_client(agent)
        try:
            session_id, send_task = await _start_cancellable_send(client, agent)
            cancel_response = await _cancel_generation(client, session_id)
            assert cancel_response.status == 200
            send_response = await asyncio.wait_for(send_task, timeout=2)
            assert send_response.status == 200

            assert agent.provider_response.closed.is_set()
            assert agent.provider_response.close_calls >= 1
            assert agent.cleaned_up.is_set()
            provider = client.server.app["agent_provider"]
            agent_lock = provider.lock_for(
                AgentScope(principal_id="test-static-agent", session_id=session_id)
            )
            await asyncio.wait_for(agent_lock.acquire(), timeout=1)
            agent_lock.release()
        finally:
            await client.close()

    asyncio.run(run())


def test_chat_cancel_partial_response_not_saved() -> None:
    async def run() -> None:
        agent = _CancellableStreamingAgent()
        client = await _create_client(agent)
        try:
            session_id, send_task = await _start_cancellable_send(client, agent)
            cancel_response = await _cancel_generation(client, session_id)
            assert cancel_response.status == 200
            send_response = await asyncio.wait_for(send_task, timeout=2)
            send_payload = await send_response.json()

            messages = send_payload["messages"]
            assert [message["role"] for message in messages] == ["user"]
            assert messages[0]["content"] == "write a deliberately long response"
            assert all("partial-" not in message["content"] for message in messages)

            history_response = await client.get(
                f"/ui/api/sessions/{session_id}/history",
                headers={"X-Slavik-Session": session_id},
            )
            assert history_response.status == 200
            history_payload = await history_response.json()
            history_messages = history_payload["messages"]
            assert [message["role"] for message in history_messages] == ["user"]
        finally:
            await client.close()

    asyncio.run(run())


class _DynamicBrainAgent(DummyAgent):
    def __init__(self, brain: LocalHttpBrain) -> None:
        super().__init__()
        self._brain = brain
        self.worker_finished = threading.Event()
        self.last_stream_response_raw = None

    def respond_stream(self, messages, cancellation_token=None):
        self.worker_finished.clear()
        try:
            text = ""
            finished = False
            for event in self._brain.generate_stream_events(
                messages, cancellation_token=cancellation_token
            ):
                if isinstance(event, TextDelta):
                    text = event.text if event.mode == "replace" else text + event.text
                if isinstance(event, Done):
                    finished = event.finish_reason not in {"cancelled", "error"}
                yield event
            if finished:
                yield ResponseProduced(AgentResponse(text))
        finally:
            self.worker_finished.set()


@pytest.mark.parametrize("stage", ["dns", "connect", "tls", "headers", "body"])
def test_custom_provider_cancel_releases_worker_and_reuses_session(monkeypatch, stage):
    from tests.fake_provider_http import HoldingProvider

    upstream = HoldingProvider(monkeypatch, stage)
    # Запрещаем старый blocking transport: regression не должна обращаться к сети.
    monkeypatch.setattr(
        "requests.Session.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("blocking provider transport")
        ),
    )

    async def run():
        scheme = "https" if stage == "tls" else "http"
        brain = LocalHttpBrain(
            ModelConfig(
                provider="custom-" + "0" * 32,
                model="m",
                base_url=f"{scheme}://provider.test/v1/chat/completions",
            ),
            native_tools=False,
        )
        agent = _DynamicBrainAgent(brain)
        client = await _create_client(agent)
        try:
            status = await client.get("/ui/api/status")
            session_id = (await status.json())["session_id"]
            await _select_local_model(client, session_id)
            headers = {"X-Slavik-Session": session_id}
            send = asyncio.create_task(
                client.post("/ui/api/chat/send", json={"content": "first"}, headers=headers)
            )
            assert await asyncio.to_thread(upstream.started.wait, 2)
            cancel = await asyncio.wait_for(client.post("/ui/api/chat/cancel", headers=headers), 2)
            assert cancel.status == 200, await cancel.text()
            assert (await cancel.json())["cancelled"] is True
            response = await asyncio.wait_for(send, 2)
            assert (await response.json())["cancelled"] is True
            assert agent.worker_finished.is_set()
            assert upstream.cleaned.is_set()
            assert upstream.active == 0
            assert upstream.calls == 1
            assert all(resolver.closed for resolver in upstream.resolvers)
            second = await asyncio.wait_for(
                client.post("/ui/api/chat/send", json={"content": "second"}, headers=headers), 2
            )
            assert second.status == 200, await second.text()
            payload = await second.json()
            assert payload.get("cancelled") is not True
            assert payload["session_id"] == session_id
            assert any(
                item["role"] == "assistant" and item["content"] == "ok"
                for item in payload["messages"]
            )
            assert upstream.calls == 2
            assert agent._brain is brain
            assert agent.worker_finished.is_set()
            assert upstream.active == 0
            assert all(resolver.closed for resolver in upstream.resolvers)
            assert all(response.closed for response in upstream.responses)
        finally:
            await client.close()

    asyncio.run(run())
