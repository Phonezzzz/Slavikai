from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import threading
import time

from llm.cancellation import bind_cancellation_resource
from llm.stream_model import Done, TextDelta
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
