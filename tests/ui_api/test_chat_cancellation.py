from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
import http.server
import socketserver
import threading
import time

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


class _HeaderHoldingHandler(http.server.BaseHTTPRequestHandler):
    """Test provider that accepts the request and holds the response headers."""

    request_count = 0
    release = threading.Event()

    def do_POST(self) -> None:  # noqa: N802
        type(self).request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        if type(self).request_count == 1:
            # Never send response headers: the client stays in the header wait.
            type(self).release.wait(timeout=30)
        body = b'{"choices":[{"message":{"content":"second-ok"}}]}'
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args) -> None:  # noqa: ANN001,ANN202
        pass


class _DynamicBrainAgent(DummyAgent):
    """Drives a real dynamic LocalHttpBrain, like core/tool_loop.py does."""

    def __init__(self, brain: LocalHttpBrain) -> None:
        super().__init__()
        self._brain = brain
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages, cancellation_token=None):  # noqa: ANN001
        yield from self._brain.generate_stream_events(
            messages,
            cancellation_token=cancellation_token,
        )


def test_chat_cancel_aborts_header_wait() -> None:
    async def run() -> None:
        _HeaderHoldingHandler.request_count = 0
        _HeaderHoldingHandler.release.clear()
        socketserver.TCPServer.allow_reuse_address = True
        server = socketserver.TCPServer(("127.0.0.1", 0), _HeaderHoldingHandler)
        server.daemon_threads = True
        port = server.server_address[1]
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            brain = LocalHttpBrain(
                default_config=ModelConfig(
                    provider="custom-0123456789abcdef0123456789abcdef",
                    model="opaque/model",
                    base_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                ),
                native_tools=False,
            )
            agent = _DynamicBrainAgent(brain)
            client = await _create_client(agent)
            try:
                status_response = await client.get("/ui/api/status")
                session_id = (await status_response.json())["session_id"]
                assert isinstance(session_id, str) and session_id
                await _select_local_model(client, session_id)
                send_task = asyncio.create_task(
                    client.post(
                        "/ui/api/chat/send",
                        json={"content": "hi"},
                        headers={"X-Slavik-Session": session_id},
                    )
                )
                # Wait until the provider is holding the response headers.
                for _ in range(200):
                    if _HeaderHoldingHandler.request_count >= 1:
                        break
                    await asyncio.sleep(0.05)
                assert _HeaderHoldingHandler.request_count == 1
                await asyncio.sleep(0.3)

                started = time.monotonic()
                cancel_response = await asyncio.wait_for(
                    client.post(
                        "/ui/api/chat/cancel",
                        headers={"X-Slavik-Session": session_id},
                    ),
                    timeout=9,
                )
                cancel_elapsed = time.monotonic() - started
                assert cancel_response.status == 200, await cancel_response.text()
                assert (await cancel_response.json())["cancelled"] is True
                assert cancel_elapsed < 9

                send_response = await asyncio.wait_for(send_task, timeout=5)
                assert send_response.status == 200
                assert (await send_response.json())["cancelled"] is True
                # No retry after cancellation.
                assert _HeaderHoldingHandler.request_count == 1

                # The session is freed: a follow-up generation runs normally.
                _HeaderHoldingHandler.release.set()
                send2 = await client.post(
                    "/ui/api/chat/send",
                    json={"content": "again"},
                    headers={"X-Slavik-Session": session_id},
                )
                assert send2.status == 200
                assert (await send2.json()).get("cancelled") is not True
            finally:
                await client.close()
        finally:
            server.shutdown()
            server.server_close()

    asyncio.run(run())
