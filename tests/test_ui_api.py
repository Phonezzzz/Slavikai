from __future__ import annotations

import asyncio

from aiohttp.test_utils import TestClient, TestServer

from server.http_api import create_app


class DummyAgent:
    def __init__(self) -> None:
        self._session_id: str | None = None
        self._approved_categories: set[str] = set()

    def set_session_context(self, session_id: str | None, approved_categories: set[str]) -> None:
        self._session_id = session_id
        self._approved_categories = set(approved_categories)

    def respond(self, messages) -> str:
        return "ok"


async def _create_client(agent: DummyAgent) -> TestClient:
    app = create_app(agent=agent, max_request_bytes=1_000_000)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


def test_ui_status_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            resp = await client.get("/ui/api/status")
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("ok") is True
            session_id = payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            assert resp.headers.get("X-Slavik-Session") == session_id
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            resp = await client.post("/ui/api/chat/send", json={"content": "Ping"})
            assert resp.status == 200
            payload = await resp.json()
            session_id = payload.get("session_id")
            messages = payload.get("messages", [])
            assert isinstance(session_id, str)
            assert session_id
            assert isinstance(messages, list)
            assert messages
            assert resp.headers.get("X-Slavik-Session") == session_id
        finally:
            await client.close()

    asyncio.run(run())
