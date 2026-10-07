from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio

from core.agent_response import AgentResponse, ResponseProduced
from llm.stream_model import Done, Error, TextDelta
from server.http.common.idempotency import IdempotencyStore

from .fakes import *


class _SequencedResponseAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.responses = iter(("first response", "replacement response"))
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages, cancellation_token=None):  # noqa: ANN001
        del messages, cancellation_token
        response = next(self.responses)
        self.last_stream_response_raw = response
        yield TextDelta(text=response)
        yield ResponseProduced(AgentResponse(self.last_stream_response_raw))
        yield Done()


class _VisibleProviderErrorAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.last_stream_response_raw: str | None = None

    def respond_stream(self, messages, cancellation_token=None):  # noqa: ANN001
        del messages, cancellation_token
        yield Error(
            message="Сетевая ошибка при обращении к провайдеру test. Повторите запрос.",
            code="provider_network_error",
        )
        yield Done(finish_reason="error")


def test_regenerate_replaces_last_response() -> None:
    async def run() -> None:
        client = await _create_client(_SequencedResponseAgent())
        try:
            status_response = await client.get("/ui/api/status")
            session_id = (await status_response.json())["session_id"]
            await _select_local_model(client, session_id)

            first_response = await client.post(
                "/ui/api/chat/send",
                json={"content": "same prompt"},
                headers={"X-Slavik-Session": session_id},
            )
            assert first_response.status == 200
            first_payload = await first_response.json()
            assert first_payload["messages"][-1]["content"] == "first response"

            delete_response = await client.delete(
                f"/ui/api/sessions/{session_id}/messages/last",
                headers={"X-Slavik-Session": session_id},
            )
            assert delete_response.status == 200
            assert (await delete_response.json())["messages"] == []

            replacement_response = await client.post(
                "/ui/api/chat/send",
                json={"content": "same prompt"},
                headers={"X-Slavik-Session": session_id},
            )
            assert replacement_response.status == 200
            messages = (await replacement_response.json())["messages"]
            assert [message["role"] for message in messages] == ["user", "assistant"]
            assert [message["content"] for message in messages] == [
                "same prompt",
                "replacement response",
            ]
        finally:
            await client.close()

    asyncio.run(run())


def test_provider_error_is_visible() -> None:
    async def run() -> None:
        client = await _create_client(_VisibleProviderErrorAgent())
        try:
            status_response = await client.get("/ui/api/status")
            session_id = (await status_response.json())["session_id"]
            await _select_local_model(client, session_id)

            response = await client.post(
                "/ui/api/chat/send",
                json={"content": "trigger provider error"},
                headers={"X-Slavik-Session": session_id},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload["generation_error"] == {
                "code": "provider_network_error",
                "kind": "network",
                "message": ("Сетевая ошибка при обращении к провайдеру test. Повторите запрос."),
            }
            assert "Сетевая ошибка" in payload["messages"][-1]["content"]
        finally:
            await client.close()

    asyncio.run(run())


def test_targeted_regeneration_retry_never_deletes_another_pair() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {"X-Slavik-Session": session}
            for content in ("earlier", "target"):
                assert (
                    await client.post(
                        "/ui/api/chat/send", headers=headers, json={"content": content}
                    )
                ).status == 200
            before = await client.app["ui_hub"].get_messages(session, lane="chat")
            target = before[-1]["message_id"]
            deletion_headers = {**headers, "If-Match": target}
            assert (
                await client.delete(
                    f"/ui/api/sessions/{session}/messages/last", headers=deletion_headers
                )
            ).status == 200
            # Lost acknowledgement / reconstructed client: same immutable target, no RAM replay.
            client.app["idempotency_store"] = IdempotencyStore()
            assert (
                await client.delete(
                    f"/ui/api/sessions/{session}/messages/last", headers=deletion_headers
                )
            ).status == 200
            remaining = await client.app["ui_hub"].get_messages(session, lane="chat")
            assert remaining == before[:2]
            assert (
                await client.post("/ui/api/chat/send", headers=headers, json={"content": "later"})
            ).status == 200
            assert (
                await client.delete(
                    f"/ui/api/sessions/{session}/messages/last",
                    headers={**headers, "If-Match": before[1]["message_id"]},
                )
            ).status == 409
            assert len(await client.app["ui_hub"].get_messages(session, lane="chat")) == 4
        finally:
            await client.close()

    asyncio.run(run())
