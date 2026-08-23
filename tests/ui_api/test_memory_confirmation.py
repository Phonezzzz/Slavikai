from __future__ import annotations

import asyncio

from core.decision.memory_save import build_memory_save_packet
from shared.models import JSONValue
from tests.ui_api.fakes import DummyAgent, _create_client, _select_local_model


class _MemoryConfirmationAgent(DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.applied: list[dict[str, JSONValue]] = []

    @staticmethod
    def _preview(text: str) -> dict[str, JSONValue]:
        normalized = text.strip()
        return {
            "text": normalized,
            "claims": [
                {
                    "claim_type": "fact",
                    "stable_key": "fact:confirmed",
                    "value_json": {"text": normalized},
                    "confidence": 0.9,
                    "summary_text": normalized,
                }
            ],
        }

    def respond(self, messages) -> str:
        text = messages[-1].content if messages else ""
        return build_memory_save_packet(self._preview(text)).to_json()

    def build_memory_save_preview(
        self,
        text: str,
        *,
        source_kind: str,
        source_id: str | None = None,
        lang_hint: str | None = None,
    ) -> dict[str, JSONValue]:
        del source_kind, source_id, lang_hint
        return self._preview(text)

    def apply_memory_save_preview(
        self,
        proposed_action: dict[str, JSONValue],
        *,
        source_kind: str,
        source_id: str | None = None,
    ) -> list[dict[str, JSONValue]]:
        stored = dict(proposed_action)
        stored["source_kind"] = source_kind
        stored["source_id"] = source_id
        self.applied.append(stored)
        return [
            {
                "stable_key": "fact:confirmed",
                "status": "active",
                "claim_type": "fact",
                "confidence": 0.9,
            }
        ]


def test_memory_save_requires_confirm_and_supports_edit_or_reject() -> None:
    async def _run() -> None:
        agent = _MemoryConfirmationAgent()
        client = await _create_client(agent)
        try:
            status = await client.get("/ui/api/status")
            session_id = (await status.json()).get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            async def _request_preview(text: str) -> dict[str, JSONValue]:
                response = await client.post(
                    "/ui/api/chat/send",
                    headers={"X-Slavik-Session": session_id},
                    json={"content": text},
                )
                assert response.status == 200
                decision = (await response.json()).get("decision")
                assert isinstance(decision, dict)
                return decision

            first = await _request_preview("remember first")
            assert first.get("decision_type") == "memory_save"
            assert first.get("status") == "pending"
            assert agent.applied == []
            first_options = first.get("options")
            assert isinstance(first_options, list)
            assert {item.get("action") for item in first_options if isinstance(item, dict)} == {
                "confirm",
                "edit_and_confirm",
                "reject",
            }

            confirmed = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": first["id"],
                    "choice": "confirm",
                },
            )
            assert confirmed.status == 200
            assert len(agent.applied) == 1
            assert agent.applied[0]["source_id"] == session_id

            duplicate_confirm = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": first["id"],
                    "choice": "confirm",
                },
            )
            assert duplicate_confirm.status == 409
            assert len(agent.applied) == 1

            rejected_preview = await _request_preview("remember reject me")
            rejected = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": rejected_preview["id"],
                    "choice": "reject",
                },
            )
            assert rejected.status == 200
            assert len(agent.applied) == 1

            edited_preview = await _request_preview("remember old text")
            edited = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": edited_preview["id"],
                    "choice": "edit_and_confirm",
                    "edited_action": {
                        "text": "edited text",
                        "claims": [{"stable_key": "fact:client_spoof"}],
                    },
                },
            )
            assert edited.status == 200
            assert len(agent.applied) == 2
            assert agent.applied[1]["text"] == "edited text"
            claims = agent.applied[1]["claims"]
            assert isinstance(claims, list)
            assert claims[0]["stable_key"] == "fact:confirmed"
        finally:
            await client.close()

    asyncio.run(_run())
