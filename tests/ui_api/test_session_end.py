from __future__ import annotations

import asyncio
import threading

import pytest

from core.agent import Agent
from core.decision.memory_save import build_memory_save_packet
from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.types import LLMResult
from shared.models import LLMMessage
from tests.ui_api.fakes import _create_client


class SummaryBrain(Brain):
    def __init__(self):
        self.calls = []
        self.text = "Обсудили контракт Memory."
        self.failure = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def generate(self, messages, config=None):
        self.calls.append(messages)
        self.started.set()
        if self.block:
            assert self.release.wait(5)
        if self.failure:
            raise RuntimeError("summary unavailable")
        return LLMResult(text=self.text)


def _agent(tmp_path, monkeypatch):
    brain = SummaryBrain()
    agent = Agent(
        brain=brain,
        memory_db_path=str(tmp_path / "memory.db"),
        vectors_db_path=str(tmp_path / "vectors.db"),
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    monkeypatch.setattr(agent, "reconfigure_models", lambda *a, **kw: None)
    vector_syncs = []
    agent.confirmed_tool_calls = []
    original_call = ToolGateway.call

    def trace_call(gateway, request):
        agent.confirmed_tool_calls.append((request.name, gateway.confirmed_decision))
        return original_call(gateway, request)

    monkeypatch.setattr(ToolGateway, "call", trace_call)
    monkeypatch.setattr(agent._atom_embedding_index, "sync_atom", vector_syncs.append)
    agent.short_term.append(LLMMessage(role="user", content="контекст остаётся до закрытия"))
    return agent, brain, vector_syncs


async def _setup(agent):
    client = await _create_client(agent)
    session_id = (await (await client.get("/ui/api/status")).json())["session_id"]
    hub = client.server.app["ui_hub"]
    message = hub.create_message(role="user", content="Обсудим Memory", lane="chat")
    await hub.append_message(session_id, message, lane="chat")
    return client, session_id, hub


async def _begin(client, sid):
    response = await client.post(
        "/ui/api/chat/send",
        headers={"X-Slavik-Session": sid},
        json={"content": "/end-session"},
    )
    assert response.status == 200, await response.text()
    return (await response.json())["decision"]


async def _choose(client, sid, decision, choice, **extra):
    return await client.post(
        "/ui/api/decision/respond",
        headers={"X-Slavik-Session": sid},
        json={"session_id": sid, "decision_id": decision["id"], "choice": choice, **extra},
    )


@pytest.mark.parametrize("choice,closed", [("reject", True), ("abort", False)])
def test_end_without_memory_needs_no_model_and_retains_history(
    tmp_path, monkeypatch, choice, closed
):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            decision = await _begin(client, sid)
            assert decision["decision_type"] == "end_session"
            assert brain.calls == []
            result = await _choose(client, sid, decision, choice)
            assert result.status == 200, await result.text()
            assert (await result.json())["resume"]["data"] == {"closed": closed, "saved": False}
            assert bool(agent.short_term) is not closed
            assert len(await hub.get_messages(sid, lane="chat")) == (3 if closed else 2)
            assert agent._canonical_store.list_atoms() == []
            assert brain.calls == syncs == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("choice", ["confirm", "edit_and_confirm", "reject"])
def test_summary_requires_second_confirm_and_exact_edit(tmp_path, monkeypatch, choice):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            first = await _begin(client, sid)
            response = await _choose(client, sid, first, "confirm")
            assert response.status == 200, await response.text()
            second = (await response.json())["decision"]
            assert second["decision_type"] == "memory_save"
            assert second["id"] != first["id"]
            assert second["proposed_action"]["text"] == brain.text
            assert "/end-session" not in brain.calls[0][-1].content
            assert agent._canonical_store.list_atoms() == syncs == []
            assert agent.short_term
            stale = await _choose(client, sid, first, "confirm")
            assert stale.status == 409
            assert (await _begin(client, sid))["id"] == second["id"]
            normal = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "другое сообщение"},
            )
            assert normal.status == 409
            edited = "remember: точные отредактированные данные"
            result = await _choose(client, sid, second, choice, edited_action={"text": edited})
            assert result.status == 200, await result.text()
            atoms = agent._canonical_store.list_atoms()
            if choice == "reject":
                assert atoms == syncs == []
                assert agent.short_term
            else:
                assert len(atoms) == len(syncs) == 1
                assert agent.confirmed_tool_calls == [("memory_save_confirmed", True)]
                assert atoms[0].value_json == {
                    "text": edited if choice == "edit_and_confirm" else brain.text
                }
                assert agent.short_term == []
                duplicate = await _choose(client, sid, second, "confirm")
                assert duplicate.status == 409
                assert len(syncs) == 1
            assert len(await hub.get_messages(sid, lane="chat")) == (2 if choice == "reject" else 3)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["empty", "exception", "write"])
def test_failed_summary_or_write_keeps_decision_retryable(tmp_path, monkeypatch, failure):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            first = await _begin(client, sid)
            if failure == "empty":
                brain.text = ""
            if failure == "exception":
                brain.failure = True
            result = await _choose(client, sid, first, "confirm")
            if failure == "write":
                second = (await result.json())["decision"]

                def fail(*a, **kw):
                    raise RuntimeError("store unavailable")

                monkeypatch.setattr(agent, "apply_confirmed_memory_save", fail)
                result = await _choose(client, sid, second, "confirm")
                pending_id = second["id"]
            else:
                pending_id = first["id"]
            assert result.status in {409, 503}, await result.text()
            pending = await hub.get_session_decision(sid)
            assert pending["id"] == pending_id and pending["status"] == "pending"
            assert agent.short_term and agent._canonical_store.list_atoms() == syncs == []
        finally:
            await client.close()

    asyncio.run(run())


def test_existing_decision_and_active_generation_are_not_overwritten(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            pending = build_memory_save_packet({"text": "keep", "claims": []}).to_dict()
            await hub.set_session_decision(sid, pending)
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "/end-session"},
            )
            assert response.status == 409
            assert (await hub.get_session_decision(sid))["id"] == pending["id"]
            await hub.set_session_decision(sid, None)
            registry = client.server.app["chat_cancellation_registry"]
            active = await registry.start(session_id=sid, stream_id="active")
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "/end-session"},
            )
            assert response.status == 409
            assert not active.token.is_set()
            assert await hub.get_session_decision(sid) is None
            await registry.finish(active)
            assert brain.calls == syncs == []
        finally:
            await client.close()

    asyncio.run(run())


def test_cancel_summary_drains_and_never_publishes_memory_preview(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        brain.block = True
        try:
            decision = await _begin(client, sid)
            pending = asyncio.create_task(_choose(client, sid, decision, "confirm"))
            assert await asyncio.to_thread(brain.started.wait, 3)
            registry = client.server.app["chat_cancellation_registry"]
            cancelled = await registry.request_cancel(sid)
            assert cancelled is not None
            assert not cancelled.generation.finished.is_set()
            second = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "/end-session"},
            )
            assert second.status == 409
            brain.release.set()
            response = await pending
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "session_end_cancelled"
            assert cancelled.generation.finished.is_set()
            current = await hub.get_session_decision(sid)
            assert current["id"] == decision["id"] and current["status"] == "pending"
            assert agent._canonical_store.list_atoms() == syncs == []
            assert agent.short_term
        finally:
            brain.release.set()
            await client.close()

    asyncio.run(run())


def test_end_session_respects_session_ownership_and_decision_binding(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            decision = await _begin(client, sid)
            foreign = await hub.create_session("foreign-principal")
            forbidden = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": foreign},
                json={"content": "/end-session"},
            )
            assert forbidden.status == 403
            forbidden = await _choose(client, foreign, decision, "confirm")
            assert forbidden.status == 403
            assert (await hub.get_session_decision(sid))["id"] == decision["id"]
            assert brain.calls == syncs == []
            assert agent.short_term
        finally:
            await client.close()

    asyncio.run(run())


def test_no_does_not_resolve_model_or_clear_approval_grants(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        agent.approved_categories = {"GIT_PUBLISH"}
        agent.runtime_workspace_root = "/unchanged-workspace"

        async def forbidden(*args):
            raise AssertionError("No attempted model resolution")

        monkeypatch.setattr(client.server.app["runtime_model_resolver"], "resolve_main", forbidden)
        try:
            decision = await _begin(client, sid)
            response = await _choose(client, sid, decision, "reject")
            assert response.status == 200
            assert agent.short_term == []
            assert agent.approved_categories == {"GIT_PUBLISH"}
            assert agent.runtime_workspace_root == "/unchanged-workspace"
            assert agent._canonical_store.list_atoms() == syncs == []
        finally:
            await client.close()

    asyncio.run(run())


def test_competing_decision_wins_atomic_start_without_overwrite(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        original = hub.transition_session_decision
        pending = build_memory_save_packet({"text": "competing", "claims": []}).to_dict()

        async def compete(session_id, **kwargs):
            await hub.set_session_decision(session_id, pending)
            return await original(session_id, **kwargs)

        monkeypatch.setattr(hub, "transition_session_decision", compete)
        try:
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "/end-session"},
            )
            assert response.status == 409
            assert (await hub.get_session_decision(sid))["id"] == pending["id"]
            assert len(await hub.get_messages(sid, lane="chat")) == 1
            assert brain.calls == syncs == agent.confirmed_tool_calls == []
        finally:
            await client.close()

    asyncio.run(run())


def test_pending_execution_owner_blocks_end_session_without_cancelling(tmp_path, monkeypatch):
    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        checkpoint = object()
        agent._mwv_checkpoints["pending"] = checkpoint
        try:
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": sid},
                json={"content": "/end-session"},
            )
            assert response.status == 409
            assert agent._mwv_checkpoints["pending"] is checkpoint
            assert agent.short_term
            assert await hub.get_session_decision(sid) is None
            assert brain.calls == syncs == []
        finally:
            agent._mwv_checkpoints.clear()
            await client.close()

    asyncio.run(run())


def test_replaced_executing_decision_cannot_write_old_preview(tmp_path, monkeypatch):
    import server.http.common.session_end as module

    async def run():
        agent, brain, syncs = _agent(tmp_path, monkeypatch)
        client, sid, hub = await _setup(agent)
        try:
            first = await _begin(client, sid)
            second_response = await _choose(client, sid, first, "confirm")
            second = (await second_response.json())["decision"]
            replacement = build_memory_save_packet({"text": "new request", "claims": []}).to_dict()

            async def replace(*args):
                await hub.set_session_decision(sid, replacement)
                return agent

            monkeypatch.setattr(module, "_resolve_agent", replace)
            response = await _choose(client, sid, second, "confirm")
            assert response.status == 409
            assert (await hub.get_session_decision(sid))["id"] == replacement["id"]
            assert agent._canonical_store.list_atoms() == syncs == agent.confirmed_tool_calls == []
            assert agent.short_term
        finally:
            await client.close()

    asyncio.run(run())
