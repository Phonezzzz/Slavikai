import asyncio

import pytest

from core.agent import Agent
from core.agent_tools import PendingChatApproval
from core.approval_policy import ApprovalPrompt, ApprovalRequest
from core.tool_loop import ToolLoopContinuation
from llm.types import ModelConfig
from server.agent_provider import AgentScope, ScopedAgentProvider
from shared.identity import principal_id_for_email
from tests.ui_api.test_memory_management import _isolate, _start


def _paused_agent(session):
    config = ModelConfig(
        provider="local",
        model="captured",
        temperature=0.23,
        max_tokens=2345,
        web_search_enabled=True,
        api_key="SECRET_KEY",
        base_url="SECRET_URL",
        system_prompt="SECRET_PROMPT",
        extra_headers={"SECRET_HEADER": "SECRET_VALUE"},
    )
    approval = ApprovalRequest(
        "network", ["network"], ApprovalPrompt("", "", "", []), "web", {}, session
    )
    continuation = ToolLoopContinuation([], [], [], approval, "SECRET_TEXT", 1, 8, [], config)
    agent = object.__new__(Agent)
    agent.session_id = session
    agent.runtime_mode = "ask"
    agent._pending_chat_approval = PendingChatApproval(
        "checkpoint", continuation, object(), "SECRET_GOAL", session, "ask"
    )
    agent.close = lambda: None
    return agent


def _decision(session):
    return {
        "id": "decision",
        "decision_type": "tool_approval",
        "status": "pending",
        "blocking": True,
        "context": {
            "session_id": session,
            "source_endpoint": "chat.tool_continue",
            "resume_payload": {"continuation_id": "checkpoint", "execution_mode": "ask"},
        },
    }


@pytest.mark.parametrize("who", ["owner", "member"])
def test_owned_paused_model_request_is_read_only_and_never_initializes(monkeypatch, tmp_path, who):
    _isolate(monkeypatch, tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Paused diagnostics must not resolve/create/initialize runtime")

    async def run():
        client = await _start()
        try:
            hub = client.app["ui_hub"]
            principal = principal_id_for_email(who + "@example.com")
            session = await hub.create_session(principal)
            agent = _paused_agent(session)
            agent._get_main_brain = forbidden
            provider = ScopedAgentProvider(factory=lambda scope, config: agent)
            await provider.get_for_current_task(AgentScope(principal, session), None)
            provider._factory = forbidden
            client.app["agent_provider"] = provider
            monkeypatch.setattr(client.app["runtime_model_resolver"], "resolve_main", forbidden)
            await hub.set_session_decision(session, _decision(session))
            before = hub._storage.load_sessions()
            headers = {"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": session}
            response = await client.get("/ui/api/session/paused-model-request", headers=headers)
            assert response.status == 200
            assert response.headers["Cache-Control"] == "no-store"
            assert "SECRET" not in await response.text()
            data = await response.json()
            assert data["session_id"] == session
            assert data["decision_id"] == "decision"
            assert data["basis"] == "paused_approval_model_request_at_refresh"
            assert data["state"] == "available"
            assert data["identity"] == {"provider": "local", "model": "captured"}
            assert data["values"]["temperature"] == 0.23
            assert data["values"]["max_tokens"] == 2345
            assert data["values"]["web_search_enabled"] is True
            assert set(data) == {
                "session_id",
                "decision_id",
                "basis",
                "state",
                "identity",
                "values",
                "unavailable_fields",
            }
            foreign = {
                **headers,
                "Cf-Access-Jwt-Assertion": "member" if who == "owner" else "owner",
            }
            assert (
                await client.get("/ui/api/session/paused-model-request", headers=foreign)
            ).status == 403
            assert (
                await client.post("/ui/api/session/paused-model-request", headers=headers, json={})
            ).status == 405
            assert hub._storage.load_sessions() == before
            # Удалённый checkpoint не заменяется текущей main_config или Settings.
            agent._pending_chat_approval = None
            assert (
                await (
                    await client.get("/ui/api/session/paused-model-request", headers=headers)
                ).json()
            )["state"] == "unavailable"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "case",
    [
        "no_agent",
        "no_decision",
        "settled",
        "wrong_source",
        "wrong_identity",
        "wrong_session",
        "wrong_mode",
        "no_config",
        "wrong_execution_mode",
        "wrong_context_session",
    ],
)
def test_paused_request_unavailable_never_falls_back(monkeypatch, tmp_path, case):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _start()
        try:
            hub = client.app["ui_hub"]
            principal = principal_id_for_email("owner@example.com")
            session = await hub.create_session(principal)
            agent = _paused_agent(session)
            provider = ScopedAgentProvider(factory=lambda scope, config: agent)
            if case != "no_agent":
                await provider.get_for_current_task(AgentScope(principal, session), None)

            def forbidden(*args, **kwargs):
                raise AssertionError("No runtime initialization or fallback")

            provider._factory = forbidden
            client.app["agent_provider"] = provider
            agent.main_config = ModelConfig(provider="local", model="WRONG_FALLBACK")
            agent._get_main_brain = forbidden
            decision = _decision(session)
            if case == "settled":
                decision["status"] = "rejected"
            if case == "wrong_source":
                decision["context"]["source_endpoint"] = "chat.send"
            if case == "wrong_identity":
                decision["context"]["resume_payload"]["continuation_id"] = "stale"
            if case == "wrong_session":
                agent.session_id = "another"
            if case == "wrong_mode":
                agent.runtime_mode = "auto"
            if case == "wrong_execution_mode":
                decision["context"]["resume_payload"]["execution_mode"] = "auto"
            if case == "wrong_context_session":
                decision["context"]["session_id"] = "another"
            if case == "no_config":
                pending = agent._pending_chat_approval
                from dataclasses import replace

                agent._pending_chat_approval = replace(
                    pending, continuation=replace(pending.continuation, config=None)
                )
            if case != "no_decision":
                await hub.set_session_decision(session, decision)
            headers = {"Cf-Access-Jwt-Assertion": "owner", "X-Slavik-Session": session}
            before = hub._storage.load_sessions()
            response = await client.get("/ui/api/session/paused-model-request", headers=headers)
            assert response.status == 200
            assert (await response.json())["state"] == "unavailable"
            assert "WRONG_FALLBACK" not in await response.text()
            assert hub._storage.load_sessions() == before
            assert (
                await client.get(
                    "/ui/api/session/paused-model-request",
                    headers={"Cf-Access-Jwt-Assertion": "owner"},
                )
            ).status == 400
            assert hub._storage.load_sessions() == before
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_real_approval_capture_is_read_without_brain_or_dispatch(monkeypatch, tmp_path, streaming):
    from dataclasses import replace

    from shared.models import LLMMessage, ToolResult
    from tests.test_agent_response import ToolLoopBrain

    brain = ToolLoopBrain()
    config = ModelConfig(provider="local", model="captured", temperature=0.31)
    agent = Agent(
        brain=brain,
        main_config=config,
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    try:
        agent.set_session_context("session", set())
        agent.runtime_mode = "ask"
        monkeypatch.setattr(agent.memory, "get_recent", lambda *args, **kwargs: [])
        monkeypatch.setattr(agent.memory, "get_user_prefs", lambda: [])
        monkeypatch.setattr(agent.vectors, "search", lambda *args, **kwargs: [])
        executed = []
        agent.tool_registry.register(
            "chat_lookup",
            lambda request: executed.append(request) or ToolResult.success({"output": "unused"}),
            enabled=True,
            capability="read",
            risk_classes=["network"],
            description="Lookup",
            parameters_schema={"type": "object"},
            chat_exposed=True,
        )
        messages = [LLMMessage(role="user", content="lookup")]
        if streaming:
            list(agent.respond_stream(messages))
        else:
            agent.respond(messages)
        assert agent.last_approval_request is not None
        identity = agent.last_approval_resume_payload["continuation_id"]

        def forbidden(*args, **kwargs):
            raise AssertionError("Read cannot initialize brain")

        monkeypatch.setattr(agent, "_get_main_brain", forbidden)
        agent.main_config = replace(config, temperature=0.99)
        assert agent.paused_chat_model_config(identity, "session", "ask") == config
        assert brain.calls == 1
        assert executed == []
        assert agent.paused_chat_model_config("stale", "session", "ask") is None
        agent.cancel_chat_approval(identity)
        assert agent.paused_chat_model_config(identity, "session", "ask") is None
    finally:
        agent.close()
