import asyncio

import pytest

from core.agent import Agent
from core.agent_tools import ChatApprovalUnavailable
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall
from shared.models import ToolResult

from .fakes import _create_client, _select_local_model


class AutoBatchBrain(Brain):
    supports_native_tools = True

    def __init__(self):
        self.network_tool = "auto_network"
        self.initial_requests = 0
        self.history = []

    def generate(self, messages, config=None, tools=None):
        self.history.append(list(messages))
        if messages[-1].role == "tool":
            return LLMResult(text="finished")
        self.initial_requests += 1
        return LLMResult(
            text="native batch",
            tool_calls=[
                ToolCall(id="first-original", name="auto_read", arguments={"index": 0}),
                ToolCall(id="network-original-1", name=self.network_tool, arguments={"index": 1}),
                ToolCall(id="network-original-2", name=self.network_tool, arguments={"index": 2}),
            ],
        )


class AutoNativeAgent(Agent):
    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True):
        # Stub только внешнего provider; runtime, Gateway и verifier настоящие.
        pass


@pytest.mark.parametrize(
    "cleanup",
    [
        None,
        "close",
        "reject",
        "security",
        "claim_race",
        "grant_race",
        "chat_cancel",
        "nested_cancel",
        "security_writer",
        "new_message",
    ],
)
def test_http_auto_native_approval_preserves_batch_and_once_scope(tmp_path, monkeypatch, cleanup):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Makefile").write_text("check:\n\t@true\n")
    brain = AutoBatchBrain()
    if cleanup in {"security", "security_writer"}:
        brain.network_tool = "web"
    agent = AutoNativeAgent(
        brain=brain,
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    executed = []
    for name, risks in [("auto_read", []), (brain.network_tool, ["network"])]:
        agent.tool_registry.register(
            name,
            lambda request: executed.append(dict(request.args))
            or ToolResult.success({"output": str(request.args)}),
            enabled=True,
            capability="read",
            risk_classes=risks,
            description="Read",
            parameters_schema={"type": "object"},
        )

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            headers = {"X-Slavik-Session": session}
            await _select_local_model(client, session)
            await client.server.app["ui_hub"].set_workspace_root(session, str(tmp_path))
            if cleanup in {"security", "security_writer"}:
                enabled = await client.post(
                    "/ui/api/session/security",
                    headers=headers,
                    json={"tools": {"state": {"web": True}}},
                )
                assert enabled.status == 200
            await client.post("/ui/api/mode", headers=headers, json={"mode": "auto"})
            response = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "read three"}
            )
            payload = await response.json()
            decision = payload["decision"]
            assert decision["context"]["source_endpoint"] == "chat.tool_continue"
            assert decision["context"]["resume_payload"]["execution_mode"] == "auto"
            assert executed == [{"index": 0}]
            original_run = agent.last_auto_state["run_id"]
            if cleanup == "new_message":
                await client.server.app["ui_hub"].set_session_workflow(session, mode="ask")
                response = await client.post(
                    "/ui/api/chat/send",
                    headers=headers,
                    json={"session_id": session, "content": "найди в интернете погоду"},
                )
                assert response.status == 200, await response.json()
                workflow = await client.server.app["ui_hub"].get_session_workflow(session)
                assert workflow["auto_state"]["status"] == "cancelled"
                assert agent.drain_auto_progress_events() == []
                assert original_run not in agent.auto_agent.orchestrator._paused_runs
                assert executed == [{"index": 0}]
                return
            if cleanup == "nested_cancel":
                from core.approval_policy import ApprovalRequired
                from core.tool_gateway import ToolGateway

                live = []
                original_resume = agent.auto_agent.resume_outcome
                original_call = ToolGateway.call

                def resumed(run_id, *, cancellation_token=None):
                    live.append(cancellation_token)
                    return original_resume(run_id, cancellation_token=cancellation_token)

                def call(gateway, request):
                    try:
                        return original_call(gateway, request)
                    except ApprovalRequired:
                        live[0].set()
                        raise

                monkeypatch.setattr(agent.auto_agent, "resume_outcome", resumed)
                monkeypatch.setattr(ToolGateway, "call", call)
                response = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session,
                        "decision_id": decision["id"],
                        "choice": "approve_once",
                    },
                )
                assert response.status == 200
                assert agent.last_auto_state["status"] == "cancelled"
                assert original_run not in agent.auto_agent.orchestrator._paused_runs
                assert agent.last_approval_request is None
                assert executed == [{"index": 0}, {"index": 1}]
                return
            if cleanup == "security_writer":
                import threading

                entered, release = threading.Event(), threading.Event()
                original_resume = agent.resume_chat_approval

                def held(identity, *, cancellation_token=None):
                    entered.set()
                    assert release.wait(timeout=3)
                    return original_resume(identity, cancellation_token=cancellation_token)

                monkeypatch.setattr(agent, "resume_chat_approval", held)
                approval = asyncio.create_task(
                    client.post(
                        "/ui/api/decision/respond",
                        headers=headers,
                        json={
                            "session_id": session,
                            "decision_id": decision["id"],
                            "choice": "approve_once",
                        },
                    )
                )
                assert await asyncio.to_thread(entered.wait, 2)
                writer = asyncio.create_task(
                    client.post(
                        "/ui/api/session/security",
                        headers=headers,
                        json={"tools": {"state": {"web": False}}},
                    )
                )
                try:
                    await asyncio.sleep(0.05)
                    assert not writer.done()
                finally:
                    release.set()
                assert (await approval).status == 200
                assert (await writer).status == 200
                assert executed == [{"index": 0}, {"index": 1}]
                return
            if cleanup == "chat_cancel":
                cancelled = await client.post(
                    "/ui/api/chat/cancel", headers=headers, json={"session_id": session}
                )
                assert cancelled.status == 200
                workflow = await client.server.app["ui_hub"].get_session_workflow(session)
                assert workflow["auto_state"]["status"] == "cancelled"
                assert agent.drain_auto_progress_events() == []
                assert original_run not in agent.auto_agent.orchestrator._paused_runs
                assert executed == [{"index": 0}]
                return
            if cleanup == "grant_race":
                store = client.server.app["session_store"]
                original_categories = store.get_categories

                async def existing_grant(scope):
                    await store.approve(scope, {"SUDO"})
                    return await original_categories(scope)

                monkeypatch.setattr(store, "get_categories", existing_grant)
                original_resume = agent.auto_agent.resume_outcome

                def cancel_before_claim(run_id, *, cancellation_token=None):
                    cancellation_token.set()
                    return original_resume(run_id, cancellation_token=cancellation_token)

                monkeypatch.setattr(agent.auto_agent, "resume_outcome", cancel_before_claim)
                rejected = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session,
                        "decision_id": decision["id"],
                        "choice": "approve_session",
                    },
                )
                assert rejected.status == 409
                assert agent.approved_categories == {"SUDO"}
                store = client.server.app["session_store"]
                for scope in store._approved:
                    assert await original_categories(scope) == {"SUDO"}
                assert original_run in agent.auto_agent.orchestrator._paused_runs
                assert executed == [{"index": 0}]
                return
            if cleanup == "security":
                disabled = await client.post(
                    "/ui/api/session/security",
                    headers=headers,
                    json={"tools": {"state": {"web": False}}},
                )
                assert disabled.status == 200
                accepted = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session,
                        "decision_id": decision["id"],
                        "choice": "approve_once",
                    },
                )
                assert accepted.status == 200
                assert agent.tools_enabled["web"] is False
                assert executed == [{"index": 0}]
                paused = agent.auto_agent.orchestrator._paused_runs[original_run]
                assert paused.continuation.executed[-1].result.ok is False
                return
            if cleanup == "claim_race":
                identity = decision["context"]["resume_payload"]["continuation_id"]
                token = asyncio.Event()
                original_resume = agent.auto_agent.resume_outcome

                def race(run_id, *, cancellation_token=None):
                    token.set()
                    return original_resume(run_id, cancellation_token=cancellation_token)

                monkeypatch.setattr(agent.auto_agent, "resume_outcome", race)
                with pytest.raises(ChatApprovalUnavailable, match="not_started"):
                    agent.resume_chat_approval(identity, cancellation_token=token)
                token.clear()
                agent.validate_chat_approval(identity)
                assert original_run in agent.auto_agent.orchestrator._paused_runs
                assert agent.last_approval_resume_payload["continuation_id"] == identity
                assert executed == [{"index": 0}]
                return
            if cleanup is not None:
                identity = decision["context"]["resume_payload"]["continuation_id"]
                if cleanup == "close":
                    agent.close()
                else:
                    rejected = await client.post(
                        "/ui/api/decision/respond",
                        headers=headers,
                        json={
                            "session_id": session,
                            "decision_id": decision["id"],
                            "choice": "reject",
                        },
                    )
                    assert rejected.status == 200
                    rejected_payload = await rejected.json()
                    assert rejected_payload["auto_state"]["status"] == "cancelled"
                    workflow = await client.server.app["ui_hub"].get_session_workflow(session)
                    assert workflow["auto_state"]["status"] == "cancelled"
                    assert agent.drain_auto_progress_events() == []
                with pytest.raises(ChatApprovalUnavailable):
                    agent.validate_chat_approval(identity)
                assert original_run not in agent.auto_agent.orchestrator._paused_runs
                assert executed == [{"index": 0}]
                return
            for expected_index in [1, 2]:
                accepted = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session,
                        "decision_id": decision["id"],
                        "choice": "approve_once",
                    },
                )
                result = await accepted.json()
                assert accepted.status == 200, result
                assert result["resume"]["ok"] is True, result
                assert executed == [{"index": i} for i in range(expected_index + 1)]
                assert agent.approved_categories == set()
                assert agent.drain_auto_progress_events() == []
                assert agent.last_auto_state["run_id"] == original_run
                if expected_index == 1:
                    following = result["decision"]
                    assert following["id"] != decision["id"]
                    assert following["status"] == "pending"
                    assert result["status"] == "pending"
                    decision = following
            assert brain.initial_requests == 1
            assert [msg.tool_call_id for msg in brain.history[-1] if msg.role == "tool"] == [
                "first-original",
                "network-original-1",
                "network-original-2",
            ]
            assert agent.last_auto_state["status"] == "completed"
            repeated = await client.post(
                "/ui/api/decision/respond",
                headers=headers,
                json={
                    "session_id": session,
                    "decision_id": decision["id"],
                    "choice": "approve_once",
                },
            )
            assert repeated.status == 409
            assert len(executed) == 3
        finally:
            await client.close()
            if cleanup != "close":
                agent.close()

    asyncio.run(run())
