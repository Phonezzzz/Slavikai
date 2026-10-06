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
                ToolCall(id="network-original-1", name="auto_network", arguments={"index": 1}),
                ToolCall(id="network-original-2", name="auto_network", arguments={"index": 2}),
            ],
        )


class AutoNativeAgent(Agent):
    def reconfigure_models(self, main_config, main_api_key=None, *, persist=True):
        # Stub только внешнего provider; runtime, Gateway и verifier настоящие.
        pass


@pytest.mark.parametrize("cleanup", [None, "close", "reject"])
def test_http_auto_native_approval_preserves_batch_and_once_scope(tmp_path, monkeypatch, cleanup):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Makefile").write_text("check:\n\t@true\n")
    brain = AutoBatchBrain()
    agent = AutoNativeAgent(
        brain=brain,
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    executed = []
    for name, risks in [("auto_read", []), ("auto_network", ["network"])]:
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
