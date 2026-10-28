from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from core.agent import Agent
from core.desktop_policy import DesktopPolicyStore
from core.task_run_storage import TaskRunStore
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig
from server.http.common.idempotency import IdempotencyStore
from shared.models import ToolResult
from shared.task_run import ProducerKind, RunScope, RunState

from .fakes import TEST_API_TOKEN, _create_client, _select_local_model


class ReportBrain(Brain):
    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, config=None, tools=None):
        self.calls += 1
        return LLMResult(text="report-r2a\n🦊", usage=LLMUsage(0, 0, 0))


@pytest.fixture
def isolated_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = sqlite3.connect

    def connect(database, *args, **kwargs):
        assert Path(database).resolve().is_relative_to(tmp_path)
        return original(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    brain = ReportBrain()
    agent = Agent(
        brain=brain,
        memory_db_path=str(tmp_path / "memory.db"),
        vectors_db_path=str(tmp_path / "vectors.db"),
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
        desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop.json"),
    )
    return agent, brain


@pytest.mark.parametrize("endpoint", ["ui", "api"])
@pytest.mark.parametrize("mode", ["ask", "auto"])
@pytest.mark.parametrize("fault", [None, "projection", "storage"])
def test_http_submission_commits_before_projection_without_replay(
    tmp_path, monkeypatch, isolated_agent, endpoint, mode, fault
):
    from server.http.handlers import chat, ui_chat

    agent, brain = isolated_agent
    monkeypatch.setattr(agent, "respond_stream", None)
    captured = []
    module = ui_chat if endpoint == "ui" else chat
    original_project = module._project_agent_response

    def project(response):
        captured.append(response)
        if fault == "projection":
            raise RuntimeError("after submission, before projection")
        return original_project(response)

    monkeypatch.setattr(module, "_project_agent_response", project)

    async def run():
        client = await _create_client(agent)
        try:
            await client.app["runtime_model_state"].set_global_main(
                ModelConfig(provider="ollama", model="test")
            )
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            scope = RunScope(principal, session)
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_workflow(session, mode=mode)
            store = client.app["task_run_store"]
            if fault == "storage":
                with sqlite3.connect(store.path) as conn:
                    conn.execute(
                        "CREATE TRIGGER fault BEFORE INSERT ON run_submissions "
                        "BEGIN SELECT RAISE(ABORT, 'submission storage fault'); END"
                    )
            headers = {
                "X-Slavik-Session": session,
                "Idempotency-Key": "durable-report",
            }
            if endpoint == "ui":
                url, payload = "/ui/api/chat/send", {"content": "hi"}
            else:
                headers["Authorization"] = f"Bearer {TEST_API_TOKEN}"
                url = "/v1/chat/completions"
                payload = {
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "hi"}],
                    "slavik_meta": {"runtime_mode": mode},
                }
            first = await client.post(url, headers=headers, json=payload)
            assert first.status == (500 if fault else 200), await first.text()
            calls = brain.calls
            assert calls > 0
            with sqlite3.connect(store.path) as conn:
                rows = conn.execute("SELECT task_run_id, state FROM task_runs").fetchall()
            assert len(rows) == 1
            run_id, state = rows[0]
            reopened = TaskRunStore(store.path)
            if fault == "storage":
                assert state == RunState.RUNNING and not captured
                with pytest.raises(LookupError):
                    reopened.get_submission(scope, run_id)
                assert len(reopened.events(scope, run_id)) == 2
            else:
                assert state == RunState.RESULT_SUBMITTED
                saved = reopened.get_submission(scope, run_id)
                assert saved.report.text == captured[-1].text
                assert saved.report.kind == (
                    ProducerKind.AUTO if mode == "auto" else ProducerKind.NATIVE_LOOP
                )
                assert saved.report.failure_code is None
                assert reopened.get(scope, run_id).terminal_at is None
            client.app["task_run_store"] = reopened
            client.app["idempotency_store"] = IdempotencyStore()
            repeated = await client.post(url, headers=headers, json=payload)
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == "task_run_continuation_unavailable"
            assert brain.calls == calls
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_approved_result_is_durable_before_resume_projection_fault(
    tmp_path, monkeypatch, isolated_agent, mode
):
    from core.agent_response import AgentResponse
    from llm.types import ToolCall
    from server.http.handlers import ui_chat

    agent, brain = isolated_agent
    dispatched = []

    def generate(messages, config=None, tools=None):
        brain.calls += 1
        if messages[-1].role == "tool":
            return LLMResult(text="observed", usage=LLMUsage(0, 0, 0))
        return LLMResult(
            text="",
            usage=LLMUsage(0, 0, 0),
            tool_calls=[ToolCall(id="approved-original", name="lookup", arguments={})],
        )

    monkeypatch.setattr(brain, "generate", generate)
    agent.tools_enabled["safe_mode"] = True
    agent.tool_registry.register(
        "lookup",
        lambda request: dispatched.append(request) or ToolResult.success({"value": "seen"}),
        capability="read",
        chat_exposed=True,
        risk_classes=["network"],
        parameters_schema={"type": "object"},
        description="Read lookup",
    )
    monkeypatch.setattr(agent, "respond_stream", None)

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_workflow(session, mode=mode)
            headers = {"X-Slavik-Session": session}
            initial = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "lookup"}
            )
            assert initial.status == 200
            decision = (await initial.json())["decision"]
            store = client.app["task_run_store"]
            with sqlite3.connect(store.path) as conn:
                run_id = conn.execute("SELECT task_run_id FROM task_runs").fetchone()[0]
            scope = RunScope(principal, session)
            accepted = store.get(scope, run_id)
            assert accepted.state == RunState.RUNNING and not dispatched
            with pytest.raises(LookupError):
                store.get_submission(scope, run_id)
            captured = []

            def fail(response: AgentResponse):
                captured.append(response)
                raise RuntimeError("after approval submission, before projection")

            monkeypatch.setattr(ui_chat, "_project_agent_response", fail)
            resumed = await client.post(
                "/ui/api/decision/respond",
                headers=headers,
                json={
                    "session_id": session,
                    "decision_id": decision["id"],
                    "choice": "approve_once",
                },
            )
            assert resumed.status == 500
            saved = TaskRunStore(store.path).get_submission(scope, run_id)
            assert saved.report.text == captured[-1].text
            assert saved.task_id == accepted.task_id and saved.attempt_id == accepted.attempt_id
            assert len(dispatched) == 1
            assert store.get(scope, run_id).state == RunState.RESULT_SUBMITTED
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_stream_error_preserves_received_producer_report(isolated_agent, monkeypatch):
    from core.agent_response import ResponseProduced

    agent, _brain = isolated_agent
    original = agent.respond_stream

    def stream(messages, cancellation_token=None):
        for event in original(messages, cancellation_token=cancellation_token):
            yield event
            if isinstance(event, ResponseProduced):
                raise RuntimeError("post-runtime stream fault")

    monkeypatch.setattr(agent, "respond_stream", stream)

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 200
            store = client.app["task_run_store"]
            with sqlite3.connect(store.path) as conn:
                run_id = conn.execute("SELECT task_run_id FROM task_runs").fetchone()[0]
            saved = TaskRunStore(store.path).get_submission(RunScope(principal, session), run_id)
            assert saved.report.failure_code is not None
            assert saved.report.failure_message
            assert store.get(RunScope(principal, session), run_id).terminal_at is None
        finally:
            await client.close()

    asyncio.run(run())
