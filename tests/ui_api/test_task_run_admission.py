from __future__ import annotations

import asyncio
import sqlite3

import pytest

from core.task_run_storage import TaskRunStore
from server.http.common.idempotency import IdempotencyStore

from .fakes import _create_client, _select_local_model
from .test_stream_and_events import _RealAgentStreamingBrain, _RealStreamingAgent


class CountingBrain(_RealAgentStreamingBrain):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001, ANN201
        self.calls += 1
        return super().generate(messages, config, tools)


@pytest.mark.parametrize("changed", [False, True])
def test_durable_admission_fences_dispatch_after_volatile_replay_is_lost(
    tmp_path,
    monkeypatch,
    changed,
) -> None:  # noqa: ANN001
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.runtime_mode = "ask"

    async def run() -> None:
        client = await _create_client(agent)  # type: ignore[arg-type]
        path = tmp_path / "runs.db"
        client.app["task_run_store"] = TaskRunStore(path)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "accepted-request"}
            first = await client.post("/ui/api/chat/send", headers=headers, json={"content": "hi"})
            assert first.status == 200
            calls = brain.calls
            assert calls > 0
            # Reopen the real SQLite store; discard only the volatile HTTP replay cache.
            client.app["task_run_store"] = TaskRunStore(path)
            client.app["idempotency_store"] = IdempotencyStore()
            repeated = await client.post(
                "/ui/api/chat/send",
                headers=headers,
                json={"content": "hi", "web_search": True} if changed else {"content": "hi"},
            )
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == (
                "idempotency_key_reused" if changed else "task_run_continuation_unavailable"
            )
            assert brain.calls == calls
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["admission", "claim"])
def test_durable_admission_storage_fault_prevents_agent_dispatch(tmp_path, monkeypatch, fault):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    async def run() -> None:
        client = await _create_client(agent)  # type: ignore[arg-type]
        path = tmp_path / "runs.db"
        client.app["task_run_store"] = TaskRunStore(path)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            with sqlite3.connect(path) as conn:
                condition = "NEW.version = 0" if fault == "admission" else "NEW.version = 1"
                conn.execute(
                    "CREATE TRIGGER fail_transition BEFORE INSERT ON run_transitions "
                    f"WHEN {condition} BEGIN SELECT RAISE(ABORT, 'storage fault'); END"
                )
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 500
            assert brain.calls == 0
            with sqlite3.connect(path) as conn:
                states = conn.execute("SELECT state FROM task_runs").fetchall()
            assert states == ([] if fault == "admission" else [("admitted",)])
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("pending", [False, True])
def test_fresh_app_cannot_repeat_committed_ask_tool_dispatch(tmp_path, monkeypatch, pending):  # noqa: ANN001, ANN201
    from aiohttp.test_utils import TestClient, TestServer

    from config.http_server_config import HttpAuthConfig
    from llm.types import LLMResult, ToolCall
    from server.http.app import create_app
    from server.ui_session_storage import SQLiteUISessionStorage
    from shared.models import ToolResult

    from .fakes import TEST_API_TOKEN, TEST_AUTH_HEADERS

    monkeypatch.chdir(tmp_path)
    executions = []

    class ToolBrain(CountingBrain):
        supports_native_tools = True
        supports_streaming_tools = True

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResult(
                    text="", tool_calls=[ToolCall(id="lookup", name="lookup", arguments={})]
                )
            return LLMResult(text="read complete")

    async def new_client():
        brain = ToolBrain()
        agent = _RealStreamingAgent(
            brain=brain,
            enable_tools={"safe_mode": True},
            memory_companion_db_path=str(tmp_path / "companion.db"),
            memory_inbox_db_path=str(tmp_path / "inbox.db"),
            canonical_atoms_db_path=str(tmp_path / "atoms.db"),
        )

        def execute(request):
            executions.append(request)
            return ToolResult(ok=True, data={"value": "observed"})

        agent.tool_registry.register(
            "lookup",
            execute,
            capability="read",
            description="Read lookup",
            parameters_schema={"type": "object"},
            chat_exposed=True,
            risk_classes=["network"] if pending else [],
        )
        app = create_app(
            agent=agent,
            ui_storage=SQLiteUISessionStorage(tmp_path / "sessions.db"),
            task_run_store=TaskRunStore(tmp_path / "runs.db"),
            auth_config=HttpAuthConfig(
                api_token=TEST_API_TOKEN,
                allow_unauth_local=False,
                browser_auth_mode="token",
            ),
        )
        client = TestClient(TestServer(app), headers=TEST_AUTH_HEADERS)
        await client.start_server()
        return client, brain, agent

    async def run():
        first, first_brain, first_agent = await new_client()
        try:
            session = (await (await first.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(first, session)
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "lookup-request"}
            response = await first.post(
                "/ui/api/chat/send", headers=headers, json={"content": "lookup"}
            )
            assert response.status == 200
            assert len(executions) == (0 if pending else 1)
            if not pending:
                assert executions[0].name == "lookup"
            assert first_brain.calls == (1 if pending else 2)
            if pending:
                checkpoint = first_agent._pending_chat_approval
                assert checkpoint is not None
                decision = await first.app["ui_hub"].get_session_decision(session)
                first.app["idempotency_store"] = IdempotencyStore()
                repeated = await first.post(
                    "/ui/api/chat/send", headers=headers, json={"content": "lookup"}
                )
                assert repeated.status == 409
                assert first_agent._pending_chat_approval is checkpoint
                assert await first.app["ui_hub"].get_session_decision(session) == decision
                assert first_brain.calls == 1
        finally:
            await first.close()
        second, second_brain, _ = await new_client()
        try:
            response = await second.post(
                "/ui/api/chat/send", headers=headers, json={"content": "lookup"}
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "task_run_continuation_unavailable"
            assert second_brain.calls == 0
            assert len(executions) == (0 if pending else 1)
            with sqlite3.connect(tmp_path / "runs.db") as conn:
                assert conn.execute("SELECT state FROM task_runs").fetchall() == [
                    ("running" if pending else "result_submitted",)
                ]
        finally:
            await second.close()

    asyncio.run(run())


def test_ask_key_cannot_dispatch_in_another_mode(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        client.app["task_run_store"] = TaskRunStore(tmp_path / "runs.db")
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "mode-key"}
            assert (
                await client.post("/ui/api/chat/send", headers=headers, json={"content": "hi"})
            ).status == 200
            calls = brain.calls
            client.app["idempotency_store"] = IdempotencyStore()
            changed = await client.post("/ui/api/mode", headers=headers, json={"mode": "auto"})
            assert changed.status == 200
            repeated = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "hi"}
            )
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == "idempotency_key_reused"
            assert brain.calls == calls
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["request.received", "context.prepared"])
def test_context_change_before_admission_or_dispatch_prevents_model_call(
    tmp_path, monkeypatch, phase
):  # noqa: ANN001, ANN201
    from server.http.handlers import ui_chat

    monkeypatch.chdir(tmp_path)
    original = ui_chat._publish_agent_activity

    async def change_context(hub, *, session_id, phase: str, detail):
        if phase == selected_phase:
            await hub.set_session_workflow(session_id, mode="auto")
        await original(hub, session_id=session_id, phase=phase, detail=detail)

    selected_phase = phase
    monkeypatch.setattr(ui_chat, "_publish_agent_activity", change_context)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        path = tmp_path / "runs.db"
        client.app["task_run_store"] = TaskRunStore(path)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "task_run_context_changed"
            assert brain.calls == 0
            with sqlite3.connect(path) as conn:
                assert conn.execute("SELECT state FROM task_runs").fetchall() == (
                    [] if phase == "request.received" else [("running",)]
                )
        finally:
            await client.close()

    asyncio.run(run())
