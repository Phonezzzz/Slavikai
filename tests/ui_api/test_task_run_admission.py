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


@pytest.mark.parametrize("change", ["grants", "upstream"])
def test_admitted_claim_retry_rejects_changed_execution_context(tmp_path, monkeypatch, change):  # noqa: ANN001, ANN201
    from dataclasses import replace

    from server.agent_provider import AgentScope
    from server.http.handlers import ui_chat

    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    original = ui_chat._resolve_agent_for_ui_session
    changed = False

    async def resolve(request, session_id):
        owner, config = await original(request, session_id)
        if changed and change == "upstream":
            config = replace(config, base_url="http://different-upstream.invalid")
            await request.app["runtime_model_state"].set_session_override(session_id, config)
        return owner, config

    monkeypatch.setattr(ui_chat, "_resolve_agent_for_ui_session", resolve)

    async def run():
        nonlocal changed
        client = await _create_client(agent)  # type: ignore[arg-type]
        path = tmp_path / "runs.db"
        client.app["task_run_store"] = TaskRunStore(path)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "claim-context"}
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TRIGGER fail_claim BEFORE INSERT ON run_transitions "
                    "WHEN NEW.version = 1 BEGIN SELECT RAISE(ABORT, 'claim fault'); END"
                )
            assert (
                await client.post("/ui/api/chat/send", headers=headers, json={"content": "hi"})
            ).status == 500
            assert brain.calls == 0
            with sqlite3.connect(path) as conn:
                conn.execute("DROP TRIGGER fail_claim")
                principal = conn.execute("SELECT principal_id FROM task_revisions").fetchone()[0]
            changed = True
            if change == "grants":
                await client.app["session_store"].approve(
                    AgentScope(principal, session), {"NETWORK_RISK"}
                )
            client.app["idempotency_store"] = IdempotencyStore()
            repeated = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "hi"}
            )
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == "idempotency_key_reused"
            assert brain.calls == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_default_http_test_apps_use_independent_temporary_run_stores(tmp_path):  # noqa: ANN001, ANN201
    from .fakes import DummyAgent

    async def run():
        first = await _create_client(DummyAgent())
        second = await _create_client(DummyAgent())
        try:
            first_path = first.app["task_run_store"].path
            second_path = second.app["task_run_store"].path
            assert first_path.is_relative_to(tmp_path)
            assert second_path.is_relative_to(tmp_path)
            assert first_path != second_path
        finally:
            await first.close()
            await second.close()

    asyncio.run(run())


def test_admitted_retry_cannot_claim_after_history_changes(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
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
        path = tmp_path / "runs.db"
        client.app["task_run_store"] = TaskRunStore(path)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "unfinished"}
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TRIGGER fail_claim BEFORE INSERT ON run_transitions "
                    "WHEN NEW.version = 1 BEGIN SELECT RAISE(ABORT, 'fault'); END"
                )
            assert (
                await client.post("/ui/api/chat/send", headers=headers, json={"content": "old"})
            ).status == 500
            with sqlite3.connect(path) as conn:
                conn.execute("DROP TRIGGER fail_claim")
            assert (
                await client.post(
                    "/ui/api/chat/send",
                    headers={"X-Slavik-Session": session},
                    json={"content": "new history"},
                )
            ).status == 200
            before = await client.app["ui_hub"].get_messages(session, lane="chat")
            calls = brain.calls
            assert calls > 0
            client.app["idempotency_store"] = IdempotencyStore()
            response = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "old"}
            )
            assert response.status == 409
            assert brain.calls == calls
            assert (await response.json())["error"]["code"] == "task_run_continuation_unavailable"
            assert await client.app["ui_hub"].get_messages(session, lane="chat") == before
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("size,count", [(50000, 1), (80000, 2)])
def test_unicode_attachment_only_admission(tmp_path, monkeypatch, size, count):  # noqa: ANN001, ANN201
    import json

    from .fakes import DummyAgent

    monkeypatch.chdir(tmp_path)

    async def run():
        client = await _create_client(DummyAgent())
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session, "Content-Type": "application/json"},
                data=json.dumps(
                    {
                        "content": "",
                        "attachments": [
                            {"name": "paste.txt", "mime": "text/plain", "content": "😀" * size}
                            for _ in range(count)
                        ],
                    },
                    ensure_ascii=False,
                ).encode(),
            )
            assert response.status == 200
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("change", ["tools", "policy", "history"])
def test_admission_snapshot_fences_security_and_early_history_mutation(
    tmp_path, monkeypatch, change
):  # noqa: ANN001, ANN201
    from server.http.handlers import ui_chat
    from server.ui_hub import UIHub

    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    enabled = False
    publish = ui_chat._publish_agent_activity
    append = UIHub._append_message

    async def change_security(hub, *, session_id, phase, detail):
        if enabled and phase == "context.prepared":
            if change == "tools":
                await hub.set_session_tools_state(session_id, tools_state={"web": False})
            elif change == "policy":
                await hub.set_session_policy(session_id, profile="yolo")
        await publish(hub, session_id=session_id, phase=phase, detail=detail)

    async def change_history(hub, session_id, message, *, lane="chat", expected_history=None):
        if enabled and change == "history" and message.get("role") == "user":
            await hub.delete_last_message_pair(session_id, lane=lane)
        return await append(hub, session_id, message, lane=lane, expected_history=expected_history)

    monkeypatch.setattr(ui_chat, "_publish_agent_activity", change_security)
    monkeypatch.setattr(UIHub, "_append_message", change_history)

    async def run():
        nonlocal enabled
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_tools_state(session, tools_state={"web": True})
            headers = {"X-Slavik-Session": session}
            assert (
                await client.post("/ui/api/chat/send", headers=headers, json={"content": "before"})
            ).status == 200
            calls = brain.calls
            enabled = True
            response = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "next"}
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "task_run_context_changed"
            assert brain.calls == calls
        finally:
            await client.close()

    asyncio.run(run())


def test_accepted_security_application_failure_prevents_dispatch(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    from server.http.handlers import ui_chat

    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )

    async def fail_application(**kwargs):
        raise RuntimeError("security application fault")

    monkeypatch.setattr(ui_chat, "_apply_agent_runtime_state", fail_application)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 500
            assert (await response.json())["error"]["code"] == "runtime_context_apply_failed"
            assert brain.calls == 0
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("scenario", ["full", "mixed", "web"])
def test_ask_dispatch_uses_canonical_history_and_runtime(tmp_path, monkeypatch, scenario):  # noqa: ANN001, ANN201
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
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            hub = client.app["ui_hub"]
            if scenario != "web":
                for index in range(500):
                    lane = "workspace" if scenario == "mixed" and index % 2 else "chat"
                    await hub.append_message(
                        session,
                        hub.create_message(role="user", content=f"prior {index}", lane=lane),
                        lane=lane,
                    )
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session, "Idempotency-Key": "canonical-ask"},
                json={
                    "content": "проверь в интернете курс биткоина" if scenario == "web" else "next"
                },
            )
            assert response.status == 200
            assert brain.calls > 0
            payload = await response.json()
            assert "/web <запрос>" not in payload["messages"][-1]["content"]
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT state FROM task_runs").fetchall() == [
                    ("result_submitted",)
                ]
            if scenario != "web":
                assert (
                    len(await hub.get_messages(session, lane="chat"))
                    + len(await hub.get_messages(session, lane="workspace"))
                    == 500
                )
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("command", ["/trace", "/end-session"])
def test_debug_command_does_not_admit_model_run(tmp_path, monkeypatch, command):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session},
                json={"content": command},
            )
            assert response.status == 200
            assert brain.calls == 0
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT state FROM task_runs").fetchall() == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_second_ask_model_context_has_no_duplicate_ui_turns(tmp_path, monkeypatch, stream):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    captured = []

    class HistoryBrain(CountingBrain):
        def generate(self, messages, config=None, tools=None):
            captured.append(list(messages))
            return super().generate(messages, config, tools)

    brain = HistoryBrain()
    agent = _RealStreamingAgent(brain=brain)
    if not stream:
        monkeypatch.setattr(agent, "respond_stream", None)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            for content in ["unique first turn", "unique second turn"]:
                assert (
                    await client.post(
                        "/ui/api/chat/send",
                        headers={"X-Slavik-Session": session},
                        json={"content": content},
                    )
                ).status == 200
            model_history = [m.content for m in captured[-1] if m.role == "user"]
            assert model_history.count("unique first turn") == 1
            assert model_history.count("unique second turn") == 1
        finally:
            await client.close()

    asyncio.run(run())
