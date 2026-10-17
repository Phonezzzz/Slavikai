from __future__ import annotations

import asyncio
import sqlite3

import pytest

from core.task_run_storage import TaskRunStore
from llm.types import ModelConfig

from .fakes import TEST_API_TOKEN, _select_local_model
from .fakes import _create_client as _create_ui_client
from .test_stream_and_events import _RealStreamingAgent
from .test_task_run_admission import CountingBrain


async def _create_client(agent):  # noqa: ANN001, ANN202
    client = await _create_ui_client(agent)
    await client.app["runtime_model_state"].set_global_main(
        ModelConfig(provider="ollama", model="test")
    )
    return client


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("explicit", [False, "ask", "auto"])
@pytest.mark.parametrize("changed", [False, True])
def test_api_ask_durable_identity_prevents_repeat_dispatch(
    tmp_path, monkeypatch, changed, explicit, scoped
):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    brain.supports_native_tools = True
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {
                "Authorization": f"Bearer {TEST_API_TOKEN}",
                "X-Slavik-Session": session,
                "Idempotency-Key": "accepted-api-request",
            }
            payload = {
                "model": "slavik",
                "messages": [{"role": "user", "content": "hi"}],
                "slavik_meta": {"runtime_mode": explicit},
            }
            if not scoped:
                headers.pop("X-Slavik-Session")
            if not explicit:
                payload.pop("slavik_meta")
                agent.runtime_mode = "auto"
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 200
            assert agent.runtime_mode == (explicit or "ask")
            calls = brain.calls
            assert calls > 0
            store = client.app["task_run_store"]
            client.app["task_run_store"] = TaskRunStore(store.path)
            if changed:
                payload["messages"] = [{"role": "user", "content": "different"}]
            repeated = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == (
                "idempotency_key_reused" if changed else "task_run_continuation_unavailable"
            )
            assert brain.calls == calls
            with sqlite3.connect(store.path) as conn:
                assert conn.execute("SELECT state FROM task_runs").fetchall() == [
                    ("result_submitted",)
                ]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
@pytest.mark.parametrize("fault", ["admission", "claim"])
def test_api_ask_admission_fault_prevents_model_call(tmp_path, monkeypatch, fault, mode):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            store = client.app["task_run_store"]
            with sqlite3.connect(store.path) as conn:
                condition = "NEW.version = 0" if fault == "admission" else "NEW.version = 1"
                conn.execute(
                    "CREATE TRIGGER fail_api_admission BEFORE INSERT ON run_transitions "
                    + f"WHEN {condition} BEGIN SELECT RAISE(ABORT, 'storage fault'); END"
                )
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}", "X-Slavik-Session": session},
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "hi"}],
                    "slavik_meta": {"runtime_mode": mode},
                },
            )
            assert response.status == 500
            assert brain.calls == 0
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
@pytest.mark.parametrize("change", ["tools", "policy", "grants"])
def test_api_context_change_before_admission_is_fenced(tmp_path, monkeypatch, change, mode):  # noqa: ANN001, ANN201
    from server.http.handlers import chat

    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)
    load = chat._load_effective_session_security
    mutated = False

    async def mutate(*, hub, session_id):
        nonlocal mutated
        result = await load(hub=hub, session_id=session_id)
        if not mutated:
            mutated = True
            if change == "tools":
                await hub.set_session_tools_state(
                    session_id, tools_state={"web": not result[0].get("web", False)}
                )
            elif change == "policy":
                await hub.set_session_policy(
                    session_id, profile="yolo" if result[1]["profile"] != "yolo" else "sandbox"
                )
            else:
                from server.agent_provider import AgentScope

                principal = await hub.get_session_principal_id(session_id)
                await client.app["session_store"].approve(
                    AgentScope(principal, session_id), {"NETWORK_RISK"}
                )
        return result

    monkeypatch.setattr(chat, "_load_effective_session_security", mutate)

    async def run():
        nonlocal client
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}", "X-Slavik-Session": session},
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "hi"}],
                    "slavik_meta": {"runtime_mode": mode},
                },
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "task_run_context_changed"
            assert brain.calls == 0
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
        finally:
            await client.close()

    client = None
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_api_reconstructed_app_does_not_repeat_gateway_dispatch(tmp_path, monkeypatch, mode):  # noqa: ANN001, ANN201
    from llm.types import LLMResult, ToolCall
    from shared.models import ToolResult

    monkeypatch.chdir(tmp_path)
    executions = []

    class LookupBrain(CountingBrain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            if not any(m.role == "tool" for m in messages):
                return LLMResult(
                    text="",
                    tool_calls=[ToolCall(id="original-lookup", name="lookup", arguments={})],
                )
            return LLMResult(text="observed")

    async def new_client():
        brain = LookupBrain()
        agent = _RealStreamingAgent(brain=brain)

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
        )
        client = await _create_client(agent)  # type: ignore[arg-type]
        return client, brain

    async def run():
        first, brain = await new_client()
        try:
            session = (await (await first.get("/ui/api/status")).json())["session_id"]
            principal = await first.app["ui_hub"].get_session_principal_id(session)
            await _select_local_model(first, session)
            headers = {
                "Authorization": f"Bearer {TEST_API_TOKEN}",
                "X-Slavik-Session": session,
                "Idempotency-Key": "gateway-request",
            }
            payload = {
                "model": "slavik",
                "messages": [{"role": "user", "content": "read"}],
                "slavik_meta": {"runtime_mode": mode},
            }
            assert (
                await first.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 200
            assert brain.calls == 2 and len(executions) == 1
            path = first.app["task_run_store"].path
        finally:
            await first.close()
        second, fresh_brain = await new_client()
        try:
            await second.app["ui_hub"].get_or_create_session(session, principal_id=principal)
            await _select_local_model(second, session)
            second.app["task_run_store"] = TaskRunStore(path)
            response = await second.post("/v1/chat/completions", headers=headers, json=payload)
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "task_run_continuation_unavailable"
            assert fresh_brain.calls == 0 and len(executions) == 1
        finally:
            await second.close()

    asyncio.run(run())


@pytest.mark.parametrize("variant", ["new_no_key", "auto_key", "command_key"])
def test_api_request_identity_cannot_change_execution_lane(tmp_path, monkeypatch, variant):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            headers = {
                "Authorization": f"Bearer {TEST_API_TOKEN}",
                "X-Slavik-Session": session,
                "Idempotency-Key": "bound-request",
            }
            payload = {"model": "slavik", "messages": [{"role": "user", "content": "hi"}]}
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 200
            previous_calls = brain.calls
            if variant == "new_no_key":
                headers.pop("Idempotency-Key")
            elif variant == "auto_key":
                payload["slavik_meta"] = {"runtime_mode": "auto"}
            else:
                payload["messages"] = [{"role": "user", "content": "/trace"}]
            response = await client.post("/v1/chat/completions", headers=headers, json=payload)
            assert response.status == (200 if variant == "new_no_key" else 409)
            assert agent.runtime_mode == "ask"
            assert (
                brain.calls > previous_calls
                if variant == "new_no_key"
                else brain.calls == previous_calls
            )
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == (
                    2 if variant == "new_no_key" else 1
                )
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("command", ["/trace", "/end-session"])
def test_api_debug_command_does_not_create_model_run(tmp_path, monkeypatch, command):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={"model": "slavik", "messages": [{"role": "user", "content": command}]},
            )
            assert response.status == 200
            assert brain.calls == 0
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", [None, "ask", "auto"])
def test_api_does_not_dispatch_injected_agent_without_selected_model(tmp_path, monkeypatch, mode):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    brain = CountingBrain()
    agent = _RealStreamingAgent(brain=brain)

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            await client.app["runtime_model_state"].set_global_main(None)
            response = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "hi"}],
                    "slavik_meta": {"runtime_mode": mode},
                },
            )
            assert response.status == 409
            assert (await response.json())["error"]["code"] == "model_not_selected"
            assert brain.calls == 0
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_api_retry_preserves_checkpoint_without_applying_ui_mode(tmp_path, monkeypatch, mode):  # noqa: ANN001, ANN201
    from llm.types import LLMResult, ToolCall
    from shared.models import ToolResult

    monkeypatch.chdir(tmp_path)

    class ApprovalBrain(CountingBrain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            self.calls += 1
            return LLMResult(
                text="", tool_calls=[ToolCall(id="original", name="lookup", arguments={})]
            )

    brain = ApprovalBrain()
    agent = _RealStreamingAgent(brain=brain)
    executions = []

    def execute(request):
        executions.append(request)
        return ToolResult(ok=True, data={})

    agent.tool_registry.register(
        "lookup",
        execute,
        capability="read",
        description="Read lookup",
        parameters_schema={"type": "object"},
        chat_exposed=True,
        risk_classes=["network"],
    )

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_workflow(session, mode="auto")
            headers = {
                "Authorization": f"Bearer {TEST_API_TOKEN}",
                "X-Slavik-Session": session,
                "Idempotency-Key": "paused-api",
            }
            payload = {
                "model": "slavik",
                "messages": [{"role": "user", "content": "lookup"}],
                "slavik_meta": {"runtime_mode": mode},
            }
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 400
            pending = agent._pending_chat_approval
            assert pending is not None and pending.task_run is not None
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 409
            assert agent._pending_chat_approval is pending
            assert agent.runtime_mode == mode
            assert brain.calls == 1 and executions == []
        finally:
            await client.close()

    asyncio.run(run())


def test_old_api_retry_does_not_change_later_execution_mode(tmp_path, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.chdir(tmp_path)
    agent = _RealStreamingAgent(brain=CountingBrain())

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            payload = {"model": "slavik", "messages": [{"role": "user", "content": "hi"}]}
            headers = {
                "Authorization": f"Bearer {TEST_API_TOKEN}",
                "Idempotency-Key": "original-ask",
            }
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 200
            auto = await client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {TEST_API_TOKEN}"},
                json={**payload, "slavik_meta": {"runtime_mode": "auto"}},
            )
            assert auto.status == 200
            assert agent.runtime_mode == "auto"
            assert (
                await client.post("/v1/chat/completions", headers=headers, json=payload)
            ).status == 409
            assert agent.runtime_mode == "auto"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("explicit", [False, True])
def test_api_implicit_scope_does_not_read_foreign_ui_context(tmp_path, monkeypatch, explicit):  # noqa: ANN001, ANN201
    from server import http_api as api

    monkeypatch.chdir(tmp_path)
    agent = _RealStreamingAgent(brain=CountingBrain())

    async def run():
        client = await _create_client(agent)  # type: ignore[arg-type]
        try:
            hub = client.app["ui_hub"]
            await hub.get_or_create_session("__principal__", principal_id="foreign-browser")
            await hub.set_workspace_root("__principal__", str(tmp_path))
            await hub.set_session_policy("__principal__", profile="yolo")
            await hub.set_session_tools_state("__principal__", tools_state={"web": True})
            headers = {"Authorization": f"Bearer {TEST_API_TOKEN}"}
            if explicit:
                headers["X-Slavik-Session"] = "__principal__"
            response = await client.post(
                "/v1/chat/completions",
                headers=headers,
                json={"model": "slavik", "messages": [{"role": "user", "content": "hi"}]},
            )
            if explicit:
                assert response.status == 403
                assert agent.brain.calls == 0
            else:
                assert response.status == 200
                assert agent.runtime_workspace_root == str(api.WORKSPACE_ROOT)
                assert agent.tools_enabled["safe_mode"] is True
                assert (
                    agent.tools_enabled["web"]
                    == api._build_effective_tools_state(session_override=None)["web"]
                )
                assert await hub.get_session_principal_id("__principal__") == "foreign-browser"
        finally:
            await client.close()

    asyncio.run(run())
