from __future__ import annotations

import asyncio
import threading

import pytest

from core.agent_response import AgentResponse
from server import http_api as api
from server.http.common import ui_settings
from server.http.handlers import plan, project, ui_chat, workspace
from tests.ui_api.test_model_qualification import client_and_session, setup


@pytest.mark.parametrize("surface", ["project_user", "project_assistant", "draft", "root", "git"])
def test_catalog_cannot_switch_during_result_publication(monkeypatch, tmp_path, surface):
    config, calls = setup(monkeypatch, tmp_path)
    catalog_entered, release_catalog = threading.Event(), threading.Event()

    def catalog(*_):
        catalog_entered.set()
        assert release_catalog.wait(5)
        return (["opaque", "other"], "ready", None)

    monkeypatch.setattr(ui_settings, "_probe_openai_models", catalog)

    async def run():
        client, app, session, _ = await client_and_session(config)
        hub = app["ui_hub"]
        await hub.set_session_model(session, config.provider, config.model)
        await hub.set_workspace_root(session, str(tmp_path))
        entered, release = asyncio.Event(), asyncio.Event()
        original_append = hub.append_message
        original_workflow = hub.set_session_workflow
        original_decision = hub.set_session_decision
        invocations = []

        async def append(session_id, message, **kwargs):
            if surface == f"project_{message.get('role')}":
                entered.set()
                await release.wait()
            return await original_append(session_id, message, **kwargs)

        async def workflow(session_id, **kwargs):
            if surface == "draft" and isinstance(kwargs.get("active_plan"), dict):
                entered.set()
                await release.wait()
            return await original_workflow(session_id, **kwargs)

        async def decision(session_id, value):
            if surface in {"root", "git"} and isinstance(value, dict):
                entered.set()
                await release.wait()
            return await original_decision(session_id, value)

        monkeypatch.setattr(hub, "append_message", append)
        monkeypatch.setattr(hub, "set_session_decision", decision)
        if surface.startswith("project"):
            original = project._resolve_agent_for_ui_session

            async def resolve(*args):
                agent, selected = await original(*args)

                def respond(messages):
                    invocations.append(messages)
                    return AgentResponse("fixture result")

                monkeypatch.setattr(agent, "respond", respond)
                return agent, selected

            monkeypatch.setattr(project, "_resolve_agent_for_ui_session", resolve)
            endpoint, payload, status = (
                "/ui/api/tools/project",
                {"command": "find", "args": "fixture"},
                200,
            )
        elif surface == "draft":
            await hub.set_session_workflow(session, mode="plan")
            original = plan._resolve_agent

            async def resolve(*args):
                agent = await original(*args)
                monkeypatch.setattr(agent, "compile_plan_steps", lambda *_: [])
                return agent

            monkeypatch.setattr(plan, "_resolve_agent", resolve)
            monkeypatch.setattr(hub, "set_session_workflow", workflow)
            endpoint, payload, status = "/ui/api/plan/draft", {"goal": "fixture"}, 200
        elif surface == "root":
            target = tmp_path / "other"
            target.mkdir()

            async def security(**_):
                return {}, {"profile": "yolo"}

            monkeypatch.setattr(workspace, "_load_effective_session_security", security)
            endpoint, payload, status = (
                "/ui/api/workspace/root/select",
                {"root_path": str(target)},
                202,
            )
        else:
            endpoint, payload, status = "/ui/api/workspace/git/stage", {"paths": ["fixture"]}, 202
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        setter = asyncio.create_task(
            client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers=headers,
            )
        )
        pending = None
        try:
            assert await asyncio.to_thread(catalog_entered.wait, 2)
            pending = asyncio.create_task(client.post(endpoint, json=payload, headers=headers))
            await asyncio.wait_for(entered.wait(), 2)
            release_catalog.set()
            assert (await asyncio.wait_for(setter, 2)).status == 409
            assert await app["runtime_model_resolver"].resolve_main(session) == config
            release.set()
            response = await asyncio.wait_for(pending, 2)
            assert response.status == status, await response.text()
            if surface.startswith("project"):
                messages = await hub.get_messages(session, lane="workspace")
                assert [item["role"] for item in messages] == ["user", "assistant"]
                assert len(invocations) == 1
            assert calls == []
        finally:
            release_catalog.set()
            release.set()
            await asyncio.gather(setter, *([pending] if pending else []), return_exceptions=True)
            await client.close()

    asyncio.run(run())


def test_root_approval_packet_observes_waiting_plan_step(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(ui_chat, "_request_requires_root_gate", lambda **_: True)

    async def run():
        client, app, session, _ = await client_and_session(config)
        hub = app["ui_hub"]
        draft = api._build_plan_draft(
            goal="fixture",
            audit_log=[],
            steps=[
                {"step_id": "s", "title": "fixture", "description": "fixture", "status": "doing"}
            ],
            verifier={},
        )
        task = {
            "task_id": "t",
            "plan_id": draft["plan_id"],
            "plan_hash": draft["plan_hash"],
            "status": "running",
            "current_step_id": "s",
        }
        await hub.set_session_workflow(session, mode="act", active_plan=draft, active_task=task)
        original = hub.set_session_decision
        observed = []

        async def packet(session_id, value):
            if isinstance(value, dict) and value.get("blocking"):
                workflow = await hub.get_session_workflow(session_id)
                observed.append(workflow["active_plan"]["steps"][0]["status"])
            return await original(session_id, value)

        monkeypatch.setattr(hub, "set_session_decision", packet)
        try:
            response = await client.post(
                "/ui/api/chat/send",
                json={"content": "fixture"},
                headers={"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session},
            )
            assert response.status == 202, await response.text()
            assert observed == ["waiting_approval"]
            assert calls == []
        finally:
            await client.close()

    asyncio.run(run())
