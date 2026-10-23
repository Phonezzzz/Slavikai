from __future__ import annotations

import asyncio
import threading

import pytest

from server import http_api as api
from server.agent_provider import AgentScope
from server.http.common import ui_settings
from tests.ui_api.test_model_qualification import client_and_session, setup


def test_busy_model_switch_conflicts_without_waiting_or_mutation(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, principal = await client_and_session(config)
        lock = app["agent_provider"].lock_for(AgentScope(principal, session))
        await lock.acquire()
        try:
            response = await asyncio.wait_for(
                client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers={"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session},
                ),
                timeout=0.5,
            )
            assert response.status == 409
            assert await app["runtime_model_resolver"].resolve_main(session) == config
            assert calls == []
        finally:
            lock.release()
            await client.close()

    asyncio.run(run())


def test_registered_generation_blocks_switch_until_cancel_settles(monkeypatch, tmp_path):
    config, _ = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, _ = await client_and_session(config)
        registry = app["chat_cancellation_registry"]
        generation = await registry.start(session_id=session, stream_id="fixture")
        try:

            async def select():
                return await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers={"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session},
                )

            assert (await select()).status == 409
            await registry.request_cancel(session)
            assert (await select()).status == 409
            assert await app["runtime_model_resolver"].resolve_main(session) == config
            await registry.finish(generation)
            assert (await select()).status == 200
            assert (await app["runtime_model_resolver"].resolve_main(session)).model == "other"
        finally:
            await registry.finish(generation)
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status", ["pending", "executing"])
def test_pending_decision_blocks_model_switch(monkeypatch, tmp_path, status):
    config, _ = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, _ = await client_and_session(config)
        decision = {"id": "fixture", "status": status, "blocking": True}
        await app["ui_hub"].set_session_decision(session, decision)
        try:
            response = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers={"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session},
            )
            assert response.status == 409
            assert await app["runtime_model_resolver"].resolve_main(session) == config
            assert (await app["ui_hub"].get_session_decision(session))["id"] == "fixture"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/ui/api/chat/send", {"content": "fixture"}),
        ("/ui/api/tools/project", {"command": "find", "args": "fixture"}),
    ],
)
def test_old_chat_snapshot_cannot_restore_previous_model(monkeypatch, tmp_path, endpoint, payload):
    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, _ = await client_and_session(config)
        hub = app["ui_hub"]
        await hub.set_session_model(session, config.provider, config.model)
        entered, release = asyncio.Event(), asyncio.Event()
        original = hub.get_session_model
        held = [False]

        async def stale_snapshot(session_id):
            value = await original(session_id)
            if session_id == session and not held[0]:
                held[0] = True
                entered.set()
                await release.wait()
            return value

        monkeypatch.setattr(hub, "get_session_model", stale_snapshot)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        pending = asyncio.create_task(client.post(endpoint, json=payload, headers=headers))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            changed = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers=headers,
            )
            assert changed.status == 200
            release.set()
            response = await pending
            assert response.status == 409
            assert (await app["runtime_model_resolver"].resolve_main(session)).model == "other"
            assert calls == []
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await client.close()

    asyncio.run(run())


def test_slow_model_catalog_cannot_apply_after_another_switch(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)
    entered, release = threading.Event(), threading.Event()
    first = [True]

    def catalog(*_):
        if first[0]:
            first[0] = False
            entered.set()
            assert release.wait(5)
        return (["opaque", "other", "slow"], "ready", None)

    monkeypatch.setattr(ui_settings, "_probe_openai_models", catalog)

    async def run():
        client, app, session, _ = await client_and_session(config)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        pending = asyncio.create_task(
            client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "slow"},
                headers=headers,
            )
        )
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            changed = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers=headers,
            )
            assert changed.status == 200
            # Return to the initial value: equality alone misses this ABA race.
            changed_back = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "opaque"},
                headers=headers,
            )
            assert changed_back.status == 200
            release.set()
            assert (await pending).status == 409
            assert (await app["runtime_model_resolver"].resolve_main(session)).model == "opaque"
            assert calls == []
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await client.close()

    asyncio.run(run())


def test_plan_registration_blocks_model_change_until_settled(monkeypatch, tmp_path):
    config, _ = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, principal = await client_and_session(config)
        registry = app["plan_cancellation_registry"]
        scope = AgentScope(principal, session)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        try:
            async with registry.running(scope, "fixture", None):
                response = await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers=headers,
                )
                assert response.status == 409
                registry.request_cancel(scope, "fixture")
                response = await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers=headers,
                )
                assert response.status == 409
            response = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers=headers,
            )
            assert response.status == 200
        finally:
            await client.close()

    asyncio.run(run())


def test_running_chat_cancel_switch_next_request(monkeypatch, tmp_path):
    from tests.fake_provider_http import HoldingProvider

    config, _ = setup(monkeypatch, tmp_path)
    upstream = HoldingProvider(monkeypatch, "body")
    import aiohttp

    requests = []
    original_request = aiohttp.ClientSession._request

    async def record(session, method, url, **kwargs):
        if "provider.test" in str(url):
            requests.append(kwargs["json"])
        return await original_request(session, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", record)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, principal = await client_and_session(config)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        await app["ui_hub"].set_session_model(session, config.provider, config.model)
        pending = asyncio.create_task(
            client.post("/ui/api/chat/send", json={"content": "first"}, headers=headers)
        )
        try:
            assert await asyncio.to_thread(upstream.started.wait, 2)
            agent = await app["agent_provider"].get_existing_for_current_task(
                AgentScope(principal, session)
            )
            brain = agent.brain
            conflict = await asyncio.wait_for(
                client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers=headers,
                ),
                1,
            )
            assert conflict.status == 409
            assert agent.brain is brain
            assert upstream.calls == 1
            cancelled = await client.post("/ui/api/chat/cancel", headers=headers)
            assert cancelled.status == 200
            assert (await (await asyncio.wait_for(pending, 2)).json())["cancelled"] is True
            assert upstream.active == 0
            changed = await client.post(
                "/ui/api/session-model",
                json={"provider": config.provider, "model": "other"},
                headers=headers,
            )
            assert changed.status == 200
            second = await client.post(
                "/ui/api/chat/send", json={"content": "second"}, headers=headers
            )
            assert second.status == 200, await second.text()
            assert upstream.calls == 2
            assert [item["model"] for item in requests] == ["opaque", "other"]
            assert requests[0]["messages"][-1]["content"] == "first"
            assert requests[1]["messages"][-1]["content"] == "second"
            assert agent.brain is not brain
            assert (await app["runtime_model_resolver"].resolve_main(session)).model == "other"
            assert upstream.active == 0
        finally:
            await client.close()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


def test_other_principal_cannot_change_owned_session(monkeypatch, tmp_path):
    config, calls = setup(monkeypatch, tmp_path)

    async def run():
        client, app, session, _ = await client_and_session(config)
        try:
            response = await client.post(
                "/ui/api/session-model",
                json={"provider": "local", "model": "other"},
                headers={"Cf-Access-Jwt-Assertion": "member", api.UI_SESSION_HEADER: session},
            )
            assert response.status == 403
            assert await app["runtime_model_resolver"].resolve_main(session) == config
            assert calls == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["plan", "project"])
@pytest.mark.parametrize("selection", ["session", "global"])
def test_model_change_before_execution_lock_fences_old_request(
    monkeypatch, tmp_path, kind, selection
):
    from server.http.handlers import plan, project

    config, calls = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(
        ui_settings, "_probe_openai_models", lambda *_: (["opaque", "other"], "ready", None)
    )

    async def run():
        client, app, session, _ = await client_and_session(config)
        hub = app["ui_hub"]
        await hub.set_session_model(session, config.provider, config.model)
        if selection == "global":
            await app["runtime_model_state"].clear_session_override(session)
            await hub.set_session_model(session, "", "")
            await app["runtime_model_state"].set_global_main(config)
        if kind == "plan":
            await hub.set_session_workflow(session, mode="plan")
        module = plan if kind == "plan" else project
        name = "_resolve_agent" if kind == "plan" else "_resolve_agent_for_ui_session"
        original = getattr(module, name)
        entered, release = asyncio.Event(), asyncio.Event()
        invocations = []

        def forbidden(*args, **kwargs):
            invocations.append(True)
            raise RuntimeError("stale invocation")

        async def hold(*args, **kwargs):
            resolved = await original(*args, **kwargs)
            agent = resolved if kind == "plan" else resolved[0]
            monkeypatch.setattr(
                agent, "compile_plan_steps" if kind == "plan" else "respond", forbidden
            )
            entered.set()
            await release.wait()
            return resolved

        monkeypatch.setattr(module, name, hold)
        headers = {"Cf-Access-Jwt-Assertion": "owner", api.UI_SESSION_HEADER: session}
        endpoint = "/ui/api/plan/draft" if kind == "plan" else "/ui/api/tools/project"
        payload = {"goal": "fixture"} if kind == "plan" else {"command": "find", "args": "fixture"}
        pending = asyncio.create_task(client.post(endpoint, json=payload, headers=headers))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if selection == "global":
                from dataclasses import replace

                await app["runtime_model_state"].set_global_main(replace(config, model="other"))
            else:
                changed = await client.post(
                    "/ui/api/session-model",
                    json={"provider": config.provider, "model": "other"},
                    headers=headers,
                )
                assert changed.status == 200
            release.set()
            assert (await pending).status == 409
            assert invocations == []
            assert calls == []
        finally:
            release.set()
            await asyncio.gather(pending, return_exceptions=True)
            await client.close()

    asyncio.run(run())
