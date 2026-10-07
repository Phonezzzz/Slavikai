from __future__ import annotations

# ruff: noqa: F403,F405
import time

import pytest

from .fakes import *


def test_ui_runtime_init_requires_confirm_flag() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            resp = await client.post(
                "/ui/api/runtime/init",
                headers={"X-Slavik-Session": session_id},
                json={},
            )
            assert resp.status == 400
            payload = await resp.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "confirm_required"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_runtime_init_resets_workflow_to_ask() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            await _enter_act_mode(client, session_id, goal="runtime init reset")
            state_before = await client.get(
                "/ui/api/state",
                headers={"X-Slavik-Session": session_id},
            )
            assert state_before.status == 200
            before_payload = await state_before.json()
            assert before_payload.get("mode") == "act"

            init_resp = await client.post(
                "/ui/api/runtime/init",
                headers={"X-Slavik-Session": session_id},
                json={"confirm": True, "force": True, "reset_reason": "test_reset"},
            )
            assert init_resp.status == 200
            init_payload = await init_resp.json()
            assert init_payload.get("mode") == "ask"
            assert init_payload.get("active_plan") is None
            assert init_payload.get("active_task") is None
            assert init_payload.get("auto_state") is None
            reset = init_payload.get("reset")
            assert isinstance(reset, dict)
            assert reset.get("workflow_reset") is True
            assert reset.get("decision_reset") is True
            assert reset.get("force") is True
            assert reset.get("reset_reason") == "test_reset"
            readiness = init_payload.get("readiness")
            assert isinstance(readiness, dict)
            assert readiness.get("verifier_available") is True
            assert isinstance(readiness.get("workspace_root_valid"), bool)
            assert isinstance(readiness.get("tool_registry_integrity"), bool)
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_runtime_init_preserves_session_history() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "Привет"},
            )
            assert send_resp.status == 200

            history_before_resp = await client.get(
                f"/ui/api/sessions/{session_id}/history",
                headers={"X-Slavik-Session": session_id},
            )
            assert history_before_resp.status == 200
            history_before_payload = await history_before_resp.json()
            messages_before = history_before_payload.get("messages")
            assert isinstance(messages_before, list)
            assert len(messages_before) >= 2

            init_resp = await client.post(
                "/ui/api/init",
                headers={"X-Slavik-Session": session_id},
                json={"confirm": True, "force": True},
            )
            assert init_resp.status == 200

            history_after_resp = await client.get(
                f"/ui/api/sessions/{session_id}/history",
                headers={"X-Slavik-Session": session_id},
            )
            assert history_after_resp.status == 200
            history_after_payload = await history_after_resp.json()
            messages_after = history_after_payload.get("messages")
            assert isinstance(messages_after, list)
            assert len(messages_after) == len(messages_before)
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_runtime_init_blocks_running_task_without_force() -> None:
    class SlowTaskAgent(DummyAgent):
        def run_task_packet(
            self, packet: TaskPacket, context: RunContext, *, cancellation_token=None
        ) -> MWVRunResult:
            time.sleep(0.2)
            return super().run_task_packet(packet, context)

    async def run() -> None:
        client = await _create_client(SlowTaskAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            await _enter_act_mode(client, session_id, goal="runtime init guard")
            init_resp = await client.post(
                "/ui/api/runtime/init",
                headers={"X-Slavik-Session": session_id},
                json={"confirm": True},
            )
            assert init_resp.status == 409
            payload = await init_resp.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "runtime_busy"
        finally:
            await client.close()

    asyncio.run(run())


def test_plan_shutdown_timeout_allows_later_cleanup(monkeypatch, caplog) -> None:
    from server.agent_provider import AgentScope

    async def run() -> None:
        original_wait_for = asyncio.wait_for
        later_cleanup = []

        async def timeout_plan_wait(awaitable, *, timeout):
            if timeout == 10:
                awaitable.cancel()
                try:
                    await awaitable
                except asyncio.CancelledError:
                    pass
                raise TimeoutError
            return await original_wait_for(awaitable, timeout=timeout)

        async def observe_cleanup(app):
            later_cleanup.append(True)

        # Signal is frozen after app setup; register before freezing via a standalone app.
        from server.http.app import create_app

        other_app = create_app(
            agent=DummyAgent(),
            auth_config=HttpAuthConfig(
                api_token=TEST_API_TOKEN, allow_unauth_local=False, browser_auth_mode="token"
            ),
            ui_storage=InMemoryUISessionStorage(),
        )
        registry = other_app["plan_cancellation_registry"]
        other_app.on_cleanup.append(observe_cleanup)
        other_app.freeze()
        try:
            async with registry.running(AgentScope("owner", "session"), "task", None) as token:
                monkeypatch.setattr(asyncio, "wait_for", timeout_plan_wait)
                await other_app.cleanup()
                assert token.is_set()
                assert later_cleanup == [True]
                assert other_app["agent_provider"]._closed
                assert "Timed out" in caplog.text
        finally:
            monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    asyncio.run(run())


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("cancel_task", [False, True])
def test_plan_shutdown_retains_active_agent_until_worker_finishes(
    monkeypatch, static, cancel_task
) -> None:
    import threading
    from contextvars import ContextVar

    from server.agent_provider import AgentScope, ScopedAgentProvider
    from server.http.app import create_app

    class Owner:
        closes = 0

        def close(self):
            self.closes += 1

    async def run():
        owner = Owner()
        provider = (
            ScopedAgentProvider.from_instance(owner)
            if static
            else ScopedAgentProvider(
                factory=lambda scope, config: owner, retirement_timeout_seconds=0.01
            )
        )
        provider._retirement_timeout_seconds = 0.01
        app = create_app(
            agent=DummyAgent(),
            auth_config=HttpAuthConfig(
                api_token=TEST_API_TOKEN, allow_unauth_local=False, browser_auth_mode="token"
            ),
            ui_storage=InMemoryUISessionStorage(),
        )
        app["agent_provider"] = provider
        registry = app["plan_cancellation_registry"]
        app.freeze()
        scope = AgentScope("principal", "session")
        entered, release = threading.Event(), threading.Event()
        token_seen = []
        marker = ContextVar("packet_scope_marker", default="missing")
        original_wait_for = asyncio.wait_for

        def worker():
            assert marker.get() == "owned-scope"
            entered.set()
            assert release.wait(2)
            assert owner.closes == 0

        async def run_worker():
            marker.set("owned-scope")
            assert await provider.get_for_current_task(scope, None) is owner
            async with registry.running(scope, "task", None) as token, provider.lock_for(scope):
                from server.http.common.workflow_runtime import _run_owned_packet_thread

                token_seen.append(token)
                await _run_owned_packet_thread(worker, token)

        async def plan_timeout(awaitable, *, timeout):
            if timeout == 10:
                awaitable.cancel()
                try:
                    await awaitable
                except asyncio.CancelledError:
                    pass
                raise TimeoutError
            return await original_wait_for(awaitable, timeout=timeout)

        task = asyncio.create_task(run_worker())
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            monkeypatch.setattr(asyncio, "wait_for", plan_timeout)
            await original_wait_for(app.cleanup(), timeout=0.3)
            assert token_seen[0].is_set()
            assert owner.closes == 0
            assert provider._closed
            assert provider._retirements
            if cancel_task:
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done()
                assert owner.closes == 0
            release.set()
            await asyncio.gather(task, return_exceptions=cancel_task)
            for _ in range(30):
                if owner.closes and not provider._retirements:
                    break
                await asyncio.sleep(0.01)
            assert owner.closes == 1
            assert registry._active == {}
            assert provider._retirements == {}
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            monkeypatch.setattr(asyncio, "wait_for", original_wait_for)

    asyncio.run(run())
