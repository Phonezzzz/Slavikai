from __future__ import annotations

# ruff: noqa: F403,F405
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
    async def run() -> None:
        client = await _create_client(DummyAgent())
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
