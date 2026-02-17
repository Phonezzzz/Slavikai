from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_state_and_mode_transitions() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            state_resp = await client.get(
                "/ui/api/state",
                headers={"X-Slavik-Session": session_id},
            )
            assert state_resp.status == 200
            state_payload = await state_resp.json()
            assert state_payload.get("mode") == "ask"
            assert state_payload.get("active_plan") is None
            assert state_payload.get("active_task") is None

            to_plan = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "plan"},
            )
            assert to_plan.status == 200
            to_plan_payload = await to_plan.json()
            assert to_plan_payload.get("mode") == "plan"

            to_act = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "act", "confirm": True},
            )
            assert to_act.status == 409
            error_payload = await to_act.json()
            error = error_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "plan_not_approved"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_plan_lifecycle_endpoints() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            mode_resp = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "plan"},
            )
            assert mode_resp.status == 200

            draft_resp = await client.post(
                "/ui/api/plan/draft",
                headers={"X-Slavik-Session": session_id},
                json={"goal": "Добавить Ask/Plan/Act"},
            )
            assert draft_resp.status == 200
            draft_payload = await draft_resp.json()
            plan_raw = draft_payload.get("active_plan")
            assert isinstance(plan_raw, dict)
            assert plan_raw.get("status") == "draft"
            plan_revision = plan_raw.get("plan_revision")
            assert isinstance(plan_revision, int)
            assert plan_revision > 0

            approve_resp = await client.post(
                "/ui/api/plan/approve",
                headers={"X-Slavik-Session": session_id},
            )
            assert approve_resp.status == 200
            approve_payload = await approve_resp.json()
            approved_plan = approve_payload.get("active_plan")
            assert isinstance(approved_plan, dict)
            assert approved_plan.get("status") == "approved"
            approved_revision = approved_plan.get("plan_revision")
            assert isinstance(approved_revision, int)

            execute_resp = await client.post(
                "/ui/api/plan/execute",
                headers={"X-Slavik-Session": session_id},
                json={"plan_revision": approved_revision},
            )
            assert execute_resp.status == 409
            execute_payload = await execute_resp.json()
            decision = execute_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)
            decision_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert decision_resp.status == 200
            decision_payload = await decision_resp.json()
            assert decision_payload.get("mode") == "act"
            task_raw = decision_payload.get("active_task")
            assert isinstance(task_raw, dict)
            assert task_raw.get("status") == "running"

            # runner skeleton доходит до completed асинхронно
            completed = False
            for _ in range(20):
                await asyncio.sleep(0.02)
                state_resp = await client.get(
                    "/ui/api/state",
                    headers={"X-Slavik-Session": session_id},
                )
                assert state_resp.status == 200
                state_payload = await state_resp.json()
                next_task = state_payload.get("active_task")
                if isinstance(next_task, dict) and next_task.get("status") == "completed":
                    completed = True
                    break
            assert completed is True
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_ask_mode_does_not_hard_block_action_intent() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            response = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "/sh ls"},
            )
            assert response.status == 200
            payload = await response.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            content = last.get("content")
            assert isinstance(content, str)
            assert "ASK_MODE_NO_ACTIONS" not in content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_requires_model_selection() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post("/ui/api/chat/send", json={"content": "Ping"})
            assert response.status == 409
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "model_not_selected"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_models_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: ([f"{provider}-1", f"{provider}-2"], None),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/models")
            assert response.status == 200
            payload = await response.json()
            providers = payload.get("providers")
            assert isinstance(providers, list)
            names = {item.get("provider") for item in providers if isinstance(item, dict)}
            assert names == {"local", "openrouter", "xai"}
            for item in providers:
                assert isinstance(item, dict)
                models = item.get("models")
                assert isinstance(models, list)
                assert len(models) == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_model_not_found_suggests_closest(monkeypatch) -> None:
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["grok-4", "grok-3-mini"], None)
        if provider == "xai"
        else (["local-default"], None),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": "session-1"},
                json={"provider": "xai", "model": "grok-4x"},
            )
            assert response.status == 404
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "model_not_found"
            message = error.get("message")
            assert isinstance(message, str)
            assert "сам придумал, сам и страдай" in message
            details = error.get("details")
            assert isinstance(details, dict)
            assert details.get("suggestion") == "grok-4"
        finally:
            await client.close()

    asyncio.run(run())
