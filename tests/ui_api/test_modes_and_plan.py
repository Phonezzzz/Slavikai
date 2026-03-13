from __future__ import annotations

# ruff: noqa: F403,F405
from server.http.common.workflow_runtime import compute_plan_completion_state
from server.http_api import PLAN_AUDIT_MAX_READ_FILES

from .fakes import *


async def _prepare_approved_plan(
    client,
    session_id: str,
    *,
    switch_to_act: bool,
) -> int:
    mode_resp = await client.post(
        "/ui/api/mode",
        headers={"X-Slavik-Session": session_id},
        json={"mode": "plan"},
    )
    assert mode_resp.status == 200

    draft_resp = await client.post(
        "/ui/api/plan/draft",
        headers={"X-Slavik-Session": session_id},
        json={"goal": "Подготовка плана для execute"},
    )
    assert draft_resp.status == 200

    approve_resp = await client.post(
        "/ui/api/plan/approve",
        headers={"X-Slavik-Session": session_id},
    )
    assert approve_resp.status == 200
    approve_payload = await approve_resp.json()
    active_plan = approve_payload.get("active_plan")
    assert isinstance(active_plan, dict)
    plan_revision = active_plan.get("plan_revision")
    assert isinstance(plan_revision, int)

    if switch_to_act:
        to_act = await client.post(
            "/ui/api/mode",
            headers={"X-Slavik-Session": session_id},
            json={"mode": "act", "confirm": True},
        )
        assert to_act.status == 200
    return plan_revision


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
            assert state_payload.get("auto_state") is None

            to_auto = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "auto"},
            )
            assert to_auto.status == 200
            to_auto_payload = await to_auto.json()
            assert to_auto_payload.get("mode") == "auto"

            back_to_ask = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "ask"},
            )
            assert back_to_ask.status == 200
            back_to_ask_payload = await back_to_ask.json()
            assert back_to_ask_payload.get("mode") == "ask"

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


def test_ui_mode_transition_clears_stale_workflow_state() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_workflow(
                session_id,
                mode="auto",
                active_plan={"status": "approved", "plan_revision": 3},
                active_task={"status": "completed", "task_id": "task-1"},
                auto_state={"status": "completed", "run_id": "auto-1"},
            )

            to_ask = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "ask"},
            )
            assert to_ask.status == 200
            ask_payload = await to_ask.json()
            assert ask_payload.get("mode") == "ask"
            assert ask_payload.get("active_plan") is None
            assert ask_payload.get("active_task") is None
            assert ask_payload.get("auto_state") is None

            await hub.set_session_workflow(
                session_id,
                mode="ask",
                active_plan={"status": "draft", "plan_revision": 4},
                active_task={"status": "running", "task_id": "task-2"},
                auto_state={"status": "failed_worker", "run_id": "auto-2"},
            )

            to_auto = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "auto"},
            )
            assert to_auto.status == 200
            auto_payload = await to_auto.json()
            assert auto_payload.get("mode") == "auto"
            assert auto_payload.get("active_plan") is None
            assert auto_payload.get("active_task") is None
            assert auto_payload.get("auto_state") is None
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


def test_ui_plan_draft_allows_audit_soft_cap(tmp_path) -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            workspace_root = tmp_path / "audit-soft-cap-workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
            for idx in range(PLAN_AUDIT_MAX_READ_FILES + 3):
                (workspace_root / f"audit_{idx}.py").write_text(
                    f"print('audit-{idx}')\n",
                    encoding="utf-8",
                )
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(workspace_root))

            mode_resp = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "plan"},
            )
            assert mode_resp.status == 200

            draft_resp = await client.post(
                "/ui/api/plan/draft",
                headers={"X-Slavik-Session": session_id},
                json={"goal": "Проверка soft-cap audit"},
            )
            assert draft_resp.status == 200
            draft_payload = await draft_resp.json()
            plan_raw = draft_payload.get("active_plan")
            assert isinstance(plan_raw, dict)
            assert plan_raw.get("status") == "draft"
            audit_usage = draft_payload.get("audit_usage")
            assert isinstance(audit_usage, dict)
            read_files = audit_usage.get("read_files")
            assert isinstance(read_files, int)
            assert read_files == PLAN_AUDIT_MAX_READ_FILES
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


def test_ui_chat_send_in_ask_mode_does_not_require_root_gate_for_action_text() -> None:
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
                json={"content": "исправь тесты в проекте"},
            )
            assert response.status == 200
            payload = await response.json()
            decision = payload.get("decision")
            assert decision is None
            output_raw = payload.get("output")
            assert isinstance(output_raw, dict)
            content_raw = output_raw.get("content")
            assert isinstance(content_raw, str)
            assert content_raw.strip()
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
            assert names == {"local", "openrouter", "xai", "inception"}
            for item in providers:
                assert isinstance(item, dict)
                models = item.get("models")
                assert isinstance(models, list)
                assert len(models) == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_in_auto_mode_returns_auto_state_and_progress_event() -> None:
    class AutoStateAgent(DummyAgent):
        def __init__(self) -> None:
            super().__init__()
            self.last_auto_state = {
                "run_id": "auto-test-1",
                "status": "completed",
                "goal": "goal",
                "pool_size": 3,
                "started_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "planner": {"status": "completed"},
                "plan": {"plan_id": "p1", "goal": "goal", "shards": []},
                "coders": [],
                "merge": {"status": "completed", "changed_paths": []},
                "verifier": {"status": "passed", "command": ["check"], "exit_code": 0},
                "approval": None,
                "error": None,
            }

        def drain_auto_progress_events(self):  # noqa: ANN001
            return [dict(self.last_auto_state)]

        def respond(self, messages) -> str:  # noqa: ANN001
            del messages
            return "auto ok"

    async def run() -> None:
        client = await _create_client(AutoStateAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            to_auto = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "auto"},
            )
            assert to_auto.status == 200

            events_response = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers={"X-Slavik-Session": session_id},
            )
            assert events_response.status == 200
            try:
                send_resp = await client.post(
                    "/ui/api/chat/send",
                    headers={"X-Slavik-Session": session_id},
                    json={"content": "исправь тесты и обнови файл src/main.py"},
                )
                assert send_resp.status == 202
                send_payload = await send_resp.json()
                decision = send_payload.get("decision")
                assert isinstance(decision, dict)
                assert decision.get("status") == "pending"
                context_raw = decision.get("context")
                assert isinstance(context_raw, dict)
                assert context_raw.get("source_endpoint") == "chat.run_root"
                decision_id = decision.get("id")
                assert isinstance(decision_id, str)
                assert decision_id

                approve_resp = await client.post(
                    "/ui/api/decision/respond",
                    headers={"X-Slavik-Session": session_id},
                    json={
                        "session_id": session_id,
                        "decision_id": decision_id,
                        "choice": "approve_once",
                    },
                )
                assert approve_resp.status == 200
                approve_payload = await approve_resp.json()
                resume = approve_payload.get("resume")
                assert isinstance(resume, dict)
                assert resume.get("source_endpoint") == "chat.run_root"
                assert resume.get("ok") is True
                auto_state = approve_payload.get("auto_state")
                assert isinstance(auto_state, dict)
                assert auto_state.get("status") == "completed"

                events = await _read_sse_events(events_response, max_events=20)
                auto_events = [event for event in events if event.get("type") == "auto.progress"]
                assert auto_events
            finally:
                events_response.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_in_auto_mode_chat_like_request_skips_root_gate() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            to_auto = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "auto"},
            )
            assert to_auto.status == 200

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "Какой софт нужен для Raspberry Pi 4 для умной колонки?"},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            assert send_payload.get("decision") is None
            output_raw = send_payload.get("output")
            assert isinstance(output_raw, dict)
            content_raw = output_raw.get("content")
            assert isinstance(content_raw, str)
            assert content_raw.strip()
            auto_state = send_payload.get("auto_state")
            if isinstance(auto_state, dict):
                assert auto_state.get("status") != "failed_worker"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_appends_guidance_for_auto_missing_target_path() -> None:
    class AutoMissingPathAgent(DummyAgent):
        def respond(self, messages) -> str:  # noqa: ANN001
            del messages
            self.last_auto_state = {
                "run_id": "auto-missing-path-1",
                "status": "failed_worker",
                "goal": "goal",
                "pool_size": 3,
                "started_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:01+00:00",
                "planner": {"status": "completed"},
                "plan": {"plan_id": "p1", "goal": "goal", "shards": []},
                "coders": [],
                "merge": {"status": "failed"},
                "verifier": None,
                "approval": None,
                "error": "Не указан путь к файлу workspace для записи.",
                "error_code": "missing_target_path",
            }
            return "AUTO не смог завершить шаг."

    async def run() -> None:
        client = await _create_client(AutoMissingPathAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            to_auto = await client.post(
                "/ui/api/mode",
                headers={"X-Slavik-Session": session_id},
                json={"mode": "auto"},
            )
            assert to_auto.status == 200

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "Подскажи архитектуру для умной колонки на Raspberry Pi."},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            output_raw = send_payload.get("output")
            assert isinstance(output_raw, dict)
            content_raw = output_raw.get("content")
            assert isinstance(content_raw, str)
            assert "нужен явный путь к файлу" in content_raw
            assert "создай docs/raspberry-setup.md" in content_raw
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
            assert "Выберите модель из списка доступных" in message
            details = error.get("details")
            assert isinstance(details, dict)
            assert details.get("suggestion") == "grok-4"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_plan_execute_rejects_second_start_when_running(monkeypatch) -> None:  # noqa: ANN001
    async def _slow_runner(**kwargs) -> None:  # noqa: ANN003
        del kwargs
        await asyncio.sleep(0.2)

    monkeypatch.setattr("server.http.handlers.plan._run_plan_runner", _slow_runner)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            revision = await _prepare_approved_plan(client, session_id, switch_to_act=True)

            first = await client.post(
                "/ui/api/plan/execute",
                headers={"X-Slavik-Session": session_id},
                json={"plan_revision": revision},
            )
            assert first.status == 200
            first_payload = await first.json()
            active_task = first_payload.get("active_task")
            assert isinstance(active_task, dict)
            assert active_task.get("status") == "running"

            second = await client.post(
                "/ui/api/plan/execute",
                headers={"X-Slavik-Session": session_id},
                json={"plan_revision": revision},
            )
            assert second.status == 409
            second_payload = await second.json()
            error = second_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "task_already_running"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_plan_execute_parallel_race_allows_only_one_runner(monkeypatch) -> None:  # noqa: ANN001
    async def _slow_runner(**kwargs) -> None:  # noqa: ANN003
        del kwargs
        await asyncio.sleep(0.2)

    monkeypatch.setattr("server.http.handlers.plan._run_plan_runner", _slow_runner)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            revision = await _prepare_approved_plan(client, session_id, switch_to_act=True)

            async def _execute_once():
                return await client.post(
                    "/ui/api/plan/execute",
                    headers={"X-Slavik-Session": session_id},
                    json={"plan_revision": revision},
                )

            first_resp, second_resp = await asyncio.gather(_execute_once(), _execute_once())
            statuses = sorted([first_resp.status, second_resp.status])
            assert statuses == [200, 409]

            payloads = [await first_resp.json(), await second_resp.json()]
            success = next((item for item in payloads if item.get("ok") is True), None)
            conflict = next(
                (item for item in payloads if isinstance(item.get("error"), dict)),
                None,
            )
            assert isinstance(success, dict)
            assert isinstance(conflict, dict)
            conflict_error = conflict.get("error")
            assert isinstance(conflict_error, dict)
            assert conflict_error.get("code") == "task_already_running"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_plan_execute_idempotency_key_matrix(monkeypatch) -> None:  # noqa: ANN001
    async def _slow_runner(**kwargs) -> None:  # noqa: ANN003
        del kwargs
        await asyncio.sleep(0.2)

    monkeypatch.setattr("server.http.handlers.plan._run_plan_runner", _slow_runner)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            revision = await _prepare_approved_plan(client, session_id, switch_to_act=True)
            headers = {
                "X-Slavik-Session": session_id,
                "Idempotency-Key": "plan-exec-k1",
            }

            first = await client.post(
                "/ui/api/plan/execute",
                headers=headers,
                json={"plan_revision": revision},
            )
            assert first.status == 200
            first_payload = await first.json()
            first_task = first_payload.get("active_task")
            assert isinstance(first_task, dict)
            first_task_id = first_task.get("task_id")
            assert isinstance(first_task_id, str)

            replay = await client.post(
                "/ui/api/plan/execute",
                headers=headers,
                json={"plan_revision": revision},
            )
            assert replay.status == 200
            replay_payload = await replay.json()
            replay_task = replay_payload.get("active_task")
            assert isinstance(replay_task, dict)
            assert replay_task.get("task_id") == first_task_id

            conflict = await client.post(
                "/ui/api/plan/execute",
                headers=headers,
                json={"plan_revision": revision, "client_nonce": "different"},
            )
            assert conflict.status == 409
            conflict_payload = await conflict.json()
            conflict_error = conflict_payload.get("error")
            assert isinstance(conflict_error, dict)
            assert conflict_error.get("code") == "idempotency_key_reused"

            no_key = await client.post(
                "/ui/api/plan/execute",
                headers={"X-Slavik-Session": session_id},
                json={"plan_revision": revision},
            )
            assert no_key.status == 409
            no_key_payload = await no_key.json()
            no_key_error = no_key_payload.get("error")
            assert isinstance(no_key_error, dict)
            assert no_key_error.get("code") == "task_already_running"
        finally:
            await client.close()

    asyncio.run(run())


def test_compute_plan_completion_state_matrix() -> None:
    assert (
        compute_plan_completion_state(
            {
                "steps": [
                    {"step_id": "s1", "status": "done"},
                    {"step_id": "s2", "status": "done"},
                ]
            }
        )
        == "completed"
    )
    assert (
        compute_plan_completion_state(
            {
                "steps": [
                    {"step_id": "s1", "status": "done"},
                    {"step_id": "s2", "status": "failed"},
                ]
            }
        )
        == "failed"
    )
    assert (
        compute_plan_completion_state(
            {
                "steps": [
                    {"step_id": "s1", "status": "done"},
                    {"step_id": "s2", "status": "waiting_approval"},
                ]
            }
        )
        == "running"
    )
    assert (
        compute_plan_completion_state(
            {
                "steps": [
                    {"step_id": "s1", "status": "done"},
                    {"step_id": "s2", "status": "blocked"},
                ]
            }
        )
        == "running"
    )
