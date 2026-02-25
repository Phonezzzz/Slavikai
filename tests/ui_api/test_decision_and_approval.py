from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_plan_edit_resets_approved_to_draft() -> None:
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
                json={"goal": "Проверить edit flow"},
            )
            assert draft_resp.status == 200
            draft_payload = await draft_resp.json()
            plan_raw = draft_payload.get("active_plan")
            assert isinstance(plan_raw, dict)

            approve_resp = await client.post(
                "/ui/api/plan/approve",
                headers={"X-Slavik-Session": session_id},
            )
            assert approve_resp.status == 200
            approve_payload = await approve_resp.json()
            approved_plan = approve_payload.get("active_plan")
            assert isinstance(approved_plan, dict)
            approved_revision = approved_plan.get("plan_revision")
            assert isinstance(approved_revision, int)

            edit_resp = await client.post(
                "/ui/api/plan/edit",
                headers={"X-Slavik-Session": session_id},
                json={
                    "plan_revision": approved_revision,
                    "operation": {
                        "op": "update_step",
                        "step_id": "step-1-audit",
                        "changes": {"title": "Новый заголовок"},
                    },
                },
            )
            assert edit_resp.status == 200
            edit_payload = await edit_resp.json()
            edited_plan = edit_payload.get("active_plan")
            assert isinstance(edited_plan, dict)
            assert edited_plan.get("status") == "draft"
            edited_revision = edited_plan.get("plan_revision")
            assert isinstance(edited_revision, int)
            assert edited_revision > approved_revision
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_workspace_write_requires_approval_and_emits_decision_packet() -> None:
    async def run() -> None:
        client = await _create_client(WorkspaceDecisionAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            events_response = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers={"X-Slavik-Session": session_id},
            )
            assert events_response.status == 200
            try:
                write_resp = await client.put(
                    "/ui/api/workspace/file",
                    headers={"X-Slavik-Session": session_id},
                    json={"path": "main.py", "content": "print('ok')\n"},
                )
                assert write_resp.status == 202
                write_payload = await write_resp.json()
                decision = write_payload.get("decision")
                assert isinstance(decision, dict)
                assert decision.get("status") == "pending"
                context = decision.get("context")
                assert isinstance(context, dict)
                assert context.get("source_endpoint") == "workspace.tool"
                resume_payload = context.get("resume_payload")
                assert isinstance(resume_payload, dict)
                assert resume_payload.get("tool_name") == "workspace_write"

                events = await _read_sse_events(events_response, max_events=10)
                decision_events = [
                    event for event in events if event.get("type") == "decision.packet"
                ]
                assert decision_events
            finally:
                events_response.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_approve_once_executes_workspace_tool() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            write_resp = await client.put(
                "/ui/api/workspace/file",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py", "content": "print('ok')\n"},
            )
            assert write_resp.status == 202
            write_payload = await write_resp.json()
            decision = write_payload.get("decision")
            assert isinstance(decision, dict)
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
            assert approve_payload.get("status") == "resolved"
            assert len(agent.tool_calls) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_approve_once_does_not_persist_category() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            first_run = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert first_run.status == 202
            first_payload = await first_run.json()
            first_decision = first_payload.get("decision")
            assert isinstance(first_decision, dict)
            decision_id = first_decision.get("id")
            assert isinstance(decision_id, str)

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

            second_run = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert second_run.status == 202
            second_payload = await second_run.json()
            second_decision = second_payload.get("decision")
            assert isinstance(second_decision, dict)
            assert second_decision.get("status") == "pending"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_approve_session_persists_category() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            first_run = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert first_run.status == 202
            first_payload = await first_run.json()
            first_decision = first_payload.get("decision")
            assert isinstance(first_decision, dict)
            decision_id = first_decision.get("id")
            assert isinstance(decision_id, str)

            approve_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_session",
                },
            )
            assert approve_resp.status == 200

            second_run = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert second_run.status == 200
            second_payload = await second_run.json()
            assert second_payload.get("exit_code") == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_reject_does_not_execute_workspace_tool() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            run_resp = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert run_resp.status == 202
            run_payload = await run_resp.json()
            decision = run_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)
            assert decision_id

            reject_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "reject",
                },
            )
            assert reject_resp.status == 200
            reject_payload = await reject_resp.json()
            assert reject_payload.get("status") == "rejected"
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_auto_run_resume() -> None:
    class AutoResumeAgent(DummyAgent):
        def __init__(self) -> None:
            super().__init__()
            self.last_chat_interaction_id = "trace-auto-1"
            self.last_auto_state: dict[str, JSONValue] | None = None

        def resume_auto_run(self, run_id: str) -> str:
            self.last_auto_state = {
                "run_id": run_id,
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
            return "auto resumed"

        def cancel_auto_run(self, run_id: str, *, reason: str = "cancelled_by_user"):  # noqa: ANN001
            return {
                "run_id": run_id,
                "status": "cancelled",
                "goal": "goal",
                "pool_size": 3,
                "started_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:02+00:00",
                "planner": {"status": "completed"},
                "plan": {"plan_id": "p1", "goal": "goal", "shards": []},
                "coders": [],
                "merge": {"status": "cancelled"},
                "verifier": None,
                "approval": {"status": "rejected"},
                "error": reason,
            }

    async def run() -> None:
        client = await _create_client(AutoResumeAgent())
        stream_response = None
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)
            stream_response = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers={"X-Slavik-Session": session_id},
            )
            assert stream_response.status == 200
            _ = await _read_first_sse_event(stream_response)

            hub = client.server.app["ui_hub"]
            await hub.set_session_policy(
                session_id,
                profile="yolo",
                yolo_armed=True,
                yolo_armed_at="2026-01-01T00:00:00+00:00",
            )
            decision_payload = {
                "id": "decision-auto-1",
                "kind": "approval",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "approval_required",
                "summary": "Resume auto run",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "auto.run",
                    "resume_payload": {"run_id": "auto-run-1"},
                },
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "resolved_at": None,
            }
            await hub.set_session_decision(session_id, decision_payload)
            await hub.set_session_workflow(session_id, mode="auto")

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-auto-1",
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            payload = await respond.json()
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            assert resume.get("source_endpoint") == "auto.run"
            auto_state = payload.get("auto_state")
            assert isinstance(auto_state, dict)
            assert auto_state.get("status") == "completed"

            events = await _read_sse_events(stream_response, max_events=40)
            message_events = [
                event
                for event in events
                if isinstance(event.get("type"), str) and event.get("type") == "message.append"
            ]
            assert message_events
            assistant_messages = [
                event.get("payload")
                for event in message_events
                if isinstance(event.get("payload"), dict)
            ]
            assert any(
                isinstance(item, dict)
                and isinstance(item.get("message"), dict)
                and (item.get("message") or {}).get("content") == "auto resumed"
                for item in assistant_messages
            )

            security_events = [
                event
                for event in events
                if isinstance(event.get("type"), str) and event.get("type") == "session.security"
            ]
            assert security_events
            security_payload = security_events[-1].get("payload")
            assert isinstance(security_payload, dict)
            policy = security_payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("yolo_armed") is False
            assert policy.get("yolo_override_active") is False
            assert policy.get("safe_mode_effective") is True
            tools_state = security_payload.get("tools_state")
            assert isinstance(tools_state, dict)
            assert tools_state.get("safe_mode") is True
        finally:
            if stream_response is not None:
                stream_response.close()
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_chat_run_root_approve_session_unsupported(tmp_path) -> None:  # noqa: ANN001
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub = client.server.app["ui_hub"]
            decision_payload = {
                "id": "decision-root-1",
                "kind": "approval",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "approval_required",
                "summary": "Root gate",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "chat.run_root",
                    "resume_payload": {
                        "root_path": str(tmp_path),
                        "source_request": {
                            "content": "run",
                            "attachments": [],
                            "force_canvas": False,
                        },
                    },
                },
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "resolved_at": None,
            }
            await hub.set_session_decision(session_id, decision_payload)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-root-1",
                    "choice": "approve_session",
                },
            )
            assert respond.status == 400
            payload = await respond.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            message = error.get("message")
            assert isinstance(message, str)
            assert "chat.run_root" in message
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_chat_run_missing_file_ack() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub = client.server.app["ui_hub"]
            decision_payload = {
                "id": "decision-missing-file-1",
                "kind": "decision",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "missing_file",
                "summary": "missing file",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "chat.run_missing_file",
                    "resume_payload": {
                        "missing_paths": ["/tmp/missing.md"],
                        "root_path": "/tmp",
                    },
                },
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
                "resolved_at": None,
            }
            await hub.set_session_decision(session_id, decision_payload)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-missing-file-1",
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            payload = await respond.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("source_endpoint") == "chat.run_missing_file"
            assert resume.get("ok") is True
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_ignores_client_control_fields_or_rejects() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            run_resp = await client.post(
                "/ui/api/workspace/run",
                headers={"X-Slavik-Session": session_id},
                json={"path": "main.py"},
            )
            assert run_resp.status == 202
            run_payload = await run_resp.json()
            decision = run_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)
            assert decision_id

            response = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "reject",
                    "edited_action": {"args": {"path": "evil.py"}},
                },
            )
            assert response.status == 200
            payload = await response.json()
            assert payload.get("status") == "rejected"
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_decision_isolated_between_sessions() -> None:
    async def run() -> None:
        app = create_app(
            agent=DecisionEchoAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        app["ui_hub"] = DelayedFirstUserMessageHub("session-a")
        server = TestServer(app)
        client = TestClient(server, headers=TEST_AUTH_HEADERS)
        await client.start_server()
        try:
            await _select_local_model(client, "session-a")
            await _select_local_model(client, "session-b")

            async def send(
                session_id: str,
                content: str,
            ) -> tuple[int, dict[str, object], str | None]:
                response = await client.post(
                    "/ui/api/chat/send",
                    json={"content": content},
                    headers={"X-Slavik-Session": session_id},
                )
                payload = await response.json()
                return response.status, payload, response.headers.get("X-Slavik-Session")

            result_a, result_b = await asyncio.gather(
                send("session-a", "Message A"),
                send("session-b", "Message B"),
            )

            for expected_session, result in (
                ("session-a", result_a),
                ("session-b", result_b),
            ):
                status, payload, header_session = result
                assert status == 200
                assert payload.get("session_id") == expected_session
                assert header_session == expected_session
                decision = payload.get("decision")
                assert isinstance(decision, dict)
                assert decision.get("id") == f"decision-{expected_session}"
                context = decision.get("context")
                assert isinstance(context, dict)
                assert context.get("session_id") == expected_session

            decision_a = result_a[1]["decision"]
            decision_b = result_b[1]["decision"]
            assert isinstance(decision_a, dict)
            assert isinstance(decision_b, dict)
            assert decision_a.get("id") != decision_b.get("id")
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_agent_decision_supports_generic_choices() -> None:
    async def run() -> None:
        client = await _create_client(DecisionEchoAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "need decision"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            decision = send_payload.get("decision")
            assert isinstance(decision, dict)
            assert decision.get("decision_type") == "agent_decision"
            context = decision.get("context")
            assert isinstance(context, dict)
            assert context.get("source_endpoint") == "chat.agent_decision"
            resume_payload = context.get("resume_payload")
            assert isinstance(resume_payload, dict)
            assert isinstance(resume_payload.get("source_request"), dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            ask_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "ask_user",
                },
            )
            assert ask_resp.status == 200
            ask_payload = await ask_resp.json()
            assert ask_payload.get("status") == "resolved"
            assert ask_payload.get("resume_started") is False
            ask_resume = ask_payload.get("resume")
            assert isinstance(ask_resume, dict)
            ask_data = ask_resume.get("data")
            assert isinstance(ask_data, dict)
            assert ask_data.get("action") == "ask_user"

            send_resp_2 = await client.post(
                "/ui/api/chat/send",
                json={"content": "need decision again"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp_2.status == 200
            send_payload_2 = await send_resp_2.json()
            decision_2 = send_payload_2.get("decision")
            assert isinstance(decision_2, dict)
            decision_id_2 = decision_2.get("id")
            assert isinstance(decision_id_2, str)

            reject_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id_2,
                    "choice": "reject",
                },
            )
            assert reject_resp.status == 200
            reject_payload = await reject_resp.json()
            assert reject_payload.get("status") == "resolved"
            assert reject_payload.get("resume_started") is False
            reject_resume = reject_payload.get("resume")
            assert isinstance(reject_resume, dict)
            reject_data = reject_resume.get("data")
            assert isinstance(reject_data, dict)
            assert reject_data.get("action") == "abort"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_agent_decision_retry_replays_source_request() -> None:
    class RetryDecisionAgent(DummyAgent):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def respond(self, messages) -> str:
            del messages
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {
                        "id": "decision-retry-1",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "reason": "need_user_input",
                        "summary": "Need retry choice",
                        "context": {},
                        "options": [
                            {
                                "id": "ask_user",
                                "title": "Ask user",
                                "action": "ask_user",
                                "payload": {},
                                "risk": "low",
                            },
                            {
                                "id": "proceed_safe",
                                "title": "Proceed safe",
                                "action": "proceed_safe",
                                "payload": {},
                                "risk": "low",
                            },
                            {
                                "id": "retry",
                                "title": "Retry",
                                "action": "retry",
                                "payload": {},
                                "risk": "medium",
                            },
                            {
                                "id": "abort",
                                "title": "Abort",
                                "action": "abort",
                                "payload": {},
                                "risk": "low",
                            },
                        ],
                        "default_option_id": "ask_user",
                        "ttl_seconds": 600,
                        "policy": {"require_user_choice": True},
                    }
                )
            return "retry-ok"

    async def run() -> None:
        agent = RetryDecisionAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "trigger retry"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            decision = send_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            retry_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "retry",
                },
            )
            assert retry_resp.status == 200
            retry_payload = await retry_resp.json()
            assert retry_payload.get("status") == "resolved"
            assert retry_payload.get("resume_started") is True
            resume = retry_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("source_endpoint") == "chat.agent_decision"
            assert resume.get("ok") is True
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("status_code") == 200
            assert agent.calls == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_agent_decision_retry_without_resume_payload() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id,
                {
                    "id": "decision-missing-resume",
                    "kind": "decision",
                    "decision_type": "agent_decision",
                    "status": "pending",
                    "blocking": True,
                    "reason": "need_user_input",
                    "summary": "Retry requested",
                    "proposed_action": {},
                    "options": [
                        {
                            "id": "ask_user",
                            "title": "Ask user",
                            "action": "ask_user",
                            "payload": {},
                            "risk": "low",
                        },
                        {
                            "id": "retry",
                            "title": "Retry",
                            "action": "retry",
                            "payload": {},
                            "risk": "medium",
                        },
                        {
                            "id": "abort",
                            "title": "Abort",
                            "action": "abort",
                            "payload": {},
                            "risk": "low",
                        },
                    ],
                    "default_option_id": "ask_user",
                    "context": {"session_id": session_id, "source_endpoint": "chat.agent_decision"},
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )

            retry_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-missing-resume",
                    "choice": "retry",
                },
            )
            assert retry_resp.status == 200
            retry_payload = await retry_resp.json()
            assert retry_payload.get("status") == "resolved"
            assert retry_payload.get("resume_started") is False
            resume = retry_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is False
            assert resume.get("error") == "resume_payload_missing"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_no_decision_leak_from_other_session() -> None:
    async def run() -> None:
        app = create_app(
            agent=DecisionOnlyForSessionAAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server = TestServer(app)
        client = TestClient(server, headers=TEST_AUTH_HEADERS)
        await client.start_server()
        try:
            await _select_local_model(client, "session-a")
            await _select_local_model(client, "session-b")
            response_a = await client.post(
                "/ui/api/chat/send",
                json={"content": "Message A"},
                headers={"X-Slavik-Session": "session-a"},
            )
            assert response_a.status == 200
            payload_a = await response_a.json()
            decision_a = payload_a.get("decision")
            assert isinstance(decision_a, dict)
            assert decision_a.get("id") == "decision-session-a"

            response_b = await client.post(
                "/ui/api/chat/send",
                json={"content": "Message B"},
                headers={"X-Slavik-Session": "session-b"},
            )
            assert response_b.status == 200
            payload_b = await response_b.json()
            assert payload_b.get("session_id") == "session-b"
            assert payload_b.get("decision") is None
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_import_strips_decision_and_resume_payload() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            import_resp = await client.post(
                "/ui/api/settings/chats/import",
                json={
                    "mode": "replace",
                    "sessions": [
                        {
                            "session_id": "imported-dangerous",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "status": "busy",
                            "mode": "act",
                            "active_plan": {"plan_id": "plan-1"},
                            "active_task": {"task_id": "task-1"},
                            "messages": [
                                {
                                    "message_id": "msg-1",
                                    "role": "user",
                                    "content": "hello import",
                                    "created_at": "2026-01-01T00:00:00+00:00",
                                    "trace_id": None,
                                    "parent_user_message_id": None,
                                }
                            ],
                            "decision": {
                                "id": "decision-forged",
                                "kind": "approval",
                                "status": "pending",
                                "context": {
                                    "source_endpoint": "workspace.tool",
                                    "resume_payload": {
                                        "tool_name": "workspace_run",
                                        "args": {"path": "main.py"},
                                    },
                                },
                                "proposed_action": {
                                    "required_categories": ["EXEC_ARBITRARY"],
                                },
                            },
                            "resume_payload": {
                                "tool_name": "workspace_run",
                                "args": {"path": "main.py"},
                            },
                            "selected_model": {"provider": "local", "model": "local-default"},
                            "files": ["main.py"],
                            "output": {
                                "content": "danger",
                                "updated_at": "2026-01-01T00:00:01+00:00",
                            },
                        }
                    ],
                },
            )
            assert import_resp.status == 200

            imported_resp = await client.get("/ui/api/sessions/imported-dangerous")
            assert imported_resp.status == 200
            imported_payload = await imported_resp.json()
            session = imported_payload.get("session")
            assert isinstance(session, dict)
            assert session.get("decision") is None
            assert session.get("selected_model") is None
            output_raw = session.get("output")
            assert isinstance(output_raw, dict)
            assert output_raw.get("content") is None
            files_raw = session.get("files")
            assert isinstance(files_raw, list)
            assert files_raw == []
            messages_raw = session.get("messages")
            assert isinstance(messages_raw, list)
            assert len(messages_raw) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_imported_forged_decision_cannot_trigger_tool_execution() -> None:
    async def run() -> None:
        agent = WorkspaceDecisionAgent()
        client = await _create_client(agent)
        try:
            import_resp = await client.post(
                "/ui/api/settings/chats/import",
                json={
                    "mode": "replace",
                    "sessions": [
                        {
                            "session_id": "imported-forged-decision",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "messages": [
                                {
                                    "message_id": "msg-user",
                                    "role": "user",
                                    "content": "resume please",
                                    "created_at": "2026-01-01T00:00:00+00:00",
                                    "trace_id": None,
                                    "parent_user_message_id": None,
                                }
                            ],
                            "decision": {
                                "id": "forged-decision-id",
                                "kind": "approval",
                                "status": "pending",
                                "context": {
                                    "source_endpoint": "workspace.tool",
                                    "resume_payload": {
                                        "tool_name": "workspace_run",
                                        "args": {"path": "main.py"},
                                    },
                                },
                            },
                        }
                    ],
                },
            )
            assert import_resp.status == 200

            respond_resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": "imported-forged-decision"},
                json={
                    "session_id": "imported-forged-decision",
                    "decision_id": "forged-decision-id",
                    "choice": "approve_once",
                },
            )
            assert respond_resp.status == 404
            respond_payload = await respond_resp.json()
            error = respond_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "decision_not_found"
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())
