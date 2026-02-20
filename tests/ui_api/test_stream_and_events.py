from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_chat_send_canvas_stream_emits_chat_status_and_canvas_stream() -> None:
    async def run() -> None:
        client = await _create_client(StreamNamedFileArtifactsAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "napiši mini app clock.py celikom"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"

            events = await _read_sse_events(stream_resp, max_events=64)
            event_types = [
                event.get("type") for event in events if isinstance(event.get("type"), str)
            ]
            assert "canvas.stream.start" in event_types
            assert "canvas.stream.delta" in event_types
            assert "canvas.stream.done" in event_types
            assert "chat.stream.start" in event_types
            assert "chat.stream.delta" in event_types
            assert "chat.stream.done" in event_types
            chat_deltas: list[str] = []
            for event in events:
                if event.get("type") != "chat.stream.delta":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                delta = payload.get("delta")
                if isinstance(delta, str):
                    chat_deltas.append(delta)
            assert chat_deltas
            assert all("import time" not in delta for delta in chat_deltas)
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_canvas_keeps_status_when_stream_started_before_canvas_detection() -> None:
    async def run() -> None:
        client = await _create_client(LateNamedFileStreamAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Сделай пример"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            messages = send_payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            assistant_text = last_message.get("content")
            assert isinstance(assistant_text, str)
            assert assistant_text.startswith("Статус: результат сформирован в Canvas")
            assert "import time" not in assistant_text
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
            first = artifacts[0]
            assert isinstance(first, dict)
            assert first.get("artifact_kind") == "file"
            assert first.get("file_name") == "clock.py"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_first_event_is_status() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            first_event = await _read_first_sse_event(stream_resp)
            assert first_event.get("type") == "status"
            payload = first_event.get("payload")
            assert isinstance(payload, dict)
            assert payload.get("session_id") == session_id
            assert payload.get("ok") is True
            assert payload.get("state") == "ok"
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_includes_agent_activity() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping with activity"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            event_types = await _read_sse_event_types(stream_resp, max_events=16)
            assert "agent.activity" in event_types
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_status_progress_is_transient_and_not_message() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "check transient status"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            messages_raw = send_payload.get("messages")
            assert isinstance(messages_raw, list)
            message_contents = [
                item.get("content")
                for item in messages_raw
                if isinstance(item, dict) and isinstance(item.get("content"), str)
            ]
            status_events = await _read_sse_events(stream_resp, max_events=32)
            progress_events = [
                event
                for event in status_events
                if event.get("type") == "status"
                and isinstance(event.get("payload"), dict)
                and isinstance((event.get("payload") or {}).get("phase"), str)
            ]
            assert progress_events
            assert any(
                (event.get("payload") or {}).get("transient") is True
                for event in progress_events
                if isinstance(event.get("payload"), dict)
            )
            assert "Готовлю ответ…" not in message_contents
            assert "Собираю контекст…" not in message_contents
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_includes_canvas_stream_events() -> None:
    async def run() -> None:
        client = await _create_client(LongCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Stream this long output"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            event_types = await _read_sse_event_types(stream_resp, max_events=64)
            assert "canvas.stream.start" in event_types
            assert "canvas.stream.delta" in event_types
            assert "canvas.stream.done" in event_types
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_includes_chat_stream_events() -> None:
    async def run() -> None:
        client = await _create_client(TracedStreamingAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Stream this chat output"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            messages = send_payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            assert len(messages) == 2
            user_message = messages[0]
            assistant_message = messages[-1]
            assert isinstance(user_message, dict)
            assert isinstance(assistant_message, dict)
            assert assistant_message.get("content") == "Hello stream"
            assert isinstance(user_message.get("message_id"), str)
            assert isinstance(assistant_message.get("message_id"), str)
            assert isinstance(user_message.get("created_at"), str)
            assert isinstance(assistant_message.get("created_at"), str)
            assert user_message.get("trace_id") is None
            assistant_trace = assistant_message.get("trace_id")
            assert isinstance(assistant_trace, str)
            assert assistant_trace.startswith("stream-trace-")
            assert assistant_message.get("parent_user_message_id") == user_message.get("message_id")
            assistant_messages = [
                item
                for item in messages
                if isinstance(item, dict) and item.get("role") == "assistant"
            ]
            assert len(assistant_messages) == 1

            event_types = await _read_sse_event_types(stream_resp, max_events=64)
            assert "chat.stream.start" in event_types
            assert "chat.stream.delta" in event_types
            assert "chat.stream.done" in event_types
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_stream_persists_assistant_only_after_stream_done() -> None:
    async def run() -> None:
        client = await _create_client(TracedStreamingAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)

            send_task = asyncio.create_task(
                client.post(
                    "/ui/api/chat/send",
                    json={"content": "Stream this chat output"},
                    headers={"X-Slavik-Session": session_id},
                ),
            )

            saw_delta = False
            while not saw_delta:
                event = await _read_first_sse_event(stream_resp)
                event_type = event.get("type")
                if event_type != "chat.stream.delta":
                    continue
                saw_delta = True

            history_mid_resp = await client.get(f"/ui/api/sessions/{session_id}/history")
            assert history_mid_resp.status == 200
            history_mid_payload = await history_mid_resp.json()
            history_mid = history_mid_payload.get("messages")
            assert isinstance(history_mid, list)
            assert len(history_mid) == 1
            only = history_mid[0]
            assert isinstance(only, dict)
            assert only.get("role") == "user"

            send_resp = await send_task
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            messages = send_payload.get("messages")
            assert isinstance(messages, list)
            assert len(messages) == 2
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_session_ownership_enforced_for_stream_workspace_decision_delete_files_output_history(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SLAVIK_ADMIN_TOKEN", "secondary-principal-token")

    async def assert_forbidden(response) -> None:
        assert response.status == 403
        payload = await response.json()
        error = payload.get("error")
        assert isinstance(error, dict)
        assert error.get("code") == "session_forbidden"

    async def run() -> None:
        app = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server = TestServer(app)
        client = TestClient(server, headers=TEST_AUTH_HEADERS)
        await client.start_server()
        try:
            session_id = "owner-session-pr3"
            status_resp = await client.get(
                "/ui/api/status",
                headers={"X-Slavik-Session": session_id},
            )
            assert status_resp.status == 200
            await _select_local_model(client, session_id)

            foreign_headers = {
                "Authorization": "Bearer secondary-principal-token",
                "X-Slavik-Session": session_id,
            }
            foreign_payload = {
                "Authorization": "Bearer secondary-principal-token",
            }

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers=foreign_headers,
            )
            await assert_forbidden(stream_resp)

            workspace_resp = await client.get(
                "/ui/api/workspace/tree",
                headers=foreign_headers,
            )
            await assert_forbidden(workspace_resp)

            decision_resp = await client.post(
                "/ui/api/decision/respond",
                headers=foreign_payload,
                json={
                    "session_id": session_id,
                    "decision_id": "non-existent-decision",
                    "choice": "reject",
                },
            )
            await assert_forbidden(decision_resp)
            runtime_decision_resp = await client.post(
                "/ui/api/decision/runtime/respond",
                headers=foreign_payload,
                json={
                    "session_id": session_id,
                    "decision_id": "non-existent-decision",
                    "action": "abort",
                },
            )
            await assert_forbidden(runtime_decision_resp)

            delete_resp = await client.delete(
                f"/ui/api/sessions/{session_id}",
                headers=foreign_payload,
            )
            await assert_forbidden(delete_resp)

            files_resp = await client.get(
                f"/ui/api/sessions/{session_id}/files",
                headers=foreign_payload,
            )
            await assert_forbidden(files_resp)

            download_resp = await client.get(
                f"/ui/api/sessions/{session_id}/files/download?path=main.py",
                headers=foreign_payload,
            )
            await assert_forbidden(download_resp)

            output_resp = await client.get(
                f"/ui/api/sessions/{session_id}/output",
                headers=foreign_payload,
            )
            await assert_forbidden(output_resp)

            history_resp = await client.get(
                f"/ui/api/sessions/{session_id}/history",
                headers=foreign_payload,
            )
            await assert_forbidden(history_resp)
        finally:
            await client.close()

    asyncio.run(run())
