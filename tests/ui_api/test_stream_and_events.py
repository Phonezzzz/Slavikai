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
            chat_lanes: set[str] = set()
            for event in events:
                event_type = event.get("type")
                if not isinstance(event_type, str) or not event_type.startswith("chat.stream."):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                lane_raw = payload.get("lane")
                if isinstance(lane_raw, str):
                    chat_lanes.add(lane_raw)
                if event_type != "chat.stream.delta":
                    continue
                delta = payload.get("delta")
                if isinstance(delta, str):
                    chat_deltas.append(delta)
            assert chat_deltas
            assert chat_lanes == {"chat"}
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

            events = await _read_sse_events(stream_resp, max_events=64)
            event_types = [
                event.get("type") for event in events if isinstance(event.get("type"), str)
            ]
            assert "chat.stream.start" in event_types
            assert "chat.stream.delta" in event_types
            assert "chat.stream.done" in event_types
            stream_lanes: set[str] = set()
            for event in events:
                event_type = event.get("type")
                if not isinstance(event_type, str) or not event_type.startswith("chat.stream."):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                lane = payload.get("lane")
                if isinstance(lane, str):
                    stream_lanes.add(lane)
            assert stream_lanes == {"chat"}
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_workspace_lane_has_chat_stream_without_canvas() -> None:
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
                json={"content": "Generate long module", "lane": "workspace"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            assert send_payload.get("lane") == "workspace"
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []

            events = await _read_sse_events(stream_resp, max_events=64)
            event_types = [
                event.get("type") for event in events if isinstance(event.get("type"), str)
            ]
            assert "chat.stream.start" in event_types
            assert "chat.stream.delta" in event_types
            assert "chat.stream.done" in event_types
            assert "canvas.stream.start" not in event_types
            assert "canvas.stream.delta" not in event_types
            assert "canvas.stream.done" not in event_types
            lanes: set[str] = set()
            for event in events:
                event_type = event.get("type")
                if not isinstance(event_type, str) or not event_type.startswith("chat.stream."):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                lane = payload.get("lane")
                if isinstance(lane, str):
                    lanes.add(lane)
            assert lanes == {"workspace"}
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


def test_workspace_stream_supports_replace_mode_chunks() -> None:
    class ReplaceStreamAgent(DummyAgent):
        def respond_stream(self, messages):  # noqa: ANN001
            del messages
            yield {"text": "hel", "mode": "replace"}
            yield {"text": "hello", "mode": "replace"}
            self.last_stream_response_raw = "hello"

        def respond(self, messages) -> str:  # noqa: ANN001
            del messages
            return "hello"

    async def run() -> None:
        client = await _create_client(ReplaceStreamAgent())
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
                json={"content": "workspace stream", "lane": "workspace"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            events = await _read_sse_events(stream_resp, max_events=32)
            replace_payloads = []
            for event in events:
                if event.get("type") != "chat.stream.delta":
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                if payload.get("mode") == "replace":
                    replace_payloads.append(payload)
            assert replace_payloads
            assert all(item.get("lane") == "workspace" for item in replace_payloads)
            assert replace_payloads[-1].get("delta") == "hello"
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_hub_subscriber_queue_is_bounded_and_coalesces_deltas() -> None:
    async def run() -> None:
        hub = UIHub(subscriber_queue_maxsize=2, subscriber_drop_policy="coalesce_deltas")
        session_id = "queue-session"
        queue = await hub.subscribe(session_id)

        await hub.publish(
            session_id,
            {
                "type": "chat.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "stream_id": "stream-1",
                    "delta": "A",
                    "mode": "append",
                    "lane": "chat",
                },
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "chat.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "stream_id": "stream-1",
                    "delta": "B",
                    "mode": "append",
                    "lane": "chat",
                },
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "chat.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "stream_id": "stream-1",
                    "delta": "C",
                    "mode": "append",
                    "lane": "chat",
                },
            },
        )
        assert queue.qsize() == 2
        queued_after_coalesce = list(queue._queue)
        assert len(queued_after_coalesce) == 2
        latest_event = queued_after_coalesce[-1]
        assert isinstance(latest_event, dict)
        assert latest_event.get("type") == "chat.stream.delta"
        latest_payload = latest_event.get("payload")
        assert isinstance(latest_payload, dict)
        assert latest_payload.get("delta") == "BC"

        await hub.publish(
            session_id,
            {
                "type": "chat.stream.done",
                "payload": {"session_id": session_id, "stream_id": "stream-1", "lane": "chat"},
            },
        )
        assert queue.qsize() == 2
        types_after_done = [
            item.get("type") for item in list(queue._queue) if isinstance(item, dict)
        ]
        assert "chat.stream.done" in types_after_done

        await hub.publish(
            session_id,
            {
                "type": "status",
                "payload": {"session_id": session_id, "state": "ok", "ok": True},
            },
        )
        assert queue.qsize() == 2
        final_types = [item.get("type") for item in list(queue._queue) if isinstance(item, dict)]
        assert "chat.stream.done" in final_types
        assert "status" in final_types

    asyncio.run(run())


def test_ui_hub_overflow_control_event_marks_resync_when_queue_has_only_control() -> None:
    async def run() -> None:
        hub = UIHub(subscriber_queue_maxsize=2, subscriber_drop_policy="coalesce_deltas")
        session_id = "control-overflow-session"
        queue = await hub.subscribe(session_id)

        await hub.publish(
            session_id,
            {
                "type": "status",
                "payload": {"session_id": session_id, "state": "ok", "ok": True},
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "session.workflow",
                "payload": {"session_id": session_id, "mode": "ask"},
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "decision.packet",
                "payload": {
                    "session_id": session_id,
                    "decision": {"id": "d-1", "status": "pending"},
                },
            },
        )

        # Освобождаем место и триггерим следующую публикацию, чтобы pending resync был доставлен.
        _ = await queue.get()
        await hub.publish(
            session_id,
            {
                "type": "agent.activity",
                "payload": {"session_id": session_id, "phase": "heartbeat"},
            },
        )
        queued_types = [item.get("type") for item in list(queue._queue) if isinstance(item, dict)]
        assert "session.resync_required" in queued_types

    asyncio.run(run())


def test_ui_hub_overflow_keeps_control_event_by_evicting_non_control() -> None:
    async def run() -> None:
        hub = UIHub(subscriber_queue_maxsize=2, subscriber_drop_policy="coalesce_deltas")
        session_id = "control-priority-session"
        queue = await hub.subscribe(session_id)

        await hub.publish(
            session_id,
            {
                "type": "chat.stream.delta",
                "payload": {
                    "session_id": session_id,
                    "stream_id": "s-1",
                    "delta": "A",
                    "mode": "append",
                    "lane": "chat",
                },
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "chat.stream.done",
                "payload": {"session_id": session_id, "stream_id": "s-1", "lane": "chat"},
            },
        )
        await hub.publish(
            session_id,
            {
                "type": "status",
                "payload": {"session_id": session_id, "state": "busy", "ok": True},
            },
        )

        queued_types = [item.get("type") for item in list(queue._queue) if isinstance(item, dict)]
        assert "status" in queued_types

    asyncio.run(run())


def test_ui_events_stream_replays_buffer_by_last_event_id() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)
            _ = await _read_sse_events(stream_resp, max_events=1)  # initial workflow

            await hub.publish(
                session_id,
                {"type": "agent.activity", "payload": {"session_id": session_id, "phase": "one"}},
            )
            await hub.publish(
                session_id,
                {"type": "agent.activity", "payload": {"session_id": session_id, "phase": "two"}},
            )
            await hub.publish(
                session_id,
                {"type": "agent.activity", "payload": {"session_id": session_id, "phase": "three"}},
            )
            first_live_events = await _read_sse_events(stream_resp, max_events=1)
            assert first_live_events
            anchor_id_raw = first_live_events[0].get("id")
            assert isinstance(anchor_id_raw, str)
            anchor_id = anchor_id_raw
            stream_resp.close()

            reconnect = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers={"Last-Event-ID": anchor_id},
                timeout=5,
            )
            assert reconnect.status == 200
            _ = await _read_first_sse_event(reconnect)
            replay_batch = await _read_sse_events(reconnect, max_events=8)
            replay_phases = []
            for event in replay_batch:
                if event.get("type") != "agent.activity":
                    continue
                payload = event.get("payload")
                if isinstance(payload, dict):
                    phase = payload.get("phase")
                    if isinstance(phase, str):
                        replay_phases.append(phase)
            assert "two" in replay_phases
            assert "three" in replay_phases
            reconnect.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_events_stream_stale_last_event_id_emits_resync_required() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)

            stream_resp = await client.get(
                f"/ui/api/events/stream?session_id={session_id}",
                headers={"Last-Event-ID": "stale-event-id"},
                timeout=5,
            )
            assert stream_resp.status == 200
            _ = await _read_first_sse_event(stream_resp)
            events = await _read_sse_events(stream_resp, max_events=8)
            resync_events = [
                event
                for event in events
                if isinstance(event, dict) and event.get("type") == "session.resync_required"
            ]
            assert resync_events
            payload = resync_events[0].get("payload")
            assert isinstance(payload, dict)
            assert payload.get("resync_only") is True
            assert payload.get("reason") == "last_event_id_stale"
            stream_resp.close()
        finally:
            await client.close()

    asyncio.run(run())
