from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_api_requires_bearer_by_default() -> None:
    async def run() -> None:
        app = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token="ui-auth-required", allow_unauth_local=False),
        )
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            response = await client.get("/ui/api/status")
            assert response.status == 401
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "unauthorized"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_policy_rejects_unknown_profile() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id = session_raw.get("session_id")
            assert isinstance(session_id, str)
            status, payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="invalid",
            )
            assert status == 400
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_policy_set_updates_session_state() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id = session_raw.get("session_id")
            assert isinstance(session_id, str)

            status, payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status == 200
            policy = payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "yolo"

            hub: UIHub = client.server.app["ui_hub"]
            stored_policy = await hub.get_session_policy(session_id)
            assert stored_policy.get("profile") == "yolo"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_security_get_and_post_updates_state() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id = session_raw.get("session_id")
            assert isinstance(session_id, str)

            update_resp = await client.post(
                "/ui/api/session/security",
                headers={"X-Slavik-Session": session_id},
                json={
                    "tools": {"state": {"shell": True, "web": True}},
                    "policy": {"profile": "sandbox"},
                },
            )
            assert update_resp.status == 200
            update_payload = await update_resp.json()
            tools_state = update_payload.get("tools_state")
            assert isinstance(tools_state, dict)
            assert tools_state.get("shell") is True
            assert tools_state.get("web") is True
            policy = update_payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "sandbox"

            get_resp = await client.get(
                "/ui/api/session/security",
                headers={"X-Slavik-Session": session_id},
            )
            assert get_resp.status == 200
            get_payload = await get_resp.json()
            get_tools = get_payload.get("tools_state")
            assert isinstance(get_tools, dict)
            assert get_tools.get("shell") is True
            assert get_tools.get("web") is True
            get_policy = get_payload.get("policy")
            assert isinstance(get_policy, dict)
            assert get_policy.get("profile") == "sandbox"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_policy_requires_confirm_for_yolo() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id = session_raw.get("session_id")
            assert isinstance(session_id, str)
            status, payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
            )
            assert status == 400
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "yolo_confirmation_required"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_session_policy_conflict_when_root_incompatible(tmp_path, monkeypatch) -> None:
    async def run() -> None:
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir(parents=True, exist_ok=True)
        inside_home = fake_home / "project"
        inside_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("server.http_api.Path.home", lambda: fake_home)

        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id = session_raw.get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(inside_home))

            status, payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="index",
            )
            assert status == 409
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "policy_root_incompatible"

            policy = await hub.get_session_policy(session_id)
            assert policy.get("profile") == "sandbox"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_status_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            resp = await client.get("/ui/api/status")
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("ok") is True
            session_id = payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            assert resp.headers.get("X-Slavik-Session") == session_id
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            selected_model = await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            response_session_id = payload.get("session_id")
            messages = payload.get("messages", [])
            assert isinstance(response_session_id, str)
            assert response_session_id
            assert isinstance(messages, list)
            assert messages
            assert resp.headers.get("X-Slavik-Session") == response_session_id
            selected = payload.get("selected_model")
            assert isinstance(selected, dict)
            assert selected.get("provider") == "local"
            assert selected.get("model") == selected_model
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            assert display.get("forced") is False
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_web_intent_returns_command_guidance() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "проверь в интернете курс биткоина"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            content = last.get("content")
            assert isinstance(content, str)
            assert "/web <запрос>" in content
            assert "После этого подтвердите approval" in content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_strips_mwv_report_block() -> None:
    async def run() -> None:
        client = await _create_client(UIReportAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            report = payload.get("mwv_report")
            assert isinstance(report, dict)
            assert report.get("route") == "chat"
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            content = last.get("content")
            assert isinstance(content, str)
            assert content == "ok"
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            output_content = output_payload.get("content")
            assert isinstance(output_content, str)
            assert output_content == "ok"
            assert "MWV_REPORT_JSON=" not in output_content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_auto_routes_long_output_to_canvas() -> None:
    async def run() -> None:
        client = await _create_client(LongCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Generate long module"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content == "Статус: результат сформирован в Canvas."
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            assert display.get("forced") is False
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            output_content = output_payload.get("content")
            assert isinstance(output_content, str)
            assert output_content.startswith("```python")
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_short_code_stays_in_chat_without_artifact() -> None:
    async def run() -> None:
        client = await _create_client(ShortCodeAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "React button sample"},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content.startswith("```ts")
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_named_files_create_file_artifacts_and_canvas() -> None:
    async def run() -> None:
        client = await _create_client(NamedFileArtifactsAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Напиши clock.py и clock.sh"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert len(artifacts) == 2
            names: set[str] = set()
            artifact_ids: list[str] = []
            for item in artifacts:
                assert isinstance(item, dict)
                assert item.get("artifact_kind") == "file"
                artifact_id = item.get("id")
                file_name = item.get("file_name")
                file_content = item.get("file_content")
                assert isinstance(artifact_id, str)
                assert isinstance(file_name, str)
                assert isinstance(file_content, str)
                artifact_ids.append(artifact_id)
                names.add(file_name)
            assert names == {"clock.py", "clock.sh"}

            messages = send_payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            assistant_text = last_message.get("content")
            assert isinstance(assistant_text, str)
            assert assistant_text.startswith("Статус: результат сформирован в Canvas")

            download_resp = await client.get(
                f"/ui/api/sessions/{session_id}/artifacts/{artifact_ids[0]}/download",
                headers={"X-Slavik-Session": session_id},
            )
            assert download_resp.status == 200
            first_download_text = await download_resp.text()
            assert "Код (`clock.py`)" not in first_download_text
            assert "```" not in first_download_text
            assert "import time" in first_download_text or "#!/bin/bash" in first_download_text
            content_disposition = download_resp.headers.get("Content-Disposition")
            assert isinstance(content_disposition, str)
            assert "attachment" in content_disposition

            download_all_resp = await client.get(
                f"/ui/api/sessions/{session_id}/artifacts/download-all",
                headers={"X-Slavik-Session": session_id},
            )
            assert download_all_resp.status == 200
            assert "application/zip" in download_all_resp.headers.get("Content-Type", "")
            archive_bytes = await download_all_resp.read()
            with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
                archive_names = set(archive.namelist())
                assert archive_names == {"clock.py", "clock.sh"}
                clock_py = archive.read("clock.py").decode("utf-8")
                clock_sh = archive.read("clock.sh").decode("utf-8")
                assert "import time" in clock_py
                assert "#!/bin/bash" in clock_sh
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_escaped_fence_named_file_artifact() -> None:
    async def run() -> None:
        client = await _create_client(EscapedFenceNamedFileAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Напиши файл clock.py целиком"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert len(artifacts) == 1
            artifact = artifacts[0]
            assert isinstance(artifact, dict)
            assert artifact.get("artifact_kind") == "file"
            assert artifact.get("file_name") == "clock.py"
            file_content = artifact.get("file_content")
            assert isinstance(file_content, str)
            assert "import time" in file_content
            assert "\\n" not in file_content

            artifact_id = artifact.get("id")
            assert isinstance(artifact_id, str)
            download_resp = await client.get(
                f"/ui/api/sessions/{session_id}/artifacts/{artifact_id}/download",
                headers={"X-Slavik-Session": session_id},
            )
            assert download_resp.status == 200
            downloaded = await download_resp.text()
            assert "import time" in downloaded
            assert "```" not in downloaded
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_force_canvas_override() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Short response", "force_canvas": True},
                headers={"X-Slavik-Session": session_id},
            )
            assert resp.status == 200
            payload = await resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last = messages[-1]
            assert isinstance(last, dict)
            last_content = last.get("content")
            assert isinstance(last_content, str)
            assert last_content.startswith("Статус: результат сформирован в Canvas")
            display = payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "canvas"
            assert display.get("forced") is True
            output_payload = payload.get("output")
            assert isinstance(output_payload, dict)
            assert output_payload.get("content") == "ok"
            artifacts = payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_api_create_send_get_history() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            session_id = created.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Ping"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            get_resp = await client.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 200
            get_payload = await get_resp.json()
            session = get_payload.get("session")
            assert isinstance(session, dict)
            messages = session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) >= 2
            first = messages[0]
            second = messages[1]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            assert first.get("role") == "user"
            assert second.get("role") == "assistant"
            assert second.get("content") == "ok"

            history_resp = await client.get(f"/ui/api/sessions/{session_id}/history")
            assert history_resp.status == 200
            history_payload = await history_resp.json()
            history_messages = history_payload.get("messages")
            assert isinstance(history_messages, list)
            assert len(history_messages) == len(messages)

            output_resp = await client.get(f"/ui/api/sessions/{session_id}/output")
            assert output_resp.status == 200
            output_payload = await output_resp.json()
            output = output_payload.get("output")
            assert isinstance(output, dict)
            assert output.get("content") == "ok"
            session_artifacts = session.get("artifacts")
            assert isinstance(session_artifacts, list)
            assert session_artifacts == []

            files_resp = await client.get(f"/ui/api/sessions/{session_id}/files")
            assert files_resp.status == 200
            files_payload = await files_resp.json()
            files = files_payload.get("files")
            assert isinstance(files, list)
            assert files == []

            list_resp = await client.get("/ui/api/sessions")
            assert list_resp.status == 200
            list_payload = await list_resp.json()
            sessions = list_payload.get("sessions")
            assert isinstance(sessions, list)
            selected = next(
                (item for item in sessions if item.get("session_id") == session_id),
                None,
            )
            assert isinstance(selected, dict)
            assert selected.get("message_count") == len(messages)
            assert selected.get("title") == "Ping"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_stores_files_from_tool_calls(monkeypatch, tmp_path) -> None:
    trace_log = tmp_path / "trace.log"
    tool_log = tmp_path / "tool_calls.log"
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_log)
    monkeypatch.setattr("server.http_api.TOOL_CALLS_LOG", tool_log)

    async def run() -> None:
        client = await _create_client(ToolCallCaptureAgent(trace_log, tool_log))
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session = create_payload.get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Generate file"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            output = send_payload.get("output")
            assert isinstance(output, dict)
            assert output.get("content") == "print('ok')"
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            files = send_payload.get("files")
            assert isinstance(files, list)
            assert "src/generated/demo.py" in files
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []

            files_resp = await client.get(f"/ui/api/sessions/{session_id}/files")
            assert files_resp.status == 200
            files_payload = await files_resp.json()
            files_from_endpoint = files_payload.get("files")
            assert isinstance(files_from_endpoint, list)
            assert "src/generated/demo.py" in files_from_endpoint
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_ignores_stale_trace_id(monkeypatch, tmp_path) -> None:
    trace_log = tmp_path / "trace.log"
    tool_log = tmp_path / "tool_calls.log"
    stale_trace_id = "stale-interaction"
    tracer = Tracer(path=trace_log)
    tool_logger = ToolCallLogger(path=tool_log)
    tracer.log("user_input", "old input")
    tool_logger.log(
        "workspace_write",
        ok=True,
        args={"path": "src/generated/old.py", "content": "print('old')"},
    )
    tracer.log(
        "interaction_logged",
        "old interaction done",
        {"interaction_id": stale_trace_id},
    )
    monkeypatch.setattr("server.http_api.TRACE_LOG", trace_log)
    monkeypatch.setattr("server.http_api.TOOL_CALLS_LOG", tool_log)

    async def run() -> None:
        client = await _create_client(StaleTraceIdAgent(stale_trace_id))
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session = create_payload.get("session")
            assert isinstance(session, dict)
            session_id = session.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Simple reply please"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            display = send_payload.get("display")
            assert isinstance(display, dict)
            assert display.get("target") == "chat"
            files = send_payload.get("files")
            assert isinstance(files, list)
            assert files == []
            artifacts = send_payload.get("artifacts")
            assert isinstance(artifacts, list)
            assert artifacts == []
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_api_delete_chat() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            session_id = created.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                json={"content": "Delete me"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200

            delete_resp = await client.delete(f"/ui/api/sessions/{session_id}")
            assert delete_resp.status == 200
            delete_payload = await delete_resp.json()
            assert delete_payload.get("session_id") == session_id
            assert delete_payload.get("deleted") is True

            get_resp = await client.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 404
            get_payload = await get_resp.json()
            error = get_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "session_not_found"

            list_resp = await client.get("/ui/api/sessions")
            assert list_resp.status == 200
            list_payload = await list_resp.json()
            sessions = list_payload.get("sessions")
            assert isinstance(sessions, list)
            assert all(
                item.get("session_id") != session_id for item in sessions if isinstance(item, dict)
            )

            delete_missing_resp = await client.delete(f"/ui/api/sessions/{session_id}")
            assert delete_missing_resp.status == 404
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_sessions_persist_after_restart(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "ui_sessions.db"

        storage_before = SQLiteUISessionStorage(db_path)
        app_before = create_app(
            agent=DecisionEchoAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_before,
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server_before = TestServer(app_before)
        client_before = TestClient(server_before, headers=TEST_AUTH_HEADERS)
        await client_before.start_server()

        session_id: str | None = None
        try:
            create_resp = await client_before.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            created = create_payload.get("session")
            assert isinstance(created, dict)
            raw_session_id = created.get("session_id")
            assert isinstance(raw_session_id, str)
            assert raw_session_id
            session_id = raw_session_id
            selected_model = await _select_local_model(client_before, session_id)

            send_resp = await client_before.post(
                "/ui/api/chat/send",
                json={"content": "Persist me"},
                headers={"X-Slavik-Session": session_id},
            )
            assert send_resp.status == 200
            send_payload = await send_resp.json()
            decision = send_payload.get("decision")
            assert isinstance(decision, dict)
            assert decision.get("id") == f"decision-{session_id}"
        finally:
            await client_before.close()

        assert isinstance(session_id, str)
        storage_after = SQLiteUISessionStorage(db_path)
        app_after = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_after,
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server_after = TestServer(app_after)
        client_after = TestClient(server_after, headers=TEST_AUTH_HEADERS)
        await client_after.start_server()
        try:
            get_resp = await client_after.get(f"/ui/api/sessions/{session_id}")
            assert get_resp.status == 200
            get_payload = await get_resp.json()
            session = get_payload.get("session")
            assert isinstance(session, dict)
            assert session.get("session_id") == session_id
            messages = session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) == 1
            first = messages[0]
            assert isinstance(first, dict)
            assert first.get("role") == "user"
            assert first.get("content") == "Persist me"
            restored_decision = session.get("decision")
            assert isinstance(restored_decision, dict)
            assert restored_decision.get("id") == f"decision-{session_id}"
            restored_output = session.get("output")
            assert isinstance(restored_output, dict)
            assert restored_output.get("content") is None
            restored_files = session.get("files")
            assert isinstance(restored_files, list)
            assert restored_files == []
            restored_artifacts = session.get("artifacts")
            assert isinstance(restored_artifacts, list)
            assert restored_artifacts == []
            restored_model = session.get("selected_model")
            assert isinstance(restored_model, dict)
            assert restored_model.get("provider") == "local"
            assert restored_model.get("model") == selected_model

            status_resp = await client_after.get(
                "/ui/api/status",
                headers={"X-Slavik-Session": session_id},
            )
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            decision_from_status = status_payload.get("decision")
            assert isinstance(decision_from_status, dict)
            assert decision_from_status.get("id") == f"decision-{session_id}"
            selected_from_status = status_payload.get("selected_model")
            assert isinstance(selected_from_status, dict)
            assert selected_from_status.get("provider") == "local"
            assert selected_from_status.get("model") == selected_model
        finally:
            await client_after.close()

    asyncio.run(run())


def test_ui_session_policy_persist_after_restart(tmp_path) -> None:
    async def run() -> None:
        db_path = tmp_path / "ui_sessions.db"

        storage_before = SQLiteUISessionStorage(db_path)
        app_before = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_before,
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server_before = TestServer(app_before)
        client_before = TestClient(server_before, headers=TEST_AUTH_HEADERS)
        await client_before.start_server()
        session_id: str | None = None
        try:
            create_resp = await client_before.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_raw = create_payload.get("session")
            assert isinstance(session_raw, dict)
            session_id_raw = session_raw.get("session_id")
            assert isinstance(session_id_raw, str)
            session_id = session_id_raw
            status, payload = await _set_session_policy_via_api(
                client_before,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status == 200
            policy = payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "yolo"
        finally:
            await client_before.close()

        assert isinstance(session_id, str)
        storage_after = SQLiteUISessionStorage(db_path)
        app_after = create_app(
            agent=DummyAgent(),
            max_request_bytes=1_000_000,
            ui_storage=storage_after,
            auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
        )
        server_after = TestServer(app_after)
        client_after = TestClient(server_after, headers=TEST_AUTH_HEADERS)
        await client_after.start_server()
        try:
            hub: UIHub = client_after.server.app["ui_hub"]
            restored_policy = await hub.get_session_policy(session_id)
            assert restored_policy.get("profile") == "yolo"
        finally:
            await client_after.close()

    asyncio.run(run())


def test_user_plane_security_mutators_always_forbidden(monkeypatch, tmp_path) -> None:
    tools_path = tmp_path / "tools.json"
    monkeypatch.setattr(
        "server.http_api.load_tools_config",
        lambda: load_tools_config_from_path(path=tools_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_tools_config",
        lambda config: save_tools_config_to_path(config, path=tools_path),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            payloads = [
                {"tools": {"state": {"shell": True}}},
                {"tools": {"state": {"safe_mode": False, "web": True}}},
                {"tools": {"state": {"web": True}}},
                {"policy": {"profile": "index"}},
                {"safe_mode": False},
                {"risk": {"categories": ["EXEC_ARBITRARY"]}},
                {"security": {"categories": ["EXEC_ARBITRARY"]}},
            ]
            for body in payloads:
                response = await client.post("/ui/api/settings", json=body)
                assert response.status == 403
                response_payload = await response.json()
                error = response_payload.get("error")
                assert isinstance(error, dict)
                assert error.get("code") == "security_fields_forbidden"

            settings_resp = await client.get("/ui/api/settings")
            assert settings_resp.status == 200
            settings_payload = await settings_resp.json()
            settings_raw = settings_payload.get("settings")
            assert isinstance(settings_raw, dict)
            tools_raw = settings_raw.get("tools")
            assert isinstance(tools_raw, dict)
            state_raw = tools_raw.get("state")
            assert isinstance(state_raw, dict)
            assert state_raw.get("safe_mode") is True
            assert state_raw.get("shell") is False
            assert state_raw.get("web") is False
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_accepts_optional_attachments() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={
                    "content": "",
                    "attachments": [
                        {
                            "name": "clock.py",
                            "mime": "text/x-python",
                            "content": "print('ok')",
                        }
                    ],
                },
            )
            assert send_resp.status == 200
            payload = await send_resp.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert len(messages) == 2
            user_message = messages[0]
            assert isinstance(user_message, dict)
            attachments = user_message.get("attachments")
            assert isinstance(attachments, list)
            assert len(attachments) == 1
            attachment = attachments[0]
            assert isinstance(attachment, dict)
            assert attachment.get("name") == "clock.py"

            history_resp = await client.get(f"/ui/api/sessions/{session_id}/history")
            assert history_resp.status == 200
            history_payload = await history_resp.json()
            history_messages = history_payload.get("messages")
            assert isinstance(history_messages, list)
            first = history_messages[0]
            assert isinstance(first, dict)
            history_attachments = first.get("attachments")
            assert isinstance(history_attachments, list)
            assert len(history_attachments) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_rejects_oversized_attachment() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={
                    "content": "small",
                    "attachments": [
                        {
                            "name": "large.txt",
                            "mime": "text/plain",
                            "content": "x" * 80001,
                        }
                    ],
                },
            )
            assert send_resp.status == 413
            payload = await send_resp.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "payload_too_large"
        finally:
            await client.close()

    asyncio.run(run())


def test_foreign_session_returns_403_consistently(monkeypatch) -> None:
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
            session_id = "owner-session-pr3-consistency"
            status_resp = await client.get(
                "/ui/api/status",
                headers={"X-Slavik-Session": session_id},
            )
            assert status_resp.status == 200
            model_id = await _select_local_model(client, session_id)

            foreign_headers = {
                "Authorization": "Bearer secondary-principal-token",
                "X-Slavik-Session": session_id,
            }
            foreign_payload = {
                "Authorization": "Bearer secondary-principal-token",
            }

            session_get_resp = await client.get(
                f"/ui/api/sessions/{session_id}",
                headers=foreign_payload,
            )
            await assert_forbidden(session_get_resp)

            title_resp = await client.patch(
                f"/ui/api/sessions/{session_id}/title",
                headers=foreign_payload,
                json={"title": "hijacked"},
            )
            await assert_forbidden(title_resp)

            folder_resp = await client.put(
                f"/ui/api/sessions/{session_id}/folder",
                headers=foreign_payload,
                json={"folder_id": None},
            )
            await assert_forbidden(folder_resp)

            session_model_resp = await client.post(
                "/ui/api/session-model",
                headers=foreign_headers,
                json={"provider": "local", "model": model_id},
            )
            await assert_forbidden(session_model_resp)

            chat_resp = await client.post(
                "/ui/api/chat/send",
                headers=foreign_headers,
                json={"content": "foreign request"},
            )
            await assert_forbidden(chat_resp)

            project_resp = await client.post(
                "/ui/api/tools/project",
                headers=foreign_headers,
                json={"command": "find", "args": "README.md"},
            )
            await assert_forbidden(project_resp)
        finally:
            await client.close()

    asyncio.run(run())
