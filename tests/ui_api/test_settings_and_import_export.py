from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_settings_unauthorized_logs_snapshot(monkeypatch) -> None:
    records: list[dict[str, object]] = []

    def _capture_warning(message: str, *args, **kwargs) -> None:
        del args
        extra = kwargs.get("extra")
        if isinstance(extra, dict):
            records.append(dict(extra))
        else:
            records.append({"message": message})

    monkeypatch.setattr("server.http_api.logger.warning", _capture_warning)

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
            response = await client.get("/ui/api/settings")
            assert response.status == 401
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "unauthorized"
        finally:
            await client.close()

    asyncio.run(run())
    assert records
    assert any("settings_snapshot" in item or "settings_snapshot_error" in item for item in records)


def test_ui_settings_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/settings")
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            personalization = settings.get("personalization")
            assert isinstance(personalization, dict)
            assert isinstance(personalization.get("tone"), str)
            assert isinstance(personalization.get("system_prompt"), str)
            composer = settings.get("composer")
            assert isinstance(composer, dict)
            assert isinstance(composer.get("long_paste_to_file_enabled"), bool)
            assert isinstance(composer.get("long_paste_threshold_chars"), int)
            memory = settings.get("memory")
            assert isinstance(memory, dict)
            assert isinstance(memory.get("auto_save_dialogue"), bool)
            assert isinstance(memory.get("inbox_max_items"), int)
            embeddings = memory.get("embeddings")
            assert isinstance(embeddings, dict)
            assert embeddings.get("provider") in {"local", "openai"}
            assert isinstance(embeddings.get("local_model"), str)
            assert isinstance(embeddings.get("openai_model"), str)
            tools = settings.get("tools")
            assert isinstance(tools, dict)
            state = tools.get("state")
            assert isinstance(state, dict)
            assert "safe_mode" in state
            providers = settings.get("providers")
            assert isinstance(providers, list)
            provider_names = {item.get("provider") for item in providers if isinstance(item, dict)}
            assert provider_names == {"local", "openrouter", "xai", "openai"}
            for provider in providers:
                assert isinstance(provider, dict)
                assert "api_key_value" not in provider
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_update_endpoint(monkeypatch, tmp_path) -> None:
    memory_path = tmp_path / "memory.json"
    tools_path = tmp_path / "tools.json"
    ui_settings_path = tmp_path / "ui_settings.json"

    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api.load_memory_config",
        lambda: load_memory_config_from_path(path=memory_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_memory_config",
        lambda config: save_memory_config_to_path(config, path=memory_path),
    )
    monkeypatch.setattr(
        "server.http_api.load_tools_config",
        lambda: load_tools_config_from_path(path=tools_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_tools_config",
        lambda config: save_tools_config_to_path(config, path=tools_path),
    )
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={
                    "personalization": {
                        "tone": "direct",
                        "system_prompt": "Focus on concise implementation details.",
                    },
                    "composer": {
                        "long_paste_to_file_enabled": False,
                        "long_paste_threshold_chars": 20000,
                    },
                    "memory": {
                        "auto_save_dialogue": True,
                        "inbox_max_items": 77,
                        "inbox_ttl_days": 14,
                        "inbox_writes_per_minute": 9,
                        "embeddings": {
                            "provider": "local",
                            "local_model": "test-embeddings-v1",
                            "openai_model": "text-embedding-3-small",
                        },
                    },
                },
            )
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            personalization = settings.get("personalization")
            assert isinstance(personalization, dict)
            assert personalization.get("tone") == "direct"
            assert (
                personalization.get("system_prompt") == "Focus on concise implementation details."
            )
            composer = settings.get("composer")
            assert isinstance(composer, dict)
            assert composer.get("long_paste_to_file_enabled") is False
            assert composer.get("long_paste_threshold_chars") == 20000
            memory = settings.get("memory")
            assert isinstance(memory, dict)
            assert memory.get("auto_save_dialogue") is True
            assert memory.get("inbox_max_items") == 77
            assert memory.get("inbox_ttl_days") == 14
            assert memory.get("inbox_writes_per_minute") == 9
            embeddings = memory.get("embeddings")
            assert isinstance(embeddings, dict)
            assert embeddings.get("provider") == "local"
            assert embeddings.get("local_model") == "test-embeddings-v1"
            assert embeddings.get("openai_model") == "text-embedding-3-small"
            tools = settings.get("tools")
            assert isinstance(tools, dict)
            state = tools.get("state")
            assert isinstance(state, dict)
            assert state.get("safe_mode") is True
            assert state.get("web") is False
            providers = settings.get("providers")
            assert isinstance(providers, list)
            provider_by_name = {
                item.get("provider"): item for item in providers if isinstance(item, dict)
            }
            xai_provider = provider_by_name.get("xai")
            assert isinstance(xai_provider, dict)
            assert xai_provider.get("api_key_set") is False
            assert xai_provider.get("api_key_source") == "missing"
            assert "api_key_value" not in xai_provider
            openrouter_provider = provider_by_name.get("openrouter")
            assert isinstance(openrouter_provider, dict)
            assert openrouter_provider.get("api_key_set") is False
            assert openrouter_provider.get("api_key_source") == "missing"
            assert "api_key_value" not in openrouter_provider

            saved_payload = json.loads(ui_settings_path.read_text(encoding="utf-8"))
            assert isinstance(saved_payload, dict)
            assert "providers" not in saved_payload
            openai_provider = provider_by_name.get("openai")
            assert isinstance(openai_provider, dict)
            assert openai_provider.get("api_key_set") is False
            assert openai_provider.get("api_key_source") == "missing"
            assert "api_key_value" not in openai_provider
        finally:
            await client.close()

    asyncio.run(run())


def test_user_plane_settings_allows_only_whitelisted_fields(monkeypatch) -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            allowed_resp = await client.post(
                "/ui/api/settings",
                json={
                    "personalization": {"tone": "strict"},
                    "composer": {"long_paste_threshold_chars": 25000},
                    "memory": {"inbox_max_items": 101},
                },
            )
            assert allowed_resp.status == 200

            forbidden_resp = await client.post(
                "/ui/api/settings",
                json={"non_security_unknown_field": {"value": True}},
            )
            assert forbidden_resp.status == 403
            forbidden_payload = await forbidden_resp.json()
            error = forbidden_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "security_fields_forbidden"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_update_rejects_providers_payload() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={"providers": {"xai": {"api_key": "xai-secret"}}},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_update_drops_legacy_provider_api_keys(tmp_path, monkeypatch) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text(
        json.dumps(
            {
                "providers": {
                    "xai": {"api_key": "legacy-xai"},
                    "openrouter": {"api_key": "legacy-openrouter"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={"personalization": {"tone": "strict"}},
            )
            assert response.status == 200
            persisted = json.loads(ui_settings_path.read_text(encoding="utf-8"))
            assert isinstance(persisted, dict)
            assert "providers" not in persisted
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_no_api_key_leak(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text(
        json.dumps(
            {
                "providers": {
                    "xai": {"api_key": "xai-secret-key"},
                    "openrouter": {"api_key": "or-secret-key"},
                    "openai": {"api_key": "openai-secret-key"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/settings")
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            providers = settings.get("providers")
            assert isinstance(providers, list)
            for provider in providers:
                assert isinstance(provider, dict)
                assert "api_key_value" not in provider
                source = provider.get("api_key_source")
                assert source in {"env", "missing"}
        finally:
            await client.close()

    asyncio.run(run())


def test_control_plane_security_settings_applies_tools_live(monkeypatch, tmp_path) -> None:
    tools_path = tmp_path / "tools.json"
    monkeypatch.setenv("SLAVIK_ADMIN_TOKEN", "admin-secret")
    monkeypatch.setattr(
        "server.http_api.load_tools_config",
        lambda: load_tools_config_from_path(path=tools_path),
    )
    monkeypatch.setattr(
        "server.http_api.save_tools_config",
        lambda config: save_tools_config_to_path(config, path=tools_path),
    )

    async def run() -> None:
        agent = LiveToolsAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            assert status_resp.status == 200
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            await _enter_act_mode(client, session_id, goal="live web tool settings")

            disabled_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "/web btc price"},
            )
            assert disabled_resp.status == 200
            disabled_payload = await disabled_resp.json()
            disabled_messages = disabled_payload.get("messages")
            assert isinstance(disabled_messages, list)
            assert disabled_messages
            disabled_last = disabled_messages[-1]
            assert isinstance(disabled_last, dict)
            disabled_text = disabled_last.get("content")
            assert isinstance(disabled_text, str)
            assert "Инструмент web отключён" in disabled_text

            settings_resp = await client.post(
                "/slavik/admin/settings/security",
                headers={"Authorization": "Bearer admin-secret"},
                json={"tools": {"state": {"web": True}}},
            )
            assert settings_resp.status == 200
            settings_payload = await settings_resp.json()
            settings = settings_payload.get("settings")
            assert isinstance(settings, dict)
            tools = settings.get("tools")
            assert isinstance(tools, dict)
            state = tools.get("state")
            assert isinstance(state, dict)
            assert state.get("web") is True
            assert agent.tools_enabled.get("web") is True
            assert agent.update_calls
            assert agent.update_calls[-1].get("web") is True

            enabled_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "/web btc price"},
            )
            assert enabled_resp.status == 200
            enabled_payload = await enabled_resp.json()
            enabled_messages = enabled_payload.get("messages")
            assert isinstance(enabled_messages, list)
            assert enabled_messages
            enabled_last = enabled_messages[-1]
            assert isinstance(enabled_last, dict)
            enabled_text = enabled_last.get("content")
            assert isinstance(enabled_text, str)
            assert enabled_text == "WEB_OK"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_settings_update_rejects_invalid_composer_threshold() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.post(
                "/ui/api/settings",
                json={"composer": {"long_paste_threshold_chars": 10}},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_settings_update_applies_embeddings_config_live(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-test-key")

    async def run() -> None:
        agent = LiveEmbeddingsAgent()
        client = await _create_client(agent)
        try:
            response = await client.post(
                "/ui/api/settings",
                json={
                    "memory": {
                        "embeddings": {
                            "provider": "openai",
                            "local_model": "all-MiniLM-L6-v2",
                            "openai_model": "text-embedding-3-small",
                        }
                    },
                },
            )
            assert response.status == 200
            payload = await response.json()
            settings = payload.get("settings")
            assert isinstance(settings, dict)
            memory = settings.get("memory")
            assert isinstance(memory, dict)
            embeddings = memory.get("embeddings")
            assert isinstance(embeddings, dict)
            assert embeddings.get("provider") == "openai"
            assert embeddings.get("local_model") == "all-MiniLM-L6-v2"
            assert embeddings.get("openai_model") == "text-embedding-3-small"
            assert agent.embeddings_provider == "openai"
            assert agent.embeddings_openai_model == "text-embedding-3-small"
            assert agent.set_calls
            assert agent.set_calls[-1] == {
                "provider": "openai",
                "local_model": "all-MiniLM-L6-v2",
                "openai_model": "text-embedding-3-small",
                "openai_api_key": "openai-env-test-key",
            }
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_memory_conflicts_endpoints() -> None:
    async def run() -> None:
        client = await _create_client(MemoryConflictAgent())
        try:
            list_resp = await client.get("/ui/api/memory/conflicts")
            assert list_resp.status == 200
            list_payload = await list_resp.json()
            conflicts = list_payload.get("conflicts")
            assert isinstance(conflicts, list)
            assert len(conflicts) == 1
            first = conflicts[0]
            assert isinstance(first, dict)
            assert first.get("stable_key") == "policy:avoid_emoji"

            resolve_resp = await client.post(
                "/ui/api/memory/conflicts/resolve",
                json={"stable_key": "policy:avoid_emoji", "action": "activate"},
            )
            assert resolve_resp.status == 200
            resolve_payload = await resolve_resp.json()
            resolved = resolve_payload.get("resolved")
            assert isinstance(resolved, dict)
            assert resolved.get("status") == "active"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_stt_transcribe_success(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-test-key")
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api.requests.post",
        lambda *args, **kwargs: FakeSttResponse(200, {"text": "привет мир"}),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            data = FormData()
            data.add_field(
                "audio",
                b"fake-audio",
                filename="recording.webm",
                content_type="audio/webm",
            )
            data.add_field("language", "ru")
            response = await client.post("/ui/api/stt/transcribe", data=data)
            assert response.status == 200
            payload = await response.json()
            assert payload.get("text") == "привет мир"
            assert payload.get("model") == "whisper-1"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_stt_transcribe_requires_openai_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            data = FormData()
            data.add_field(
                "audio",
                b"fake-audio",
                filename="recording.webm",
                content_type="audio/webm",
            )
            response = await client.post("/ui/api/stt/transcribe", data=data)
            assert response.status == 409
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "stt_api_key_missing"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_stt_transcribe_handles_unsupported_format(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env-test-key")
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api.requests.post",
        lambda *args, **kwargs: FakeSttResponse(
            400,
            {"error": {"message": "Unsupported file format"}},
        ),
    )

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            data = FormData()
            data.add_field(
                "audio",
                b"fake-audio",
                filename="recording.bin",
                content_type="application/octet-stream",
            )
            response = await client.post("/ui/api/stt/transcribe", data=data)
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "unsupported_audio_format"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chat_send_uses_api_key_from_env(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-env-test-key")
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.setattr(
        "server.http_api._fetch_provider_models",
        lambda provider: (["grok-4x"], None)
        if provider == "openrouter"
        else (["local-default"], None),
    )

    async def run() -> None:
        agent = CaptureConfigAgent()
        client = await _create_client(agent)
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_payload = create_payload.get("session")
            assert isinstance(session_payload, dict)
            session_id = session_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id

            select_resp = await client.post(
                "/ui/api/session-model",
                headers={"X-Slavik-Session": session_id},
                json={"provider": "openrouter", "model": "grok-4x"},
            )
            assert select_resp.status == 200

            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "ping"},
            )
            assert send_resp.status == 200
            assert agent.last_provider == "openrouter"
            assert agent.last_model == "grok-4x"
            assert agent.last_api_key == "or-env-test-key"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_chats_export_import_endpoints() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            create_resp = await client.post("/ui/api/sessions")
            assert create_resp.status == 200
            create_payload = await create_resp.json()
            session_payload = create_payload.get("session")
            assert isinstance(session_payload, dict)
            session_id = session_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            send_resp = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "export me"},
            )
            assert send_resp.status == 200

            export_resp = await client.get("/ui/api/settings/chats/export")
            assert export_resp.status == 200
            export_payload = await export_resp.json()
            exported_sessions = export_payload.get("sessions")
            assert isinstance(exported_sessions, list)
            assert len(exported_sessions) >= 1

            import_resp = await client.post(
                "/ui/api/settings/chats/import",
                json={
                    "mode": "replace",
                    "sessions": [
                        {
                            "session_id": "imported-session",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "status": "ok",
                            "messages": [
                                {
                                    "message_id": "msg-user-import",
                                    "role": "user",
                                    "content": "hello import",
                                    "created_at": "2026-01-01T00:00:00+00:00",
                                    "trace_id": None,
                                    "parent_user_message_id": None,
                                },
                                {
                                    "message_id": "msg-assistant-import",
                                    "role": "assistant",
                                    "content": "import ok",
                                    "created_at": "2026-01-01T00:00:01+00:00",
                                    "trace_id": "trace-import",
                                    "parent_user_message_id": "msg-user-import",
                                },
                            ],
                            "decision": None,
                            "selected_model": {"provider": "local", "model": "local-default"},
                        },
                    ],
                },
            )
            assert import_resp.status == 200
            import_payload = await import_resp.json()
            assert import_payload.get("imported") == 1
            assert import_payload.get("mode") == "replace"

            imported_resp = await client.get("/ui/api/sessions/imported-session")
            assert imported_resp.status == 200
            imported_payload = await imported_resp.json()
            imported_session = imported_payload.get("session")
            assert isinstance(imported_session, dict)
            messages = imported_session.get("messages")
            assert isinstance(messages, list)
            assert len(messages) == 2
            first = messages[0]
            second = messages[1]
            assert isinstance(first, dict)
            assert isinstance(second, dict)
            assert first.get("content") == "hello import"
            assert second.get("content") == "import ok"
        finally:
            await client.close()

    asyncio.run(run())
