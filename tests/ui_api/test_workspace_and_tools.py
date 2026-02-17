from __future__ import annotations

# ruff: noqa: F403,F405
from .fakes import *


def test_ui_workspace_root_returns_root_payload() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/ui/api/workspace/root")
            assert response.status == 200
            payload = await response.json()
            assert isinstance(payload.get("session_id"), str)
            assert isinstance(payload.get("root_path"), str)
            policy = payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "sandbox"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_workspace_root_select_rejects_outside_workspace_in_sandbox() -> None:
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

            status, policy_payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="sandbox",
            )
            assert status == 200
            policy = policy_payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "sandbox"

            response = await client.post(
                "/ui/api/workspace/root/select",
                headers={"X-Slavik-Session": session_id},
                json={"root_path": str(Path.cwd())},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            message = error.get("message")
            assert isinstance(message, str)
            assert "sandbox директории" in message
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_workspace_root_select_applies_without_approval_in_index(tmp_path) -> None:
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

            outside_home = tmp_path / "outside-index-root"
            outside_home.mkdir(parents=True, exist_ok=True)
            next_outside_home = tmp_path / "outside-index-next"
            next_outside_home.mkdir(parents=True, exist_ok=True)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(outside_home))

            status, policy_payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="index",
            )
            assert status == 200
            policy = policy_payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "index"

            response = await client.post(
                "/ui/api/workspace/root/select",
                headers={"X-Slavik-Session": session_id},
                json={"root_path": str(next_outside_home)},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload.get("applied") is True
            assert payload.get("root_path") == str(next_outside_home)
            assert payload.get("decision") is None
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_workspace_root_select_requires_approval_in_yolo() -> None:
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

            status, policy_payload = await _set_session_policy_via_api(
                client,
                session_id=session_id,
                policy_profile="yolo",
                confirm_yolo=True,
            )
            assert status == 200
            policy = policy_payload.get("policy")
            assert isinstance(policy, dict)
            assert policy.get("profile") == "yolo"

            response = await client.post(
                "/ui/api/workspace/root/select",
                headers={"X-Slavik-Session": session_id},
                json={"root_path": "/"},
            )
            assert response.status == 202
            payload = await response.json()
            decision = payload.get("decision")
            assert isinstance(decision, dict)
            proposed_action = decision.get("proposed_action")
            assert isinstance(proposed_action, dict)
            required_categories = proposed_action.get("required_categories")
            assert required_categories == ["FS_OUTSIDE_WORKSPACE"]
        finally:
            await client.close()

    asyncio.run(run())


def test_resolve_workspace_root_candidate_sandbox_accepts_and_rejects(
    monkeypatch, tmp_path
) -> None:
    sandbox_root = tmp_path / "sandbox-root"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    inside = sandbox_root / "inside"
    inside.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("server.http_api.WORKSPACE_ROOT", sandbox_root)

    accepted = _resolve_workspace_root_candidate(str(inside), policy_profile="sandbox")
    assert accepted == inside.resolve()

    try:
        _resolve_workspace_root_candidate(str(outside), policy_profile="sandbox")
    except ValueError as exc:
        assert "sandbox директории" in str(exc)
    else:
        raise AssertionError("sandbox profile must reject outside directories")


def test_resolve_workspace_root_candidate_index_rejects_home(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(parents=True, exist_ok=True)
    inside_home = fake_home / "project"
    inside_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("server.http_api.Path.home", lambda: fake_home)

    try:
        _resolve_workspace_root_candidate(str(inside_home), policy_profile="index")
    except ValueError as exc:
        assert "домашней директории" in str(exc)
    else:
        raise AssertionError("index profile must reject home subtree")


def test_resolve_workspace_root_candidate_index_accepts_outside_home(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(parents=True, exist_ok=True)
    outside_home = tmp_path / "outside-home"
    outside_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("server.http_api.Path.home", lambda: fake_home)

    accepted = _resolve_workspace_root_candidate(str(outside_home), policy_profile="index")
    assert accepted == outside_home.resolve()


def test_resolve_workspace_root_candidate_yolo_accepts_outside_home(monkeypatch, tmp_path) -> None:
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir(parents=True, exist_ok=True)
    outside_home = tmp_path / "outside-home-yolo"
    outside_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("server.http_api.Path.home", lambda: fake_home)

    accepted = _resolve_workspace_root_candidate(str(outside_home), policy_profile="yolo")
    assert accepted == outside_home.resolve()


def test_ui_workspace_index_respects_index_enabled_env(monkeypatch) -> None:
    monkeypatch.setenv("SLAVIK_INDEX_ENABLED", "false")

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

            response = await client.post(
                "/ui/api/workspace/index",
                headers={"X-Slavik-Session": session_id},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload.get("ok") is False
            assert payload.get("message") == "INDEX disabled"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_workspace_index_openai_provider_requires_api_key(monkeypatch, tmp_path) -> None:
    ui_settings_path = tmp_path / "ui_settings.json"
    ui_settings_path.write_text(
        json.dumps(
            {
                "memory": {
                    "embeddings": {
                        "provider": "openai",
                        "local_model": "all-MiniLM-L6-v2",
                        "openai_model": "text-embedding-3-small",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", ui_settings_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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

            response = await client.post(
                "/ui/api/workspace/index",
                headers={"X-Slavik-Session": session_id},
            )
            assert response.status == 500
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            message = error.get("message")
            assert isinstance(message, str)
            assert "API key" in message
        finally:
            await client.close()

    asyncio.run(run())


def test_create_app_keeps_workspace_root_and_policy_on_boot(tmp_path) -> None:
    db_path = tmp_path / "ui_sessions.db"
    storage = SQLiteUISessionStorage(db_path)
    storage.save_session(
        PersistedSession(
            session_id="session-reset",
            principal_id="principal-reset",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            status="ok",
            decision=None,
            messages=[],
            workspace_root="/tmp",
            policy_profile="yolo",
        )
    )

    _ = create_app(
        agent=DummyAgent(),
        max_request_bytes=1_000_000,
        ui_storage=storage,
        auth_config=HttpAuthConfig(api_token=TEST_API_TOKEN, allow_unauth_local=False),
    )

    sessions = storage.load_sessions()
    assert len(sessions) == 1
    restored = sessions[0]
    assert restored.workspace_root == "/tmp"
    assert restored.policy_profile == "yolo"


def test_ui_project_tool_endpoint() -> None:
    async def run() -> None:
        client = await _create_client(ProjectCommandAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            await _enter_act_mode(client, session_id, goal="project find command")

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={"command": "find", "args": "README"},
            )
            assert response.status == 200
            payload = await response.json()
            assert payload.get("session_id") == session_id
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert len(messages) >= 2
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            content = last_message.get("content")
            assert isinstance(content, str)
            assert "Командный режим (без MWV)" in content
            assert "/project find README" in content
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_tool_endpoint_rejects_unknown_command() -> None:
    async def run() -> None:
        client = await _create_client(ProjectCommandAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            await _enter_act_mode(client, session_id, goal="project import requires approval")

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={"command": "drop_db", "args": ""},
            )
            assert response.status == 400
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_github_import_requires_approval() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            await _enter_act_mode(client, session_id, goal="project import requires approval")

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={
                    "command": "github_import",
                    "args": "https://github.com/example/repo",
                },
            )
            assert response.status == 200
            payload = await response.json()
            approval_request = payload.get("approval_request")
            assert isinstance(approval_request, dict)
            required_categories = approval_request.get("required_categories")
            assert isinstance(required_categories, list)
            assert "NETWORK_RISK" in required_categories
            assert "EXEC_ARBITRARY" in required_categories
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_project_github_import_runs_after_approval(monkeypatch, tmp_path) -> None:
    async def fake_clone(**kwargs):
        del kwargs
        return True, "ok"

    monkeypatch.setattr("server.http_api._clone_github_repository", fake_clone)
    monkeypatch.setattr(
        "server.http_api._resolve_github_target",
        lambda repo_url: (tmp_path / "repo", "github/example/repo"),
    )
    monkeypatch.setattr(
        "server.http_api._index_imported_project",
        lambda relative_path: (True, f"Code=1, Docs=1 ({relative_path})"),
    )
    monkeypatch.setenv("SLAVIK_ADMIN_TOKEN", "admin-secret")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            assert session_id
            await _select_local_model(client, session_id)
            await _enter_act_mode(client, session_id, goal="project import after approval")

            approve_resp = await client.post(
                "/slavik/approve-session",
                headers={"Authorization": "Bearer admin-secret"},
                json={
                    "session_id": session_id,
                    "categories": ["NETWORK_RISK", "EXEC_ARBITRARY"],
                },
            )
            assert approve_resp.status == 200

            response = await client.post(
                "/ui/api/tools/project",
                headers={"X-Slavik-Session": session_id},
                json={
                    "command": "github_import",
                    "args": "https://github.com/example/repo",
                },
            )
            assert response.status == 200
            payload = await response.json()
            messages = payload.get("messages")
            assert isinstance(messages, list)
            assert messages
            last_message = messages[-1]
            assert isinstance(last_message, dict)
            content = last_message.get("content")
            assert isinstance(content, str)
            assert "GitHub import completed" in content
            assert "Code=1, Docs=1" in content
        finally:
            await client.close()

    asyncio.run(run())
