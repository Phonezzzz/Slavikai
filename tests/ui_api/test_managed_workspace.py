from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from core.desktop_policy import DesktopPolicyStore
from core.task_run_storage import TaskRunStore
from server.http.common.auth import _owner_principal_id
from server.http_api import create_app
from server.ui_session_storage import InMemoryUISessionStorage

from .fakes import TEST_API_TOKEN, TEST_AUTH_HEADERS, DummyAgent


def test_private_engine_roots_and_foreign_workspace_refused_by_http(tmp_path, monkeypatch):
    state = tmp_path / "rootless-state"
    state.mkdir(mode=0o700)
    (state / "storage").mkdir(mode=0o700)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SLAVIK_COMPUTER_BACKEND", "container")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_ENGINE", "podman")
    # Неизвестный image должен сохранить fence, не обращаться к engine/registry.
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_IMAGE", "unqualified-synthetic")
    monkeypatch.setenv("SLAVIK_COMPUTER_CONTAINER_WORKSPACE", "/workspace")
    monkeypatch.setenv("SLAVIK_COMPUTER_ROOTLESS_STATE", str(state))
    monkeypatch.setenv("SLAVIK_COMPUTER_STAGING_SLOT_BYTES", "2097152")
    monkeypatch.setenv("SLAVIK_COMPUTER_STAGING_SLOT_INODES", "64")
    monkeypatch.setenv("SLAVIK_COMPUTER_STAGING_SLOTS", "2")
    for suffix, value in {
        "FILE_BYTES": "1024",
        "TOTAL_BYTES": "4096",
        "ENTRIES": "20",
        "DEPTH": "8",
        "OUTPUT_BYTES": "1024",
    }.items():
        monkeypatch.setenv("SLAVIK_COMPUTER_CAPTURE_" + suffix, value)
    monkeypatch.setattr("server.http.app.load_model_configs", lambda: None)
    original_connect = sqlite3.connect
    opened = []

    def guarded_connect(database, *args, **kwargs):
        raw = str(database)
        assert not raw.startswith("file:")
        assert raw == ":memory:" or Path(raw).resolve().is_relative_to(tmp_path)
        opened.append(raw)
        return original_connect(database, *args, **kwargs)

    def forbidden_engine(*args, **kwargs):
        raise AssertionError("unqualified image must not contact engine")

    monkeypatch.setattr(sqlite3, "connect", guarded_connect)
    monkeypatch.setattr("subprocess.run", forbidden_engine)
    monkeypatch.setattr("subprocess.Popen", forbidden_engine)

    async def run():
        auth = HttpAuthConfig(
            api_token=TEST_API_TOKEN, allow_unauth_local=False, browser_auth_mode="token"
        )
        app = create_app(
            agent=DummyAgent(),
            auth_config=auth,
            ui_storage=InMemoryUISessionStorage(),
            task_run_store=TaskRunStore(tmp_path / "task-runs.db"),
            desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop.json"),
        )
        pool = app["managed_workspace_pool"]
        own = pool.for_principal(_owner_principal_id(auth))
        foreign = pool.for_principal("synthetic-foreign")
        client = TestClient(TestServer(app), headers=TEST_AUTH_HEADERS)
        await client.start_server()
        try:
            response = await client.post("/ui/api/sessions")
            assert response.status == 200
            session_id = (await response.json())["session"]["session_id"]
            headers = {"X-Slavik-Session": session_id}
            for root in (state, state / "storage", foreign.root):
                denied = await client.post(
                    "/ui/api/workspace/root/select",
                    headers=headers,
                    json={"root_path": str(root)},
                )
                assert denied.status == 409, await denied.text()
                assert (await denied.json())["error"]["code"] == "workspace_unavailable"
            allowed = await client.post(
                "/ui/api/workspace/root/select",
                headers=headers,
                json={"root_path": str(own.root)},
            )
            assert allowed.status == 200, await allowed.text()
            assert (await allowed.json())["root_path"] == str(own.root)
            assert await app["ui_hub"].get_workspace_root(session_id) == str(own.root)
        finally:
            await client.close()
        assert not pool.owns(_owner_principal_id(auth), own)

    asyncio.run(run())
    assert opened
