from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from config.memory_config import MemoryConfig
from core.agent import Agent
from core.desktop_policy import DesktopPolicyStore
from core.managed_workspace import ManagedWorkspacePool, WorkspaceAccess, WorkspaceLimits
from core.rootless_podman import RootlessPodmanProfile
from core.rootless_staging import RootlessStagePool, StagingLimits
from core.task_run_storage import TaskRunStore
from core.workspace_python_execution import RootlessPythonExecutor
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig, ToolCall
from server.agent_provider import ScopedAgentProvider
from server.http.app import create_app
from server.http.common.auth import UI_AUTH_COOKIE, _ui_auth_cookie_value
from server.ui_session_storage import InMemoryUISessionStorage
from shared.task_run import RunScope, RunState


@pytest.mark.parametrize(
    "endpoint,approval,fault",
    [
        ("ui_sync", False, None),
        ("ui_stream", False, None),
        ("api", False, None),
        ("ui_sync", True, None),
        ("ui_stream", True, None),
        ("ui_sync", False, "intent"),
        ("ui_sync", False, "observation"),
    ],
)
def test_admitted_capture_commits_original_run_before_model_view(
    tmp_path, monkeypatch, endpoint, approval, fault
):
    import server.http.app as app_module
    import server.http_api as api

    monkeypatch.chdir(tmp_path)
    original_connect = sqlite3.connect

    def connect(database, *args, **kwargs):
        assert not str(database).startswith("file:")
        assert Path(database).resolve().is_relative_to(tmp_path)
        return original_connect(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(app_module, "_DOTENV_LOAD_ATTEMPTED", True)
    monkeypatch.setattr("core.agent.load_memory_config", lambda: MemoryConfig())
    state = tmp_path / "engine"
    state.mkdir(mode=0o700)
    profile = RootlessPodmanProfile(state, os.geteuid(), os.getegid(), "synthetic-profile", (), ())
    staging = RootlessStagePool(profile, StagingLimits(2097152, 64, 2))
    staging.root.mkdir(mode=0o700)
    staging._view = staging.root
    staging._revision = "synthetic-bounded-environment"
    slots = []
    for index in range(2):
        slot = staging.root / f"slot-{index}"
        slot.mkdir(mode=0o700)
        slots.append((slot, (slot.stat().st_dev, slot.stat().st_ino)))
    staging._slots = tuple(slots)
    # Заменена только OS/engine boundary: pytest не запускает host worker или engine.
    # Real qualification и admitted HTTP engine proof находятся в отдельных artifacts.
    monkeypatch.setattr(RootlessStagePool, "check_current", lambda self: None)
    monkeypatch.setattr(RootlessPodmanProfile, "check_current", lambda self: None)
    monkeypatch.setattr(RootlessPodmanProfile, "verify_created", lambda *args: None)
    monkeypatch.setattr(
        RootlessPodmanProfile,
        "control",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"a" * 64, stderr=b""),
    )
    monkeypatch.setattr(RootlessPodmanProfile, "finished_exit", lambda *args, **kwargs: 0)
    monkeypatch.setattr(RootlessPodmanProfile, "remove_owned", lambda *args: True)
    starts = []
    active_store = []

    def start(args, **kwargs):
        with sqlite3.connect(active_store[0].path) as conn:
            assert conn.execute("SELECT count(*) FROM tool_operations").fetchone()[0] == 1
            assert conn.execute("SELECT count(*) FROM tool_observations").fetchone()[0] == 0
        starts.append(args)
        out, write_out = os.pipe()
        err, write_err = os.pipe()
        os.write(write_out, b"http\xff\r\n")
        os.write(write_err, b"diagnostic\x00")
        os.close(write_out)
        os.close(write_err)
        return SimpleNamespace(
            stdout=os.fdopen(out, "rb"),
            stderr=os.fdopen(err, "rb"),
            returncode=0,
            poll=lambda: 0,
            wait=lambda **kwargs: 0,
        )

    monkeypatch.setattr("subprocess.Popen", start)
    pool = ManagedWorkspacePool(state / "workspaces", protected_root=state)
    received = []
    created_agents = []
    model_requests = []

    class LocalBrain(Brain):
        supports_native_tools = True

        def generate(self, messages, config=None, tools=None):
            model_requests.append(True)
            if messages[-1].role == "tool":
                received.append(json.loads(messages[-1].content))
                return LLMResult(text="observed", usage=LLMUsage(0, 0, 0))
            return LLMResult(
                text="",
                usage=LLMUsage(0, 0, 0),
                tool_calls=[ToolCall("original-label", "workspace_run", {"path": "emit.py"})],
            )

    def factory(scope, config):
        owner = pool.for_principal(scope.principal_id)
        (owner.root / "emit.py").write_text("# synthetic bytes; never executed on host\n")
        agent = Agent(
            brain=LocalBrain(),
            user_id=scope.principal_id,
            main_config=config,
            enable_tools={"safe_mode": True, "workspace_run": True},
            memory_db_path=str(tmp_path / "memory.db"),
            vectors_db_path=str(tmp_path / "vectors.db"),
            memory_companion_db_path=str(tmp_path / "companion.db"),
            memory_inbox_db_path=str(tmp_path / "inbox.db"),
            canonical_atoms_db_path=str(tmp_path / "atoms.db"),
            desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop.json"),
            desktop_home=tmp_path,
            workspace_pool=pool,
            workspace_access=WorkspaceAccess(
                owner, scope.principal_id, scope.session_id, staging.execution_revision
            ),
            workspace_executor=RootlessPythonExecutor(
                profile, owner, WorkspaceLimits(32768, 131072, 60, 12), 1048576, staging
            ),
        )
        if endpoint == "ui_sync":
            monkeypatch.setattr(agent, "respond_stream", None)
        source_root = Path(__file__).resolve().parents[2]
        for resource in ("shared/tool_capture.py", "core/task_run_storage.py"):
            assert (
                agent.desktop_security.resolve(
                    str(source_root / resource), mutation=True
                ).protection
                == "deny"
            )
        created_agents.append(agent)
        return agent

    async def run():
        token = "synthetic-capture-token"
        store = TaskRunStore(tmp_path / "runs.db")
        active_store.append(store)
        if fault is not None:
            table = "tool_operations" if fault == "intent" else "tool_observations"
            with sqlite3.connect(store.path) as conn:
                conn.execute(
                    f"CREATE TRIGGER synthetic_fault BEFORE INSERT ON {table} "
                    "BEGIN SELECT RAISE(ABORT, 'synthetic capture write fault'); END"
                )
        app = create_app(
            ui_storage=InMemoryUISessionStorage(),
            task_run_store=store,
            auth_config=HttpAuthConfig(
                api_token=token, browser_auth_mode="token", allow_unauth_local=False
            ),
            desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop.json"),
        )
        app["agent_provider"] = ScopedAgentProvider(factory=factory)
        client = TestClient(
            TestServer(app),
            headers={
                "Cookie": f"{UI_AUTH_COOKIE}={_ui_auth_cookie_value(token)}",
                "Authorization": f"Bearer {token}",
            },
        )
        await client.start_server()
        try:
            await app["runtime_model_state"].set_global_main(
                ModelConfig(provider="local", model="synthetic")
            )
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            hub = app["ui_hub"]
            principal = await hub.get_session_principal_id(session)
            await hub.set_session_model(session, "local", "synthetic")
            await hub.set_workspace_root(session, str(pool.for_principal(principal).root))
            await hub.set_session_workflow(session, mode="auto")
            if not approval:
                await hub.set_session_policy(session, profile="yolo")
            headers = {"X-Slavik-Session": session, "Idempotency-Key": "capture"}
            url, body = "/ui/api/chat/send", {"content": "run synthetic script"}
            if endpoint == "api":
                url = "/v1/chat/completions"
                body = {
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "run synthetic script"}],
                    "slavik_meta": {"runtime_mode": "auto"},
                }
            response = await client.post(url, headers=headers, json=body)
            assert response.status == 200, await response.text()
            with sqlite3.connect(store.path) as conn:
                run_id = conn.execute("SELECT task_run_id FROM task_runs").fetchone()[0]
            if approval:
                decision = (await response.json())["decision"]
                assert decision and not starts
                response = await client.post(
                    "/ui/api/decision/respond",
                    headers=headers,
                    json={
                        "session_id": session,
                        "decision_id": decision["id"],
                        "choice": "approve_once",
                    },
                )
                assert response.status == 200, await response.text()
            if fault is not None:
                assert not received and len(model_requests) == 1
                assert len(starts) == (0 if fault == "intent" else 1)
                with sqlite3.connect(store.path) as conn:
                    assert conn.execute("SELECT count(*) FROM tool_operations").fetchone()[0] == (
                        0 if fault == "intent" else 1
                    )
                    assert conn.execute("SELECT count(*) FROM tool_observations").fetchone()[0] == 0
                scope = RunScope(principal, session)
                report = store.get_submission(scope, run_id).report
                assert created_agents[-1].last_auto_state["error_code"] == "tool_capture_failed"
                assert created_agents[-1].last_auto_state["status"] == "failed_internal"
                assert "no automatic replay" in report.text
                if fault == "observation":
                    assert "observed_exit_code=0" in report.text
                return
            assert len(starts) == len(received) == 1
            ref = received[0]["data"]["capture"]
            scope = RunScope(principal, session)
            reopened = TaskRunStore(store.path)
            saved = reopened.get_tool_observation(
                scope, run_id, ref["operation_id"], ref["tool_attempt_id"]
            )
            assert saved.result_id == ref["result_id"]
            assert saved.observation.stdout == b"http\xff\r\n"
            assert saved.observation.stderr == b"diagnostic\x00"
            assert saved.operation.model_call_id == "original-label"
            assert saved.operation.run_attempt_id == reopened.get(scope, run_id).attempt_id
            assert reopened.get(scope, run_id).state == RunState.RESULT_SUBMITTED
            assert reopened.get(scope, run_id).terminal_at is None
            assert not staging._allocations
        finally:
            await client.close()
            pool.close()
            staging.close()

    asyncio.run(run())
