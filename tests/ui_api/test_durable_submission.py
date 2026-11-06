from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from core.agent import Agent
from core.desktop_policy import DesktopPolicyStore
from core.task_run_storage import TaskRunStore
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig
from server.http.common.idempotency import IdempotencyStore
from shared.models import ToolResult
from shared.task_run import ProducerKind, ProducerReport, RunScope, RunState

from .fakes import TEST_API_TOKEN, _create_client, _select_local_model


class ReportBrain(Brain):
    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, config=None, tools=None):
        self.calls += 1
        return LLMResult(text="report-r2a\n🦊", usage=LLMUsage(0, 0, 0))


@pytest.fixture
def isolated_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original = sqlite3.connect

    def connect(database, *args, **kwargs):
        assert Path(database).resolve().is_relative_to(tmp_path)
        return original(database, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", connect)
    brain = ReportBrain()
    agent = Agent(
        brain=brain,
        memory_db_path=str(tmp_path / "memory.db"),
        vectors_db_path=str(tmp_path / "vectors.db"),
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
        desktop_policy_store=DesktopPolicyStore(tmp_path / "desktop.json"),
    )
    return agent, brain


@pytest.mark.parametrize("endpoint", ["ui", "api"])
@pytest.mark.parametrize("mode", ["ask", "auto"])
@pytest.mark.parametrize("fault", [None, "projection", "storage"])
def test_http_submission_commits_before_projection_without_replay(
    tmp_path, monkeypatch, isolated_agent, endpoint, mode, fault
):
    from server.http.handlers import chat, ui_chat

    agent, brain = isolated_agent
    monkeypatch.setattr(agent, "respond_stream", None)
    captured = []
    module = ui_chat if endpoint == "ui" else chat
    original_project = module._project_agent_response

    def project(response):
        captured.append(response)
        if fault == "projection":
            raise RuntimeError("after submission, before projection")
        return original_project(response)

    monkeypatch.setattr(module, "_project_agent_response", project)

    async def run():
        client = await _create_client(agent)
        try:
            await client.app["runtime_model_state"].set_global_main(
                ModelConfig(provider="ollama", model="test")
            )
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            scope = RunScope(principal, session)
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_workflow(session, mode=mode)
            store = client.app["task_run_store"]
            if fault == "storage":
                with sqlite3.connect(store.path) as conn:
                    conn.execute(
                        "CREATE TRIGGER fault BEFORE INSERT ON run_submissions "
                        "BEGIN SELECT RAISE(ABORT, 'submission storage fault'); END"
                    )
            headers = {
                "X-Slavik-Session": session,
                "Idempotency-Key": "durable-report",
            }
            if endpoint == "ui":
                url, payload = "/ui/api/chat/send", {"content": "hi"}
            else:
                headers["Authorization"] = f"Bearer {TEST_API_TOKEN}"
                url = "/v1/chat/completions"
                payload = {
                    "model": "slavik",
                    "messages": [{"role": "user", "content": "hi"}],
                    "slavik_meta": {"runtime_mode": mode},
                }
            first = await client.post(url, headers=headers, json=payload)
            assert first.status == (500 if fault else 200), await first.text()
            calls = brain.calls
            assert calls > 0
            with sqlite3.connect(store.path) as conn:
                rows = conn.execute("SELECT task_run_id, state FROM task_runs").fetchall()
            assert len(rows) == 1
            run_id, state = rows[0]
            reopened = TaskRunStore(store.path)
            if fault == "storage":
                assert state == RunState.RUNNING and not captured
                with pytest.raises(LookupError):
                    reopened.get_submission(scope, run_id)
                assert len(reopened.events(scope, run_id)) == 2
            else:
                ordinary_ask = endpoint == "ui" and mode == "ask" and fault is None
                assert state == ("completed" if ordinary_ask else RunState.RESULT_SUBMITTED)
                saved = reopened.get_submission(scope, run_id)
                assert saved.report.text == captured[-1].text
                assert saved.report.kind == (
                    ProducerKind.AUTO if mode == "auto" else ProducerKind.NATIVE_LOOP
                )
                assert saved.report.failure_code is None
                assert (reopened.get(scope, run_id).terminal_at is not None) is ordinary_ask
                if ordinary_ask:
                    assert [event.state for event in reopened.events(scope, run_id)][-3:] == [
                        "result_submitted",
                        "accepted",
                        "completed",
                    ]
            client.app["task_run_store"] = reopened
            client.app["idempotency_store"] = IdempotencyStore()
            repeated = await client.post(url, headers=headers, json=payload)
            assert repeated.status == 409
            assert (await repeated.json())["error"]["code"] == "task_run_continuation_unavailable"
            assert brain.calls == calls
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["ask", "auto"])
def test_approved_result_is_durable_before_resume_projection_fault(
    tmp_path, monkeypatch, isolated_agent, mode
):
    from core.agent_response import AgentResponse
    from llm.types import ToolCall
    from server.http.handlers import ui_chat

    agent, brain = isolated_agent
    dispatched = []

    def generate(messages, config=None, tools=None):
        brain.calls += 1
        if messages[-1].role == "tool":
            return LLMResult(text="observed", usage=LLMUsage(0, 0, 0))
        return LLMResult(
            text="",
            usage=LLMUsage(0, 0, 0),
            tool_calls=[ToolCall(id="approved-original", name="lookup", arguments={})],
        )

    monkeypatch.setattr(brain, "generate", generate)
    agent.tools_enabled["safe_mode"] = True
    agent.tool_registry.register(
        "lookup",
        lambda request: dispatched.append(request) or ToolResult.success({"value": "seen"}),
        capability="read",
        chat_exposed=True,
        risk_classes=["network"],
        parameters_schema={"type": "object"},
        description="Read lookup",
    )
    monkeypatch.setattr(agent, "respond_stream", None)

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            await _select_local_model(client, session)
            await client.app["ui_hub"].set_session_workflow(session, mode=mode)
            headers = {"X-Slavik-Session": session}
            initial = await client.post(
                "/ui/api/chat/send", headers=headers, json={"content": "lookup"}
            )
            assert initial.status == 200
            decision = (await initial.json())["decision"]
            store = client.app["task_run_store"]
            with sqlite3.connect(store.path) as conn:
                run_id = conn.execute("SELECT task_run_id FROM task_runs").fetchone()[0]
            scope = RunScope(principal, session)
            accepted = store.get(scope, run_id)
            assert accepted.state == RunState.RUNNING and not dispatched
            with pytest.raises(LookupError):
                store.get_submission(scope, run_id)
            captured = []

            def fail(response: AgentResponse):
                captured.append(response)
                raise RuntimeError("after approval submission, before projection")

            monkeypatch.setattr(ui_chat, "_project_agent_response", fail)
            resumed = await client.post(
                "/ui/api/decision/respond",
                headers=headers,
                json={
                    "session_id": session,
                    "decision_id": decision["id"],
                    "choice": "approve_once",
                },
            )
            assert resumed.status == 500
            saved = TaskRunStore(store.path).get_submission(scope, run_id)
            assert saved.report.text == captured[-1].text
            assert saved.task_id == accepted.task_id and saved.attempt_id == accepted.attempt_id
            assert len(dispatched) == 1
            assert store.get(scope, run_id).state == RunState.RESULT_SUBMITTED
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_stream_error_preserves_received_producer_report(isolated_agent, monkeypatch):
    from core.agent_response import ResponseProduced

    agent, _brain = isolated_agent
    original = agent.respond_stream

    def stream(messages, cancellation_token=None):
        for event in original(messages, cancellation_token=cancellation_token):
            yield event
            if isinstance(event, ResponseProduced):
                raise RuntimeError("post-runtime stream fault")

    monkeypatch.setattr(agent, "respond_stream", stream)

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 200
            store = client.app["task_run_store"]
            with sqlite3.connect(store.path) as conn:
                run_id = conn.execute("SELECT task_run_id FROM task_runs").fetchone()[0]
            saved = TaskRunStore(store.path).get_submission(RunScope(principal, session), run_id)
            assert saved.report.failure_code is not None
            assert saved.report.failure_message
            assert store.get(RunScope(principal, session), run_id).terminal_at is None
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("finish_reason", ["stop", "length", None])
def test_browser_ask_stream_requires_provider_stop_for_acceptance(
    isolated_agent, monkeypatch, finish_reason
):
    from llm.stream_model import Done, TextDelta

    agent, brain = isolated_agent

    def stream(*args, **kwargs):
        yield TextDelta(text="ordinary answer")
        if finish_reason is not None:
            yield Done(finish_reason=finish_reason)

    monkeypatch.setattr(brain, "generate_stream_events", stream)

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            await _select_local_model(client, session)
            response = await client.post(
                "/ui/api/chat/send", headers={"X-Slavik-Session": session}, json={"content": "hi"}
            )
            assert response.status == 200
            with sqlite3.connect(client.app["task_run_store"].path) as conn:
                (state,) = conn.execute("SELECT state FROM task_runs").fetchone()
            assert state == (
                RunState.COMPLETED if finish_reason == "stop" else RunState.RESULT_SUBMITTED
            )
        finally:
            await client.close()

    asyncio.run(run())


def test_browser_recovery_evidence_is_scoped_bounded_and_read_only(isolated_agent, tmp_path):
    from core.managed_workspace import ManagedWorkspacePool, WorkspaceLimits
    from shared.task_run import TransitionAuthority
    from shared.tool_capture import ProcessObservation, WorkspacePublicationEvidence

    agent, brain = isolated_agent

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            scope = RunScope(principal, session)
            store = client.app["task_run_store"]
            admitted = store.admit(
                scope,
                request_key="recovery-evidence",
                goal="private goal",
                mode="auto",
                request_fingerprint="a" * 64,
            )
            running = store.transition(
                scope,
                admitted.task_run_id,
                expected_version=admitted.version,
                state=RunState.RUNNING,
                authority=TransitionAuthority.RUNTIME,
                reason_code="start",
            )
            operations = [
                store.begin_tool_operation(running, model_call_id=str(i), request_path="run.py")
                for i in range(51)
            ]
            first = min(operations, key=lambda op: op.operation_id)
            result_id = store.record_tool_observation(
                first,
                ProcessObservation(b"private bytes", b"", 13, 0, True, True, 0, False),
                publication=WorkspacePublicationEvidence(
                    cleanup_warning=True, snapshot_sha256="a" * 64
                ),
            )
            store.record_tool_observation(
                sorted(operations, key=lambda op: op.operation_id)[1],
                ProcessObservation(b"", b"", 0, 0, True, True, 0, False),
            )
            # Simulate process loss after exchange, before the observation insert.
            private = tmp_path / "managed"
            private.mkdir(mode=0o700)
            pool = ManagedWorkspacePool(private / "workspaces")
            client.app["managed_workspace_pool"] = pool
            owner = pool.for_principal(principal)
            limits = WorkspaceLimits(1024, 4096, 20, 8)
            binding = owner.bind(principal_id=principal, session_id=session)
            snapshot = owner.capture(binding, limits=limits)
            worker = tmp_path / "worker"
            worker.mkdir(mode=0o700)
            stage = owner.stage(snapshot, storage=worker)
            stage.begin_process()
            (stage.root / "run.py").write_text("# published")
            stage.mark_settled()
            marker_operation = sorted(operations, key=lambda op: op.operation_id)[2]
            owner.publish_bounded(
                snapshot, stage, limits=limits, publication_operation=marker_operation
            )
            submitted = store.submit_result(
                scope,
                running.task_run_id,
                expected_version=running.version,
                report=ProducerReport(ProducerKind.AUTO, "private producer text"),
                reason_code="result_submitted",
            )
            url = f"/ui/api/sessions/{session}/runs/{running.task_run_id}/recovery-evidence"
            assert (await client.get(url)).status == 404
            recovery = store.transition(
                scope,
                running.task_run_id,
                expected_version=submitted.version,
                state=RunState.RECOVERY_REQUIRED,
                authority=TransitionAuthority.RECOVERY,
                reason_code="server_restart_requires_reconciliation",
            )
            response = await client.get(url)
            assert response.status == 200, await response.text()
            assert response.headers["Cache-Control"] == "no-store"
            body = await response.json()
            assert body["external_effects"] == "unreconciled"
            assert body["run_version"] == recovery.version
            assert body["recovery_prior_state"] == "result_submitted"
            assert (
                body["recovery_occurred_at"]
                == store.events(scope, running.task_run_id)[-1].occurred_at
            )
            assert body["submission"]["kind"] == "auto"
            assert len(body["operations"]) == 50
            assert body["operations"][0]["observation"]["result_id"] == result_id
            assert body["operations"][0]["publication"] == "published"
            assert body["operations"][0]["publication_source"] == "observation"
            assert body["operations"][0]["publication_cleanup_warning"] is True
            assert body["operations"][0]["publication_snapshot_sha256"] == "a" * 64
            assert body["operations"][1]["observation"]["exit_code"] == 0
            assert body["operations"][1]["publication"] == "unknown"
            assert body["operations"][1]["publication_cleanup_warning"] is None
            assert body["operations"][1]["publication_snapshot_sha256"] is None
            assert body["operations"][2]["observation"] is None
            assert body["operations"][2]["publication"] == "published"
            assert body["operations"][2]["publication_source"] == "workspace_root_marker"
            assert body["operations"][2]["publication_cleanup_warning"] is None
            assert body["operations"][3]["publication"] == "unknown"
            assert "private" not in await response.text()
            following = await (await client.get(url, params={"after": body["next_cursor"]})).json()
            assert len(following["operations"]) == 1 and following["next_cursor"] is None
            assert (await client.get(url, params={"after": "bad"})).status == 400

            other = await client.app["ui_hub"].create_session(principal_id=principal)
            other_url = url.replace(f"/sessions/{session}/", f"/sessions/{other}/")
            assert (await client.get(other_url)).status == 404
            assert store.get(scope, running.task_run_id) == recovery
            assert brain.calls == 0
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("after_exchange", [False, True])
def test_browser_recovery_after_workspace_publisher_is_killed(
    isolated_agent, tmp_path, after_exchange
):
    import signal
    import subprocess
    import sys

    from core.managed_workspace import ManagedWorkspacePool
    from core.task_run_storage import open_task_run_server

    agent, brain = isolated_agent
    crash_script = """
import os
import signal
import sys
from pathlib import Path

import core.managed_workspace as managed
from core.task_run_storage import open_task_run_server
from shared.task_run import RunScope, RunState, TransitionAuthority

database, workspaces, worker, principal, session, after_exchange = sys.argv[1:]
scope = RunScope(principal, session)
with open_task_run_server(Path(database)) as store:
    pool = managed.ManagedWorkspacePool(Path(workspaces))
    owner = pool.for_principal(principal)
    admitted = store.admit(
        scope, request_key="crash", goal="run", mode="auto", request_fingerprint="a" * 64
    )
    running = store.transition(
        scope, admitted.task_run_id, expected_version=admitted.version,
        state=RunState.RUNNING, authority=TransitionAuthority.RUNTIME, reason_code="start",
    )
    operation = store.begin_tool_operation(running, model_call_id="tool", request_path="run.py")
    limits = managed.WorkspaceLimits(1024, 4096, 20, 8)
    snapshot = owner.capture(owner.bind(principal_id=principal, session_id=session), limits=limits)
    stage = owner.stage(snapshot, storage=Path(worker))
    stage.begin_process()
    (stage.root / "run.py").write_text("# published")
    stage.mark_settled()
    exchange = managed._exchange

    def crash(parent_fd, current, candidate):
        if after_exchange == "1":
            exchange(parent_fd, current, candidate)
        print(running.task_run_id, operation.operation_id, running.version, flush=True)
        os.kill(os.getpid(), signal.SIGKILL)

    managed._exchange = crash
    owner.publish_bounded(snapshot, stage, limits=limits, publication_operation=operation)
"""

    async def run():
        client = await _create_client(agent)
        try:
            session = (await (await client.get("/ui/api/status")).json())["session_id"]
            principal = await client.app["ui_hub"].get_session_principal_id(session)
            scope = RunScope(principal, session)
            private = tmp_path / "private"
            private.mkdir(mode=0o700)
            worker = tmp_path / "worker"
            worker.mkdir(mode=0o700)
            database = tmp_path / "crash.db"
            workspaces = private / "workspaces"
            killed = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable,
                    "-c",
                    crash_script,
                    str(database),
                    str(workspaces),
                    str(worker),
                    principal,
                    session,
                    str(int(after_exchange)),
                ],
                cwd=Path(__file__).resolve().parents[2],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert killed.returncode == -signal.SIGKILL, killed.stderr
            run_id, operation_id, version = killed.stdout.split()
            pool = ManagedWorkspacePool(workspaces)
            try:
                with open_task_run_server(database) as restarted:
                    client.app["task_run_store"] = restarted
                    client.app["managed_workspace_pool"] = pool
                    recovered = restarted.get(scope, run_id)
                    assert recovered.state == RunState.RECOVERY_REQUIRED
                    assert recovered.version == int(version) + 1
                    url = f"/ui/api/sessions/{session}/runs/{run_id}/recovery-evidence"
                    for _ in range(2):
                        response = await client.get(url)
                        assert response.status == 200, await response.text()
                        body = await response.json()
                        assert body["external_effects"] == "unreconciled"
                        assert body["run_version"] == recovered.version
                        assert body["recovery_prior_state"] == "running"
                        assert (
                            body["recovery_occurred_at"]
                            == restarted.events(scope, run_id)[-1].occurred_at
                        )
                        [operation] = body["operations"]
                        assert operation["operation_id"] == operation_id
                        assert operation["observation"] is None
                        digest = operation["publication_snapshot_sha256"]
                        if after_exchange:
                            marker = pool.publication_marker(principal)
                            assert marker is not None
                            assert isinstance(digest, str) and len(digest) == 64
                            assert digest == marker.snapshot_sha256
                        else:
                            assert digest is None
                        assert operation["publication"] == (
                            "published" if after_exchange else "unknown"
                        )
                        assert operation["publication_source"] == (
                            "workspace_root_marker" if after_exchange else None
                        )
                    assert restarted.get(scope, run_id) == recovered
                    assert brain.calls == 0
            finally:
                pool.close()
        finally:
            await client.close()

    asyncio.run(run())
