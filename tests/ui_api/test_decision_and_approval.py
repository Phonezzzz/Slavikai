from __future__ import annotations

# ruff: noqa: F403,F405
import pytest

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
                f"/ui/api/chat/events/{session_id}",
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
            transitions = approve_payload.get("mode_transitions")
            assert isinstance(transitions, dict)
            assert transitions.get("current_mode") == "ask"
            targets = transitions.get("targets")
            assert isinstance(targets, dict)
            ask_target = targets.get("ask")
            assert isinstance(ask_target, dict)
            assert ask_target.get("allowed") is False
            assert ask_target.get("reason_code") == "already_active"
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


@pytest.mark.behavior
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
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            hub = client.server.app["ui_hub"]
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
        finally:
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


def test_ui_decision_respond_workspace_root_select_returns_root_path(tmp_path) -> None:  # noqa: ANN001
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub = client.server.app["ui_hub"]
            target_root = Path.cwd() / "sandbox" / "project" / "approved-root"
            target_root.mkdir(parents=True, exist_ok=True)
            decision_payload = {
                "id": "decision-workspace-root-1",
                "kind": "approval",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "approval_required",
                "summary": "workspace root change",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "workspace.root_select",
                    "resume_payload": {
                        "root_path": str(target_root),
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
                    "decision_id": "decision-workspace-root-1",
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            payload = await respond.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            assert resume.get("source_endpoint") == "workspace.root_select"
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("root_path") == str(target_root)
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_project_command_github_import_resumes_clone_and_index(
    monkeypatch, tmp_path
) -> None:
    target_path = tmp_path / "github-imported-repo"
    target_path.mkdir(parents=True, exist_ok=True)

    def fake_parse(args_raw: str) -> tuple[str, str | None]:
        parts = args_raw.strip().split()
        return parts[0], parts[1] if len(parts) > 1 else None

    def fake_resolve(repo_url: str) -> tuple[Path, str]:
        return target_path, "github/example/repo"

    async def fake_clone(
        *, repo_url: str, branch: str | None, target_path: Path
    ) -> tuple[bool, str]:
        (target_path / "README.md").write_text("# test\n", encoding="utf-8")
        return True, "ok"

    def fake_index(relative_path: str) -> tuple[bool, str]:
        return True, "Code=1, Docs=1"

    monkeypatch.setattr("server.http.handlers.decision._parse_github_import_args", fake_parse)
    monkeypatch.setattr("server.http.handlers.decision._resolve_github_target", fake_resolve)
    monkeypatch.setattr("server.http.handlers.decision._clone_github_repository", fake_clone)
    monkeypatch.setattr("server.http.handlers.decision._index_imported_project", fake_index)

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]

            decision_payload = {
                "id": "decision-project-import-1",
                "kind": "approval",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "approval_required",
                "summary": "GitHub import request",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "project.command",
                    "resume_payload": {
                        "source_request": {
                            "command": "github_import",
                            "args": "https://github.com/example/repo",
                        },
                        "user_message_id": None,
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
                    "decision_id": "decision-project-import-1",
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            payload = await respond.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            assert resume.get("source_endpoint") == "project.command"
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("command") == "github_import"
            assert data.get("repo_url") == "https://github.com/example/repo"
            assert data.get("root_applied") == str(target_path)
            assert "GitHub import completed" in str(data.get("result"))

            workspace_root = await hub.get_workspace_root(session_id)
            assert workspace_root == str(target_path)

            workspace_messages = await hub.get_messages(session_id, lane="workspace")
            assert len(workspace_messages) >= 1
            last_ws = workspace_messages[-1]
            assert isinstance(last_ws, dict)
            assert last_ws.get("role") == "assistant"
            assert "GitHub import completed" in str(last_ws.get("content"))
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_stage_approve_executes(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        capture_output=True,
        check=True,
    )
    (repo / "to-stage.txt").write_text("data\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": ["to-stage.txt"]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            assert resume.get("source_endpoint") == "workspace.git"

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert "to-stage.txt" in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_stage_reject_does_nothing(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    (repo / "no-stage.txt").write_text("x\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": ["no-stage.txt"]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "reject",
                },
            )
            assert respond.status == 200

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert "no-stage.txt" not in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_stage_dash_filename(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        capture_output=True,
        check=True,
    )
    (repo / "-weird.txt").write_text("x\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": ["-weird.txt"]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert "-weird.txt" in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_invalid_paths_rejected(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            for bad_payload in (
                {"paths": []},
                {"paths": [1, 2]},
                {"paths": [""]},
                {"all": "yes"},
                {"all": 1},
                {},
                {"all": False},
                {"all": True, "paths": ["a.txt"]},
            ):
                stage_resp = await client.post(
                    "/ui/api/workspace/git/stage",
                    headers={"X-Slavik-Session": session_id},
                    json=bad_payload,
                )
                assert stage_resp.status == 400
                body = await stage_resp.json()
                error = body.get("error")
                assert isinstance(error, dict)
                assert (
                    "список" in str(error.get("message"))
                    or "строк" in str(error.get("message"))
                    or "boolean" in str(error.get("message"))
                    or "paths" in str(error.get("message"))
                )
                decision_after = await hub.get_session_decision(session_id)
                assert decision_after is None or decision_after.get("status") != "pending"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_stage_preserves_leading_trailing_spaces(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        capture_output=True,
        check=True,
    )
    (repo / " leading.txt").write_text("x\n")
    (repo / "trailing.txt ").write_text("y\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": [" leading.txt", "trailing.txt "]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert " leading.txt" in staged
            assert "trailing.txt " in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_malformed_resume_not_executed(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    (repo / "malformed.txt").write_text("x\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            decision_payload = {
                "id": "decision-malformed-resume",
                "kind": "approval",
                "decision_type": "tool_approval",
                "status": "pending",
                "blocking": True,
                "reason": "approval_required",
                "summary": "Git stage (malformed resume)",
                "proposed_action": {},
                "options": [],
                "default_option_id": None,
                "context": {
                    "session_id": session_id,
                    "source_endpoint": "workspace.git",
                    "resume_payload": {
                        "operation": "stage",
                        "paths": "malformed.txt",
                        "all": False,
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
                    "decision_id": "decision-malformed-resume",
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is False
            assert "paths" in str(resume.get("error"))

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert "malformed.txt" not in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_works_without_selected_model(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@test"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "test"],
        capture_output=True,
        check=True,
    )
    (repo / "nomodel.txt").write_text("x\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            # intentionally NOT selecting a model
            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": ["nomodel.txt"]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True

            from server.http.common.workspace_git import git_status

            status = git_status(repo)
            staged = [e["path"] for e in status["staged"]]
            assert "nomodel.txt" in staged
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_workspace_git_reapprove_does_not_run_twice(
    tmp_path,
) -> None:
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], capture_output=True, check=True)
    (repo / "once.txt").write_text("x\n")

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_workspace_root(session_id, str(repo))

            stage_resp = await client.post(
                "/ui/api/workspace/git/stage",
                headers={"X-Slavik-Session": session_id},
                json={"paths": ["once.txt"]},
            )
            assert stage_resp.status == 202
            stage_payload = await stage_resp.json()
            decision = stage_payload.get("decision")
            assert isinstance(decision, dict)
            decision_id = decision.get("id")
            assert isinstance(decision_id, str)

            respond = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert respond.status == 200
            respond_payload = await respond.json()
            resume = respond_payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True

            from server.http.common.workspace_git import git_status

            status_after_first = git_status(repo)
            staged_after_first = [e["path"] for e in status_after_first["staged"]]
            assert "once.txt" in staged_after_first

            reapprove = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": decision_id,
                    "choice": "approve_once",
                },
            )
            assert reapprove.status == 409
            reapprove_payload = await reapprove.json()
            error = reapprove_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") in {"decision_already_resolved", "decision_not_pending"}

            status_after_reapprove = git_status(repo)
            staged_after_reapprove = [e["path"] for e in status_after_reapprove["staged"]]
            assert staged_after_reapprove == staged_after_first
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
            source_request = resume_payload.get("source_request")
            assert isinstance(source_request, dict)
            assert source_request.get("lane") == "chat"
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


@pytest.mark.behavior
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


def test_ui_decision_respond_agent_decision_retry_uses_chat_lane() -> None:
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
                        "id": "decision-retry-workspace",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "reason": "need_user_input",
                        "summary": "Need retry choice",
                        "context": {},
                        "options": [
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
                        "default_option_id": "retry",
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
            assert send_payload.get("lane") == "chat"
            chat_messages = send_payload.get("messages")
            assert isinstance(chat_messages, list)
            assert len(chat_messages) == 1
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
            assert resume.get("ok") is True
            assert agent.calls == 2

            chat_history_resp = await client.get(f"/ui/api/sessions/{session_id}/history")
            assert chat_history_resp.status == 200
            chat_history_payload = await chat_history_resp.json()
            chat_history = chat_history_payload.get("messages")
            assert isinstance(chat_history, list)
            assert len(chat_history) == 3
            last_item = chat_history[-1]
            assert isinstance(last_item, dict)
            assert last_item.get("role") == "assistant"
            assert last_item.get("content") == "retry-ok"
            assert all(
                isinstance(item, dict) and item.get("lane") == "chat" for item in chat_history
            )
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
            assert retry_resp.status == 409
            retry_payload = await retry_resp.json()
            error = retry_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "resume_payload_missing"
            decision_after = await hub.get_session_decision(session_id)
            assert isinstance(decision_after, dict)
            assert decision_after.get("status") == "pending"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_returns_409_for_already_resolved_decision() -> None:
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
                    "id": "decision-resolved-1",
                    "kind": "decision",
                    "decision_type": "agent_decision",
                    "status": "pending",
                    "blocking": True,
                    "reason": "need_user_input",
                    "summary": "Resolve once",
                    "proposed_action": {},
                    "options": [
                        {
                            "id": "abort",
                            "title": "Abort",
                            "action": "abort",
                            "payload": {},
                            "risk": "low",
                        }
                    ],
                    "default_option_id": "abort",
                    "context": {
                        "session_id": session_id,
                        "source_endpoint": "chat.agent_decision",
                        "resume_payload": {
                            "source_request": {
                                "content": "x",
                                "force_canvas": False,
                                "lane": "chat",
                            },
                            "selected_model_snapshot": {
                                "provider": "local",
                                "model": "local-default",
                            },
                        },
                    },
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )
            first = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-resolved-1",
                    "choice": "abort",
                },
            )
            assert first.status == 200

            second = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-resolved-1",
                    "choice": "abort",
                },
            )
            assert second.status == 409
            second_payload = await second.json()
            error = second_payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "decision_already_resolved"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_returns_409_for_non_pending_status() -> None:
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
                    "id": "decision-executing-1",
                    "kind": "decision",
                    "decision_type": "agent_decision",
                    "status": "executing",
                    "blocking": True,
                    "reason": "need_user_input",
                    "summary": "Executing decision",
                    "proposed_action": {},
                    "options": [],
                    "default_option_id": None,
                    "context": {"session_id": session_id, "source_endpoint": "chat.agent_decision"},
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )

            response = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-executing-1",
                    "choice": "reject",
                },
            )
            assert response.status == 409
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "decision_not_pending"
        finally:
            await client.close()

    asyncio.run(run())


def test_ui_decision_respond_rejects_expired_packet() -> None:
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
                    "id": "decision-expired-1",
                    "kind": "decision",
                    "decision_type": "agent_decision",
                    "status": "pending",
                    "blocking": True,
                    "reason": "need_user_input",
                    "summary": "Expired packet",
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
                    "context": {"session_id": session_id},
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "expires_at": "2026-01-01T00:00:01+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "resolved_at": None,
                },
            )

            response = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "decision-expired-1",
                    "choice": "abort",
                },
            )
            assert response.status == 410
            payload = await response.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "decision_expired"

            current = await hub.get_session_decision(session_id)
            assert isinstance(current, dict)
            assert current.get("status") == "expired"
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.behavior
def test_ui_chat_send_idempotency_key_matrix() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            status_payload = await status_resp.json()
            session_id = status_payload.get("session_id")
            assert isinstance(session_id, str)
            await _select_local_model(client, session_id)

            headers = {
                "X-Slavik-Session": session_id,
                "Idempotency-Key": "chat-send-k1",
            }
            first = await client.post(
                "/ui/api/chat/send",
                headers=headers,
                json={"content": "Ping idempotency"},
            )
            assert first.status == 200
            first_payload = await first.json()
            first_messages = first_payload.get("messages")
            assert isinstance(first_messages, list)
            first_count = len(first_messages)
            assert first_count >= 2

            replay = await client.post(
                "/ui/api/chat/send",
                headers=headers,
                json={"content": "Ping idempotency"},
            )
            assert replay.status == 200
            replay_payload = await replay.json()
            replay_messages = replay_payload.get("messages")
            assert isinstance(replay_messages, list)
            assert len(replay_messages) == first_count
            assert replay_messages == first_messages

            conflict = await client.post(
                "/ui/api/chat/send",
                headers=headers,
                json={"content": "Ping idempotency changed"},
            )
            assert conflict.status == 409
            conflict_payload = await conflict.json()
            conflict_error = conflict_payload.get("error")
            assert isinstance(conflict_error, dict)
            assert conflict_error.get("code") == "idempotency_key_reused"

            no_key = await client.post(
                "/ui/api/chat/send",
                headers={"X-Slavik-Session": session_id},
                json={"content": "Ping idempotency"},
            )
            assert no_key.status == 200
            no_key_payload = await no_key.json()
            no_key_messages = no_key_payload.get("messages")
            assert isinstance(no_key_messages, list)
            assert len(no_key_messages) > first_count
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


# ── PR-21: computer_commit decision flow ─────────────────────────────────────


class _ComputerCommitAgent(DummyAgent):
    """Records all call_tool invocations; returns success for workspace_terminal_run."""

    def __init__(self, *, terminal_ok: bool = True) -> None:
        super().__init__()
        self.tool_calls: list[tuple[str, dict[str, JSONValue]]] = []
        self._terminal_ok = terminal_ok

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult:
        del raw_input
        call_args = dict(args or {})
        self.tool_calls.append((name, call_args))
        if name == "workspace_terminal_run":
            if self._terminal_ok:
                return ToolResult.success({"output": "ok", "stderr": "", "exit_code": 0})
            return ToolResult.failure("simulated terminal failure")
        return ToolResult.failure(f"unsupported tool {name}")


def _make_computer_commit_packet(
    *,
    decision_id: str = "ccommit-1",
    session_id: str,
    changed_files: list[str] | None = None,
    commit_message: str = "feat: test commit",
    diff_summary: str = "",
) -> dict[str, JSONValue]:
    return {
        "id": decision_id,
        "kind": "decision",
        "decision_type": "computer_commit",
        "status": "pending",
        "blocking": True,
        "reason": "computer_changes_review",
        "summary": f"Changes ready to commit: {commit_message}",
        "proposed_action": {
            "category": "computer_changes_review",
            "changed_files": changed_files if changed_files is not None else ["core/foo.py"],
            "diff_summary": diff_summary,
            "proposed_commit_message": commit_message,
        },
        "options": [
            {
                "id": "approve_once",
                "title": "Commit",
                "action": "approve",
                "payload": {},
                "risk": "low",
            },
            {"id": "reject", "title": "Cancel", "action": "reject", "payload": {}, "risk": "low"},
        ],
        "default_option_id": "approve_once",
        "context": {"session_id": session_id},
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "resolved_at": None,
    }


def test_computer_commit_approve_calls_git_add_then_commit() -> None:
    """approve_once → agent.call_tool with git add then git commit, in that order."""

    async def run() -> None:
        agent = _ComputerCommitAgent(terminal_ok=True)
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id,
                _make_computer_commit_packet(
                    session_id=session_id,
                    changed_files=["core/foo.py", "tests/test_foo.py"],
                    commit_message="feat: wired commit",
                ),
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "approve_once",
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            assert resume.get("source_endpoint") == "computer.commit"
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("committed") is True

            assert len(agent.tool_calls) == 2
            add_name, add_args = agent.tool_calls[0]
            commit_name, commit_args = agent.tool_calls[1]
            assert add_name == "workspace_terminal_run"
            assert commit_name == "workspace_terminal_run"
            add_cmd = add_args.get("command", "")
            commit_cmd = commit_args.get("command", "")
            assert isinstance(add_cmd, str) and "git add" in add_cmd
            assert isinstance(commit_cmd, str) and "git commit" in commit_cmd
            assert "core/foo.py" in add_cmd
            assert "tests/test_foo.py" in add_cmd
            assert "feat: wired commit" in commit_cmd
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_approve_no_push_merge_checkout() -> None:
    """approve_once must not issue git push, merge, checkout, or switch."""

    async def run() -> None:
        agent = _ComputerCommitAgent(terminal_ok=True)
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id, _make_computer_commit_packet(session_id=session_id)
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "approve_once",
                },
            )
            assert resp.status == 200

            for _name, args in agent.tool_calls:
                cmd = str(args.get("command", ""))
                assert "push" not in cmd.lower(), f"Unexpected push: {cmd!r}"
                assert "merge" not in cmd.lower(), f"Unexpected merge: {cmd!r}"
                assert "checkout" not in cmd.lower(), f"Unexpected checkout: {cmd!r}"
                assert "switch" not in cmd.lower(), f"Unexpected switch: {cmd!r}"
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_reject_does_not_call_gateway() -> None:
    """reject → no tool calls, decision resolves with action=reject acknowledged."""

    async def run() -> None:
        agent = _ComputerCommitAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id, _make_computer_commit_packet(session_id=session_id)
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "reject",
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is True
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("action") == "reject"
            assert data.get("acknowledged") is True
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_invalid_choice_returns_400() -> None:
    """Choices other than approve_once|reject return 400 for computer_commit decisions."""

    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id, _make_computer_commit_packet(session_id=session_id)
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "ask_user",
                },
            )
            assert resp.status == 400
            payload = await resp.json()
            error = payload.get("error")
            assert isinstance(error, dict)
            assert error.get("code") == "invalid_request_error"
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_empty_files_resolves_with_invalid_data() -> None:
    """Packet with empty changed_files → decision resolves, ok=False, error=invalid_commit_data."""

    async def run() -> None:
        agent = _ComputerCommitAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id,
                _make_computer_commit_packet(session_id=session_id, changed_files=[]),
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "approve_once",
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is False
            assert resume.get("error") == "invalid_commit_data"
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_blank_message_resolves_with_invalid_data() -> None:
    """Packet with blank proposed_commit_message → ok=False, error=invalid_commit_data."""

    async def run() -> None:
        agent = _ComputerCommitAgent()
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id,
                _make_computer_commit_packet(session_id=session_id, commit_message="   "),
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "approve_once",
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is False
            assert resume.get("error") == "invalid_commit_data"
            assert len(agent.tool_calls) == 0
        finally:
            await client.close()

    asyncio.run(run())


def test_computer_commit_add_failure_resolves_with_committed_false() -> None:
    """git add failure → decision resolves, committed=False, git commit NOT called."""

    async def run() -> None:
        agent = _ComputerCommitAgent(terminal_ok=False)
        client = await _create_client(agent)
        try:
            status_resp = await client.get("/ui/api/status")
            session_id = (await status_resp.json()).get("session_id")
            assert isinstance(session_id, str)

            hub: UIHub = client.server.app["ui_hub"]
            await hub.set_session_decision(
                session_id, _make_computer_commit_packet(session_id=session_id)
            )

            resp = await client.post(
                "/ui/api/decision/respond",
                headers={"X-Slavik-Session": session_id},
                json={
                    "session_id": session_id,
                    "decision_id": "ccommit-1",
                    "choice": "approve_once",
                },
            )
            assert resp.status == 200
            payload = await resp.json()
            assert payload.get("status") == "resolved"
            resume = payload.get("resume")
            assert isinstance(resume, dict)
            assert resume.get("ok") is False
            data = resume.get("data")
            assert isinstance(data, dict)
            assert data.get("committed") is False
            # Only git add attempted; commit must NOT have been called
            assert len(agent.tool_calls) == 1
            assert "git add" in agent.tool_calls[0][1].get("command", "")
        finally:
            await client.close()

    asyncio.run(run())
