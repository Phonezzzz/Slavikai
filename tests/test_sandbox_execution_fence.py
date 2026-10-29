from __future__ import annotations

import subprocess

import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import RunCodeTool, workspace_root_context


def test_unqualified_backend_and_model_arguments_cannot_enable_native_process(
    tmp_path, monkeypatch
):
    from core.container_computer_backend import ContainerComputerBackend

    (tmp_path / "script.py").write_text("print('synthetic')")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("unqualified backend must not spawn")

    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    backend = ContainerComputerBackend(image="image", project_root=tmp_path, runtime="podman")
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_run", RunCodeTool(backend=backend), capability="exec")
    registry.set_execution_policy(mode="auto")
    gateway = ToolGateway(registry=registry)
    with workspace_root_context(tmp_path):
        result = gateway.call(
            ToolRequest(
                "workspace_run",
                {
                    "path": "script.py",
                    "isolation_verified": True,
                    "workspace_binding": {"principal_id": "alice", "revision": "forged"},
                },
            )
        )
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
    assert result.meta["execution_started"] is False
    assert "exit_code" not in result.data
    assert calls == []


def test_approved_sandbox_run_cannot_spawn_host_process(tmp_path, monkeypatch) -> None:
    script = tmp_path / "script.py"
    script.write_text('print("private data")\n')
    calls = []

    def run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(a[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "tools.log"))
    registry.register("workspace_run", RunCodeTool(), capability="exec")
    registry.set_execution_policy(mode="auto")
    context = ApprovalContext(safe_mode=True, session_id="s", approved_categories=set())
    gateway = ToolGateway(registry=registry, approval_context=context)
    request = ToolRequest(name="workspace_run", args={"path": "script.py"})
    with workspace_root_context(tmp_path):
        with pytest.raises(ApprovalRequired) as pending:
            gateway.call(request)
        assert calls == []
        foreign = ToolGateway(
            registry=registry,
            approval_context=ApprovalContext(
                safe_mode=True,
                session_id="foreign",
                approved_categories=set(),
            ),
        )
        with pytest.raises(ValueError, match="approval_subject_mismatch"):
            foreign.call_approved_once(request, pending.value.request)
        result = gateway.call_approved_once(request, pending.value.request)
    assert context.approved_categories == set()
    assert calls == []
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
    assert result.meta["execution_started"] is False
    assert "exit_code" not in result.data


def test_required_verifier_does_not_spawn_or_report_pass(tmp_path, monkeypatch) -> None:
    from core.mwv.models import RunContext, TaskPacket, VerificationStatus
    from core.mwv.verifier_runtime import VerifierRuntime

    calls = []

    def run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(a[0], 0, stdout="ok", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    task = TaskPacket(
        task_id="task",
        session_id="s",
        trace_id="t",
        goal="verify",
        verifier={"command": ["make", "check"]},
    )
    context = RunContext(
        session_id="s", trace_id="t", workspace_root=str(tmp_path), safe_mode=False
    )
    result = VerifierRuntime().run(task, context)
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"
    assert result.exit_code is None


def test_manual_git_and_container_do_not_bypass_fence(tmp_path, monkeypatch) -> None:
    from core.container_computer_backend import ContainerComputerBackend
    from server.http.common.workspace_git import git_status

    calls = []

    def run(*a, **k):
        calls.append(a)
        return subprocess.CompletedProcess(a[0], 0, stdout="", stderr="")

    monkeypatch.setattr("subprocess.run", run)
    result = git_status(tmp_path)
    assert calls == []
    assert result["ok"] is False

    class Runner:
        def run(self, args):
            calls.append(args)
            return 0, "ok", ""

    backend = ContainerComputerBackend(image="unqualified", project_root=tmp_path, runner=Runner())
    result = backend.run_command("echo data")
    assert calls == []
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"


def test_manual_terminal_does_not_spawn_or_accept_input(tmp_path, monkeypatch) -> None:
    import asyncio

    from tools.terminal_tool import TerminalTool

    calls = []
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))

    async def run():
        terminal = TerminalTool()
        for operation in (
            terminal.create_or_get("s", workspace_root=str(tmp_path)),
            terminal.write_input("s", "echo private\n"),
        ):
            try:
                await operation
            except RuntimeError as exc:
                assert "изолированная среда" in str(exc)
            else:
                raise AssertionError("unqualified terminal operation accepted")
        assert calls == []
        assert await terminal.get_snapshot("s") is None

    asyncio.run(run())


def test_workspace_import_diff_and_direct_container_runner_cannot_spawn(tmp_path, monkeypatch):
    import asyncio

    from core.container_computer_backend import SubprocessContainerRunner
    from server.http.common.github_import import clone_github_repository
    from server.http.common.workspace_index import workspace_git_diff
    from shared.sandbox_execution import SandboxExecutionUnavailable

    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))
    marker = tmp_path / "preserve.txt"
    marker.write_text("owned fixture")
    ok, message = asyncio.run(
        clone_github_repository(
            repo_url="https://example.invalid/repo",
            branch="main",
            target_path=tmp_path,
        )
    )
    assert not ok
    assert "изолированная среда" in message
    assert workspace_git_diff(tmp_path)[0] == ""
    assert "изолированная среда" in workspace_git_diff(tmp_path)[1]
    with pytest.raises(SandboxExecutionUnavailable):
        SubprocessContainerRunner().run(["docker", "run", "unqualified"])
    assert calls == []
    assert marker.read_text() == "owned fixture"


def test_manual_agent_run_entrypoint_cannot_use_category_only_execution_resume(monkeypatch):
    from core.agent import Agent
    from shared.models import ToolResult

    agent = Agent.__new__(Agent)
    calls = []

    def category_only_dispatch(*args, **kwargs):
        calls.append(args)
        return ToolResult.success({"exit_code": 0})

    monkeypatch.setattr(agent, "_call_tool_logged", category_only_dispatch)
    result = agent.call_tool("workspace_run", {"path": "script.py"}, raw_input="ui:workspace_run")
    assert not result.ok
    assert result.meta["reason_code"] == "sandbox_execution_unavailable"
    assert result.meta["execution_started"] is False
    assert calls == []
