"""Штатный process verifier: qualified command execution (E04-T03 slice).

Unit-тесты — на fake executor/profile без engine. Real-backend E2E —
opt-in через SLAVIK_RUN_QUALIFIED_VERIFIER_E2E=1: строит настоящий rootless
state, пуллит оба pinned образа, квалифицирует профиль и гоняет make check
в контейнере. В CI пропускается.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.auto_preset import AutoBudgets, AutoPresetRunner
from core.managed_workspace import ManagedWorkspace, WorkspaceAccess, WorkspaceLimits
from core.mwv.models import RunContext, TaskPacket, VerificationStatus
from core.mwv.verifier_runtime import (
    CANONICAL_CHECK_COMMAND as VERIFIER_CANONICAL_CHECK_COMMAND,
)
from core.mwv.verifier_runtime import (
    VerifierRuntime,
    has_canonical_check_definition,
)
from core.rootless_podman import (
    CANONICAL_CHECK_COMMAND,
    QUALIFIED_AMD64_DIGEST,
    QUALIFIED_COMMAND_AMD64_DIGEST,
    QUALIFIED_COMMAND_DIGEST,
    QUALIFIED_COMMAND_IMAGE,
    QUALIFIED_PYTHON_DIGEST,
    QUALIFIED_PYTHON_IMAGE,
    PreparedCommandContainer,
    RootlessPodmanProfile,
)
from core.rootless_staging import RootlessStagePool, StagingLimits
from core.workspace_python_execution import (
    RootlessCommandExecutor,
    RootlessPythonExecutor,
    WorkspacePythonOutcome,
)
from shared.workspace_binding import WorkspaceUnavailable

CHECK_MAKEFILE = (
    "check:\n"
    "\t@test -f result.txt\n"
    '\t@grep -qx "expected ok" result.txt\n'
    '\t@grep -qx "42" result.txt\n'
)
CORRECT_RESULT = "expected ok\n42\n"
WRONG_RESULT = "expected ok\n43\n"


def _profile(tmp_path, *, command_revision="cmd-rev"):
    return RootlessPodmanProfile(
        tmp_path,
        os.geteuid(),
        os.getegid(),
        "python-rev",
        (),
        (),
        command_revision=command_revision,
    )


def _fake_executor(outcome=None, *, calls=None):
    class _Fake:
        def __init__(self):
            self.calls = [] if calls is None else calls

        def is_available(self):
            return True

        def execute_command(self, binding, argv, *, timeout, criterion_sha256=None):
            self.calls.append((binding, list(argv), timeout, criterion_sha256))
            assert outcome is not None, "fake executor must not be called here"
            return outcome

    return _Fake()


def _finished(exit_code, *, stdout=b"", stderr=b""):
    return WorkspacePythonOutcome("finished", stdout, stderr, exit_code, True, None, "none")


def _task(tmp_path, verifier):
    return TaskPacket(task_id="t", session_id="s", trace_id="x", goal="check", verifier=verifier)


def _context(tmp_path):
    return RunContext(session_id="s", trace_id="x", workspace_root=str(tmp_path), safe_mode=False)


def _forbidden_engine(*args, **kwargs):
    raise AssertionError("engine must not be contacted")


def _runtime_with_executor(tmp_path, executor, *, binding=True):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700, exist_ok=True)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    workspace_binding = (
        workspace.bind(principal_id="alice", session_id="chat-a") if binding else None
    )
    return VerifierRuntime(executor=executor, binding=workspace_binding), workspace


# --- podman: allowlist и command-ревизия -------------------------------


def test_command_container_refuses_non_allowlisted_argv_before_engine(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    monkeypatch.setattr("subprocess.run", _forbidden_engine)
    for argv in (["make", "check", "extra"], ["make"], ["python", "--", "/workspace/x.py"], []):
        with pytest.raises(WorkspaceUnavailable, match="allowlisted"):
            profile.create_command_container(SimpleNamespace(), argv)


def test_command_container_refuses_without_qualified_command_profile(tmp_path, monkeypatch):
    profile = _profile(tmp_path, command_revision=None)
    monkeypatch.setattr("subprocess.run", _forbidden_engine)
    with pytest.raises(WorkspaceUnavailable, match="not qualified"):
        profile.create_command_container(
            SimpleNamespace(), ["make", "-B", "-f", "Makefile", "check"]
        )


def _command_prepared(tmp_path, *, command=("make", "-B", "-f", "Makefile", "check")):
    stage_root = tmp_path / "stage"
    stage_root.mkdir(mode=0o700)
    info = os.stat(stage_root)
    binding = SimpleNamespace(environment_revision="env-rev")
    stage = SimpleNamespace(root=stage_root, identity=(info.st_dev, info.st_ino), binding=binding)
    pool = SimpleNamespace(check_stage=lambda bounded: None, execution_revision="env-rev")
    bounded = SimpleNamespace(pool=pool, stage=stage, engine_root=stage_root)
    return PreparedCommandContainer(
        "c" * 64, "slavikai-workspace-test", bounded, command, "cmd-rev"
    )


def _inspect_record(tmp_path, *, command):
    profile = _profile(tmp_path)
    return {
        "Id": "c" * 64,
        "Name": "slavikai-workspace-test",
        "Image": QUALIFIED_COMMAND_IMAGE,
        "OCIRuntime": "runc",
        "State": {"Status": "created", "Running": False, "Pid": 0},
        "Config": {
            "WorkingDir": "/workspace",
            "User": f"{profile.uid}:{profile.gid}",
            "Cmd": list(command),
        },
    }


def test_verify_created_command_rejects_cmd_mismatch(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    prepared = _command_prepared(tmp_path)
    record = _inspect_record(tmp_path, command=["make", "check", "extra"])
    monkeypatch.setattr(RootlessPodmanProfile, "json_control", lambda self, args: [record])
    with pytest.raises(WorkspaceUnavailable, match="differs from qualified profile"):
        profile.verify_created_command(prepared)


def test_verify_created_command_passes_canonical_cmd_then_checks_rest(tmp_path, monkeypatch):
    # Тот же record, но с каноническим Cmd: проверка Cmd пройдена, отказ —
    # уже на следующем отсутствующем поле (другое сообщение).
    profile = _profile(tmp_path)
    prepared = _command_prepared(tmp_path)
    record = _inspect_record(tmp_path, command=["make", "-B", "-f", "Makefile", "check"])
    monkeypatch.setattr(RootlessPodmanProfile, "json_control", lambda self, args: [record])
    with pytest.raises(WorkspaceUnavailable, match="incomplete effective container profile"):
        profile.verify_created_command(prepared)


# --- executor: derivation и availability -------------------------------


def test_command_executor_derivation_none_cases(tmp_path):
    assert RootlessCommandExecutor.from_python_executor(None) is None
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    profile = _profile(tmp_path)
    executor = RootlessPythonExecutor(
        profile=profile,
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=None,
    )
    assert RootlessCommandExecutor.from_python_executor(executor) is None


def test_command_executor_is_available_requires_command_revision(tmp_path, monkeypatch):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    limits = WorkspaceLimits(1024, 4096, 20, 8)
    monkeypatch.setattr("subprocess.run", _forbidden_engine)
    unqualified = RootlessCommandExecutor(
        profile=_profile(tmp_path, command_revision=None),
        workspace=workspace,
        limits=limits,
        max_output_bytes=1024,
        staging_pool=None,
    )
    assert not unqualified.is_available()
    no_pool = RootlessCommandExecutor(
        profile=_profile(tmp_path),
        workspace=workspace,
        limits=limits,
        max_output_bytes=1024,
        staging_pool=None,
    )
    assert not no_pool.is_available()


def test_command_executor_unavailable_returns_error_without_staging(tmp_path, monkeypatch):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    executor = RootlessCommandExecutor(
        profile=_profile(tmp_path, command_revision=None),
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=None,
    )
    monkeypatch.setattr("subprocess.run", _forbidden_engine)
    outcome = executor.execute_command(
        binding, ["make", "-B", "-f", "Makefile", "check"], timeout=60
    )
    assert outcome.state == "unavailable"
    assert outcome.execution_started is False
    assert outcome.reason_code == "sandbox_execution_unavailable"
    assert outcome.publication == "none"


# --- VerifierRuntime: маппинг исходов ----------------------------------


def test_canonical_check_commands_agree():
    assert CANONICAL_CHECK_COMMAND == ("make", "-B", "-f", "Makefile", "check")
    assert VERIFIER_CANONICAL_CHECK_COMMAND == CANONICAL_CHECK_COMMAND


def test_verifier_maps_finished_zero_to_passed(tmp_path):
    executor = _fake_executor(_finished(0, stdout=b"ok\n"))
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert result.status == VerificationStatus.PASSED
    assert result.exit_code == 0
    assert result.command == ["make", "-B", "-f", "Makefile", "check"]
    assert result.stdout == "ok\n"
    assert result.verifier_profile == "canonical"
    assert result.error is None


def test_verifier_maps_finished_nonzero_to_failed(tmp_path):
    executor = _fake_executor(_finished(2, stderr=b"make: *** [Makefile:2: check] Error 1\n"))
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert result.status == VerificationStatus.FAILED
    assert result.exit_code == 2
    assert result.error == "canonical check failed"
    assert result.fail_type == "check_failed"


def test_verifier_maps_unavailable_to_error(tmp_path):
    executor = _fake_executor(
        WorkspacePythonOutcome(
            "unavailable", b"", b"", None, False, "sandbox_execution_unavailable", "none"
        )
    )
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"
    assert result.fail_type == "execution_unavailable"
    assert result.exit_code is None
    assert result.stderr != ""


def test_verifier_maps_unknown_and_conflict_to_error_never_failed(tmp_path):
    for state, fail_type in (
        ("unknown", "execution_unknown"),
        ("conflict", "execution_conflict"),
    ):
        executor = _fake_executor(
            WorkspacePythonOutcome(
                state, b"out", b"err", None, None, "workspace_conflict", "retained"
            )
        )
        runtime, _ = _runtime_with_executor(tmp_path, executor)
        result = runtime.run(
            _task(
                tmp_path,
                {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)},
            ),
            _context(tmp_path),
        )
        assert result.status == VerificationStatus.ERROR
        assert result.fail_type == fail_type
        assert result.exit_code is None


def test_verifier_without_executor_unchanged(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr("subprocess.Popen", lambda *a, **k: calls.append(a))

    class Runner:
        def run(self):
            calls.append("runner")
            raise AssertionError("unqualified fallback runner")

    result = VerifierRuntime(runner=Runner()).run(
        _task(tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"]}), _context(tmp_path)
    )
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"
    assert result.exit_code is None


def test_verifier_rejects_non_canonical_command_without_calling_executor(tmp_path):
    calls = []
    executor = _fake_executor(_finished(0), calls=calls)
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    result = runtime.run(
        _task(tmp_path, {"command": ["make", "test"], "cwd": str(tmp_path)}),
        _context(tmp_path),
    )
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error.startswith("invalid_verifier_config")


def test_verifier_requires_binding_with_executor(tmp_path):
    calls = []
    executor = _fake_executor(_finished(0), calls=calls)
    runtime, _ = _runtime_with_executor(tmp_path, executor, binding=False)
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error == "verifier_binding_required"


def test_verifier_rejects_cwd_outside_workspace_root(tmp_path):
    calls = []
    executor = _fake_executor(_finished(0), calls=calls)
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    subdir = tmp_path / "sub"
    subdir.mkdir()
    result = runtime.run(
        _task(tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(subdir)}),
        _context(tmp_path),
    )
    assert calls == []
    assert result.status == VerificationStatus.ERROR
    assert result.error.startswith("invalid_verifier_config")


def test_verifier_truncates_long_output(tmp_path):
    executor = _fake_executor(_finished(0, stdout=b"x" * 9000))
    runtime, _ = _runtime_with_executor(tmp_path, executor)
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert result.status == VerificationStatus.PASSED
    assert len(result.stdout) == 8000 + len("\n[truncated]")
    assert result.stdout.endswith("[truncated]")


def test_has_canonical_check_definition_ignores_host_make(tmp_path, monkeypatch):
    assert not has_canonical_check_definition(tmp_path)
    (tmp_path / "Makefile").write_text(CHECK_MAKEFILE)
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert has_canonical_check_definition(tmp_path)


# --- _verify_after_loop: fail-closed и хэш критерия --------------------


def _stub_parent(tmp_path):
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    profile = _profile(tmp_path)
    python_executor = RootlessPythonExecutor(
        profile=profile,
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=None,
    )
    access = WorkspaceAccess(workspace, "alice", "chat-a")
    stub = SimpleNamespace(
        user_id="alice",
        session_id="chat-a",
        tools_enabled={"safe_mode": True},
        approved_categories=set(),
        _workspace_executor=python_executor,
        _workspace_access=access,
    )
    return stub, workspace


def _loop_result(*, with_calls=True):
    return SimpleNamespace(
        tool_calls=[object()] if with_calls else [],
        text="done",
    )


def _budgets():
    return AutoBudgets(900, 80, 120, 120000, 0)


def test_verify_after_loop_fails_closed_without_backend(tmp_path):
    stub, workspace = _stub_parent(tmp_path)
    run_root = workspace.root
    (run_root / "Makefile").write_text(CHECK_MAKEFILE)
    criterion = hashlib.sha256(CHECK_MAKEFILE.encode()).hexdigest()
    preset = AutoPresetRunner(stub, workspace_root=run_root)
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=run_root,
        budgets=_budgets(),
        loop_result=_loop_result(),
        criterion_sha256=criterion,
    )
    assert result is not None
    assert result.status == VerificationStatus.ERROR
    assert result.error == "sandbox_execution_unavailable"


def test_verify_after_loop_passes_with_fake_executor_when_hash_matches(tmp_path, monkeypatch):
    stub, workspace = _stub_parent(tmp_path)
    run_root = workspace.root
    (run_root / "Makefile").write_text(CHECK_MAKEFILE)
    (run_root / "result.txt").write_text(CORRECT_RESULT)
    criterion = hashlib.sha256(CHECK_MAKEFILE.encode()).hexdigest()
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    calls = []
    executor = _fake_executor(_finished(0, stdout=b"ok\n"), calls=calls)
    monkeypatch.setattr(
        AutoPresetRunner,
        "_verifier_command_executor",
        lambda self, root, **kwargs: (executor, binding),
    )
    preset = AutoPresetRunner(stub, workspace_root=run_root)
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=run_root,
        budgets=_budgets(),
        loop_result=_loop_result(),
        criterion_sha256=criterion,
    )
    assert result is not None
    assert result.status == VerificationStatus.PASSED
    assert len(calls) == 1
    assert calls[0][1] == ["make", "-B", "-f", "Makefile", "check"]


def test_verify_after_loop_rejects_changed_makefile(tmp_path, monkeypatch):
    stub, workspace = _stub_parent(tmp_path)
    run_root = workspace.root
    (run_root / "Makefile").write_text(CHECK_MAKEFILE)
    criterion = hashlib.sha256(CHECK_MAKEFILE.encode()).hexdigest()
    # Агент подменил критерий после фиксации хэша.
    (run_root / "Makefile").write_text("check:\n\t@true\n")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    calls = []
    executor = _fake_executor(_finished(0), calls=calls)
    monkeypatch.setattr(
        AutoPresetRunner,
        "_verifier_command_executor",
        lambda self, root, **kwargs: (executor, binding),
    )
    preset = AutoPresetRunner(stub, workspace_root=run_root)
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=run_root,
        budgets=_budgets(),
        loop_result=_loop_result(),
        criterion_sha256=criterion,
    )
    assert calls == []
    assert result is not None
    assert result.status == VerificationStatus.ERROR
    assert result.error == "verifier_criterion_changed"


def test_verify_after_loop_missing_makefile_non_repo_returns_none(tmp_path):
    stub, workspace = _stub_parent(tmp_path)
    preset = AutoPresetRunner(stub, workspace_root=workspace.root)
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=workspace.root,
        budgets=_budgets(),
        loop_result=_loop_result(),
    )
    assert result is None


def test_verify_after_loop_repo_without_makefile_fails_honestly(tmp_path):
    stub, workspace = _stub_parent(tmp_path)
    (workspace.root / ".git").mkdir()
    preset = AutoPresetRunner(stub, workspace_root=workspace.root)
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=workspace.root,
        budgets=_budgets(),
        loop_result=_loop_result(),
    )
    assert result is not None
    assert result.status == VerificationStatus.ERROR
    assert result.error == "canonical_repo_verifier_unavailable"


def test_verify_after_loop_deleted_criterion_fails(tmp_path, monkeypatch):
    # Критерий зафиксирован до loop, Makefile удалён к verify-time:
    # обязательная проверка потеряна — ERROR, executor не вызван.
    stub, workspace = _stub_parent(tmp_path)
    (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
    criterion = hashlib.sha256(CHECK_MAKEFILE.encode()).hexdigest()
    (workspace.root / "Makefile").unlink()
    calls = []
    executor = _fake_executor(_finished(0, stdout=b"ok\n"), calls=calls)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    preset = AutoPresetRunner(stub, workspace_root=workspace.root)
    monkeypatch.setattr(
        AutoPresetRunner,
        "_verifier_command_executor",
        lambda self, root, **kwargs: (executor, binding),
    )
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=workspace.root,
        budgets=_budgets(),
        loop_result=_loop_result(),
        criterion_sha256=criterion,
    )
    assert result is not None
    assert result.status == VerificationStatus.ERROR
    assert result.error == "verifier_criterion_changed"
    assert result.fail_type == "criterion_changed"
    assert calls == []


def test_verify_after_loop_planted_makefile_without_criterion_returns_none(tmp_path, monkeypatch):
    # Критерий до loop не зафиксирован, а Makefile появился во время
    # loop: он мог быть подложен агентом — не исполнять, вернуть None.
    stub, workspace = _stub_parent(tmp_path)
    (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
    calls = []
    executor = _fake_executor(_finished(0, stdout=b"ok\n"), calls=calls)
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    preset = AutoPresetRunner(stub, workspace_root=workspace.root)
    monkeypatch.setattr(
        AutoPresetRunner,
        "_verifier_command_executor",
        lambda self, root, **kwargs: (executor, binding),
    )
    result = preset._verify_after_loop(
        run_id="r",
        goal="check",
        run_root=workspace.root,
        budgets=_budgets(),
        loop_result=_loop_result(),
    )
    assert result is None
    assert calls == []


def test_verifier_command_executor_rejects_foreign_binding(tmp_path):
    stub, workspace = _stub_parent(tmp_path)
    profile = _profile(tmp_path)
    pool = SimpleNamespace(profile=profile, execution_revision="env", check_current=lambda: None)
    python_executor = RootlessPythonExecutor(
        profile=profile,
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=pool,
    )
    stub._workspace_executor = python_executor
    stub._workspace_access = WorkspaceAccess(workspace, "alice", "chat-a", "env")
    preset = AutoPresetRunner(stub, workspace_root=workspace.root)
    # Чужой principal, чужая сессия и биндинг без revision — отказ.
    assert preset._verifier_command_executor(
        workspace.root, principal_id="bob", session_id="chat-a"
    ) == (None, None)
    assert preset._verifier_command_executor(
        workspace.root, principal_id="alice", session_id="other"
    ) == (None, None)
    stub._workspace_access = WorkspaceAccess(workspace, "alice", "chat-a")
    assert preset._verifier_command_executor(
        workspace.root, principal_id="alice", session_id="chat-a"
    ) == (None, None)
    # Свой principal/session и revision пула — (executor, binding).
    stub._workspace_access = WorkspaceAccess(workspace, "alice", "chat-a", "env")
    executor, binding = preset._verifier_command_executor(
        workspace.root, principal_id="alice", session_id="chat-a"
    )
    assert executor is not None
    assert binding is not None
    assert binding.principal_id == "alice"
    assert binding.session_id == "chat-a"
    assert binding.environment_revision == "env"


def test_verifier_maps_criterion_changed_to_error(tmp_path):
    outcome = WorkspacePythonOutcome(
        "conflict", b"", b"", None, False, "verifier_criterion_changed", "none"
    )
    executor = _fake_executor(outcome)
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700, exist_ok=True)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    runtime = VerifierRuntime(executor=executor, binding=binding, criterion_sha256="abc")
    result = runtime.run(
        _task(
            tmp_path, {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)}
        ),
        _context(tmp_path),
    )
    assert result.status == VerificationStatus.ERROR
    assert result.error == "verifier_criterion_changed"
    assert result.fail_type == "criterion_changed"


def test_verifier_maps_cleanup_failure_to_settlement_error(tmp_path):
    # Settlement провалился после отработанной проверки: exit code
    # известен и сохраняется, но результат — ERROR, а не тихий PASSED.
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700, exist_ok=True)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    binding = workspace.bind(principal_id="alice", session_id="chat-a")
    for exit_code in (0, 2):
        outcome = WorkspacePythonOutcome(
            "finished",
            b"ok\n",
            b"",
            exit_code,
            True,
            None,
            "none",
            tmp_path / "retained-stage",
            cleanup_warning=True,
        )
        runtime = VerifierRuntime(
            executor=_fake_executor(outcome), binding=binding, criterion_sha256="abc"
        )
        result = runtime.run(
            _task(
                tmp_path,
                {"command": ["make", "-B", "-f", "Makefile", "check"], "cwd": str(tmp_path)},
            ),
            _context(tmp_path),
        )
        assert result.status == VerificationStatus.ERROR
        assert result.error == "workspace_cleanup_failed"
        assert result.fail_type == "settlement_failed"
        assert result.exit_code == exit_code


def test_command_executor_rejects_stale_workspace_after_execution(tmp_path, monkeypatch):
    # Другая сессия того же principal пишет в workspace, пока идёт
    # проверка: PASSED обязан относиться к актуальному workspace, а не
    # к устаревшему snapshot — исход conflict, а не finished.
    from core.rootless_podman import PreparedCommandContainer
    from core.rootless_staging import BoundedStage

    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
    (workspace.root / "result.txt").write_text(CORRECT_RESULT)
    access = WorkspaceAccess(workspace, "alice", "chat-a", "env")
    binding = access.bind()
    profile = _profile(tmp_path)
    prepared = []

    def prepare(workspace_arg, snapshot):
        stage = workspace_arg.stage(snapshot)
        info = stage.root.stat()
        bounded = BoundedStage(stage, pool, stage.root, stage.root, (info.st_dev, info.st_ino))
        prepared.append(bounded)
        return bounded

    pool = SimpleNamespace(
        profile=profile,
        execution_revision="env",
        check_current=lambda: None,
        check_stage=lambda bounded: None,
        prepare=prepare,
        release_unpublished=lambda bounded: None,
    )
    executor = RootlessCommandExecutor(
        profile=profile,
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=pool,
    )
    monkeypatch.setattr(RootlessPodmanProfile, "check_current", lambda *args: None)
    monkeypatch.setattr(RootlessPodmanProfile, "verify_created_command", lambda *args: None)
    monkeypatch.setattr(RootlessPodmanProfile, "finished_exit", lambda *args, **kwargs: 0)
    monkeypatch.setattr(RootlessPodmanProfile, "remove_owned", lambda *args: True)
    monkeypatch.setattr(RootlessPodmanProfile, "argv", lambda *args: ["synthetic-podman"])
    monkeypatch.setattr(RootlessPodmanProfile, "environment", lambda *args: {})
    monkeypatch.setattr(
        RootlessPodmanProfile,
        "create_command_container",
        lambda self, bounded, argv: PreparedCommandContainer(
            "c" * 64, "slavikai-workspace-test", bounded, list(argv), "cmd-rev"
        ),
    )

    def start(*args, **kwargs):
        # Мутация — во время исполнения процесса, после предстартовой
        # проверки свежести.
        with workspace.mutation(binding):
            (workspace.root / "result.txt").write_text(WRONG_RESULT)
        streams = []
        for raw in (b"", b""):
            read_fd, write_fd = os.pipe()
            os.write(write_fd, raw)
            os.close(write_fd)
            streams.append(os.fdopen(read_fd, "rb"))
        return SimpleNamespace(
            stdout=streams[0],
            stderr=streams[1],
            returncode=0,
            poll=lambda: 0,
            wait=lambda **kw: 0,
        )

    monkeypatch.setattr("subprocess.Popen", start)
    criterion = hashlib.sha256(CHECK_MAKEFILE.encode()).hexdigest()
    outcome = executor.execute_command(
        binding, ["make", "-B", "-f", "Makefile", "check"], timeout=60, criterion_sha256=criterion
    )
    assert prepared != []
    assert outcome.state == "conflict"
    assert outcome.reason_code == "workspace_conflict"
    assert outcome.execution_started is True
    assert outcome.publication == "none"


def test_command_executor_rejects_snapshot_criterion_mismatch(tmp_path, monkeypatch):
    # Хэш сверяется на том же snapshot, который пойдёт в контейнер:
    # расхождение — conflict/verifier_criterion_changed до pool.prepare.
    parent = tmp_path / "managed"
    parent.mkdir(mode=0o700)
    workspace = ManagedWorkspace.create(parent, principal_id="alice")
    (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
    access = WorkspaceAccess(workspace, "alice", "chat-a", "env")
    binding = access.bind()
    profile = _profile(tmp_path)
    pool = SimpleNamespace(
        profile=profile,
        execution_revision="env",
        check_current=lambda: None,
        check_stage=lambda bounded: None,
        prepare=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("prepare must not run on criterion mismatch")
        ),
    )
    executor = RootlessCommandExecutor(
        profile=profile,
        workspace=workspace,
        limits=WorkspaceLimits(1024, 4096, 20, 8),
        max_output_bytes=1024,
        staging_pool=pool,
    )
    # Квалификация профиля — не предмет этого теста: stub, чтобы
    # is_available() дошёл до validate и проверял именно критерий.
    monkeypatch.setattr(RootlessPodmanProfile, "check_current", lambda *args: None)
    monkeypatch.setattr("subprocess.run", _forbidden_engine)
    wrong = hashlib.sha256(b"check:\n\t@true\n").hexdigest()
    outcome = executor.execute_command(
        binding, ["make", "-B", "-f", "Makefile", "check"], timeout=60, criterion_sha256=wrong
    )
    assert outcome.state == "conflict"
    assert outcome.reason_code == "verifier_criterion_changed"
    assert outcome.publication == "none"
    assert outcome.execution_started is False


# --- real backend (opt-in) ----------------------------------------------
#
# vfs storage дублирует слои на каждый свежий state root (~5 ГБ на оба
# образа), поэтому state живёт НЕ в tmp_path (/tmp — tmpfs 16 ГБ), а в
# постоянном каталоге на диске: образы пуллятся один раз и переиспользуются.
# Переопределить: SLAVIK_VERIFIER_E2E_STATE.

E2E_STATE_ENV = "SLAVIK_VERIFIER_E2E_STATE"
E2E_STATE_DEFAULT = "/home/ki/.cache/slavikai-qualified-verifier-e2e"


def _qualified_state():
    state_root = Path(os.environ.get(E2E_STATE_ENV, E2E_STATE_DEFAULT))
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for name in ("home", "config", "data", "storage", "runroot", "tmp", "hooks"):
        (state_root / name).mkdir(mode=0o700, exist_ok=True)
    (state_root / "containers.conf").write_text('[engine]\nruntime = "runc"\n')
    # umask 002 даёт 664 — qualify требует не group/world-writable.
    os.chmod(state_root / "containers.conf", 0o600)
    probe = RootlessPodmanProfile(state_root, os.geteuid(), os.getegid(), "unqualified", (), ())
    # Чистим только свои контейнеры от упавших прогонов; чужие не трогаем.
    names = (
        subprocess.run(
            probe.argv() + ["ps", "--all", "--format", "{{.Names}}"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        .stdout.decode()
        .split()
    )
    for name in names:
        if name.startswith("slavikai-workspace-"):
            subprocess.run(
                probe.argv() + ["rm", "--force", "--time=1", "--ignore", name],
                check=True,
                timeout=60,
            )
    # Пулл строго по digest'ам: qualify требует в RepoDigests ОБА digest
    # (index + amd64 manifest). Пулл по тегу ненадёжен — теги двигаются
    # (python:3.12.16-slim-bookworm уже указывает на другой индекс), а пулл
    # одного digest'а записывает только одну запись. Поэтому тянем оба
    # digest'а: индекс + amd64-манифест, оба указывают на тот же образ.
    # Пиннинг проверяет сам qualify (Id/Digest/RepoDigests).
    for image_id, digests in (
        (
            QUALIFIED_PYTHON_IMAGE,
            (QUALIFIED_PYTHON_DIGEST, QUALIFIED_AMD64_DIGEST),
        ),
        (
            QUALIFIED_COMMAND_IMAGE,
            (QUALIFIED_COMMAND_DIGEST, QUALIFIED_COMMAND_AMD64_DIGEST),
        ),
    ):
        inspected = subprocess.run(
            probe.argv() + ["image", "inspect", "--", image_id, "--format", "{{.RepoDigests}}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        bound = inspected.returncode == 0 and all(
            f"docker.io/library/python@{digest}" in inspected.stdout for digest in digests
        )
        if not bound:
            subprocess.run(
                probe.argv() + ["rmi", "--force", "--ignore", image_id],
                check=False,
                timeout=120,
            )
            for digest in digests:
                subprocess.run(
                    probe.argv() + ["pull", "--", f"docker.io/library/python@{digest}"],
                    check=True,
                    timeout=300,
                )
    return RootlessPodmanProfile.qualify(state_root)


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_QUALIFIED_VERIFIER_E2E") != "1",
    reason="qualified verifier E2E needs a real rootless Podman state (opt-in)",
)
def test_qualified_command_executor_verifies_fixture_makefile(tmp_path):
    profile = _qualified_state()
    assert profile.command_revision is not None
    pool = RootlessStagePool.open(profile, StagingLimits(16 * 1024 * 1024, 1024, 2))
    try:
        parent = tmp_path / "managed"
        parent.mkdir(mode=0o700)
        workspace = ManagedWorkspace.create(parent, principal_id="alice")
        (workspace.root / "Makefile").write_text(CHECK_MAKEFILE)
        (workspace.root / "result.txt").write_text(CORRECT_RESULT)
        executor = RootlessCommandExecutor(
            profile,
            workspace,
            WorkspaceLimits(1024, 65536, 50, 8),
            65536,
            pool,
        )
        assert executor.is_available()
        # Binding — как в production (app.py): через WorkspaceAccess с
        # execution_revision пула, иначе executor вернёт conflict.
        access = WorkspaceAccess(
            workspace,
            "alice",
            "chat-a",
            executor.environment_revision,
        )
        binding = access.bind()

        outcome = executor.execute_command(
            binding, ["make", "-B", "-f", "Makefile", "check"], timeout=120
        )
        assert outcome.state == "finished"
        assert outcome.exit_code == 0
        assert outcome.publication == "none"
        assert outcome.changed_paths is None
        assert outcome.snapshot_sha256 is None
        assert (workspace.root / "result.txt").read_text(encoding="utf-8") == CORRECT_RESULT
        assert sorted(p.name for p in workspace.root.iterdir()) == ["Makefile", "result.txt"]

        (workspace.root / "result.txt").write_text(WRONG_RESULT)
        outcome = executor.execute_command(
            binding, ["make", "-B", "-f", "Makefile", "check"], timeout=120
        )
        assert outcome.state == "finished"
        assert outcome.exit_code != 0
        assert outcome.publication == "none"
        assert (workspace.root / "result.txt").read_text(encoding="utf-8") == WRONG_RESULT

        # GNUmakefile-подмена: агент кладёт рядом файл, который make
        # выбрал бы по умолчанию вместо Makefile. Явный -f Makefile
        # фиксирует проверяемый файл: неверный результат по-прежнему
        # проваливает проверку.
        (workspace.root / "GNUmakefile").write_text("check:\n\t@true\n")
        outcome = executor.execute_command(
            binding, ["make", "-B", "-f", "Makefile", "check"], timeout=120
        )
        assert outcome.state == "finished"
        assert outcome.exit_code != 0
        assert outcome.publication == "none"

        # Файл `check` в workspace: без -B make решил бы, что цель
        # up to date, и вернул бы 0, не выполняя recipe. С -B recipe
        # выполняется всегда: неверный результат проваливается, а
        # верный — проходит.
        (workspace.root / "check").write_text("")
        outcome = executor.execute_command(
            binding, ["make", "-B", "-f", "Makefile", "check"], timeout=120
        )
        assert outcome.state == "finished"
        assert outcome.exit_code != 0
        assert outcome.publication == "none"
        (workspace.root / "result.txt").write_text(CORRECT_RESULT)
        outcome = executor.execute_command(
            binding, ["make", "-B", "-f", "Makefile", "check"], timeout=120
        )
        assert outcome.state == "finished"
        assert outcome.exit_code == 0
        assert outcome.publication == "none"

        names = profile.control(["ps", "--all", "--format", "{{.Names}}"]).stdout.decode().split()
        assert not any(name.startswith("slavikai-workspace-") for name in names)
    finally:
        pool.close()
