from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.mwv.models import RunContext, TaskPacket, VerificationStatus
from core.mwv.verifier_runtime import NON_REPO_VERIFIER_REQUIRED_ERROR, VerifierRuntime


def _context(workspace_root: Path) -> RunContext:
    return RunContext(
        session_id="session",
        trace_id="trace",
        workspace_root=str(workspace_root),
        safe_mode=True,
    )


def _task(workspace_root: Path, *, verifier: dict[str, object] | None = None) -> TaskPacket:
    return TaskPacket(
        task_id="task-1",
        session_id="session",
        trace_id="trace",
        goal="verify",
        scope={"workspace_root": str(workspace_root)},
        verifier=verifier or {},
    )


def test_verifier_runtime_fallback_pass_for_repo_like_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    calls: list[list[str]] = []

    def _run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = (cwd, capture_output, text, timeout, check)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        fallback_commands=(
            ("python", "-m", "ruff", "check", "."),
            ("python", "-m", "pytest", "-q"),
        ),
        project_root=tmp_path,
    )
    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.PASSED
    assert result.exit_code == 0
    assert calls == [["python", "-m", "ruff", "check", "."], ["python", "-m", "pytest", "-q"]]


def test_verifier_runtime_fallback_fail_for_repo_like_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()

    def _run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = (cwd, capture_output, text, timeout, check)
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        fallback_commands=(("python", "-m", "ruff", "check", "."),),
        project_root=tmp_path,
    )
    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.FAILED
    assert result.command == ["python", "-m", "ruff", "check", "."]
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_verifier_runtime_disables_fallback_for_non_repo_workspace(tmp_path: Path) -> None:
    runtime = VerifierRuntime(project_root=tmp_path)
    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.ERROR
    assert result.error == NON_REPO_VERIFIER_REQUIRED_ERROR
    assert result.command == []


def test_verifier_runtime_runs_explicit_command_in_non_repo_workspace(tmp_path: Path) -> None:
    runtime = VerifierRuntime(project_root=tmp_path)
    result = runtime.run(
        _task(
            tmp_path,
            verifier={
                "command": [sys.executable, "-c", "print('explicit-ok')"],
                "cwd": ".",
                "timeout_seconds": 5,
            },
        ),
        _context(tmp_path),
    )

    assert result.status == VerificationStatus.PASSED
    assert result.command[0] == sys.executable
    assert "explicit-ok" in result.stdout


def test_verifier_runtime_fallback_os_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()

    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("cannot execute")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        fallback_commands=(("python", "-m", "ruff", "check", "."),),
        project_root=tmp_path,
    )
    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.ERROR
    assert result.error and "fallback_failed" in result.error
