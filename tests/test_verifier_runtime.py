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
    monkeypatch.setattr("core.mwv.verifier_runtime.is_repo_workspace", lambda _root: True)
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
    assert result.verifier_profile == "fallback"
    assert calls == [["python", "-m", "ruff", "check", "."], ["python", "-m", "pytest", "-q"]]


def test_verifier_runtime_default_fallback_uses_make_check(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("core.mwv.verifier_runtime.is_repo_workspace", lambda _root: True)
    (tmp_path / "Makefile").write_text(".PHONY: check\ncheck:\n\t@true\n", encoding="utf-8")
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
    runtime = VerifierRuntime(project_root=tmp_path)

    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.PASSED
    assert result.command == ["make", "check"]
    assert result.verifier_profile == "fallback"
    assert calls == [["make", "check"]]


def test_verifier_runtime_fallback_fail_for_repo_like_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("core.mwv.verifier_runtime.is_repo_workspace", lambda _root: True)

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
    assert result.fail_type == "stderr"
    assert result.excerpt is not None and "boom" in result.excerpt


def test_verifier_runtime_disables_fallback_for_non_repo_workspace(tmp_path: Path) -> None:
    runtime = VerifierRuntime(project_root=tmp_path)
    result = runtime.run(_task(tmp_path), _context(tmp_path))

    assert result.status == VerificationStatus.ERROR
    assert result.error == NON_REPO_VERIFIER_REQUIRED_ERROR
    assert result.command == []
    assert result.fail_type == "non_repo_workspace"


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
    assert result.verifier_profile == "explicit"


def test_verifier_runtime_fallback_os_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("core.mwv.verifier_runtime.is_repo_workspace", lambda _root: True)

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


@pytest.mark.parametrize("detached", [False, True])
def test_verifier_cancellation_stops_owned_process_group(tmp_path: Path, detached: bool) -> None:
    import threading
    import time

    ready = tmp_path / "ready"
    cancelled = threading.Event()
    script = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], "
        f"start_new_session={detached}); "
        "print('critical diagnostic',flush=True); "
        f"pathlib.Path({str(ready)!r}).write_text(str(child.pid)); time.sleep(60)"
    )

    def cancel_when_started() -> None:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancelled.set()

    worker = threading.Thread(target=cancel_when_started)
    worker.start()
    started = time.monotonic()
    try:
        result = VerifierRuntime(project_root=tmp_path).run(
            _task(tmp_path, verifier={"command": [sys.executable, "-c", script]}),
            _context(tmp_path),
            cancelled=cancelled.is_set,
        )
    finally:
        worker.join(timeout=4)
        if detached and ready.exists():
            import os
            import signal

            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    if detached:
        assert "capture incomplete" in result.stderr
    assert time.monotonic() - started < 4
    assert result.error == "verifier_cancelled"
    assert result.status == VerificationStatus.ERROR
    assert "critical diagnostic" in result.stdout
    assert ready.exists()
    child_status = Path(f"/proc/{ready.read_text()}/status")
    deadline = time.monotonic() + 1
    while True:
        try:
            status = child_status.read_text()
        except (FileNotFoundError, ProcessLookupError):
            status = "State:\tZ"
        if "State:\tZ" in status or time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    assert "State:\tZ" in status
