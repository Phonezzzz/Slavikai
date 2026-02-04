from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from core.mwv.models import RunContext, VerificationResult, VerificationStatus
from core.mwv.verifier_runtime import SCRIPT_NOT_FOUND_PREFIX, VerifierRuntime


@dataclass(frozen=True)
class StubRunner:
    result: VerificationResult

    def run(self) -> VerificationResult:
        return self.result


def _context(workspace_root: Path) -> RunContext:
    return RunContext(
        session_id="session",
        trace_id="trace",
        workspace_root=str(workspace_root),
        safe_mode=True,
    )


def _missing_script_result() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.ERROR,
        command=["bash", "scripts/check.sh"],
        exit_code=None,
        stdout="",
        stderr="",
        duration_seconds=0.0,
        error=f"{SCRIPT_NOT_FOUND_PREFIX} scripts/check.sh",
    )


def test_verifier_runtime_fallback_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def _run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = (cwd, capture_output, text, check)
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        runner=StubRunner(_missing_script_result()),
        fallback_commands=(
            ("python", "-m", "ruff", "check", "."),
            ("python", "-m", "pytest", "-q"),
        ),
        project_root=tmp_path,
    )
    result = runtime.run(_context(tmp_path))

    assert result.status == VerificationStatus.PASSED
    assert result.exit_code == 0
    assert calls == [["python", "-m", "ruff", "check", "."], ["python", "-m", "pytest", "-q"]]


def test_verifier_runtime_fallback_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        _ = (cwd, capture_output, text, check)
        return subprocess.CompletedProcess(command, 3, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        runner=StubRunner(_missing_script_result()),
        fallback_commands=(("python", "-m", "ruff", "check", "."),),
        project_root=tmp_path,
    )
    result = runtime.run(_context(tmp_path))

    assert result.status == VerificationStatus.FAILED
    assert result.command == ["python", "-m", "ruff", "check", "."]
    assert result.exit_code == 3
    assert "boom" in result.stderr


def test_verifier_runtime_no_fallback_for_other_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("fallback should not run")

    monkeypatch.setattr(subprocess, "run", _run)
    initial = VerificationResult(
        status=VerificationStatus.ERROR,
        command=["bash", "scripts/check.sh"],
        exit_code=None,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        error="verifier_timeout",
    )
    runtime = VerifierRuntime(runner=StubRunner(initial))
    result = runtime.run(_context(tmp_path))

    assert result == initial


def test_verifier_runtime_fallback_os_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("cannot execute")

    monkeypatch.setattr(subprocess, "run", _run)
    runtime = VerifierRuntime(
        runner=StubRunner(_missing_script_result()),
        fallback_commands=(("python", "-m", "ruff", "check", "."),),
        project_root=tmp_path,
    )
    result = runtime.run(_context(tmp_path))

    assert result.status == VerificationStatus.ERROR
    assert result.error and "fallback_failed" in result.error
