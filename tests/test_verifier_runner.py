from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.mwv.models import VerificationStatus
from core.mwv.verifier import VerifierRunner, _coerce_output


def test_verifier_runner_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["bash", "/tmp/check.sh"],
            returncode=0,
            stdout="ok",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    script_path = tmp_path / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner = VerifierRunner(script_path=script_path)
    result = runner.run()

    assert result.status == VerificationStatus.PASSED
    assert result.exit_code == 0
    assert result.stdout == "ok"
    assert result.stderr == ""
    assert result.command[:1] == ["bash"]


def test_verifier_runner_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=["bash", "/tmp/check.sh"],
            returncode=1,
            stdout="",
            stderr="lint error",
        )

    monkeypatch.setattr(subprocess, "run", _fake_run)
    script_path = tmp_path / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    runner = VerifierRunner(script_path=script_path)
    result = runner.run()

    assert result.status == VerificationStatus.FAILED
    assert result.exit_code == 1
    assert result.stderr == "lint error"


def test_verifier_runner_missing_script(tmp_path: Path) -> None:
    runner = VerifierRunner(script_path=tmp_path / "missing.sh")
    result = runner.run()

    assert result.status == VerificationStatus.ERROR
    assert result.exit_code is None
    assert result.error is not None
    assert "not found" in result.error


def test_verifier_runner_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script_path = tmp_path / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def _raise_timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(  # noqa: TRY003
            cmd=["bash", str(script_path)],
            timeout=1,
            output=b"out",
            stderr=b"err",
        )

    monkeypatch.setattr(subprocess, "run", _raise_timeout)
    runner = VerifierRunner(script_path=script_path, timeout_seconds=1)
    result = runner.run()

    assert result.status == VerificationStatus.ERROR
    assert result.error == "verifier_timeout"
    assert result.stdout == "out"
    assert result.stderr == "err"


def test_verifier_runner_os_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script_path = tmp_path / "check.sh"
    script_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    def _raise_os_error(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise OSError("boom")  # noqa: TRY003

    monkeypatch.setattr(subprocess, "run", _raise_os_error)
    runner = VerifierRunner(script_path=script_path)
    result = runner.run()

    assert result.status == VerificationStatus.ERROR
    assert result.error and "boom" in result.error


def test_coerce_output_variants() -> None:
    assert _coerce_output(None) == ""
    assert _coerce_output("text") == "text"
