from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from core.mwv.models import RunContext, VerificationResult, VerificationStatus
from core.mwv.verifier import VerifierRunner


class VerifierRunnerProtocol(Protocol):
    def run(self) -> VerificationResult: ...


def _default_runner() -> VerifierRunnerProtocol:
    return VerifierRunner()


DEFAULT_FALLBACK_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("python", "-m", "ruff", "check", "."),
    ("python", "-m", "ruff", "format", "--check", "."),
    ("python", "skills/tools/lint_skills.py"),
    ("python", "skills/tools/build_manifest.py", "--check"),
    ("python", "-m", "mypy", "."),
    ("python", "-m", "pytest", "--cov", "--cov-fail-under=80"),
)
SCRIPT_NOT_FOUND_PREFIX = "Verifier script not found:"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VerifierRuntime:
    runner: VerifierRunnerProtocol = field(default_factory=_default_runner)
    fallback_commands: tuple[tuple[str, ...], ...] = DEFAULT_FALLBACK_COMMANDS
    project_root: Path = field(default_factory=_default_project_root)

    def run(self, context: RunContext) -> VerificationResult:
        _ = context
        result = self.runner.run()
        if not _should_run_fallback(result):
            return result
        return self._run_fallback()

    def _run_fallback(self) -> VerificationResult:
        start = time.monotonic()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        last_command: list[str] = []
        for command_tuple in self.fallback_commands:
            command = list(command_tuple)
            last_command = command
            try:
                completed = subprocess.run(
                    command,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except OSError as exc:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    command=command,
                    exit_code=None,
                    stdout=_join_output(stdout_parts),
                    stderr=_join_output([*stderr_parts, str(exc)]),
                    duration_seconds=time.monotonic() - start,
                    error=f"fallback_failed: {exc}",
                )

            stdout_parts.extend((f"$ {' '.join(command)}", completed.stdout or ""))
            stderr_parts.extend((f"$ {' '.join(command)}", completed.stderr or ""))
            if completed.returncode != 0:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    command=command,
                    exit_code=completed.returncode,
                    stdout=_join_output(stdout_parts),
                    stderr=_join_output(stderr_parts),
                    duration_seconds=time.monotonic() - start,
                    error=None,
                )
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=last_command,
            exit_code=0,
            stdout=_join_output(stdout_parts),
            stderr=_join_output(stderr_parts),
            duration_seconds=time.monotonic() - start,
            error=None,
        )


def _join_output(parts: list[str]) -> str:
    filtered = [part for part in parts if part]
    return "\n".join(filtered)


def _should_run_fallback(result: VerificationResult) -> bool:
    if result.status != VerificationStatus.ERROR:
        return False
    if result.error is None:
        return False
    return result.error.startswith(SCRIPT_NOT_FOUND_PREFIX)
