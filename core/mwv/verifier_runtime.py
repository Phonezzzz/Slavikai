from __future__ import annotations

import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from core.mwv.models import RunContext, TaskPacket, VerificationResult, VerificationStatus
from core.mwv.verifier import VerifierRunner
from core.mwv.verifier_summary import extract_verifier_excerpt, verifier_fail_type


class VerifierRunnerProtocol(Protocol):
    def run(self) -> VerificationResult: ...


def _default_runner() -> VerifierRunnerProtocol:
    return VerifierRunner()


_PYTHON_EXECUTABLE = sys.executable or "python"
_DEFAULT_TIMEOUT_SECONDS = 60 * 30

DEFAULT_FALLBACK_COMMANDS: tuple[tuple[str, ...], ...] = (
    (_PYTHON_EXECUTABLE, "-m", "ruff", "check", "."),
    (_PYTHON_EXECUTABLE, "-m", "ruff", "format", "--check", "."),
    (_PYTHON_EXECUTABLE, "skills/tools/lint_skills.py"),
    (_PYTHON_EXECUTABLE, "skills/tools/build_manifest.py", "--check"),
    (_PYTHON_EXECUTABLE, "-m", "mypy", "."),
    (_PYTHON_EXECUTABLE, "-m", "pytest", "--cov", "--cov-fail-under=80"),
)
SCRIPT_NOT_FOUND_PREFIX = "Verifier script not found:"
NON_REPO_VERIFIER_REQUIRED_ERROR = "verifier_command_required_for_non_repo_workspace"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class VerifierRuntime:
    runner: VerifierRunnerProtocol = field(default_factory=_default_runner)
    fallback_commands: tuple[tuple[str, ...], ...] = DEFAULT_FALLBACK_COMMANDS
    project_root: Path = field(default_factory=_default_project_root)

    def run(self, task: TaskPacket, context: RunContext) -> VerificationResult:
        start = time.monotonic()
        try:
            workspace_root = _resolve_workspace_root(context.workspace_root)
            command = _resolve_packet_command(task.verifier)
            timeout_seconds = _resolve_timeout(task.verifier)
            cwd = _resolve_cwd(task.verifier, workspace_root=workspace_root)
        except ValueError as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=[],
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error=f"invalid_verifier_config: {exc}",
                fail_type="invalid_config",
                excerpt=str(exc),
                verifier_profile=(
                    "explicit"
                    if isinstance(task.verifier, dict) and task.verifier.get("command") is not None
                    else "fallback"
                ),
            )

        if command is not None:
            return self._run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)

        if not _is_repo_like_workspace(workspace_root):
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=[],
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error=NON_REPO_VERIFIER_REQUIRED_ERROR,
                fail_type="non_repo_workspace",
                excerpt=NON_REPO_VERIFIER_REQUIRED_ERROR,
                verifier_profile="fallback",
            )
        return self._run_fallback(cwd=cwd, timeout_seconds=timeout_seconds)

    def _run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: int,
    ) -> VerificationResult:
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                exit_code=None,
                stdout=_coerce_output(exc.stdout),
                stderr=_coerce_output(exc.stderr),
                duration_seconds=time.monotonic() - start,
                error="verifier_timeout",
                fail_type="timeout",
                excerpt="verifier_timeout",
                verifier_profile="explicit",
            )
        except OSError as exc:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error=f"verifier_os_error: {exc}",
                fail_type="os_error",
                excerpt=str(exc),
                verifier_profile="explicit",
            )

        status = (
            VerificationStatus.PASSED if completed.returncode == 0 else VerificationStatus.FAILED
        )
        result = VerificationResult(
            status=status,
            command=command,
            exit_code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            duration_seconds=time.monotonic() - start,
            error=None,
            fail_type=None
            if status == VerificationStatus.PASSED
            else verifier_fail_type(
                VerificationResult(
                    status=status,
                    command=command,
                    exit_code=completed.returncode,
                    stdout=completed.stdout or "",
                    stderr=completed.stderr or "",
                    duration_seconds=0.0,
                )
            ),
            excerpt=None,
            verifier_profile="explicit",
        )
        if status != VerificationStatus.PASSED:
            return VerificationResult(
                status=result.status,
                command=result.command,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_seconds=result.duration_seconds,
                error=result.error,
                fail_type=result.fail_type,
                excerpt=extract_verifier_excerpt(result),
                verifier_profile=result.verifier_profile,
            )
        return result

    def _run_fallback(
        self,
        *,
        cwd: Path,
        timeout_seconds: int,
    ) -> VerificationResult:
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
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    command=command,
                    exit_code=None,
                    stdout=_join_output(stdout_parts),
                    stderr=_join_output([*stderr_parts, _coerce_output(exc.stderr)]),
                    duration_seconds=time.monotonic() - start,
                    error="verifier_timeout",
                    fail_type="timeout",
                    excerpt="verifier_timeout",
                    verifier_profile="fallback",
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
                    fail_type="os_error",
                    excerpt=str(exc),
                    verifier_profile="fallback",
                )

            stdout_parts.extend((f"$ {' '.join(command)}", completed.stdout or ""))
            stderr_parts.extend((f"$ {' '.join(command)}", completed.stderr or ""))
            if completed.returncode != 0:
                result = VerificationResult(
                    status=VerificationStatus.FAILED,
                    command=command,
                    exit_code=completed.returncode,
                    stdout=_join_output(stdout_parts),
                    stderr=_join_output(stderr_parts),
                    duration_seconds=time.monotonic() - start,
                    error=None,
                    fail_type="stderr",
                    excerpt=None,
                    verifier_profile="fallback",
                )
                return VerificationResult(
                    status=result.status,
                    command=result.command,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    duration_seconds=result.duration_seconds,
                    error=result.error,
                    fail_type=verifier_fail_type(result),
                    excerpt=extract_verifier_excerpt(result),
                    verifier_profile=result.verifier_profile,
                )

        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=last_command,
            exit_code=0,
            stdout=_join_output(stdout_parts),
            stderr=_join_output(stderr_parts),
            duration_seconds=time.monotonic() - start,
            error=None,
            fail_type=None,
            excerpt=None,
            verifier_profile="fallback",
        )


def _resolve_packet_command(verifier: object) -> list[str] | None:
    if not isinstance(verifier, dict):
        return None
    command_raw = verifier.get("command")
    if command_raw is None:
        return None
    if isinstance(command_raw, str):
        stripped = command_raw.strip()
        if not stripped:
            return None
        parsed = shlex.split(stripped)
        if not parsed:
            raise ValueError("verifier.command пустой.")
        return parsed
    if isinstance(command_raw, list):
        command: list[str] = []
        for item in command_raw:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("verifier.command list должен содержать непустые строки.")
            command.append(item.strip())
        if not command:
            raise ValueError("verifier.command list пустой.")
        return command
    raise ValueError("verifier.command должен быть string или list[string].")


def _resolve_timeout(verifier: object) -> int:
    if not isinstance(verifier, dict):
        return _DEFAULT_TIMEOUT_SECONDS
    timeout_raw = verifier.get("timeout_seconds")
    if timeout_raw is None:
        return _DEFAULT_TIMEOUT_SECONDS
    if not isinstance(timeout_raw, int):
        raise ValueError("verifier.timeout_seconds должен быть int.")
    if timeout_raw <= 0:
        raise ValueError("verifier.timeout_seconds должен быть > 0.")
    return timeout_raw


def _resolve_workspace_root(workspace_root: str) -> Path:
    root = Path(workspace_root).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"workspace_root недоступен: {root}")
    return root


def _resolve_cwd(verifier: object, *, workspace_root: Path) -> Path:
    root = workspace_root
    if not isinstance(verifier, dict):
        return root
    cwd_raw = verifier.get("cwd")
    if cwd_raw is None:
        return root
    if not isinstance(cwd_raw, str) or not cwd_raw.strip():
        raise ValueError("verifier.cwd должен быть непустой строкой.")
    candidate_raw = Path(cwd_raw.strip()).expanduser()
    candidate = candidate_raw if candidate_raw.is_absolute() else (root / candidate_raw)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"verifier.cwd вне workspace_root: {resolved}") from exc
    if not resolved.exists() or not resolved.is_dir():
        raise ValueError(f"verifier.cwd недоступен: {resolved}")
    return resolved


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _join_output(parts: list[str]) -> str:
    filtered = [part for part in parts if part]
    return "\n".join(filtered)


def _is_repo_like_workspace(root: Path) -> bool:
    if (root / ".git").exists():
        return True
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0
