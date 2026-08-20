from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from core.mwv.models import RunContext, TaskPacket, VerificationResult, VerificationStatus
from core.mwv.verifier import VerifierRunner
from core.mwv.verifier_summary import extract_verifier_excerpt, verifier_fail_type
from shared.models import JSONValue, ToolResult


class VerifierRunnerProtocol(Protocol):
    def run(self) -> VerificationResult: ...


def _default_runner() -> VerifierRunnerProtocol:
    return VerifierRunner()


_DEFAULT_TIMEOUT_SECONDS = 60 * 30
CANONICAL_CHECK_COMMAND: tuple[str, ...] = ("make", "check")

DEFAULT_FALLBACK_COMMANDS: tuple[tuple[str, ...], ...] = (CANONICAL_CHECK_COMMAND,)
SCRIPT_NOT_FOUND_PREFIX = "Verifier script not found:"
NON_REPO_VERIFIER_REQUIRED_ERROR = "verifier_command_required_for_non_repo_workspace"


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _desktop_observation_result(
    *,
    status: VerificationStatus,
    reason: str,
    duration_seconds: float,
) -> VerificationResult:
    passed = status == VerificationStatus.PASSED
    return VerificationResult(
        status=status,
        command=[],
        exit_code=0 if passed else 1,
        stdout=reason if passed else "",
        stderr="" if passed else reason,
        duration_seconds=duration_seconds,
        error=None if passed else reason,
        fail_type=None if passed else "desktop_observation",
        excerpt=reason,
        verifier_profile="desktop_observation",
    )


def canonical_check_command() -> list[str]:
    return list(CANONICAL_CHECK_COMMAND)


def has_canonical_repo_verifier(project_root: Path) -> bool:
    return (project_root / "Makefile").is_file() and (
        shutil.which(CANONICAL_CHECK_COMMAND[0]) is not None
    )


def _desktop_change_is_verified(
    name: str,
    args: dict[str, JSONValue],
    result: ToolResult,
    later: Sequence[tuple[str, dict[str, JSONValue], ToolResult]],
) -> bool:
    if result.data.get("verified") is True:
        return True
    verifications = [
        (verify_args, verify_result)
        for verify_name, verify_args, verify_result in later
        if verify_name == "desktop_verify" and verify_result.ok
    ]
    if name == "desktop_file_write":
        path = _result_string(result, "path")
        content = args.get("content")
        if path is None or not isinstance(content, str):
            return False
        if not content:
            return _has_path_check(verifications, path=path, check="path_exists")
        return any(
            verify_args.get("check") == "file_contains"
            and verify_result.data.get("path") == path
            and verify_args.get("expected") == content
            for verify_args, verify_result in verifications
        )
    if name == "desktop_file_transfer":
        destination = _result_string(result, "destination")
        source = _result_string(result, "source")
        operation = args.get("operation")
        if destination is None or not _has_path_check(
            verifications,
            path=destination,
            check="path_exists",
        ):
            return False
        if operation in {"move", "rename"}:
            return source is not None and _has_path_check(
                verifications,
                path=source,
                check="path_missing",
            )
        return True
    if name == "desktop_file_delete":
        path = _result_string(result, "path")
        return path is not None and _has_path_check(
            verifications,
            path=path,
            check="path_missing",
        )
    if name == "desktop_archive_extract":
        destination = _result_string(result, "destination")
        return destination is not None and _has_path_check(
            verifications,
            path=destination,
            check="path_exists",
        )
    if name in {"desktop_launch", "desktop_process"}:
        pid = result.data.get("pid")
        return isinstance(pid, int) and any(
            verify_args.get("check") == "process_running" and verify_args.get("pid") == pid
            for verify_args, _ in verifications
        )
    if name == "desktop_browser":
        page_id = result.data.get("page_id")
        if not isinstance(page_id, str):
            return False
        semantic_observations = {"find", "read", "snapshot", "wait"}
        return any(
            later_name == "desktop_browser"
            and later_result.ok
            and later_args.get("operation") in semantic_observations
            and later_result.data.get("page_id") == page_id
            for later_name, later_args, later_result in later
        )
    if name == "desktop_gui":
        observations = {"windows", "active_window", "observe", "screenshot"}
        return any(
            later_name == "desktop_gui"
            and later_result.ok
            and later_args.get("operation") in observations
            for later_name, later_args, later_result in later
        )
    return bool(verifications)


def _desktop_call_changes_state(name: str, args: dict[str, JSONValue]) -> bool:
    if name in {
        "desktop_file_write",
        "desktop_file_transfer",
        "desktop_file_delete",
        "desktop_archive_extract",
        "desktop_shell",
        "desktop_launch",
    }:
        return True
    operation = args.get("operation")
    if not isinstance(operation, str):
        return False
    if name == "desktop_clipboard":
        return operation in {"write", "clear"}
    if name == "desktop_process":
        return operation in {"launch", "terminate", "kill"}
    if name == "desktop_systemd":
        return operation in {"start", "stop", "restart", "enable", "disable"}
    if name == "desktop_package":
        return operation in {"install", "remove", "update_metadata"}
    if name == "desktop_session":
        return operation in {"notify", "lock"}
    if name == "desktop_browser":
        return operation in {
            "open",
            "new_tab",
            "navigate",
            "click",
            "input",
            "select",
            "submit",
            "close_tab",
            "download",
            "close",
        }
    if name == "desktop_gui":
        return operation in {"focus", "invoke", "set_text", "click", "type", "shortcut"}
    return False


def _has_path_check(
    verifications: Sequence[tuple[dict[str, JSONValue], ToolResult]],
    *,
    path: str,
    check: str,
) -> bool:
    return any(
        verify_args.get("check") == check and verify_result.data.get("path") == path
        for verify_args, verify_result in verifications
    )


def _result_string(result: ToolResult, key: str) -> str | None:
    value = result.data.get(key)
    return value if isinstance(value, str) else None


@dataclass(frozen=True)
class VerifierRuntime:
    runner: VerifierRunnerProtocol = field(default_factory=_default_runner)
    fallback_commands: tuple[tuple[str, ...], ...] = DEFAULT_FALLBACK_COMMANDS
    project_root: Path = field(default_factory=_default_project_root)

    def verify_desktop_observations(
        self,
        calls: Sequence[tuple[str, dict[str, JSONValue], ToolResult]],
    ) -> VerificationResult:
        start = time.monotonic()
        if not calls:
            return _desktop_observation_result(
                status=VerificationStatus.FAILED,
                reason="desktop_no_observable_tool_action",
                duration_seconds=time.monotonic() - start,
            )
        change_attempts = [
            (index, name, args, result)
            for index, (name, args, result) in enumerate(calls)
            if _desktop_call_changes_state(name, args)
        ]
        successful_changes = [
            (index, name, args, result)
            for index, name, args, result in change_attempts
            if result.ok
        ]
        if successful_changes:
            for index, name, args, result in successful_changes:
                later = calls[index + 1 :]
                if _desktop_change_is_verified(name, args, result, later):
                    continue
                return _desktop_observation_result(
                    status=VerificationStatus.FAILED,
                    reason=f"desktop_result_verification_required:{name}",
                    duration_seconds=time.monotonic() - start,
                )
        elif change_attempts or not any(result.ok for _, _, result in calls):
            last_error = next(
                (result.error for _, _, result in reversed(calls) if result.error),
                "desktop_no_successful_observation",
            )
            return _desktop_observation_result(
                status=VerificationStatus.FAILED,
                reason=last_error,
                duration_seconds=time.monotonic() - start,
            )
        return _desktop_observation_result(
            status=VerificationStatus.PASSED,
            reason=f"Verified {len(calls)} Desktop tool observation(s).",
            duration_seconds=time.monotonic() - start,
        )

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

        if not is_repo_workspace(workspace_root):
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


def is_repo_workspace(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return False
    return Path(result.stdout.strip()).resolve() == root.resolve()
