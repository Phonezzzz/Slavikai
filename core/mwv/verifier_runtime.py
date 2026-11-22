from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from core.mwv.models import RunContext, TaskPacket, VerificationResult, VerificationStatus
from core.mwv.verifier import (
    MAX_OUTPUT_CHARS,
    TRUNCATION_SUFFIX,
    VerifierRunner,
    sandbox_verification_unavailable,
)
from shared.models import JSONValue, ToolResult
from shared.sandbox_execution import SANDBOX_EXECUTION_MESSAGE, SandboxExecutionReason
from shared.workspace_binding import WorkspaceUnavailable

if TYPE_CHECKING:
    from core.workspace_python_execution import WorkspacePythonOutcome
    from shared.workspace_binding import WorkspaceBinding


class CommandExecutorProtocol(Protocol):
    """Квалифицированный исполнитель canonical check на staged snapshot.

    Реализация — RootlessCommandExecutor; протокол держит VerifierRuntime
    независимым от rootless-механизма и тестируемым с fake executor.
    """

    def is_available(self) -> bool: ...

    def execute_command(
        self,
        binding: WorkspaceBinding,
        argv: list[str],
        *,
        timeout: float,
        criterion_sha256: str | None = None,
    ) -> WorkspacePythonOutcome: ...


class VerifierRunnerProtocol(Protocol):
    def run(self) -> VerificationResult: ...


def _default_runner() -> VerifierRunnerProtocol:
    return VerifierRunner()


_DEFAULT_TIMEOUT_SECONDS = 60 * 30
CANONICAL_CHECK_COMMAND: tuple[str, ...] = ("make", "-B", "-r", "-f", "Makefile", "check")

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


def has_canonical_check_definition(project_root: Path) -> bool:
    """Makefile в workspace — критерий проверки для штатного verifier-пути.

    В отличие от has_canonical_repo_verifier, host make не проверяется:
    исполнение идёт только внутри квалифицированного контейнера.
    """
    return (project_root / "Makefile").is_file()


def _browser_observation_confirms(args: dict[str, JSONValue], result: ToolResult) -> bool:
    """Семантическая observation браузера, пригодная как подтверждение.

    ``find`` с ``total=0`` — успешный поиск, который ничего не нашёл:
    он не подтверждает ничего (E02-T06, дыра acceptance). Подтверждает
    только find с total>=1; read/snapshot/wait подтверждают факт
    наблюдения страницы.
    """
    operation = args.get("operation")
    if operation == "find":
        total = result.data.get("total")
        return isinstance(total, int) and not isinstance(total, bool) and total >= 1
    return operation in {"read", "snapshot", "wait"}


def _browser_state_confirms(args: dict[str, JSONValue], result: ToolResult) -> bool:
    """E02-T10: объявленное ``browser_state`` подтверждено захватом tool.

    Tool — свидетель: внутри вызова ``click``/``submit`` он захватывает
    факты состояния сущности ``entity_id`` до и после действия и кладёт
    их в ``result.data["browser_state"]`` (для ``text`` — только факты
    точного сравнения с объявленным ожиданием, без сырых строк; для
    ``count`` — числа). Verifier — судья: подтверждение только при
    точном совпадении объявления и фактов и при доказанном изменении
    (``before`` не совпадал с ``expect.after``). Всё остальное —
    неподтверждено (UNVERIFIED в канонической деривации), не failure.
    """
    declaration = args.get("browser_state")
    if not isinstance(declaration, dict):
        return False
    entity_id = declaration.get("entity_id")
    expect = declaration.get("expect")
    if not isinstance(entity_id, str) or not entity_id or not isinstance(expect, dict):
        return False
    kind = expect.get("kind")
    after = expect.get("after")
    before_declared = "before" in expect
    before = expect.get("before")
    facts = result.data.get("browser_state")
    if not isinstance(facts, dict):
        return False
    if facts.get("entity_id") != entity_id or facts.get("kind") != kind:
        return False
    if kind == "text":
        if not isinstance(after, str):
            return False
        if before_declared and not isinstance(before, str):
            return False
        if facts.get("after_matches_after") is not True:
            return False
        if facts.get("before_matches_after") is not False:
            return False
        if before_declared and facts.get("before_matches_expected_before") is not True:
            return False
        return True
    if kind == "count":
        if not isinstance(after, int) or isinstance(after, bool) or after != 1:
            return False
        if before_declared and (not isinstance(before, int) or isinstance(before, bool)):
            return False
        before_count = facts.get("before_count")
        after_count = facts.get("after_count")
        if not isinstance(before_count, int) or isinstance(before_count, bool):
            return False
        if not isinstance(after_count, int) or isinstance(after_count, bool):
            return False
        if before_count != 0 or after_count != after:
            return False
        if before_declared and before_count != before:
            return False
        return True
    return False


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
        return any(
            later_name == "desktop_browser"
            and later_result.ok
            and _browser_observation_confirms(later_args, later_result)
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


def desktop_last_state_change_succeeded(
    calls: Sequence[tuple[str, dict[str, JSONValue], ToolResult]],
) -> bool:
    """Последнее state-changing действие завершилось успешно.

    Критерий канонической деривации desktop-run (E02-T06): исчерпание
    final_gate нормализуется в UNVERIFIED только когда работа реально
    выполнена, но не подтверждена. Если последнее state-changing
    действие вернуло ошибку и не исправлено последующим успешным
    действием, выполнение завершилось отказом — исходная ошибка и
    FAILED сохраняются. Ошибка, исправленная последующими успешными
    действиями, run не проваливает.
    """
    change_attempts = [
        result for name, args, result in calls if _desktop_call_changes_state(name, args)
    ]
    return bool(change_attempts) and change_attempts[-1].ok


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
    # Штатный исполнитель canonical check. None — поведение без исполнителя:
    # run() возвращает sandbox_verification_unavailable, как раньше.
    executor: CommandExecutorProtocol | None = None
    binding: WorkspaceBinding | None = None
    # Хэш доверенного критерия (Makefile), зафиксированный до loop.
    # None — критерий не задан; executor его не сверяет.
    criterion_sha256: str | None = None

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

    def verify_browser_postconditions(
        self,
        calls: Sequence[tuple[str, dict[str, JSONValue], ToolResult]],
    ) -> VerificationResult | None:
        """Каноническая проверка browser postconditions (E02-T06, семантика B).

        Отвечает на вопрос канонического outcome — «подтверждено ли
        целевое состояние», а не на вопрос desktop gate «была ли
        re-observation перед финишем» (это ``verify_desktop_observations``;
        семантика desktop-пути не меняется).

        Правила (решение владельца 2026-11-22):
        - self-verified операции (``verified=True`` в результате tool:
          navigate/input/select/download/close_tab/close — read-back
          значения, файл на диске, final_url) подтверждены самим tool.
        - ``click``/``submit`` без self-verification НЕ подтверждаются
          фактом повторной observation: snapshot/read доказывают факт
          наблюдения, find — факт наличия элемента, но не достижение
          целевого состояния конкретного действия.
        - ``click``/``submit`` с объявленным ``browser_state`` (E02-T10,
          контракт B): tool внутри того же вызова захватывает факты
          состояния сущности ``entity_id`` до и после действия;
          подтверждение — только по ``_browser_state_confirms``
          (точное совпадение объявления и фактов + доказанное
          изменение). Tool не ставит ``verified=True`` для
          click/submit; без объявления или без захвата click/submit
          остаются неподтверждёнными (FAILED -> UNVERIFIED в
          канонической деривации).
        - setup-операции без self-verification (open/new_tab пустой
          вкладки) подтверждаются последующей observation той же
          страницы — как в desktop-семантике.
        - нет state-changing browser-вызовов -> ``None``: browser
          postcondition неприменим и не блокирует DONE.
        """
        start = time.monotonic()
        change_attempts = [
            (index, args, result)
            for index, (name, args, result) in enumerate(calls)
            if name == "desktop_browser" and _desktop_call_changes_state(name, args)
        ]
        if not change_attempts:
            return None
        successful_changes = [
            (index, args, result) for index, args, result in change_attempts if result.ok
        ]
        if not successful_changes:
            last_error = next(
                (result.error for _, _, result in reversed(change_attempts) if result.error),
                "browser_no_successful_change",
            )
            return _desktop_observation_result(
                status=VerificationStatus.FAILED,
                reason=last_error,
                duration_seconds=time.monotonic() - start,
            )
        for index, args, result in successful_changes:
            if result.data.get("verified") is True:
                continue
            operation = args.get("operation")
            if operation in {"click", "submit"}:
                if _browser_state_confirms(args, result):
                    continue
                return _desktop_observation_result(
                    status=VerificationStatus.FAILED,
                    reason=f"browser_postcondition_unconfirmed:{operation}",
                    duration_seconds=time.monotonic() - start,
                )
            later = calls[index + 1 :]
            if _desktop_change_is_verified("desktop_browser", args, result, later):
                continue
            return _desktop_observation_result(
                status=VerificationStatus.FAILED,
                reason="browser_postcondition_unconfirmed:desktop_browser",
                duration_seconds=time.monotonic() - start,
            )
        return _desktop_observation_result(
            status=VerificationStatus.PASSED,
            reason=f"Verified {len(successful_changes)} browser postcondition(s).",
            duration_seconds=time.monotonic() - start,
        )

    def run(
        self, task: TaskPacket, context: RunContext, *, cancelled: Callable[[], bool] | None = None
    ) -> VerificationResult:
        start = time.monotonic()
        try:
            workspace_root = _resolve_workspace_root(context.workspace_root)
            command = _resolve_packet_command(task.verifier)
            timeout = _resolve_timeout(task.verifier)
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

        if cancelled is not None and cancelled():
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command or [],
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=0.0,
                error="verifier_cancelled",
                fail_type="cancelled",
                verifier_profile="unavailable",
            )
        if self.executor is None:
            return sandbox_verification_unavailable(command or [])
        resolved_command = command if command is not None else list(CANONICAL_CHECK_COMMAND)
        if resolved_command != list(CANONICAL_CHECK_COMMAND):
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=resolved_command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error="invalid_verifier_config: verifier.command must be the canonical check.",
                fail_type="invalid_config",
                excerpt="verifier.command must be the canonical check",
                verifier_profile="explicit",
            )
        if cwd != workspace_root:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=resolved_command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error="invalid_verifier_config: verifier.cwd must be the workspace root.",
                fail_type="invalid_config",
                excerpt="verifier.cwd must be the workspace root",
                verifier_profile="explicit",
            )
        if self.binding is None:
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=resolved_command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error="verifier_binding_required",
                fail_type="invalid_config",
                excerpt="verifier_binding_required",
                verifier_profile="unavailable",
            )
        try:
            outcome = self.executor.execute_command(
                self.binding,
                resolved_command,
                timeout=float(timeout),
                criterion_sha256=self.criterion_sha256,
            )
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            reason = (
                exc.reason_code
                if isinstance(exc, WorkspaceUnavailable)
                else "workspace_unavailable"
            )
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=resolved_command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=time.monotonic() - start,
                error=reason,
                fail_type="execution_unavailable",
                excerpt=reason,
                verifier_profile="unavailable",
            )
        return _map_command_outcome(outcome, command=resolved_command, started=start)


def _truncate_command_text(data: bytes) -> str:
    text = data.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT_CHARS:
        return text[:MAX_OUTPUT_CHARS] + TRUNCATION_SUFFIX
    return text


def _map_command_outcome(
    outcome: WorkspacePythonOutcome, *, command: list[str], started: float
) -> VerificationResult:
    """Исход изолированного make check → VerificationResult. Без фейкового PASSED.

    finished+0 → PASSED; finished+≠0 → FAILED; unavailable → ERROR зеркалом
    sandbox_verification_unavailable; unknown/conflict → ERROR, никогда FAILED.
    """
    duration = time.monotonic() - started
    stdout = _truncate_command_text(outcome.stdout)
    stderr = _truncate_command_text(outcome.stderr)
    excerpt = (stdout + stderr)[-500:]
    if outcome.state == "finished" and (
        outcome.cleanup_warning or outcome.retained_stage is not None
    ):
        # Settlement провалился после того, как проверка отработала:
        # exit code известен и сохраняется, но результат — ERROR, а не
        # тихий PASSED. Retained stage блокирует следующие запуски
        # principal и требует вмешательства оператора.
        return VerificationResult(
            status=VerificationStatus.ERROR,
            command=command,
            exit_code=outcome.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            error="workspace_cleanup_failed",
            fail_type="settlement_failed",
            excerpt=excerpt,
            verifier_profile="canonical",
        )
    if outcome.state == "finished" and outcome.exit_code == 0:
        return VerificationResult(
            status=VerificationStatus.PASSED,
            command=command,
            exit_code=0,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            error=None,
            fail_type=None,
            excerpt=excerpt,
            verifier_profile="canonical",
        )
    if outcome.state == "finished":
        return VerificationResult(
            status=VerificationStatus.FAILED,
            command=command,
            exit_code=outcome.exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            error="canonical check failed",
            fail_type="check_failed",
            excerpt=excerpt,
            verifier_profile="canonical",
        )
    if outcome.state == "unavailable":
        return VerificationResult(
            status=VerificationStatus.ERROR,
            command=command,
            exit_code=None,
            stdout="",
            stderr=SANDBOX_EXECUTION_MESSAGE,
            duration_seconds=duration,
            error=SandboxExecutionReason.UNAVAILABLE.value,
            fail_type="execution_unavailable",
            excerpt=SANDBOX_EXECUTION_MESSAGE,
            verifier_profile="unavailable",
        )
    reason = outcome.reason_code or "workspace_unavailable"
    if outcome.reason_code == "verifier_criterion_changed":
        fail_type = "criterion_changed"
    elif outcome.state == "unknown":
        fail_type = "execution_unknown"
    else:
        fail_type = "execution_conflict"
    return VerificationResult(
        status=VerificationStatus.ERROR,
        command=command,
        exit_code=None,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        error=reason,
        fail_type=fail_type,
        excerpt=excerpt,
        verifier_profile="canonical",
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


def is_repo_workspace(root: Path) -> bool:
    """Marker корня repo/worktree без исполнения Git и repository config.

    Существующий критерий относится к самому workspace root, не его родителям.
    Положительный marker или неоднозначный Git env требуют repository verification.
    """
    ambiguous_git_context = any(
        name in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR"} or name.startswith("GIT_CONFIG_")
        for name in os.environ
    )
    return ambiguous_git_context or (root.resolve() / ".git").exists()
