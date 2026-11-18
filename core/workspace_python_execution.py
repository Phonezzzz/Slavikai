from __future__ import annotations

import hashlib
import logging
import math
import os
import selectors
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from core.managed_workspace import ManagedWorkspace, WorkspaceLimits, WorkspaceSnapshot
from core.rootless_podman import (
    CANONICAL_CHECK_COMMAND,
    PreparedContainer,
    PreparedPythonContainer,
    RootlessPodmanProfile,
)
from core.rootless_staging import BoundedStage, RootlessStagePool
from shared.tool_capture import (
    RawProcessCapture,
    ToolCaptureContext,
    ToolCaptureFault,
    ToolCaptureReceipt,
    ToolOperation,
    WorkspacePublicationEvidence,
)
from shared.workspace_binding import (
    WorkspaceBinding,
    WorkspaceConflict,
    WorkspaceCriterionChanged,
    WorkspaceFileBudgetExceeded,
    WorkspaceUnavailable,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkspacePythonOutcome:
    state: Literal["finished", "unavailable", "unknown", "conflict"]
    stdout: bytes
    stderr: bytes
    exit_code: int | None
    execution_started: bool | None
    reason_code: str | None
    publication: Literal["none", "published", "retained"]
    retained_stage: Path | None = None
    cleanup_warning: bool = False
    container_id: str | None = None
    capture_receipt: ToolCaptureReceipt | None = None
    changed_paths: frozenset[str] | None = None
    snapshot_sha256: str | None = None


@dataclass
class _CaptureAttempt:
    context: ToolCaptureContext
    script: str
    streams: RawProcessCapture = field(default_factory=RawProcessCapture)
    operation: ToolOperation | None = None
    observed_child_exit: int | None = None

    def begin(self) -> None:
        try:
            run = self.context.binding.run
            operation = self.context.binding.store.begin_tool_operation(
                run,
                model_call_id=self.context.model_call_id,
                request_path=self.script,
            )
        except Exception as exc:
            raise ToolCaptureFault("workspace_capture_intent_failed") from exc
        if not isinstance(operation, ToolOperation) or (
            operation.scope,
            operation.task_id,
            operation.task_revision,
            operation.task_run_id,
            operation.run_attempt_id,
            operation.run_version,
            operation.model_call_id,
            operation.request_path,
        ) != (
            run.scope,
            run.task_id,
            run.task_revision,
            run.task_run_id,
            run.attempt_id,
            run.version,
            self.context.model_call_id,
            self.script,
        ):
            raise ToolCaptureFault("workspace_capture_intent_invalid")
        self.operation = operation

    def finish(self, outcome: WorkspacePythonOutcome) -> WorkspacePythonOutcome:
        operation = self.operation
        if operation is None:
            if outcome.execution_started is not False:
                raise ToolCaptureFault("workspace_capture_intent_unavailable")
            return outcome
        observation = self.streams.observation(
            exit_code=self.observed_child_exit,
            timed_out=outcome.reason_code == "workspace_process_timeout",
            capture_error=(
                "output_limit"
                if outcome.reason_code == "workspace_output_limit"
                else "read_error"
                if outcome.reason_code == "workspace_transport_unknown"
                else "cleanup_incomplete"
                if outcome.reason_code == "workspace_settlement_unknown" or outcome.cleanup_warning
                else None
            ),
        )
        try:
            publication = (
                WorkspacePublicationEvidence(outcome.cleanup_warning, outcome.snapshot_sha256)
                if outcome.publication == "published"
                else None
            )
            result_id = self.context.binding.store.record_tool_observation(
                operation, observation, publication=publication
            )
        except Exception as exc:
            raise ToolCaptureFault("workspace_capture_store_failed", observation) from exc
        return replace(
            outcome, capture_receipt=ToolCaptureReceipt(operation, result_id, observation)
        )


def unavailable_python_execution() -> WorkspacePythonOutcome:
    return WorkspacePythonOutcome(
        "unavailable", b"", b"", None, False, "sandbox_execution_unavailable", "none"
    )


@dataclass(frozen=True)
class _AttachedCapture:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    reason: str | None


def _collect(
    process: subprocess.Popen[bytes],
    *,
    timeout: float,
    limit: int,
    capture: RawProcessCapture | None = None,
) -> _AttachedCapture:
    output = bytearray()
    errors = bytearray()
    reason: str | None = None
    deadline = time.monotonic() + timeout
    try:
        if process.stdout is None or process.stderr is None:
            raise OSError("isolated process output pipes unavailable")
        buffers = {process.stdout.fileno(): output, process.stderr.fileno(): errors}
        streams: dict[int, Literal["stdout", "stderr"]] = {
            process.stdout.fileno(): "stdout",
            process.stderr.fileno(): "stderr",
        }
        with selectors.DefaultSelector() as selector:
            for fd in buffers:
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = "workspace_process_timeout"
                    break
                for key, _ in selector.select(min(remaining, 0.05)):
                    try:
                        chunk = os.read(key.fd, 65536)
                    except BlockingIOError:
                        continue
                    if capture is not None:
                        capture.feed(streams[key.fd], chunk)
                    if not chunk:
                        selector.unregister(key.fd)
                        continue
                    target = buffers[key.fd]
                    available = limit - len(target)
                    target.extend(chunk[:available])
                    if len(chunk) > available:
                        reason = "workspace_output_limit"
                        break
                if reason is not None:
                    break
        if reason is None:
            try:
                process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                reason = "workspace_process_timeout"
    except OSError:
        reason = "workspace_transport_unknown"
    finally:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                reason = "workspace_transport_unknown"
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return _AttachedCapture(bytes(output), bytes(errors), process.returncode, reason)


@dataclass(frozen=True)
class _IsolatedRunner:
    """Один механизм изолированного запуска для Python- и command-путей.

    Различаются только admission (что допускается в staged snapshot),
    verify созданного контейнера и settlement: Python публикует результат
    обратно в workspace, command (verifier) — только наблюдает.
    """

    profile: RootlessPodmanProfile
    workspace: ManagedWorkspace
    limits: WorkspaceLimits
    max_output_bytes: int
    staging_pool: RootlessStagePool | None = None

    @property
    def environment_revision(self) -> str:
        pool = self.staging_pool
        if pool is None or pool.profile is not self.profile:
            raise WorkspaceUnavailable("bounded staging environment unavailable")
        return pool.execution_revision

    def _execute_isolated(
        self,
        binding: WorkspaceBinding,
        *,
        timeout: float,
        capture: _CaptureAttempt | None,
        admit: Callable[[WorkspaceSnapshot, BoundedStage], PreparedContainer],
        publish: bool,
        validate: Callable[[WorkspaceSnapshot], None] | None = None,
        max_files_touched: int | None = None,
        already_touched: frozenset[str] = frozenset(),
    ) -> WorkspacePythonOutcome:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("process timeout must be finite and positive")
        stage = None
        try:
            snapshot = self.workspace.capture(binding, limits=self.limits)
            pool = self.staging_pool
            if pool is None or pool.profile is not self.profile:
                raise WorkspaceUnavailable("bounded staging environment unavailable")
            if binding.environment_revision != self.environment_revision:
                raise WorkspaceConflict("workspace staging environment binding changed")
            # Валидация — до pool.prepare: предзапусковой отказ не должен
            # занимать staging slot (иначе следующие запуски principal
            # блокируются retained stage без механизма восстановления).
            if validate is not None:
                validate(snapshot)
            bounded = pool.prepare(self.workspace, snapshot)
            stage = bounded.stage
            prepared = admit(snapshot, bounded)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            return WorkspacePythonOutcome(
                "conflict" if isinstance(exc, WorkspaceConflict) else "unavailable",
                b"",
                b"",
                None,
                False,
                exc.reason_code
                if isinstance(exc, WorkspaceUnavailable)
                else "workspace_unavailable",
                "none" if stage is None else "retained",
                None if stage is None else stage.root,
            )

        attached = _AttachedCapture(b"", b"", None, "workspace_transport_unknown")
        child_exit: int | None = None
        reason: str | None = None
        start_attempted = False
        removed = False
        try:
            if self.workspace.capture(binding, limits=self.limits) != snapshot:
                raise WorkspaceConflict("workspace changed before process admission")
            if isinstance(prepared, PreparedPythonContainer):
                self.profile.verify_created(prepared)
            else:
                self.profile.verify_created_command(prepared)
            stage.begin_process()
            pool.check_stage(bounded)
            self.profile.check_current()
            if capture is not None:
                capture.begin()
            start_attempted = True
            process = subprocess.Popen(
                self.profile.argv() + ["start", "--attach", prepared.container_id],
                env=self.profile.environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            attached = _collect(
                process,
                timeout=timeout,
                limit=self.max_output_bytes,
                capture=capture.streams if capture is not None else None,
            )
            reason = attached.reason
            if reason is None and attached.returncode is not None:
                child_exit = self.profile.finished_exit(
                    prepared, cli_returncode=attached.returncode
                )
                if capture is not None:
                    capture.observed_child_exit = child_exit
            if child_exit is None and reason is None:
                reason = "workspace_transport_unknown"
        except ToolCaptureFault:
            raise
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            reason = (
                exc.reason_code
                if isinstance(exc, WorkspaceUnavailable)
                else "workspace_transport_unknown"
            )
        except Exception:
            logger.exception("Unexpected isolated workspace execution fault")
            reason = "workspace_transport_unknown"
        finally:
            try:
                removed = self.profile.remove_owned(prepared)
            except Exception:
                logger.exception("Isolated workspace container settlement failed")
            if not removed or reason is not None or child_exit is None:
                stage.mark_unknown()
        if not removed or reason is not None or child_exit is None:
            if not start_attempted:
                return WorkspacePythonOutcome(
                    "conflict" if reason == "workspace_conflict" else "unavailable",
                    attached.stdout,
                    attached.stderr,
                    None,
                    False,
                    reason or "workspace_unavailable",
                    "retained",
                    stage.root,
                    container_id=None if removed else prepared.container_id,
                )
            return WorkspacePythonOutcome(
                "unknown",
                attached.stdout,
                attached.stderr,
                None,
                None,
                reason or "workspace_settlement_unknown",
                "retained",
                stage.root,
                container_id=None if removed else prepared.container_id,
            )

        stage.mark_settled()
        if not publish:
            # Verifier только наблюдает: staged snapshot освобождается без
            # записи чего-либо обратно в workspace.
            try:
                pool.check_stage(bounded)
                pool.release_unpublished(bounded)
            except (OSError, WorkspaceUnavailable):
                return WorkspacePythonOutcome(
                    "finished",
                    attached.stdout,
                    attached.stderr,
                    child_exit,
                    True,
                    None,
                    "none",
                    stage.root,
                    cleanup_warning=True,
                )
            # Freshness после исполнения: успешный результат обязан
            # относиться к актуальному workspace, а не к устаревшему
            # snapshot. Другая сессия того же principal может писать
            # напрямую, без staging slot, пока идёт проверка.
            try:
                fresh = self.workspace.capture(binding, limits=self.limits)
            except WorkspaceConflict as exc:
                return WorkspacePythonOutcome(
                    "conflict",
                    attached.stdout,
                    attached.stderr,
                    child_exit,
                    True,
                    exc.reason_code,
                    "none",
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                return WorkspacePythonOutcome(
                    "unavailable",
                    attached.stdout,
                    attached.stderr,
                    child_exit,
                    True,
                    "workspace_unavailable",
                    "none",
                )
            if fresh != snapshot:
                return WorkspacePythonOutcome(
                    "conflict",
                    attached.stdout,
                    attached.stderr,
                    child_exit,
                    True,
                    "workspace_conflict",
                    "none",
                )
            return WorkspacePythonOutcome(
                "finished",
                attached.stdout,
                attached.stderr,
                child_exit,
                True,
                None,
                "none",
            )
        try:
            pool.check_stage(bounded)
            publication = self.workspace.publish_bounded(
                snapshot,
                stage,
                limits=self.limits,
                max_files_touched=max_files_touched,
                already_touched=already_touched,
                publication_operation=capture.operation if capture is not None else None,
            )
        except WorkspaceFileBudgetExceeded as exc:
            try:
                pool.discard_budget_rejected(bounded)
            except (OSError, WorkspaceUnavailable):
                return WorkspacePythonOutcome(
                    "unavailable",
                    attached.stdout,
                    attached.stderr,
                    child_exit,
                    True,
                    exc.reason_code,
                    "retained",
                    stage.root,
                    cleanup_warning=True,
                )
            return WorkspacePythonOutcome(
                "unavailable",
                attached.stdout,
                attached.stderr,
                child_exit,
                True,
                exc.reason_code,
                "none",
            )
        except (OSError, WorkspaceUnavailable) as exc:
            return WorkspacePythonOutcome(
                "conflict" if isinstance(exc, WorkspaceConflict) else "unavailable",
                attached.stdout,
                attached.stderr,
                child_exit,
                True,
                exc.reason_code
                if isinstance(exc, WorkspaceUnavailable)
                else "workspace_publication_unavailable",
                "retained",
                stage.root,
            )
        try:
            if publication.retired_host is not None:
                raise WorkspaceUnavailable("retired host workspace cleanup unavailable")
            pool.release(bounded)
        except (OSError, WorkspaceUnavailable):
            return WorkspacePythonOutcome(
                "finished",
                attached.stdout,
                attached.stderr,
                child_exit,
                True,
                None,
                "published",
                stage.root,
                cleanup_warning=True,
                changed_paths=publication.changed_paths,
                snapshot_sha256=publication.snapshot_sha256,
            )
        return WorkspacePythonOutcome(
            "finished",
            attached.stdout,
            attached.stderr,
            child_exit,
            True,
            None,
            "published",
            changed_paths=publication.changed_paths,
            snapshot_sha256=publication.snapshot_sha256,
        )


@dataclass(frozen=True)
class RootlessPythonExecutor:
    """Byte primitive ниже ComputerBackend/Gateway, без собственного tool loop."""

    profile: RootlessPodmanProfile
    workspace: ManagedWorkspace
    limits: WorkspaceLimits
    max_output_bytes: int
    staging_pool: RootlessStagePool | None = None

    def __post_init__(self) -> None:
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise ValueError("output byte limit must be a positive integer")

    @property
    def environment_revision(self) -> str:
        pool = self.staging_pool
        if pool is None or pool.profile is not self.profile:
            raise WorkspaceUnavailable("bounded staging environment unavailable")
        return pool.execution_revision

    def execute_python(
        self,
        binding: WorkspaceBinding,
        script: str,
        *,
        timeout: float,
        capture: ToolCaptureContext | None = None,
        max_files_touched: int | None = None,
        already_touched: frozenset[str] = frozenset(),
    ) -> WorkspacePythonOutcome:
        if capture is not None and (
            not isinstance(capture, ToolCaptureContext)
            or capture.binding.run.scope.principal_id != binding.principal_id
            or capture.binding.run.scope.session_id != binding.session_id
        ):
            raise ToolCaptureFault("workspace_capture_scope_mismatch")
        attempt = _CaptureAttempt(capture, script) if capture is not None else None
        outcome = self._execute_python(
            binding,
            script,
            timeout=timeout,
            capture=attempt,
            max_files_touched=max_files_touched,
            already_touched=already_touched,
        )
        return attempt.finish(outcome) if attempt is not None else outcome

    def _execute_python(
        self,
        binding: WorkspaceBinding,
        script: str,
        *,
        timeout: float,
        capture: _CaptureAttempt | None,
        max_files_touched: int | None,
        already_touched: frozenset[str],
    ) -> WorkspacePythonOutcome:
        relative = Path(script)

        def validate(snapshot: WorkspaceSnapshot) -> None:
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
                raise WorkspaceUnavailable("invalid workspace Python script")
            if relative.as_posix() in snapshot.directories:
                raise WorkspaceUnavailable("workspace Python script unavailable")
            if not any(file.path == relative.as_posix() for file in snapshot.files):
                raise WorkspaceUnavailable("workspace Python script unavailable")

        def admit(snapshot: WorkspaceSnapshot, bounded: BoundedStage) -> PreparedContainer:
            return self.profile.create_python_container(bounded, relative.as_posix())

        return self._runner()._execute_isolated(
            binding,
            timeout=timeout,
            capture=capture,
            admit=admit,
            publish=True,
            validate=validate,
            max_files_touched=max_files_touched,
            already_touched=already_touched,
        )

    def _runner(self) -> _IsolatedRunner:
        return _IsolatedRunner(
            profile=self.profile,
            workspace=self.workspace,
            limits=self.limits,
            max_output_bytes=self.max_output_bytes,
            staging_pool=self.staging_pool,
        )


@dataclass(frozen=True)
class RootlessCommandExecutor:
    """No-publish executor штатного process verifier: только
    ["make", "-f", "Makefile", "check"].

    Исполняет canonical check на staged snapshot managed workspace; ничего
    не публикует обратно в workspace. Без квалифицированного command-профиля
    недоступен (is_available False; execute_command — WorkspaceUnavailable).
    """

    profile: RootlessPodmanProfile
    workspace: ManagedWorkspace
    limits: WorkspaceLimits
    max_output_bytes: int
    staging_pool: RootlessStagePool | None = None

    def __post_init__(self) -> None:
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise ValueError("output byte limit must be a positive integer")

    @property
    def environment_revision(self) -> str:
        pool = self.staging_pool
        if pool is None or pool.profile is not self.profile:
            raise WorkspaceUnavailable("bounded staging environment unavailable")
        return pool.execution_revision

    @classmethod
    def from_python_executor(
        cls, executor: RootlessPythonExecutor | None
    ) -> RootlessCommandExecutor | None:
        """Вывести command executor из существующего Python executor.

        None — когда Python executor отсутствует или staging pool не задан:
        command-путь тогда недоступен, Python-путь это не затрагивает.
        """
        if executor is None or executor.staging_pool is None:
            return None
        return cls(
            profile=executor.profile,
            workspace=executor.workspace,
            limits=executor.limits,
            max_output_bytes=executor.max_output_bytes,
            staging_pool=executor.staging_pool,
        )

    def is_available(self) -> bool:
        """Живая проверка command-пути без запуска процессов."""
        if self.profile.command_revision is None:
            return False
        pool = self.staging_pool
        if pool is None or pool.profile is not self.profile:
            return False
        try:
            self.profile.check_current()
            pool.check_current()
        except (OSError, ValueError, subprocess.SubprocessError):
            return False
        return True

    def execute_command(
        self,
        binding: WorkspaceBinding,
        argv: list[str],
        *,
        timeout: float,
        criterion_sha256: str | None = None,
    ) -> WorkspacePythonOutcome:
        if list(argv) != list(CANONICAL_CHECK_COMMAND):
            raise WorkspaceUnavailable("isolated command is not allowlisted")
        if not self.is_available():
            # Fail-closed до capture/staging/engine: недоступный backend
            # никогда не запускает процесс.
            return WorkspacePythonOutcome(
                "unavailable",
                b"",
                b"",
                None,
                False,
                "sandbox_execution_unavailable",
                "none",
            )

        def validate(snapshot: WorkspaceSnapshot) -> None:
            # Хэш сверяется на том же snapshot, который пойдёт в контейнер:
            # Makefile из snapshot — это ровно те байты, которые make
            # прочитает по -f. Любое расхождение с доверенным критерием
            # (включая удаление файла) — verifier_criterion_changed.
            criterion = next((file for file in snapshot.files if file.path == "Makefile"), None)
            if criterion is None:
                raise WorkspaceCriterionChanged(
                    "canonical check definition unavailable in staged snapshot"
                )
            if criterion_sha256 is not None:
                actual = hashlib.sha256(criterion.content).hexdigest()
                if actual != criterion_sha256:
                    raise WorkspaceCriterionChanged(
                        "staged Makefile differs from the trusted criterion"
                    )

        def admit(snapshot: WorkspaceSnapshot, bounded: BoundedStage) -> PreparedContainer:
            return self.profile.create_command_container(bounded, list(argv))

        return self._runner()._execute_isolated(
            binding,
            timeout=timeout,
            capture=None,
            admit=admit,
            publish=False,
            validate=validate,
        )

    def _runner(self) -> _IsolatedRunner:
        return _IsolatedRunner(
            profile=self.profile,
            workspace=self.workspace,
            limits=self.limits,
            max_output_bytes=self.max_output_bytes,
            staging_pool=self.staging_pool,
        )
