from __future__ import annotations

import logging
import math
import os
import selectors
import subprocess
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from core.managed_workspace import ManagedWorkspace, WorkspaceLimits
from core.rootless_podman import RootlessPodmanProfile
from core.rootless_staging import RootlessStagePool
from shared.tool_capture import (
    RawProcessCapture,
    ToolCaptureContext,
    ToolCaptureFault,
    ToolCaptureReceipt,
    ToolOperation,
)
from shared.workspace_binding import WorkspaceBinding, WorkspaceConflict, WorkspaceUnavailable

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
            result_id = self.context.binding.store.record_tool_observation(operation, observation)
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
    ) -> WorkspacePythonOutcome:
        if capture is not None and (
            not isinstance(capture, ToolCaptureContext)
            or capture.binding.run.scope.principal_id != binding.principal_id
            or capture.binding.run.scope.session_id != binding.session_id
        ):
            raise ToolCaptureFault("workspace_capture_scope_mismatch")
        attempt = _CaptureAttempt(capture, script) if capture is not None else None
        outcome = self._execute_python(binding, script, timeout=timeout, capture=attempt)
        return attempt.finish(outcome) if attempt is not None else outcome

    def _execute_python(
        self,
        binding: WorkspaceBinding,
        script: str,
        *,
        timeout: float,
        capture: _CaptureAttempt | None,
    ) -> WorkspacePythonOutcome:
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("process timeout must be finite and positive")
        stage = None
        try:
            snapshot = self.workspace.capture(binding, limits=self.limits)
            relative = Path(script)
            if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
                raise WorkspaceUnavailable("invalid workspace Python script")
            if not any(file.path == relative.as_posix() for file in snapshot.files):
                raise WorkspaceUnavailable("workspace Python script unavailable")
            pool = self.staging_pool
            if pool is None or pool.profile is not self.profile:
                raise WorkspaceUnavailable("bounded staging environment unavailable")
            if binding.environment_revision != self.environment_revision:
                raise WorkspaceConflict("workspace staging environment binding changed")
            bounded = pool.prepare(self.workspace, snapshot)
            stage = bounded.stage
            prepared = self.profile.create_python_container(bounded, relative.as_posix())
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
            self.profile.verify_created(prepared)
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
        try:
            pool.check_stage(bounded)
            retained_host = self.workspace.publish_bounded(snapshot, stage, limits=self.limits)
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
            if retained_host is not None:
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
            )
        return WorkspacePythonOutcome(
            "finished", attached.stdout, attached.stderr, child_exit, True, None, "published"
        )
