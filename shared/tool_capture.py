from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from shared.task_run import RunScope, RunState, TaskRun

MAX_CAPTURE_BYTES = 1024 * 1024
MAX_VIEW_BYTES = 32 * 1024


@dataclass(frozen=True)
class ProcessObservation:
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)
    stdout_seen: int
    stderr_seen: int
    stdout_eof: bool
    stderr_eof: bool
    exit_code: int | None
    timed_out: bool
    capture_error: str | None = None

    def __post_init__(self) -> None:
        for data, seen in ((self.stdout, self.stdout_seen), (self.stderr, self.stderr_seen)):
            if not isinstance(data, bytes) or len(data) > MAX_CAPTURE_BYTES:
                raise ValueError("Invalid retained capture bytes")
            if (
                type(seen) is not int
                or seen < len(data)
                or len(data) != min(seen, MAX_CAPTURE_BYTES)
            ):
                raise ValueError("Invalid capture byte count")
        if any(
            type(value) is not bool for value in (self.stdout_eof, self.stderr_eof, self.timed_out)
        ):
            raise ValueError("Invalid capture completeness")
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("Invalid observed exit code")
        if self.capture_error not in {
            None,
            "spawn_error",
            "read_error",
            "cleanup_incomplete",
            "output_limit",
        }:
            raise ValueError("Invalid capture error")

    @property
    def stdout_complete(self) -> bool:
        return self.stdout_eof and self.stdout_seen == len(self.stdout)

    @property
    def stderr_complete(self) -> bool:
        return self.stderr_eof and self.stderr_seen == len(self.stderr)


class RawProcessCapture:
    """Ограниченное наблюдение одного request, отдельно от backend view limits."""

    def __init__(self) -> None:
        self._buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self._seen = {"stdout": 0, "stderr": 0}
        self._eof = {"stdout": False, "stderr": False}

    def feed(self, stream: Literal["stdout", "stderr"], chunk: bytes) -> None:
        if not chunk:
            self._eof[stream] = True
            return
        self._seen[stream] += len(chunk)
        target = self._buffers[stream]
        target.extend(chunk[: MAX_CAPTURE_BYTES - len(target)])

    def observation(
        self, *, exit_code: int | None, timed_out: bool, capture_error: str | None
    ) -> ProcessObservation:
        return ProcessObservation(
            bytes(self._buffers["stdout"]),
            bytes(self._buffers["stderr"]),
            self._seen["stdout"],
            self._seen["stderr"],
            self._eof["stdout"],
            self._eof["stderr"],
            exit_code,
            timed_out,
            capture_error,
        )


@dataclass(frozen=True)
class ToolOperation:
    scope: RunScope
    task_id: str
    task_revision: int
    task_run_id: str
    run_attempt_id: str
    run_version: int
    operation_id: str
    tool_attempt_id: str
    model_call_id: str
    request_path: str
    created_at: str


@dataclass(frozen=True)
class StoredToolObservation:
    operation: ToolOperation
    result_id: str
    observation: ProcessObservation
    stdout_digest: str
    stderr_digest: str
    captured_at: str


class ToolCaptureStore(Protocol):
    def begin_tool_operation(
        self, run: TaskRun, *, model_call_id: str, request_path: str
    ) -> ToolOperation: ...

    def record_tool_observation(
        self, operation: ToolOperation, observation: ProcessObservation
    ) -> str: ...


@dataclass(frozen=True)
class CaptureBinding:
    store: ToolCaptureStore = field(repr=False)
    run: TaskRun

    def __post_init__(self) -> None:
        if not isinstance(self.run, TaskRun) or self.run.state != RunState.RUNNING:
            raise ValueError("Capture requires a trusted running admission")

    def __deepcopy__(self, memo: dict[int, object]) -> CaptureBinding:
        # Immutable binding: continuation сохраняет исходного владельца Store.
        return self


@dataclass(frozen=True)
class ToolCaptureContext:
    binding: CaptureBinding
    model_call_id: str


@dataclass(frozen=True)
class ToolCaptureReceipt:
    operation: ToolOperation
    result_id: str
    observation: ProcessObservation


class ToolCaptureFault(RuntimeError):
    def __init__(self, code: str, observation: ProcessObservation | None = None) -> None:
        self.observation = observation
        self.code = code
        # Исключение не содержит payload; известный exit не означает успешный capture.
        detail = f"; observed_exit_code={observation.exit_code}" if observation is not None else ""
        super().__init__(f"{code}{detail}; no automatic replay")
