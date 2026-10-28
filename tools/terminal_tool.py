from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import subprocess
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from config.shell_config import DEFAULT_SHELL_CONFIG_PATH, load_shell_config
from shared.command_safety import is_hard_unsafe_command as _is_unsafe_shell_command
from shared.models import JSONValue, ToolRequest, ToolResult
from shared.sandbox_execution import SandboxExecutionUnavailable, sandbox_execution_failure
from tools.shell_tool import _validate_args as _validate_shell_args


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


@dataclass
class _EventRecord:
    event_id: str
    event: dict[str, JSONValue]


@dataclass
class _TerminalState:
    terminal_id: str
    session_id: str
    workspace_root: str
    spawn_cwd: str
    rows: int
    cols: int
    status: str = "running"
    created_at: str = field(default_factory=_utc_iso_now)
    updated_at: str = field(default_factory=_utc_iso_now)
    closed_at: str | None = None
    exit_code: int | None = None
    output: str = ""
    event_buffer: deque[_EventRecord] = field(default_factory=deque)
    subscribers: set[asyncio.Queue[dict[str, JSONValue]]] = field(default_factory=set)
    process: subprocess.Popen[bytes] | None = None
    master_fd: int | None = None
    read_task: asyncio.Task[None] | None = None
    wait_task: asyncio.Task[None] | None = None
    close_requested: bool = False


class TerminalTool:
    """Single terminal implementation for oneshot commands and PTY sessions."""

    def __init__(
        self,
        *,
        output_max_chars: int = 64_000,
        event_buffer_size: int = 256,
        subscriber_queue_maxsize: int = 256,
        default_rows: int = 24,
        default_cols: int = 80,
    ) -> None:
        self._output_max_chars = max(4_096, output_max_chars)
        self._event_buffer_size = max(16, event_buffer_size)
        self._subscriber_queue_maxsize = max(16, subscriber_queue_maxsize)
        self._default_rows = max(1, default_rows)
        self._default_cols = max(1, default_cols)
        self._states: dict[str, _TerminalState] = {}
        self._lock = asyncio.Lock()

    def handle(self, request: ToolRequest) -> ToolResult:
        mode = str(request.args.get("mode") or "oneshot").strip().lower()
        if mode != "oneshot":
            return ToolResult.failure(
                "PTY terminal mode is managed through the async terminal runtime."
            )
        command = str(request.args.get("command") or "").strip()
        cwd_mode = str(request.args.get("cwd_mode") or "session_root").strip().lower()
        workspace_root_raw = request.args.get("workspace_root")
        sandbox_root_raw = request.args.get("sandbox_root")
        workspace_root = (
            Path(workspace_root_raw).expanduser().resolve()
            if isinstance(workspace_root_raw, str) and workspace_root_raw.strip()
            else (Path("sandbox") / "project").resolve()
        )
        sandbox_root = (
            Path(sandbox_root_raw).expanduser().resolve()
            if isinstance(sandbox_root_raw, str) and sandbox_root_raw.strip()
            else Path("sandbox").resolve()
        )
        return self.run_oneshot(
            command=command,
            cwd_mode=cwd_mode,
            workspace_root=workspace_root,
            sandbox_root=sandbox_root,
        )

    def run_oneshot(
        self,
        *,
        command: str,
        cwd_mode: str,
        workspace_root: Path,
        sandbox_root: Path,
    ) -> ToolResult:
        command = command.strip()
        cwd_mode = cwd_mode.strip().lower()
        if not command:
            return ToolResult.failure("Не указана команда.")
        if _is_unsafe_shell_command(command):
            return ToolResult.failure("Опасная команда заблокирована.")

        try:
            cfg = load_shell_config(DEFAULT_SHELL_CONFIG_PATH)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Ошибка загрузки shell config: {exc}")

        try:
            args = shlex.split(command)
        except ValueError as exc:
            return ToolResult.failure(f"Ошибка парсинга команды: {exc}")
        validation_error = _validate_shell_args(args, set(cfg.allowed_commands))
        if validation_error:
            return ToolResult.failure(validation_error)

        if cwd_mode == "session_root":
            cwd = workspace_root
        elif cwd_mode == "sandbox":
            cwd = sandbox_root
        else:
            return ToolResult.failure("cwd_mode должен быть session_root|sandbox.")
        if not cwd.exists() or not cwd.is_dir():
            return ToolResult.failure("Рабочая директория для команды не найдена.")

        return sandbox_execution_failure()

    async def create_or_get(
        self,
        session_id: str,
        *,
        workspace_root: str,
        rows: int | None = None,
        cols: int | None = None,
    ) -> dict[str, JSONValue]:
        raise SandboxExecutionUnavailable()

    async def get_snapshot(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return None
            return self._snapshot_for_state(state)

    async def write_input(self, session_id: str, data: str) -> dict[str, JSONValue]:
        raise SandboxExecutionUnavailable()

    async def resize(
        self,
        session_id: str,
        *,
        rows: int,
        cols: int,
    ) -> dict[str, JSONValue]:
        raise SandboxExecutionUnavailable()

    async def close(self, session_id: str) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            if state.status != "running":
                return self._snapshot_for_state(state)
            state.close_requested = True
            process = state.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), timeout=2)
            except TimeoutError:
                process.kill()
                await asyncio.to_thread(process.wait)
        return await self.require_snapshot(session_id)

    async def delete_session(self, session_id: str) -> None:
        snapshot = await self.get_snapshot(session_id)
        if snapshot is None:
            return
        await self.close(session_id)
        async with self._lock:
            state = self._states.pop(session_id, None)
            if state is None:
                return
            for queue in list(state.subscribers):
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(
                        {
                            "id": uuid.uuid4().hex,
                            "type": "terminal.closed",
                            "ts": _utc_iso_now(),
                            "payload": {
                                "session_id": session_id,
                                "terminal_id": state.terminal_id,
                                "status": state.status,
                            },
                        }
                    )

    async def shutdown(self) -> None:
        async with self._lock:
            session_ids = list(self._states.keys())
        for session_id in session_ids:
            with contextlib.suppress(KeyError, RuntimeError):
                await self.close(session_id)

    async def subscribe(self, session_id: str) -> asyncio.Queue[dict[str, JSONValue]]:
        queue: asyncio.Queue[dict[str, JSONValue]] = asyncio.Queue(
            maxsize=self._subscriber_queue_maxsize
        )
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            state.subscribers.add(queue)
        return queue

    async def unsubscribe(
        self,
        session_id: str,
        queue: asyncio.Queue[dict[str, JSONValue]],
    ) -> None:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return
            state.subscribers.discard(queue)

    async def get_events_since(
        self,
        session_id: str,
        *,
        last_event_id: str | None,
    ) -> tuple[list[dict[str, JSONValue]], bool]:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            if last_event_id is None:
                return [], False
            events = list(state.event_buffer)
            replay_start = -1
            for idx, record in enumerate(events):
                if record.event_id == last_event_id:
                    replay_start = idx + 1
                    break
            if replay_start < 0:
                return [], True
            return [dict(record.event) for record in events[replay_start:]], False

    async def require_snapshot(self, session_id: str) -> dict[str, JSONValue]:
        snapshot = await self.get_snapshot(session_id)
        if snapshot is None:
            raise KeyError("terminal not started")
        return snapshot

    async def _read_loop(self, session_id: str, terminal_id: str) -> None:
        while True:
            async with self._lock:
                state = self._states.get(session_id)
                if state is None or state.terminal_id != terminal_id:
                    return
                master_fd = state.master_fd
            if master_fd is None:
                return
            try:
                chunk = await asyncio.to_thread(os.read, master_fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            async with self._lock:
                state = self._states.get(session_id)
                if state is None or state.terminal_id != terminal_id:
                    return
                state.output = self._trim_output(state.output + text)
                state.updated_at = _utc_iso_now()
                self._append_event_locked(
                    state,
                    "terminal.output",
                    {
                        "session_id": session_id,
                        "terminal_id": state.terminal_id,
                        "data": text,
                    },
                )
                self._publish_to_subscribers_locked(state, state.event_buffer[-1].event)

    async def _wait_loop(self, session_id: str, terminal_id: str) -> None:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None or state.terminal_id != terminal_id or state.process is None:
                return
            process = state.process
        exit_code = await asyncio.to_thread(process.wait)
        async with self._lock:
            state = self._states.get(session_id)
            if state is None or state.terminal_id != terminal_id:
                return
            if state.status == "closed":
                return
            state.exit_code = exit_code
            state.closed_at = _utc_iso_now()
            state.updated_at = state.closed_at
            state.status = "closed" if state.close_requested else "exited"
            self._cleanup_process_handles_locked(state)
            event_type = "terminal.closed" if state.close_requested else "terminal.exited"
            self._append_event_locked(
                state,
                event_type,
                {
                    "session_id": session_id,
                    "terminal_id": state.terminal_id,
                    "status": state.status,
                    "exit_code": exit_code,
                },
            )
            self._publish_to_subscribers_locked(state, state.event_buffer[-1].event)

    def _append_event_locked(
        self,
        state: _TerminalState,
        event_type: str,
        payload: dict[str, JSONValue],
    ) -> None:
        event: dict[str, JSONValue] = {
            "id": uuid.uuid4().hex,
            "type": event_type,
            "ts": _utc_iso_now(),
            "payload": payload,
        }
        event_id = str(event["id"])
        state.event_buffer.append(_EventRecord(event_id=event_id, event=event))
        while len(state.event_buffer) > self._event_buffer_size:
            state.event_buffer.popleft()

    def _publish_to_subscribers_locked(
        self,
        state: _TerminalState,
        event: dict[str, JSONValue],
    ) -> None:
        stale: list[asyncio.Queue[dict[str, JSONValue]]] = []
        for queue in state.subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                stale.append(queue)
        for queue in stale:
            state.subscribers.discard(queue)

    def _snapshot_for_state(self, state: _TerminalState) -> dict[str, JSONValue]:
        return {
            "terminal_id": state.terminal_id,
            "status": state.status,
            "workspace_root": state.workspace_root,
            "spawn_cwd": state.spawn_cwd,
            "rows": state.rows,
            "cols": state.cols,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "closed_at": state.closed_at,
            "exit_code": state.exit_code,
            "output": state.output,
        }

    def _cleanup_process_handles_locked(self, state: _TerminalState) -> None:
        if state.master_fd is not None:
            with contextlib.suppress(OSError):
                os.close(state.master_fd)
            state.master_fd = None

    def _trim_output(self, text: str) -> str:
        if len(text) <= self._output_max_chars:
            return text
        return text[-self._output_max_chars :]
