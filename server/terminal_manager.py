from __future__ import annotations

import asyncio
import contextlib
import os
import pty
import shutil
import subprocess
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from shared.models import JSONValue

if TYPE_CHECKING:
    from collections.abc import Sequence


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


class TerminalManager:
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

    async def create_or_get(
        self,
        session_id: str,
        *,
        workspace_root: str,
        rows: int | None = None,
        cols: int | None = None,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            existing = self._states.get(session_id)
            if existing is not None and existing.status == "running":
                return self._snapshot_for_state(existing)
            state = self._spawn_state(
                session_id=session_id,
                workspace_root=workspace_root,
                rows=rows,
                cols=cols,
            )
            self._states[session_id] = state
            self._append_event_locked(
                state,
                "terminal.started",
                {
                    "session_id": session_id,
                    "terminal_id": state.terminal_id,
                    "status": state.status,
                    "workspace_root": state.workspace_root,
                    "spawn_cwd": state.spawn_cwd,
                    "rows": state.rows,
                    "cols": state.cols,
                },
            )
            self._publish_to_subscribers_locked(state, state.event_buffer[-1].event)
            return self._snapshot_for_state(state)

    async def get_snapshot(self, session_id: str) -> dict[str, JSONValue] | None:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                return None
            return self._snapshot_for_state(state)

    async def write_input(self, session_id: str, data: str) -> dict[str, JSONValue]:
        encoded = data.encode("utf-8")
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            if state.status != "running":
                raise RuntimeError("terminal is not running")
            master_fd = state.master_fd
            if master_fd is None:
                raise RuntimeError("terminal PTY is unavailable")
        await asyncio.to_thread(os.write, master_fd, encoded)
        return await self.require_snapshot(session_id)

    async def resize(
        self,
        session_id: str,
        *,
        rows: int,
        cols: int,
    ) -> dict[str, JSONValue]:
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            if state.status != "running":
                raise RuntimeError("terminal is not running")
            master_fd = state.master_fd
            if master_fd is None:
                raise RuntimeError("terminal PTY is unavailable")
        await asyncio.to_thread(self._apply_winsize, master_fd, rows, cols)
        async with self._lock:
            state = self._states.get(session_id)
            if state is None:
                raise KeyError("terminal not started")
            state.rows = rows
            state.cols = cols
            state.updated_at = _utc_iso_now()
            self._append_event_locked(
                state,
                "terminal.resized",
                {
                    "session_id": session_id,
                    "terminal_id": state.terminal_id,
                    "rows": rows,
                    "cols": cols,
                },
            )
            self._publish_to_subscribers_locked(state, state.event_buffer[-1].event)
            return self._snapshot_for_state(state)

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

    def _spawn_state(
        self,
        *,
        session_id: str,
        workspace_root: str,
        rows: int | None,
        cols: int | None,
    ) -> _TerminalState:
        master_fd, slave_fd = pty.openpty()
        spawn_cwd = workspace_root
        shell_argv = self._shell_argv()
        env = dict(os.environ)
        env["TERM"] = env.get("TERM", "xterm-256color")
        env["PS1"] = ""
        env["PROMPT_COMMAND"] = ""
        process = subprocess.Popen(
            shell_argv,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            cwd=spawn_cwd,
            env=env,
            start_new_session=True,
        )
        os.close(slave_fd)
        os.set_blocking(master_fd, True)
        state = _TerminalState(
            terminal_id=uuid.uuid4().hex,
            session_id=session_id,
            workspace_root=workspace_root,
            spawn_cwd=spawn_cwd,
            rows=max(1, rows or self._default_rows),
            cols=max(1, cols or self._default_cols),
            process=process,
            master_fd=master_fd,
        )
        self._apply_winsize(master_fd, state.rows, state.cols)
        state.read_task = asyncio.create_task(
            self._read_loop(session_id, state.terminal_id),
            name=f"terminal-read-{session_id}",
        )
        state.wait_task = asyncio.create_task(
            self._wait_loop(session_id, state.terminal_id),
            name=f"terminal-wait-{session_id}",
        )
        return state

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

    def _shell_argv(self) -> Sequence[str]:
        bash_path = shutil.which("bash")
        if bash_path:
            return [bash_path, "--noprofile", "--norc", "-i"]
        sh_path = shutil.which("sh")
        if sh_path:
            return [sh_path, "-i"]
        raise RuntimeError("No interactive shell found for PTY terminal runtime.")

    @staticmethod
    def _apply_winsize(master_fd: int, rows: int, cols: int) -> None:
        import fcntl
        import struct
        import termios

        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
