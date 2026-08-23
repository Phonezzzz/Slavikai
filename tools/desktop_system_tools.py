from __future__ import annotations

import hashlib
import json
import os
import platform
import pwd
import re
import shutil
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import psutil

from core.desktop_security import DesktopPathSecurity
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.desktop_tools import DesktopLaunchTool

MAX_STRUCTURED_ITEMS = 200
MAX_CLIPBOARD_CHARS = 64_000
MAX_LOG_RECORDS = 200
PACKAGE_RE = re.compile(r"^[a-z0-9][a-z0-9+.-]{0,127}(?::[a-z0-9]+)?$")
UNIT_RE = re.compile(r"^[A-Za-z0-9@_.:-]{1,192}$")


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: int,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


class ClipboardBackend(Protocol):
    @property
    def name(self) -> str: ...

    def read(self) -> str: ...

    def write(self, text: str) -> None: ...

    def clear(self) -> None: ...


class DesktopLaunchControl(Protocol):
    def drain_launches(self) -> list[subprocess.Popen[bytes]]: ...

    def restore_launches(self, processes: Sequence[subprocess.Popen[bytes]]) -> None: ...


class DesktopProcessTracker(Protocol):
    def forget_launched_process(self, pid: int) -> None: ...


@dataclass(frozen=True, slots=True)
class TrackedProcess:
    process: subprocess.Popen[bytes]
    create_time: float
    argv: tuple[str, ...]


class SubprocessClipboardBackend:
    def __init__(
        self,
        *,
        backend_name: str,
        read_argv: Sequence[str],
        write_argv: Sequence[str],
        runner: CommandRunner,
    ) -> None:
        self._name = backend_name
        self._read_argv = tuple(read_argv)
        self._write_argv = tuple(write_argv)
        self._runner = runner

    @property
    def name(self) -> str:
        return self._name

    def read(self) -> str:
        completed = self._runner(self._read_argv, timeout=5)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "clipboard read failed")
        return completed.stdout

    def write(self, text: str) -> None:
        completed = self._runner(self._write_argv, timeout=5, input_text=text)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "clipboard write failed")

    def clear(self) -> None:
        self.write("")


def detect_clipboard_backend(
    *,
    environ: Mapping[str, str] | None = None,
    runner: CommandRunner | None = None,
) -> ClipboardBackend | None:
    env = environ if environ is not None else os.environ
    run = runner or _run_command
    if env.get("WAYLAND_DISPLAY") and shutil.which("wl-copy") and shutil.which("wl-paste"):
        return SubprocessClipboardBackend(
            backend_name="wayland-wl-clipboard",
            read_argv=("wl-paste", "--no-newline"),
            write_argv=("wl-copy", "--type", "text/plain;charset=utf-8"),
            runner=run,
        )
    if env.get("DISPLAY") and shutil.which("xclip"):
        return SubprocessClipboardBackend(
            backend_name="x11-xclip",
            read_argv=("xclip", "-selection", "clipboard", "-o"),
            write_argv=("xclip", "-selection", "clipboard", "-i"),
            runner=run,
        )
    if env.get("DISPLAY") and shutil.which("xsel"):
        return SubprocessClipboardBackend(
            backend_name="x11-xsel",
            read_argv=("xsel", "--clipboard", "--output"),
            write_argv=("xsel", "--clipboard", "--input"),
            runner=run,
        )
    return None


class DesktopClipboardTool:
    def __init__(self, backend: ClipboardBackend | None = None) -> None:
        self._backend = backend

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request)
        backend = self._backend or detect_clipboard_backend()
        if backend is None:
            return ToolResult.failure(
                "Clipboard unavailable: no usable Wayland/X11 clipboard backend in this session."
            )
        try:
            if operation == "read":
                text = backend.read()
                truncated = len(text) > MAX_CLIPBOARD_CHARS
                visible = text[:MAX_CLIPBOARD_CHARS]
                return ToolResult.success(
                    {
                        "output": visible,
                        "text": visible,
                        "backend": backend.name,
                        "truncated": truncated,
                        "total_chars": len(text),
                    }
                )
            if operation == "write":
                text = _required_string(request, "text")
                backend.write(text)
                observed = backend.read()
                verified = observed == text
                if not verified:
                    return ToolResult.failure("Clipboard write verification failed.")
                return ToolResult.success(
                    {
                        "output": "Clipboard text written and verified.",
                        "backend": backend.name,
                        "verified": True,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "chars": len(text),
                    }
                )
            if operation == "clear":
                backend.clear()
                verified = backend.read() == ""
                if not verified:
                    return ToolResult.failure("Clipboard clear verification failed.")
                return ToolResult.success(
                    {
                        "output": "Clipboard cleared and verified.",
                        "backend": backend.name,
                        "verified": True,
                    }
                )
        except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as exc:
            return ToolResult.failure(f"Clipboard operation failed: {exc}")
        return ToolResult.failure("operation must be read|write|clear")


class DesktopSystemInfoTool:
    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request, default="summary")
        limit = _bounded_int(request.args.get("limit"), 100, 1, MAX_STRUCTURED_ITEMS)
        offset = _bounded_int(request.args.get("offset"), 0, 0, 1_000_000)
        try:
            if operation == "summary":
                data: dict[str, JSONValue] = {
                    "os": _os_info(),
                    "hostname": socket.gethostname(),
                    "cpu": _cpu_info(),
                    "memory": _memory_info(),
                    "session": _session_info(),
                }
                return ToolResult.success({"output": "Structured host summary.", **data})
            if operation == "os":
                return ToolResult.success({"output": "Structured OS information.", **_os_info()})
            if operation == "cpu":
                return ToolResult.success({"output": "Structured CPU information.", **_cpu_info()})
            if operation == "memory":
                return ToolResult.success(
                    {"output": "Structured memory information.", **_memory_info()}
                )
            if operation in {"disks", "mounts"}:
                items = _disk_items()
                return _paginated_result(operation, items, offset=offset, limit=limit)
            if operation == "network":
                items = _network_items()
                return _paginated_result(operation, items, offset=offset, limit=limit)
            if operation == "processes":
                items = _process_items(query=_optional_string(request, "query"))
                return _paginated_result(operation, items, offset=offset, limit=limit)
            if operation == "session":
                return ToolResult.success(
                    {"output": "Safe desktop session information.", **_session_info()}
                )
        except (OSError, psutil.Error) as exc:
            return ToolResult.failure(f"System information failed: {exc}")
        return ToolResult.failure(
            "operation must be summary|os|cpu|memory|disks|mounts|network|processes|session"
        )


class DesktopProcessTool:
    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        launcher: DesktopLaunchTool,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._security = security
        self._launcher = launcher
        self._cancelled = cancelled or (lambda: False)
        self._protected_pids = _process_ancestor_pids()
        self._launched: dict[int, TrackedProcess] = {}

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request)
        try:
            if operation in {"list", "find"}:
                limit = _bounded_int(request.args.get("limit"), 100, 1, MAX_STRUCTURED_ITEMS)
                offset = _bounded_int(request.args.get("offset"), 0, 0, 1_000_000)
                query = _optional_string(request, "query")
                if operation == "find" and query is None:
                    return ToolResult.failure("query is required for process find")
                return _paginated_result(
                    "processes",
                    self._process_items(query=query),
                    offset=offset,
                    limit=limit,
                )
            if operation == "launch":
                launch_request = ToolRequest(
                    "desktop_launch",
                    {
                        "argv": request.args.get("argv", []),
                        "cwd": request.args.get("cwd", str(self._security.home)),
                    },
                )
                result, process = self._launcher.start(launch_request)
                if not result.ok:
                    return result
                pid = result.data.get("pid")
                if not isinstance(pid, int) or process is None:
                    return ToolResult.failure("Process launch returned no PID.")
                create_time = time.time()
                identity_source = "launcher_handle"
                identity = _process_identity(pid)
                if identity is not None:
                    observed = identity.get("create_time")
                    if isinstance(observed, (int, float)):
                        create_time = float(observed)
                        identity_source = "psutil"
                argv_raw = request.args.get("argv")
                assert isinstance(argv_raw, list)
                tracked = TrackedProcess(
                    process=process,
                    create_time=create_time,
                    argv=tuple(str(item) for item in argv_raw),
                )
                self._launched[pid] = tracked
                identity = {
                    "pid": pid,
                    "create_time": create_time,
                    "running": process.poll() is None,
                    "status": "running" if process.poll() is None else "exited",
                    "name": Path(tracked.argv[0]).name if tracked.argv else "",
                    "identity_source": identity_source,
                }
                if process.poll() is not None:
                    self._launched.pop(pid, None)
                    return ToolResult.failure("Process exited before launch verification.")
                result.data.update(identity)
                result.data["verified"] = True
                result.data["operation"] = "launch"
                return result
            if operation in {"inspect", "status", "wait", "terminate", "kill"}:
                pid = _required_pid(request)
                if operation in {"terminate", "kill"}:
                    return self._stop_process(request, pid=pid, operation=operation)
                if operation == "wait":
                    timeout = _bounded_int(request.args.get("timeout_seconds"), 30, 1, 120)
                    return self._wait_process(request, pid=pid, timeout=timeout)
                identity = self._identity(pid)
                if identity is None:
                    return ToolResult.success(
                        {
                            "output": f"Process {pid} is not running.",
                            "pid": pid,
                            "running": False,
                            "operation": operation,
                        }
                    )
                expected_raw = request.args.get("expected_create_time")
                observed_create_time = identity.get("create_time")
                if (
                    isinstance(expected_raw, (int, float))
                    and not isinstance(expected_raw, bool)
                    and isinstance(observed_create_time, (int, float))
                    and abs(float(observed_create_time) - float(expected_raw)) > 0.01
                ):
                    return ToolResult.success(
                        {
                            "output": (
                                f"Original process {pid} is no longer running; the PID was reused."
                            ),
                            "operation": operation,
                            "pid": pid,
                            "expected_create_time": float(expected_raw),
                            "observed_create_time": float(observed_create_time),
                            "running": False,
                            "pid_reused": True,
                        }
                    )
                if operation == "status":
                    return ToolResult.success(
                        {"output": f"Process {pid} is running.", **identity, "operation": operation}
                    )
                return ToolResult.success(
                    {"output": f"Process {pid} inspected.", **self._inspect(pid)}
                )
        except (OSError, ValueError, psutil.Error) as exc:
            return ToolResult.failure(f"Process operation failed: {exc}")
        return ToolResult.failure(
            "operation must be list|find|inspect|launch|status|wait|terminate|kill"
        )

    def forget_launched_process(self, pid: int) -> None:
        self._launched.pop(pid, None)

    def _stop_process(self, request: ToolRequest, *, pid: int, operation: str) -> ToolResult:
        if pid in self._protected_pids or pid <= 1:
            return ToolResult.failure("Refusing to stop SlavikAI or an ancestor/system process.")
        expected = _required_float(request, "expected_create_time")
        tracked = self._launched.get(pid)
        if tracked is not None:
            if abs(tracked.create_time - expected) > 0.01:
                return ToolResult.failure("Stale PID identity: process create_time does not match.")
            if operation == "kill":
                tracked.process.kill()
            else:
                tracked.process.terminate()
            try:
                tracked.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                return ToolResult.failure(f"Process {pid} did not exit after {operation}.")
            self._launched.pop(pid, None)
            return ToolResult.success(
                {
                    "output": f"Process {pid} {operation} completed and verified.",
                    "operation": operation,
                    "pid": pid,
                    "create_time": tracked.create_time,
                    "running": False,
                    "verified": tracked.process.poll() is not None,
                }
            )
        process = psutil.Process(pid)
        actual = process.create_time()
        if abs(actual - expected) > 0.01:
            return ToolResult.failure("Stale PID identity: process create_time does not match.")
        if operation == "kill":
            process.kill()
        else:
            process.terminate()
        try:
            process.wait(timeout=3)
        except psutil.TimeoutExpired:
            return ToolResult.failure(f"Process {pid} did not exit after {operation}.")
        return ToolResult.success(
            {
                "output": f"Process {pid} {operation} completed and verified.",
                "operation": operation,
                "pid": pid,
                "create_time": actual,
                "running": False,
                "verified": not psutil.pid_exists(pid),
            }
        )

    def _wait_process(self, request: ToolRequest, *, pid: int, timeout: int) -> ToolResult:
        expected_raw = request.args.get("expected_create_time")
        tracked = self._launched.get(pid)
        if tracked is not None:
            if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool):
                if abs(tracked.create_time - float(expected_raw)) > 0.01:
                    return ToolResult.failure("Stale PID identity while waiting.")
            try:
                tracked.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return ToolResult.failure(f"Process {pid} is still running after {timeout}s.")
            self._launched.pop(pid, None)
            return ToolResult.success(
                {
                    "output": f"Process {pid} exited.",
                    "operation": "wait",
                    "pid": pid,
                    "running": False,
                }
            )
        process = psutil.Process(pid)
        if isinstance(expected_raw, (int, float)) and not isinstance(expected_raw, bool):
            if abs(process.create_time() - float(expected_raw)) > 0.01:
                return ToolResult.failure("Stale PID identity while waiting.")
        deadline = time.monotonic() + timeout
        while process.is_running() and time.monotonic() < deadline:
            if self._cancelled():
                return ToolResult.failure("Process wait cancelled.")
            try:
                process.wait(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
            except psutil.TimeoutExpired:
                continue
        running = process.is_running() and process.status() != psutil.STATUS_ZOMBIE
        if running:
            return ToolResult.failure(f"Process {pid} is still running after {timeout}s.")
        return ToolResult.success(
            {
                "output": f"Process {pid} exited.",
                "operation": "wait",
                "pid": pid,
                "running": False,
            }
        )

    def _identity(self, pid: int) -> dict[str, JSONValue] | None:
        tracked = self._launched.get(pid)
        if tracked is not None:
            running = tracked.process.poll() is None
            return {
                "pid": pid,
                "create_time": tracked.create_time,
                "running": running,
                "status": "running" if running else "exited",
                "name": Path(tracked.argv[0]).name if tracked.argv else "",
                "identity_source": "launcher_handle",
            }
        return _process_identity(pid)

    def _inspect(self, pid: int) -> dict[str, JSONValue]:
        tracked = self._launched.get(pid)
        if tracked is not None:
            identity = self._identity(pid)
            assert identity is not None
            return {**identity, "argv": list(tracked.argv)}
        return _inspect_process(pid)

    def _process_items(self, *, query: str | None) -> list[dict[str, JSONValue]]:
        items = _process_items(query=query)
        known_pids = {cast(int, item["pid"]) for item in items}
        needle = query.casefold() if query is not None else None
        for pid, tracked in list(self._launched.items()):
            if tracked.process.poll() is not None:
                self._launched.pop(pid, None)
                continue
            name = Path(tracked.argv[0]).name if tracked.argv else ""
            if pid in known_pids or (
                needle is not None and needle not in name.casefold() and needle not in str(pid)
            ):
                continue
            items.append(
                {
                    "pid": pid,
                    "name": name,
                    "username": pwd.getpwuid(os.getuid()).pw_name,
                    "status": "running",
                    "create_time": tracked.create_time,
                    "identity_source": "launcher_handle",
                }
            )
        return sorted(items, key=lambda item: cast(int, item["pid"]))


class DesktopUnverifiedLaunchCleanupTool:
    def __init__(
        self,
        control: DesktopLaunchControl,
        process_tracker: DesktopProcessTracker,
    ) -> None:
        self._control = control
        self._process_tracker = process_tracker

    def handle(self, request: ToolRequest) -> ToolResult:
        if request.args:
            return ToolResult.failure("Desktop launch cleanup does not accept arguments.")
        terminated: list[int] = []
        killed: list[int] = []
        already_exited: list[int] = []
        failed: list[int] = []
        retry: list[subprocess.Popen[bytes]] = []
        for process in self._control.drain_launches():
            pid = process.pid
            try:
                if process.poll() is None:
                    os.killpg(pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(pid, signal.SIGKILL)
                            process.wait(timeout=3)
                        except (OSError, subprocess.TimeoutExpired):
                            if process.poll() is None:
                                failed.append(pid)
                                retry.append(process)
                            else:
                                killed.append(pid)
                        else:
                            killed.append(pid)
                    else:
                        terminated.append(pid)
                else:
                    already_exited.append(pid)
            except OSError:
                if process.poll() is None:
                    failed.append(pid)
                    retry.append(process)
                else:
                    already_exited.append(pid)
            if pid not in failed:
                self._process_tracker.forget_launched_process(pid)
        self._control.restore_launches(retry)
        details: dict[str, JSONValue] = {
            "operation": "rollback",
            "terminated_pids": terminated,
            "killed_pids": killed,
            "already_exited_pids": already_exited,
            "failed_pids": failed,
            "retained_pids": failed,
            "verified": not failed,
        }
        if failed:
            return ToolResult.failure(
                "Failed to terminate one or more unverified Desktop launches.",
                meta=details,
            )
        return ToolResult.success(
            {
                "output": "Unverified Desktop launches were rolled back through ToolGateway.",
                **details,
            }
        )


class DesktopSystemdTool:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or _run_command

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request)
        scope = _optional_string(request, "scope") or "system"
        if scope not in {"system", "user"}:
            return ToolResult.failure("scope must be system|user")
        unit = _optional_string(request, "unit")
        if operation != "logs" and unit is None:
            return ToolResult.failure("unit is required")
        if unit is not None and UNIT_RE.fullmatch(unit) is None:
            return ToolResult.failure("Invalid systemd unit name.")
        try:
            if operation == "status":
                assert unit is not None
                return self._status(unit, scope=scope)
            if operation == "logs":
                if unit is None:
                    return ToolResult.failure("unit is required for logs")
                return self._logs(unit, scope=scope, request=request)
            if operation in {"start", "stop", "restart", "enable", "disable"}:
                assert unit is not None
                argv = ["systemctl"]
                if scope == "user":
                    argv.append("--user")
                argv.extend((operation, "--", unit))
                completed = self._runner(argv, timeout=60)
                if completed.returncode != 0:
                    return ToolResult.failure(
                        f"systemd {operation} failed.",
                        meta={"stderr": completed.stderr[:8000], "exit_code": completed.returncode},
                    )
                status = self._status(unit, scope=scope)
                if not status.ok:
                    return ToolResult.failure(
                        f"systemd {operation} verification failed: {status.error or 'unknown'}"
                    )
                if not _systemd_expected(operation, status.data):
                    return ToolResult.failure(
                        f"systemd {operation} completed but resulting unit state is unexpected.",
                        meta={"status": status.data},
                    )
                status.data.update(
                    {
                        "output": f"systemd {operation} completed and verified for {unit}.",
                        "operation": operation,
                        "verified": True,
                    }
                )
                return status
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            return ToolResult.failure(f"systemd operation failed: {exc}")
        return ToolResult.failure("operation must be status|logs|start|stop|restart|enable|disable")

    def _status(self, unit: str, *, scope: str) -> ToolResult:
        argv = ["systemctl"]
        if scope == "user":
            argv.append("--user")
        argv.extend(
            (
                "show",
                "--no-pager",
                "--property=Id,LoadState,ActiveState,SubState,UnitFileState,MainPID,Result",
                "--",
                unit,
            )
        )
        completed = self._runner(argv, timeout=20)
        values = _parse_key_values(completed.stdout)
        if completed.returncode != 0 or values.get("LoadState") == "not-found":
            return ToolResult.failure(
                f"systemd unit not available: {unit}",
                meta={"stderr": completed.stderr[:8000], "exit_code": completed.returncode},
            )
        return ToolResult.success(
            {
                "output": f"Structured systemd status for {unit}.",
                "operation": "status",
                "unit": unit,
                "scope": scope,
                "load_state": values.get("LoadState", ""),
                "active_state": values.get("ActiveState", ""),
                "sub_state": values.get("SubState", ""),
                "unit_file_state": values.get("UnitFileState", ""),
                "main_pid": _safe_int(values.get("MainPID")),
                "result": values.get("Result", ""),
            }
        )

    def _logs(self, unit: str, *, scope: str, request: ToolRequest) -> ToolResult:
        limit = _bounded_int(request.args.get("limit"), 50, 1, MAX_LOG_RECORDS)
        argv = ["journalctl"]
        if scope == "user":
            argv.append("--user")
        argv.extend(("--unit", unit, "--no-pager", "--output=json", "--lines", str(limit)))
        completed = self._runner(argv, timeout=20)
        if completed.returncode != 0:
            return ToolResult.failure(
                "journalctl failed.",
                meta={"stderr": completed.stderr[:8000], "exit_code": completed.returncode},
            )
        records: list[JSONValue] = []
        for line in completed.stdout.splitlines()[-limit:]:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(raw, dict):
                continue
            records.append(
                {
                    "timestamp": str(raw.get("__REALTIME_TIMESTAMP", "")),
                    "priority": str(raw.get("PRIORITY", "")),
                    "message": str(raw.get("MESSAGE", ""))[:4000],
                    "pid": str(raw.get("_PID", "")),
                }
            )
        return ToolResult.success(
            {
                "output": f"Returned {len(records)} structured journal record(s).",
                "operation": "logs",
                "unit": unit,
                "scope": scope,
                "records": records,
                "truncated": len(records) >= limit,
            }
        )


class DesktopPackageTool:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or _run_command

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request)
        package = _optional_string(request, "package")
        if operation != "update_metadata":
            if package is None:
                return ToolResult.failure("package is required")
            if PACKAGE_RE.fullmatch(package) is None:
                return ToolResult.failure("Invalid Ubuntu package name.")
        try:
            if operation == "search":
                assert package is not None
                return self._search(package, request=request)
            if operation == "query":
                assert package is not None
                return self._query(package)
            if operation in {"install", "remove"}:
                assert package is not None
                env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
                argv = ["apt-get", "-y"]
                if operation == "install":
                    argv.extend(("--no-install-recommends", "install", package))
                else:
                    argv.extend(("remove", package))
                completed = self._runner(argv, timeout=120, env=env)
                if completed.returncode != 0:
                    return ToolResult.failure(
                        f"Package {operation} failed.",
                        meta={
                            "exit_code": completed.returncode,
                            "stdout": completed.stdout[-8000:],
                            "stderr": completed.stderr[-8000:],
                        },
                    )
                state = self._query(package)
                if not state.ok:
                    return ToolResult.failure(
                        f"Package {operation} state verification failed: {state.error or 'unknown'}"
                    )
                installed = state.data.get("installed") is True
                expected = operation == "install"
                if installed != expected:
                    return ToolResult.failure(
                        f"Package {operation} completed but installed state is {installed}."
                    )
                state.data.update(
                    {
                        "output": f"Package {operation} completed and verified: {package}",
                        "operation": operation,
                        "verified": True,
                    }
                )
                return state
            if operation == "update_metadata":
                completed = self._runner(["apt-get", "update"], timeout=120)
                if completed.returncode != 0:
                    return ToolResult.failure(
                        "Package metadata update failed.",
                        meta={"stderr": completed.stderr[-8000:]},
                    )
                metadata_state = _apt_metadata_state()
                if metadata_state["file_count"] == 0:
                    return ToolResult.failure(
                        "Package metadata update returned success, but no apt list state exists."
                    )
                return ToolResult.success(
                    {
                        "output": "Ubuntu package metadata updated and list state verified.",
                        "operation": operation,
                        "verified": True,
                        "metadata": metadata_state,
                    }
                )
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            return ToolResult.failure(f"Package operation failed: {exc}")
        return ToolResult.failure("operation must be search|query|install|remove|update_metadata")

    def _search(self, query: str, *, request: ToolRequest) -> ToolResult:
        limit = _bounded_int(request.args.get("limit"), 50, 1, MAX_STRUCTURED_ITEMS)
        completed = self._runner(["apt-cache", "search", "--names-only", query], timeout=30)
        if completed.returncode != 0:
            return ToolResult.failure("apt-cache search failed.")
        items: list[JSONValue] = []
        lines = completed.stdout.splitlines()
        for line in lines[:limit]:
            name, separator, description = line.partition(" - ")
            if not separator:
                continue
            items.append({"package": name, "description": description[:1000]})
        return ToolResult.success(
            {
                "output": f"Found {len(items)} package result(s).",
                "operation": "search",
                "query": query,
                "items": items,
                "truncated": len(lines) > limit,
                "total": len(lines),
            }
        )

    def _query(self, package: str) -> ToolResult:
        installed_result = self._runner(
            ["dpkg-query", "-W", "-f=${db:Status-Abbrev}\t${Version}\n", package],
            timeout=20,
        )
        installed = installed_result.returncode == 0 and installed_result.stdout.startswith("ii ")
        installed_version = ""
        if installed:
            _, _, installed_version = installed_result.stdout.strip().partition("\t")
        policy = self._runner(["apt-cache", "policy", package], timeout=20)
        if policy.returncode != 0:
            return ToolResult.failure(f"Package not found: {package}")
        values = _parse_apt_policy(policy.stdout)
        if values["candidate"] == "(none)" and not installed:
            return ToolResult.failure(f"Package not found: {package}")
        return ToolResult.success(
            {
                "output": f"Structured package state for {package}.",
                "operation": "query",
                "package": package,
                "installed": installed,
                "installed_version": installed_version or values["installed"],
                "available_version": values["candidate"],
            }
        )


class DesktopSessionTool:
    def __init__(self, runner: CommandRunner | None = None) -> None:
        self._runner = runner or _run_command

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request, default="capabilities")
        if operation == "capabilities":
            return ToolResult.success(
                {"output": "Detected desktop integration capabilities.", **_desktop_capabilities()}
            )
        if operation == "notify":
            title = _required_string(request, "title")[:200]
            body = _required_string(request, "body")[:4000]
            if not os.environ.get("DBUS_SESSION_BUS_ADDRESS") or not shutil.which("gdbus"):
                return ToolResult.failure("Desktop notification DBus session is unavailable.")
            argv = [
                "gdbus",
                "call",
                "--session",
                "--dest",
                "org.freedesktop.Notifications",
                "--object-path",
                "/org/freedesktop/Notifications",
                "--method",
                "org.freedesktop.Notifications.Notify",
                "SlavikAI",
                "0",
                "",
                title,
                body,
                "[]",
                "{}",
                "5000",
            ]
            try:
                completed = self._runner(argv, timeout=10)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return ToolResult.failure(f"Desktop notification failed: {exc}")
            if completed.returncode != 0:
                return ToolResult.failure(
                    "Desktop notification DBus call failed.",
                    meta={"stderr": completed.stderr[:8000]},
                )
            return ToolResult.success(
                {
                    "output": "Desktop notification submitted.",
                    "operation": "notify",
                    "verified": True,
                    "backend": "freedesktop-notifications-dbus",
                }
            )
        if operation == "lock":
            session_id = os.environ.get("XDG_SESSION_ID", "").strip()
            if not session_id or not shutil.which("loginctl"):
                return ToolResult.failure(
                    "Session lock unavailable: XDG_SESSION_ID/loginctl missing."
                )
            try:
                completed = self._runner(["loginctl", "lock-session", session_id], timeout=10)
                if completed.returncode != 0:
                    return ToolResult.failure(
                        "Session lock failed.", meta={"stderr": completed.stderr[:8000]}
                    )
                status = self._runner(
                    ["loginctl", "show-session", session_id, "--property=LockedHint", "--value"],
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                return ToolResult.failure(f"Session lock failed: {exc}")
            verified = status.returncode == 0 and status.stdout.strip().lower() == "yes"
            if not verified:
                return ToolResult.failure("Session lock could not be verified.")
            return ToolResult.success(
                {
                    "output": "Desktop session locked and verified.",
                    "operation": "lock",
                    "verified": True,
                    "session_id": session_id,
                }
            )
        return ToolResult.failure("operation must be capabilities|notify|lock")


def _run_command(
    argv: Sequence[str],
    *,
    timeout: int,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=dict(env) if env is not None else None,
        input=input_text,
    )


def _operation(request: ToolRequest, *, default: str = "") -> str:
    raw = request.args.get("operation")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else default


def _required_string(request: ToolRequest, key: str) -> str:
    value = _optional_string(request, key)
    if value is None:
        raise ValueError(f"{key} is required")
    return value


def _optional_string(request: ToolRequest, key: str) -> str | None:
    raw = request.args.get(key)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _required_pid(request: ToolRequest) -> int:
    raw = request.args.get("pid")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise ValueError("positive integer pid is required")
    return raw


def _required_float(request: ToolRequest, key: str) -> float:
    raw = request.args.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(raw)


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _safe_int(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def _os_info() -> dict[str, JSONValue]:
    release: dict[str, str] = {}
    os_release = Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                release[key] = value.strip().strip('"')
    return {
        "name": release.get("PRETTY_NAME", platform.platform()),
        "id": release.get("ID", ""),
        "version_id": release.get("VERSION_ID", ""),
        "kernel": platform.release(),
        "architecture": platform.machine(),
    }


def _cpu_info() -> dict[str, JSONValue]:
    frequency = psutil.cpu_freq()
    return {
        "physical_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cores": psutil.cpu_count(logical=True) or 0,
        "percent": psutil.cpu_percent(interval=0.05),
        "frequency_mhz": round(frequency.current, 1) if frequency is not None else None,
        "model": platform.processor(),
    }


def _memory_info() -> dict[str, JSONValue]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_bytes": memory.total,
        "available_bytes": memory.available,
        "used_bytes": memory.used,
        "percent": memory.percent,
        "swap_total_bytes": swap.total,
        "swap_used_bytes": swap.used,
    }


def _disk_items() -> list[dict[str, JSONValue]]:
    items: list[dict[str, JSONValue]] = []
    for partition in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(partition.mountpoint)
        except (OSError, PermissionError):
            continue
        items.append(
            {
                "device": partition.device,
                "mountpoint": partition.mountpoint,
                "filesystem": partition.fstype,
                "options": partition.opts,
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
                "percent": usage.percent,
            }
        )
    return items


def _network_items() -> list[dict[str, JSONValue]]:
    stats = psutil.net_if_stats()
    items: list[dict[str, JSONValue]] = []
    for name, addresses in psutil.net_if_addrs().items():
        interface_stats = stats.get(name)
        items.append(
            {
                "name": name,
                "up": interface_stats.isup if interface_stats is not None else False,
                "speed_mbps": interface_stats.speed if interface_stats is not None else 0,
                "addresses": [
                    {
                        "family": str(address.family),
                        "address": address.address,
                        "netmask": address.netmask,
                    }
                    for address in addresses
                ],
            }
        )
    return items


def _process_items(*, query: str | None) -> list[dict[str, JSONValue]]:
    needle = query.casefold() if query is not None else None
    items: list[dict[str, JSONValue]] = []
    for process in psutil.process_iter(("pid", "name", "username", "status", "create_time")):
        try:
            info = process.info
            name = str(info.get("name") or "")
            if (
                needle is not None
                and needle not in name.casefold()
                and needle not in str(info.get("pid", ""))
            ):
                continue
            items.append(
                {
                    "pid": int(info["pid"]),
                    "name": name,
                    "username": str(info.get("username") or ""),
                    "status": str(info.get("status") or ""),
                    "create_time": float(info.get("create_time") or 0.0),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, KeyError, TypeError, ValueError):
            continue
    return sorted(items, key=lambda item: cast(int, item["pid"]))


def _process_identity(pid: int) -> dict[str, JSONValue] | None:
    try:
        process = psutil.Process(pid)
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return None
        return {
            "pid": pid,
            "create_time": process.create_time(),
            "running": True,
            "status": process.status(),
            "name": process.name(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def _inspect_process(pid: int) -> dict[str, JSONValue]:
    process = psutil.Process(pid)
    identity = _process_identity(pid)
    if identity is None:
        return {"pid": pid, "running": False}
    details: dict[str, JSONValue] = dict(identity)
    for key, getter in (
        ("exe", process.exe),
        ("cwd", process.cwd),
        ("username", process.username),
    ):
        try:
            details[key] = getter()
        except (psutil.AccessDenied, psutil.NoSuchProcess, OSError):
            details[key] = None
    try:
        details["cmdline"] = process.cmdline()[:64]
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        details["cmdline"] = []
    try:
        details["memory_rss_bytes"] = process.memory_info().rss
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        details["memory_rss_bytes"] = None
    return details


def _process_ancestor_pids() -> set[int]:
    protected = {os.getpid()}
    try:
        process = psutil.Process(os.getpid())
        protected.update(parent.pid for parent in process.parents())
    except psutil.Error:
        pass
    return protected


def _session_info() -> dict[str, JSONValue]:
    keys = (
        "XDG_SESSION_TYPE",
        "XDG_SESSION_DESKTOP",
        "XDG_CURRENT_DESKTOP",
        "XDG_SESSION_ID",
        "DISPLAY",
        "WAYLAND_DISPLAY",
        "LANG",
    )
    return {
        "environment": {key: os.environ[key] for key in keys if key in os.environ},
        "uid": os.getuid(),
        "gid": os.getgid(),
        "username": pwd.getpwuid(os.getuid()).pw_name,
        "dbus_session_available": bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")),
    }


def _paginated_result(
    label: str,
    items: list[dict[str, JSONValue]],
    *,
    offset: int,
    limit: int,
) -> ToolResult:
    page = items[offset : offset + limit]
    next_offset = offset + len(page)
    truncated = next_offset < len(items)
    return ToolResult.success(
        {
            "output": f"Returned {len(page)} of {len(items)} {label} item(s).",
            "items": page,
            "total": len(items),
            "offset": offset,
            "limit": limit,
            "truncated": truncated,
            "next_offset": next_offset if truncated else None,
        }
    )


def _parse_key_values(output: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            result[key] = value
    return result


def _systemd_expected(operation: str, data: dict[str, JSONValue]) -> bool:
    active = data.get("active_state")
    enabled = data.get("unit_file_state")
    if operation in {"start", "restart"}:
        return active == "active"
    if operation == "stop":
        return active in {"inactive", "failed"}
    if operation == "enable":
        return enabled in {"enabled", "enabled-runtime", "linked", "linked-runtime"}
    if operation == "disable":
        return enabled in {"disabled", "indirect", "static", "masked"}
    return False


def _parse_apt_policy(output: str) -> dict[str, str]:
    values = {"installed": "", "candidate": ""}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Installed:"):
            values["installed"] = stripped.partition(":")[2].strip()
        elif stripped.startswith("Candidate:"):
            values["candidate"] = stripped.partition(":")[2].strip()
    return values


def _apt_metadata_state() -> dict[str, JSONValue]:
    root = Path("/var/lib/apt/lists")
    try:
        files = [path for path in root.iterdir() if path.is_file() and path.name != "lock"]
    except OSError:
        files = []
    newest = max((path.stat().st_mtime for path in files), default=0.0)
    return {"path": str(root), "file_count": len(files), "newest_mtime": newest}


def _desktop_capabilities() -> dict[str, JSONValue]:
    session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    clipboard_backend = detect_clipboard_backend()
    return {
        "session_type": session_type,
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", ""),
        "display_available": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "dbus_session_available": bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS")),
        "notification_available": bool(
            os.environ.get("DBUS_SESSION_BUS_ADDRESS") and shutil.which("gdbus")
        ),
        "clipboard_backend": clipboard_backend.name if clipboard_backend is not None else None,
    }
