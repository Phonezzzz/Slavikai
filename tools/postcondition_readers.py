"""Concrete :class:`PostconditionReader` implementations (E05-T01).

- :class:`DesktopPostconditionReader`: reads desktop state through
  the desktop security layer (path canonicalization first), psutil
  (pid liveness + create_time) and ``systemctl show`` (unit state).
  The file read cap defaults to the ``DesktopVerifyTool`` limit; the
  tool passes it explicitly so the constant stays in one place.
- :class:`BridgePostconditionReader`: bridge-backed process checks
  (test-only until T11 adoption — there is no ``host.process.poll``
  client helper yet, so the poll is an injected callable). File,
  service and canonical-check reads delegate to a wrapped desktop
  reader.

Neither reader trusts a tool's ``ok`` flag: every answer comes from
a fresh read of real state.
"""

from __future__ import annotations

from collections.abc import Callable

import psutil

from core.desktop_security import DesktopPathSecurity
from core.mwv.models import VerificationResult
from core.mwv.verifier import sandbox_verification_unavailable
from core.mwv.verifier_runtime import CANONICAL_CHECK_COMMAND
from core.postconditions import FileInfo, FileTooLargeError, PathDeniedError
from shared.models import JSONValue
from tools.desktop_system_tools import (
    UNIT_RE,
    CommandRunner,
    _parse_key_values,
    _run_command,
    _safe_int,
)

_SYSTEMCTL_PROPERTIES = "Id,LoadState,ActiveState,SubState,UnitFileState,MainPID,Result"


class DesktopPostconditionReader:
    """Desktop-state reader: security-checked files, psutil, systemd."""

    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        systemd_runner: CommandRunner | None = None,
        canonical_check: Callable[[], VerificationResult] | None = None,
        max_read_bytes: int = 1_048_576,
    ) -> None:
        self._security = security
        self._systemd_runner = systemd_runner if systemd_runner is not None else _run_command
        self._canonical_check = canonical_check
        self._max_read_bytes = max_read_bytes

    def describe_file(self, path: str) -> FileInfo:
        try:
            canonical = self._security.require_not_denied(path).canonical
        except (PermissionError, ValueError) as exc:
            raise PathDeniedError(str(exc)) from exc
        exists = canonical.exists()
        if not exists:
            return FileInfo(exists=False, is_regular_file=False)
        return FileInfo(exists=True, is_regular_file=canonical.is_file())

    def read_file(self, path: str) -> bytes | None:
        try:
            canonical = self._security.require_not_denied(path).canonical
        except (PermissionError, ValueError) as exc:
            raise PathDeniedError(str(exc)) from exc
        try:
            size = canonical.stat().st_size
        except FileNotFoundError:
            return None
        if size > self._max_read_bytes:
            raise FileTooLargeError(f"file exceeds the read limit of {self._max_read_bytes} bytes")
        try:
            return canonical.read_bytes()
        except FileNotFoundError:
            return None

    def process_status(self, process_id: str) -> dict[str, JSONValue] | None:
        # The desktop reader has no bridge/registry access: there is no
        # record to report. Bridge process checks use
        # BridgePostconditionReader (test-only until T11 adoption).
        del process_id
        return None

    def os_process_running(self, pid: int) -> tuple[bool, float | None]:
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return (False, None)
            return (True, process.create_time())
        except psutil.NoSuchProcess:
            return (False, None)
        except psutil.AccessDenied as exc:
            raise PermissionError(str(exc)) from exc

    def service_status(self, unit: str, scope: str) -> dict[str, JSONValue] | None:
        if scope not in ("user", "system"):
            raise ValueError(f"unknown systemd scope: {scope!r}")
        if not UNIT_RE.match(unit):
            raise ValueError(f"invalid systemd unit name: {unit!r}")
        argv: list[str] = ["systemctl"]
        if scope == "user":
            argv.append("--user")
        argv += [
            "show",
            "--no-pager",
            f"--property={_SYSTEMCTL_PROPERTIES}",
            "--",
            unit,
        ]
        # Same invocation shape as DesktopSystemdTool._status (same
        # runner, same parser); the reader owns the not-found vs
        # error distinction _status collapses into one failure.
        completed = self._systemd_runner(argv, timeout=20)
        if completed.returncode != 0:
            raise OSError(f"systemctl show failed for {unit}: exit_code={completed.returncode}")
        values = _parse_key_values(completed.stdout)
        if values.get("LoadState") == "not-found":
            return None
        return {
            "unit": values.get("Id", unit),
            "load_state": values.get("LoadState", ""),
            "active_state": values.get("ActiveState", ""),
            "sub_state": values.get("SubState", ""),
            "unit_file_state": values.get("UnitFileState", ""),
            "main_pid": _safe_int(values.get("MainPID")),
            "result": values.get("Result", ""),
            "scope": scope,
        }

    def canonical_check(self) -> VerificationResult:
        if self._canonical_check is not None:
            return self._canonical_check()
        return sandbox_verification_unavailable(list(CANONICAL_CHECK_COMMAND))


class BridgePostconditionReader:
    """Bridge-backed reader for process checks (test-only until T11).

    ``poll`` answers ``process_status`` from the host bridge
    (``host.process.poll`` record shape: ``{"state": ...,
    "exit_code": ...}``). Every other read delegates to the wrapped
    desktop reader, so the bridge reader never bypasses the desktop
    security layer for files.
    """

    def __init__(
        self,
        poll: Callable[[str], dict[str, JSONValue] | None],
        desktop: DesktopPostconditionReader,
    ) -> None:
        self._poll = poll
        self._desktop = desktop

    def describe_file(self, path: str) -> FileInfo:
        return self._desktop.describe_file(path)

    def read_file(self, path: str) -> bytes | None:
        return self._desktop.read_file(path)

    def process_status(self, process_id: str) -> dict[str, JSONValue] | None:
        result = self._poll(process_id)
        if result is None:
            return None
        if not isinstance(result, dict):
            raise ValueError("bridge poll must return a dict or None")
        return result

    def os_process_running(self, pid: int) -> tuple[bool, float | None]:
        return self._desktop.os_process_running(pid)

    def service_status(self, unit: str, scope: str) -> dict[str, JSONValue] | None:
        return self._desktop.service_status(unit, scope)

    def canonical_check(self) -> VerificationResult:
        return self._desktop.canonical_check()


__all__ = ["BridgePostconditionReader", "DesktopPostconditionReader"]
