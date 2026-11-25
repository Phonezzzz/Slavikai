"""Read-only host diagnostics (E03-T07, ADR-0013 §6).

Four capabilities — ``host.diag.service_status`` / ``process_list`` /
``disk`` / ``host`` — answering "what is the state of the bridge and
the host" as structured JSON from the daemon, never by parsing
``systemctl`` / ``ps`` / ``df`` output in a shell.

Read-only by construction (schema §4.6): no file is written, no run
dir / scope / config is created, the process registry is only read
(through the read-only snapshot API of ``HostProcessService``), and
the only subprocess is ``systemctl --user show`` — a read-only state
query with a hard timeout. Nothing here can create, kill, restart or
change anything on the host; service control is forbidden by
ADR-0013 §10 and ``service.restart`` does not exist in the registry.

Degradation instead of errors (schema §4.9): an unavailable source
degrades its field to ``null`` while the request still answers
``ok: true``. Diagnostic sources are deadline-bounded (owner
clarification 7): ``systemctl`` gets a subprocess timeout and
``disk_usage`` runs under a shared budget, so a hung source can
never push the handler past the per-request deadline — the daemon's
deadline backstop (``request_timeout``) is the last resort, not the
plan.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final

from host_bridge.fs import (
    CancelToken,
    FsError,
    HostFsService,
    Root,
    _check_params,
    _str_param,
)
from host_bridge.protocol import (
    BRIDGE_VERSION,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    encode_success,
)

if TYPE_CHECKING:  # typing only; avoids a runtime import cycle (diag <- daemon)
    from host_bridge.process import HostProcessService

# The systemd unit this daemon runs under (deploy file
# ``slavikai-host-bridge.service.example``). The unit block of
# ``service_status`` is reported only when ``systemctl show`` proves
# the unit's MainPID is this daemon instance (owner clarification 1):
# the state of another instance is never reported as our own.
_UNIT_NAME: Final[str] = "slavikai-host-bridge.service"
_SERVICE_NAME: Final[str] = "slavikai-host-bridge"

# Hard timeout for the ``systemctl --user show`` query (schema §4.1.1:
# 2 s). The effective timeout is min(remaining request deadline,
# this value), so the query can never push the handler past the
# daemon's per-request deadline.
_SYSTEMCTL_TIMEOUT_S: Final[float] = 2.0

# Shared budget for all ``disk_usage`` calls of one ``disk`` request
# (owner clarification 7): ``statvfs`` is not assumed instant, so
# every root is probed concurrently under one deadline and the
# unfinished ones degrade to nulls instead of blocking the handler.
_DISK_BUDGET_S: Final[float] = 2.0

# Longest run_id the envelope accepts (protocol.MAX_ATTRIBUTION_CHARS);
# used by the frame-budget test, not enforced here (the envelope
# already enforces it).
_MAX_RUN_ID_CHARS: Final[int] = 256


class DiagError(BridgeProtocolError):
    """Typed diagnostics failure; ``code`` is the wire error code."""

    def __init__(self, code: str, message: str, *, request_id: str | None = None) -> None:
        super().__init__(code, message, request_id=request_id)


def _invalid_params(message: str) -> DiagError:
    return DiagError("invalid_params", message)


def _run_systemctl_show(timeout_s: float) -> str | None:
    """Raw ``systemctl --user show`` output, or None on any failure.

    ``None`` covers: no systemctl binary, bus unreachable, non-zero
    exit, timeout. Callers degrade to ``unit: null``; they never
    distinguish the reasons (the unit state is unknown either way).
    """
    try:
        out = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                _UNIT_NAME,
                "--property=MainPID,ActiveState,SubState,NRestarts",
            ],
            capture_output=True,
            timeout=timeout_s,
            text=True,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _parse_systemctl_show(text: str) -> dict[str, str] | None:
    """Parse ``systemctl show`` ``KEY=value`` output; None when the
    shape is unknown (defensive: an unparseable unit is not our
    unit — it degrades to ``unit: null``, never to a guess)."""
    props: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            return None
        key, _, value = line.partition("=")
        props[key.strip()] = value.strip()
    if set(props) != {"MainPID", "ActiveState", "SubState", "NRestarts"}:
        return None
    if not props["ActiveState"] or not props["SubState"]:
        return None
    try:
        int(props["MainPID"])
        int(props["NRestarts"])
    except ValueError:
        return None
    return props


def _read_meminfo() -> dict[str, int] | None:
    """``/proc/meminfo`` bytes, or None when unreadable/unparseable.

    Only ``MemTotal`` / ``MemAvailable`` are used; a missing key
    degrades the whole ``memory`` block to null (schema §4.9).
    """
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            try:
                values[parts[0].rstrip(":")] = int(parts[1]) * 1024
            except ValueError:
                return None
    if "MemTotal" not in values or "MemAvailable" not in values:
        return None
    return values


def _read_uptime_s() -> float | None:
    """First field of ``/proc/uptime`` (seconds), or None on failure."""
    try:
        text = Path("/proc/uptime").read_text(encoding="utf-8")
        return float(text.split()[0])
    except (OSError, UnicodeDecodeError, ValueError, IndexError):
        return None


def _get_loadavg() -> list[float] | None:
    """``os.getloadavg()`` as a list, or None where unavailable."""
    try:
        return [float(v) for v in os.getloadavg()]
    except (OSError, AttributeError):
        return None


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat()


class HostDiagService:
    """Read-only diagnostics for the Owner Host Bridge (E03-T07).

    Built by the daemon from its own services; ``daemon_pid`` and the
    start timestamps are the daemon's own facts. ``systemctl_fn`` and
    ``disk_usage_fn`` are injectable seams for tests (same pattern as
    ``peer_cred_fn`` / ``dispatch_fn`` in the daemon): the defaults
    call the real system.
    """

    def __init__(
        self,
        *,
        fs_service: HostFsService,
        process_service: HostProcessService,
        daemon_pid: int,
        daemon_started_at: str | None = None,
        daemon_started_mono: float | None = None,
        systemctl_fn: Callable[[float], str | None] | None = None,
        disk_usage_fn: Callable[[Path], tuple[int, int, int]] | None = None,
    ) -> None:
        self._fs_service = fs_service
        self._process_service = process_service
        self._daemon_pid = daemon_pid
        # Construction-time fallback: the daemon overwrites both via
        # note_started() in start(), so service_status always has
        # values even when the service is used directly in tests.
        now_mono = time.monotonic()
        self._daemon_started_at = daemon_started_at if daemon_started_at is not None else _iso_now()
        self._daemon_started_mono = (
            daemon_started_mono if daemon_started_mono is not None else now_mono
        )
        self._systemctl_fn = systemctl_fn if systemctl_fn is not None else _run_systemctl_show
        self._disk_usage_fn = disk_usage_fn if disk_usage_fn is not None else shutil.disk_usage

    def note_started(self) -> None:
        """Record the daemon's start time (called by ``start()``)."""
        self._daemon_started_at = _iso_now()
        self._daemon_started_mono = time.monotonic()

    # -- dispatch ------------------------------------------------------------

    def handle(
        self,
        capability: str,
        params: dict[str, object],
        request_id: str,
        token: CancelToken,
        *,
        principal_id: str | None = None,
    ) -> dict[str, object]:
        """Dispatch one diag capability (same shape as the FS/process
        ``handle`` methods: params are validated here, ``FsError``
        from the shared validators is converted to ``DiagError`` so
        the client sees the typed code with the request echoed).
        ``principal_id`` comes from the request envelope (E03-T06
        ownership key); it is never read from params."""
        try:
            if capability == "host.diag.service_status":
                _check_params(params, frozenset())
                return self.service_status(token=token)
            if capability == "host.diag.process_list":
                _check_params(params, frozenset({"run_id"}))
                run_id = _str_param(params, "run_id", required=False)
                if run_id is not None and not 1 <= len(run_id) <= _MAX_RUN_ID_CHARS:
                    raise _invalid_params(
                        f"params.run_id must be a string of 1..{_MAX_RUN_ID_CHARS} characters"
                    )
                return self.process_list(
                    run_id=run_id,
                    principal_id=principal_id,
                    request_id=request_id,
                    token=token,
                )
            if capability == "host.diag.disk":
                _check_params(params, frozenset({"root_id"}))
                root_id = _str_param(params, "root_id", required=False)
                return self.disk(root_id=root_id, token=token)
            if capability == "host.diag.host":
                _check_params(params, frozenset())
                return self.host()
        except FsError as exc:
            # The shared validators raise FsError, not DiagError:
            # convert and keep the correlation (same pattern as
            # HostProcessService.handle).
            raise DiagError(exc.code, str(exc), request_id=request_id) from exc
        except DiagError as exc:
            if exc.request_id is None:
                exc.request_id = request_id
            raise
        raise DiagError(
            "invalid_params", f"unknown diag capability {capability}", request_id=request_id
        )

    # -- service_status ------------------------------------------------------

    def service_status(self, *, token: CancelToken | None = None) -> dict[str, object]:
        """Daemon facts + systemd unit state (owner clarification 1:
        the unit block is reported only when ``systemctl show``
        proves the unit's MainPID is this daemon instance; any
        other outcome — absent systemd, timeout, non-zero exit,
        unparseable output, a MainPID that is not ours — degrades
        to ``unit: null`` and the request still answers ok)."""
        remaining = token.remaining_s if token is not None else _SYSTEMCTL_TIMEOUT_S
        timeout_s = max(0.05, min(remaining, _SYSTEMCTL_TIMEOUT_S))
        raw = self._systemctl_fn(timeout_s)
        unit: dict[str, object] | None = None
        props = _parse_systemctl_show(raw) if raw is not None else None
        if props is not None and int(props["MainPID"]) == self._daemon_pid:
            unit = {
                "name": _UNIT_NAME,
                "active_state": props["ActiveState"],
                "sub_state": props["SubState"],
                "restarts": int(props["NRestarts"]),
            }
        return {
            "service": _SERVICE_NAME,
            "daemon": {
                "pid": self._daemon_pid,
                "started_at": self._daemon_started_at,
                "uptime_s": time.monotonic() - self._daemon_started_mono,
                "protocol": PROTOCOL_VERSION,
                "bridge": BRIDGE_VERSION,
            },
            "process_available": self._process_service.is_available(),
            "unit": unit,
        }

    # -- process_list --------------------------------------------------------

    def process_list(
        self,
        *,
        run_id: str | None = None,
        principal_id: str | None = None,
        request_id: str = "",
        token: CancelToken | None = None,
    ) -> dict[str, object]:
        """Processes of this daemon from the read-only snapshot API
        of ``HostProcessService`` (owner clarification 2: diag never
        touches the private ``_records`` / ``_lock``; there is no
        second registry).

        Ownership (E03-T06 key): with ``run_id`` given, only the
        records bound to the envelope's ``principal_id`` and that
        ``run_id`` are returned; an unknown run is a successful
        empty answer. Without ``run_id`` — all records. Daemon-side
        this is the full registry: in v1 every run is the owner's,
        and scoping for untrusted contexts is a T11 adoption
        obligation — the server-side ownership policy applies to
        ``process_list`` there (see ROADMAP.md §E03-T11, owner
        clarification 3). No argv/env/cwd/stdout/scope names ever
        leave the daemon.

        Frame budget (owner clarification 5): entries are dropped
        oldest-first until the serialized frame fits (the codebase
        pattern from ``host.fs.list``), ``count`` keeps the total
        number of matching records and ``truncated`` marks the
        trimming — the client is never handed a partial list it
        mistakes for complete, and the daemon backstop
        (``internal_error`` on an over-limit frame) never fires for
        diag.
        """
        # The snapshot API takes the envelope principal from the
        # caller (handle() passes request.principal_id through).
        entries = self._process_service.list_processes(
            run_id=run_id,
            principal_id=principal_id,
        )
        result: dict[str, object] = {
            "processes": entries,
            "count": len(entries),
            "truncated": False,
        }
        while entries and not self._frame_fits(result, request_id):
            entries.pop(0)
            result["truncated"] = True
        return result

    # -- disk ----------------------------------------------------------------

    def disk(
        self, *, root_id: str | None = None, token: CancelToken | None = None
    ) -> dict[str, object]:
        """Per-root disk usage (owner clarification 4: no
        ``total == used + free`` requirement — reserved blocks make
        the equality false on real filesystems; the numbers are
        reported as the OS gives them).

        Partial degradation: a root whose usage cannot be read
        (missing path, I/O error, or the deadline budget spent)
        degrades to null numbers while the other roots still answer
        and the request stays ok. ``statvfs`` is not assumed
        instant (owner clarification 7): every root is probed
        concurrently under one shared budget of
        ``min(remaining request deadline, 2 s)``.
        """
        registry = self._fs_service.registry
        if root_id is not None:
            root = registry.by_id(root_id)
            if root is None:
                raise DiagError("invalid_params", f"unknown root_id: {root_id!r}")
            roots = [root]
        else:
            roots = list(registry.roots)
        remaining = token.remaining_s if token is not None else _DISK_BUDGET_S
        budget_s = max(0.05, min(remaining, _DISK_BUDGET_S))
        usage = self._disk_usage_all(roots, budget_s)
        disks: list[dict[str, object]] = []
        for root in roots:
            numbers = usage.get(root.id)
            if numbers is None:
                disks.append(
                    {
                        "root_id": root.id,
                        "total_bytes": None,
                        "used_bytes": None,
                        "free_bytes": None,
                    }
                )
            else:
                total, used, free = numbers
                disks.append(
                    {
                        "root_id": root.id,
                        "total_bytes": total,
                        "used_bytes": used,
                        "free_bytes": free,
                    }
                )
        return {"disks": disks}

    def _disk_usage_all(
        self, roots: list[Root], budget_s: float
    ) -> dict[str, tuple[int, int, int] | None]:
        """Probe every root concurrently under one shared deadline.

        Returns ``{root_id: (total, used, free) | None}``; ``None``
        means the probe failed or the budget ran out before it
        finished. A worker thread stuck in ``statvfs`` past the
        budget is abandoned (daemon thread, read-only syscall —
        harmless); the handler never waits for it.
        """
        results: dict[str, tuple[int, int, int] | None] = {}

        def work(root: Root) -> None:
            try:
                total, used, free = self._disk_usage_fn(root.path)
            except OSError:
                results[root.id] = None
            else:
                results[root.id] = (total, used, free)

        threads = [threading.Thread(target=work, args=(root,), daemon=True) for root in roots]
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + budget_s
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        for root in roots:
            results.setdefault(root.id, None)
        return results

    # -- host ----------------------------------------------------------------

    def host(self) -> dict[str, object]:
        """Host facts from kernel-provided values (uname, loadavg,
        /proc pseudo-files): these are syscalls and pseudo-file
        reads, not disk I/O, so no timeout applies to them; each
        still degrades to null on failure instead of failing the
        request."""
        uname = os.uname()
        meminfo = _read_meminfo()
        uptime_s = _read_uptime_s()
        return {
            "hostname": socket.gethostname(),
            "os": {
                "sysname": uname.sysname,
                "release": uname.release,
                "machine": uname.machine,
            },
            "cpu_count": os.cpu_count(),
            "load_average": _get_loadavg(),
            "memory": (
                {"total_bytes": meminfo["MemTotal"], "available_bytes": meminfo["MemAvailable"]}
                if meminfo is not None
                else None
            ),
            "uptime_s": uptime_s,
        }

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _frame_fits(result: dict[str, object], request_id: str) -> bool:
        """Serialized frame (escaping counts) within the 64 KiB limit."""
        return len(encode_success(result, request_id)) - 1 <= MAX_FRAME_BYTES
