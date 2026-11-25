"""Read-only host diagnostics (E03-T07, ADR-0013 §6).

Four capabilities over the daemon wire — ``host.diag.service_status``
/ ``process_list`` / ``disk`` / ``host`` — answering "what is the
state of the bridge and the host" as structured JSON. Read-only by
construction: the tests assert absence of mutation (no files created,
no registry records changed, no process spawned) rather than
identical dynamic host metrics — the metrics move, the state must
not (owner clarification 6).

Service-level tests drive ``HostDiagService.handle`` directly with a
fake ``systemctl`` seam and seeded process records (no systemd, no
enforcement); wire tests run a real daemon socket in a thread and
call it through ``host_bridge.client``.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from host_bridge import client
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.diag import DiagError, HostDiagService
from host_bridge.fs import CancelToken, HostFsService, Root, RootsRegistry
from host_bridge.process import HostProcessService, _Record
from host_bridge.protocol import (
    BRIDGE_VERSION,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    encode_success,
)

_OWNER = "owner"
_OTHER = "someone-else"


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    fake_home = tmp_path / "home"
    work = fake_home / "Project" / "work"
    work.mkdir(parents=True)
    (fake_home / "Documents" / "docs").mkdir(parents=True)
    (work / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    return fake_home


@pytest.fixture()
def registry(home: Path) -> RootsRegistry:
    return RootsRegistry(
        roots=(
            Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=home / ".config" / "slavikai" / "host-bridge.json",
        socket_dir=None,
    )


@pytest.fixture()
def fs_service(registry: RootsRegistry, tmp_path: Path) -> HostFsService:
    return HostFsService(registry, trash_root=tmp_path / "Trash")


@pytest.fixture()
def proc_service(
    fs_service: HostFsService, registry: RootsRegistry, tmp_path: Path
) -> HostProcessService:
    """A process service with no profiles: ``is_available()`` is
    False and the registry starts empty; tests seed ``_records``
    directly (the run -> record path is covered by E03-T06)."""
    return HostProcessService(fs_service, registry, runtime_dir=tmp_path / "run", profiles=())


@pytest.fixture()
def diag(fs_service: HostFsService, proc_service: HostProcessService) -> HostDiagService:
    """Diag service with a systemctl seam that reports no unit:
    unit behaviour is covered by tests that inject their own seam."""
    return HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        systemctl_fn=lambda timeout_s: None,
    )


@pytest.fixture()
def bridge(
    fs_service: HostFsService,
    proc_service: HostProcessService,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, HostProcessService]]:
    """A real daemon on a real socket in a thread.

    The daemon's startup orphan sweep needs a working user systemd
    bus: link the real manager sockets into this test's runtime dir
    (same pattern as the live crash test in
    test_host_bridge_process.py) instead of overriding them away.
    Without these, systemctl --user fails and the sweep refuses to
    start the daemon. Skipped where no user manager exists.
    """
    runtime_dir = tmp_path / "run"
    real_runtime = Path(f"/run/user/{os.getuid()}")
    if not (real_runtime / "systemd" / "private").exists():
        pytest.skip("no user systemd manager for the wire daemon to sweep with")
    (runtime_dir / "systemd").symlink_to(real_runtime / "systemd", target_is_directory=True)
    (runtime_dir / "bus").symlink_to(real_runtime / "bus")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=fs_service, process_service=proc_service)
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        yield socket_path, proc_service
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _token(deadline_s: float = 5.0) -> CancelToken:
    return CancelToken(deadline=time.monotonic() + deadline_s, request_id="test-req")


def _call(
    diag: HostDiagService, capability: str, params: dict[str, object] | None = None
) -> dict[str, object]:
    return diag.handle(capability, params or {}, "test-req", _token())


def _seed(
    svc: HostProcessService,
    *,
    count: int = 1,
    run_id: str | None = "run-1",
    principal_id: str | None = _OWNER,
    profile_id: str = "true",
    state: str = "running",
) -> list[str]:
    """Seed ``count`` records with increasing started_at; returns
    their process ids in seed order (oldest first)."""
    base = datetime.now(UTC)
    ids: list[str] = []
    for i in range(count):
        # Unique ids across _seed calls: repeated seeds must not
        # overwrite each other's records (the registry is keyed by
        # process_id).
        process_id = uuid.uuid4().hex
        # Distinct started_at keeps the snapshot order deterministic.
        started_at = base.replace(microsecond=i).isoformat()
        record = _Record(
            process_id=process_id,
            profile_id=profile_id,
            state=state,
            run_id=run_id,
            principal_id=principal_id,
            started_at=started_at,
            finished_at=None,
            exit_code=None,
            exit_signal=None,
            started_mono=time.monotonic(),
        )
        svc._records[process_id] = record
        ids.append(process_id)
    return ids


def _snapshot_tree(home: Path) -> dict[str, bytes]:
    """File bytes under home (read-only check input)."""
    return {
        str(path.relative_to(home)): path.read_bytes()
        for path in sorted(home.rglob("*"))
        if path.is_file()
    }


# --- service_status ----------------------------------------------------------


def test_service_status_reports_daemon_facts(diag: HostDiagService) -> None:
    result = _call(diag, "host.diag.service_status")
    daemon = result["daemon"]
    assert isinstance(daemon, dict)
    assert daemon["pid"] == os.getpid()
    assert daemon["protocol"] == PROTOCOL_VERSION
    assert daemon["bridge"] == BRIDGE_VERSION
    assert isinstance(daemon["started_at"], str) and daemon["started_at"]
    assert isinstance(daemon["uptime_s"], float) and daemon["uptime_s"] >= 0.0
    assert result["service"] == "slavikai-host-bridge"
    # No profiles in this fixture: the process subsystem is down.
    assert result["process_available"] is False
    # No systemctl seam output: the unit degrades to null, the
    # request still answers ok.
    assert result["unit"] is None


def test_service_status_unit_reported_for_own_instance(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """Owner clarification 1: the unit block appears only when
    systemctl proves the unit's MainPID is this daemon instance."""
    pid = os.getpid()
    output = f"MainPID={pid}\nActiveState=active\nSubState=running\nNRestarts=2\n"
    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=pid,
        systemctl_fn=lambda timeout_s: output,
    )
    result = svc.handle("host.diag.service_status", {}, "test-req", _token())
    assert result["unit"] == {
        "name": "slavikai-host-bridge.service",
        "active_state": "active",
        "sub_state": "running",
        "restarts": 2,
    }


def test_service_status_unit_other_instance_degrades_to_null(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """Owner clarification 1: another instance's state is never
    reported as our own — MainPID != daemon pid degrades to null."""
    output = "MainPID=999999\nActiveState=active\nSubState=running\nNRestarts=0\n"
    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        systemctl_fn=lambda timeout_s: output,
    )
    result = svc.handle("host.diag.service_status", {}, "test-req", _token())
    assert result["unit"] is None


def test_service_status_unit_unparseable_output_degrades_to_null(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """A unit we cannot parse is not our unit: degrade, never guess."""
    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        systemctl_fn=lambda timeout_s: "garbage without equals signs\n",
    )
    result = svc.handle("host.diag.service_status", {}, "test-req", _token())
    assert result["unit"] is None


def test_service_status_real_systemctl_never_reports_foreign_unit(
    diag: HostDiagService,
) -> None:
    """With the real systemctl query (no seam): the unit is null
    unless the test process itself is the unit's MainPID — it never
    is, so the answer is null in every environment, with or without
    a user systemd manager."""
    from host_bridge.diag import _run_systemctl_show

    real = HostDiagService(
        fs_service=diag._fs_service,
        process_service=diag._process_service,
        daemon_pid=os.getpid(),
        systemctl_fn=_run_systemctl_show,
    )
    result = real.handle("host.diag.service_status", {}, "test-req", _token())
    assert result["unit"] is None
    assert result["daemon"]["pid"] == os.getpid()


def test_service_status_deadline_bounds_systemctl(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """Owner clarification 7: the systemctl seam receives
    min(remaining deadline, 2 s) — a query that honours the passed
    timeout can never push the handler past the request deadline."""
    seen: list[float] = []

    def slow_systemctl(timeout_s: float) -> None:
        seen.append(timeout_s)
        time.sleep(timeout_s)
        return None

    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        systemctl_fn=slow_systemctl,
    )
    started = time.monotonic()
    result = svc.handle("host.diag.service_status", {}, "test-req", _token(5.0))
    elapsed = time.monotonic() - started
    assert seen == [pytest.approx(2.0)]
    assert elapsed < 3.0
    assert result["unit"] is None


# --- process_list ------------------------------------------------------------


def test_process_list_returns_seeded_records(
    diag: HostDiagService, proc_service: HostProcessService
) -> None:
    ids = _seed(proc_service, count=2, state="exited")
    result = _call(diag, "host.diag.process_list")
    assert result["count"] == 2
    assert result["truncated"] is False
    processes = result["processes"]
    assert isinstance(processes, list)
    assert [p["process_id"] for p in processes] == ids  # started_at order
    entry = processes[0]
    assert entry["process_id"] == ids[0]
    assert entry["profile_id"] == "true"
    assert entry["state"] == "exited"
    assert entry["run_id"] == "run-1"
    assert isinstance(entry["started_at"], str) and entry["started_at"]
    assert entry["finished_at"] is None
    assert entry["exit_code"] is None
    assert entry["signal"] is None
    # No argv/env/cwd/stdout/scope names ever leave the daemon.
    assert set(entry) == {
        "process_id",
        "profile_id",
        "state",
        "run_id",
        "started_at",
        "finished_at",
        "exit_code",
        "signal",
    }


def test_process_list_filters_by_run_and_principal(
    diag: HostDiagService, proc_service: HostProcessService
) -> None:
    _seed(proc_service, count=1, run_id="run-1", principal_id=_OWNER)
    _seed(proc_service, count=1, run_id="run-1", principal_id=_OTHER)
    _seed(proc_service, count=1, run_id="run-2", principal_id=_OWNER)
    # Only the (principal_id, run_id) pair from the envelope matches.
    result = diag.handle(
        "host.diag.process_list",
        {"run_id": "run-1"},
        "test-req",
        _token(),
        principal_id=_OWNER,
    )
    assert result["count"] == 1
    assert result["processes"][0]["run_id"] == "run-1"
    # The same run under another principal is a different binding.
    other = diag.handle(
        "host.diag.process_list",
        {"run_id": "run-1"},
        "test-req",
        _token(),
        principal_id=_OTHER,
    )
    assert other["count"] == 1
    # An unknown run is a successful empty answer, not an error.
    missing = diag.handle(
        "host.diag.process_list",
        {"run_id": "no-such-run"},
        "test-req",
        _token(),
        principal_id=_OWNER,
    )
    assert missing == {"processes": [], "count": 0, "truncated": False}


def test_process_list_without_run_id_returns_all(
    diag: HostDiagService, proc_service: HostProcessService
) -> None:
    _seed(proc_service, count=2, run_id="run-1", principal_id=_OWNER)
    _seed(proc_service, count=1, run_id="run-2", principal_id=_OTHER)
    result = _call(diag, "host.diag.process_list")
    assert result["count"] == 3


def test_process_list_frame_budget_trims_oldest_first(
    diag: HostDiagService, proc_service: HostProcessService
) -> None:
    """Owner clarification 5: worst case — 40 records with a
    256-char multi-byte run_id (JSON-escaped, ~1.5 KiB each) — must
    still fit the 64 KiB frame: oldest entries drop first, ``count``
    keeps the total, ``truncated`` marks the trimming."""
    run_id = "é" * 256
    ids = _seed(proc_service, count=40, run_id=run_id, principal_id=_OWNER)
    result = _call(diag, "host.diag.process_list")
    assert result["count"] == 40
    assert result["truncated"] is True
    assert len(encode_success(result, "test-req")) - 1 <= MAX_FRAME_BYTES
    kept = [p["process_id"] for p in result["processes"]]
    assert kept[0] != ids[0]  # the oldest entry was trimmed
    assert kept[-1] == ids[-1]  # the newest entry survives


# --- disk --------------------------------------------------------------------


def test_disk_reports_per_root_numbers(diag: HostDiagService) -> None:
    result = _call(diag, "host.diag.disk")
    disks = result["disks"]
    assert isinstance(disks, list)
    assert {d["root_id"] for d in disks} == {"work", "docs"}
    for entry in disks:
        assert entry["total_bytes"] is not None and entry["total_bytes"] > 0
        assert entry["used_bytes"] is not None and entry["used_bytes"] >= 0
        assert entry["free_bytes"] is not None and entry["free_bytes"] >= 0
    # Total == used + free is NOT required (reserved blocks):
    # the numbers are reported as the OS gives them.


def test_disk_single_root(diag: HostDiagService) -> None:
    result = _call(diag, "host.diag.disk", {"root_id": "docs"})
    assert [d["root_id"] for d in result["disks"]] == ["docs"]


def test_disk_partial_degradation_per_root(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """Owner clarification 4: an unreadable root degrades to null
    numbers while the other roots still answer and the request
    stays ok."""

    def failing_usage(path: Path) -> tuple[int, int, int]:
        if path.name == "docs":
            raise OSError("cannot stat this root")
        return (1000, 400, 600)

    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        disk_usage_fn=failing_usage,
    )
    result = svc.handle("host.diag.disk", {}, "test-req", _token())
    by_id = {d["root_id"]: d for d in result["disks"]}
    assert by_id["docs"] == {
        "root_id": "docs",
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
    }
    assert by_id["work"] == {
        "root_id": "work",
        "total_bytes": 1000,
        "used_bytes": 400,
        "free_bytes": 600,
    }


def test_disk_deadline_bounds_slow_roots(
    fs_service: HostFsService, proc_service: HostProcessService
) -> None:
    """Owner clarification 7: statvfs is not assumed instant — a
    root that hangs past the shared 2 s budget degrades to null
    instead of blocking the handler past the request deadline."""

    def hanging_usage(path: Path) -> tuple[int, int, int]:
        if path.name == "docs":
            time.sleep(30.0)
        return (1000, 400, 600)

    svc = HostDiagService(
        fs_service=fs_service,
        process_service=proc_service,
        daemon_pid=os.getpid(),
        disk_usage_fn=hanging_usage,
    )
    started = time.monotonic()
    result = svc.handle("host.diag.disk", {}, "test-req", _token(5.0))
    elapsed = time.monotonic() - started
    assert elapsed < 4.0
    by_id = {d["root_id"]: d for d in result["disks"]}
    assert by_id["docs"]["total_bytes"] is None
    assert by_id["work"]["total_bytes"] == 1000


def test_disk_many_roots_fit_frame(tmp_path: Path) -> None:
    """Owner clarification 5: 64 roots — the maximum the config
    allows in practice — still fit one frame with no trimming."""
    home = tmp_path / "home64"
    roots = tuple(
        Root(id=f"r{i:02d}", path=(home / f"r{i:02d}").resolve(), mode="ro") for i in range(64)
    )
    for root in roots:
        root.path.mkdir(parents=True)
    registry = RootsRegistry(
        roots=roots,
        write_excluded=(),
        home=home.resolve(),
        config_path=home / "host-bridge.json",
        socket_dir=None,
    )
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    svc = HostDiagService(
        fs_service=fs,
        process_service=HostProcessService(
            fs,
            registry,
            runtime_dir=tmp_path / "run",
            profiles=(),
        ),
        daemon_pid=os.getpid(),
    )
    result = svc.handle("host.diag.disk", {}, "test-req", _token())
    assert len(result["disks"]) == 64
    assert len(encode_success(result, "test-req")) - 1 <= MAX_FRAME_BYTES


# --- host --------------------------------------------------------------------


def test_host_reports_facts(diag: HostDiagService) -> None:
    result = _call(diag, "host.diag.host")
    assert isinstance(result["hostname"], str) and result["hostname"]
    assert result["os"]["sysname"]
    assert result["os"]["release"]
    assert result["os"]["machine"]
    assert isinstance(result["cpu_count"], int) and result["cpu_count"] >= 1
    load = result["load_average"]
    assert isinstance(load, list) and len(load) == 3
    assert all(isinstance(v, float) and v >= 0.0 for v in load)
    memory = result["memory"]
    assert isinstance(memory, dict)
    assert memory["total_bytes"] > 0
    assert 0 <= memory["available_bytes"] <= memory["total_bytes"]
    assert isinstance(result["uptime_s"], float) and result["uptime_s"] > 0


def test_host_degrades_load_average_when_unavailable(
    diag: HostDiagService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable source degrades its field to null; the
    request still answers ok."""

    def _no_loadavg() -> tuple[float, float, float]:
        raise OSError("no loadavg")

    monkeypatch.setattr(os, "getloadavg", _no_loadavg)
    result = _call(diag, "host.diag.host")
    assert result["load_average"] is None
    assert result["hostname"]


# --- read-only: no mutation --------------------------------------------------


def test_diag_calls_do_not_mutate(
    diag: HostDiagService,
    proc_service: HostProcessService,
    home: Path,
) -> None:
    """Owner clarification 6: read-only means no mutation — not
    identical metrics (they move), but identical state: no files
    created or changed, no registry record touched, no process
    spawned. Checked across all four capabilities."""
    _seed(proc_service, count=2, run_id="run-1", principal_id=_OWNER)
    before_files = _snapshot_tree(home)
    before_records = {
        pid: (r.profile_id, r.state, r.run_id, r.principal_id, r.started_at)
        for pid, r in proc_service._records.items()
    }
    before_list = _call(diag, "host.diag.process_list")
    _call(diag, "host.diag.service_status")
    _call(diag, "host.diag.disk")
    _call(diag, "host.diag.host")
    after_list = _call(diag, "host.diag.process_list")
    after_files = _snapshot_tree(home)
    after_records = {
        pid: (r.profile_id, r.state, r.run_id, r.principal_id, r.started_at)
        for pid, r in proc_service._records.items()
    }
    assert after_files == before_files
    assert after_records == before_records
    assert after_list == before_list


# --- wire --------------------------------------------------------------------


def test_capabilities_include_diag_even_when_process_hidden(
    bridge: tuple[Path, HostProcessService],
) -> None:
    """Diag capabilities are always visible — unlike the process
    capabilities, which hide when the enforcement probe fails (here:
    no profiles, so process is unavailable). The diagnosis must
    answer exactly when the diagnosed subsystem is broken."""
    socket_path, _svc = bridge
    caps = client.capabilities(socket_path)
    assert isinstance(caps, list)
    assert "host.diag.service_status" in caps
    assert "host.diag.process_list" in caps
    assert "host.diag.disk" in caps
    assert "host.diag.host" in caps
    assert "host.process.run" not in caps
    assert "host.process.poll" not in caps


def test_wire_service_status_disk_host(
    bridge: tuple[Path, HostProcessService],
) -> None:
    socket_path, _svc = bridge
    status = client.service_status(socket_path)
    # The daemon runs in a thread of this test process.
    assert status["daemon"]["pid"] == os.getpid()
    assert status["process_available"] is False
    assert status["unit"] is None
    disks = client.disk(socket_path)["disks"]
    assert {d["root_id"] for d in disks} == {"work", "docs"}
    host = client.host(socket_path)
    assert host["hostname"]
    assert host["cpu_count"] >= 1


def test_wire_process_list_empty_then_seeded(
    bridge: tuple[Path, HostProcessService],
) -> None:
    socket_path, svc = bridge
    assert client.process_list(socket_path) == {
        "processes": [],
        "count": 0,
        "truncated": False,
    }
    ids = _seed(svc, count=2, run_id="run-1", principal_id=_OWNER)
    result = client.process_list(socket_path)
    assert result["count"] == 2
    assert [p["process_id"] for p in result["processes"]] == ids
    # Envelope principal + params run_id filter end to end.
    filtered = client.process_list(socket_path, "run-1", principal_id=_OWNER)
    assert filtered["count"] == 2
    other = client.process_list(socket_path, "run-1", principal_id=_OTHER)
    assert other == {"processes": [], "count": 0, "truncated": False}


def test_wire_process_list_unknown_run_is_empty(
    bridge: tuple[Path, HostProcessService],
) -> None:
    socket_path, _svc = bridge
    assert client.process_list(socket_path, "no-such-run", principal_id=_OWNER) == {
        "processes": [],
        "count": 0,
        "truncated": False,
    }


# --- negative ------------------------------------------------------------------


def test_unknown_diag_capability_rejected(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        diag.handle("host.diag.nope", {}, "test-req", _token())
    assert excinfo.value.code == "invalid_params"


def test_service_status_rejects_params(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.service_status", {"unexpected": 1})
    assert excinfo.value.code == "invalid_params"


def test_host_rejects_params(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.host", {"unexpected": 1})
    assert excinfo.value.code == "invalid_params"


def test_process_list_rejects_unknown_param(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.process_list", {"bogus": 1})
    assert excinfo.value.code == "invalid_params"


def test_process_list_rejects_non_string_run_id(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.process_list", {"run_id": 5})
    assert excinfo.value.code == "invalid_params"


def test_process_list_rejects_empty_run_id(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.process_list", {"run_id": ""})
    assert excinfo.value.code == "invalid_params"


def test_process_list_rejects_overlong_run_id(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.process_list", {"run_id": "x" * 257})
    assert excinfo.value.code == "invalid_params"


def test_disk_rejects_unknown_root(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.disk", {"root_id": "no-such-root"})
    assert excinfo.value.code == "invalid_params"


def test_disk_rejects_unknown_param(diag: HostDiagService) -> None:
    with pytest.raises(DiagError) as excinfo:
        _call(diag, "host.diag.disk", {"bogus": 1})
    assert excinfo.value.code == "invalid_params"
