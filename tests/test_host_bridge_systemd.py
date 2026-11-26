"""E03-T09: systemd hardening profile of the host-bridge daemon.

The profile lives in ``deploy/slavikai-host-bridge.service.example`` and
is the ONLY production artifact of this ticket (plus this file and the
claim). No daemon code is touched: the enforcement probe is the
detector, never the thing being tuned.

What the profile is (verified live on the host 2026-11-26, systemd 259):

- 9 active settings, each proven to apply behaviourally AND to keep the
  enforcement probe green: NoNewPrivileges, RestrictAddressFamilies=
  AF_UNIX, SystemCallFilter=@system-service, SystemCallArchitectures=
  native, MemoryDenyWriteExecute, RestrictRealtime, RestrictSUIDSGID,
  LockPersonality, UMask=0077. ``systemd-analyze security``: 6.7 MEDIUM
  (25 checks green); exposure is an auxiliary metric, not the goal.
- Mount settings are DELIBERATELY ABSENT: they do not apply for user
  services on this host AND each of them breaks the enforcement probe
  (fresh user namespace -> spawn helper's
  unshare(CLONE_NEWUSER|NEWNET|NEWIPC) fails with EPERM).
- DevicePolicy/DeviceAllow are DELIBERATELY NOT active (directors'
  verdict Q4): the device filter does not bind for user services here.
  They stay in the file as commented-out documentation only.
- The OS-level filesystem isolation the ticket originally asked for is
  NOT provided by this file; the filesystem boundary stays in daemon
  code (RootsRegistry + fs.py). The gap is recorded in ROADMAP.md
  §E03-T09 as an open architectural gap, carried into T11 acceptance.

Directors' verdict adjustments over schema rev. 1 (all in this file):

- Condition 1 (skip semantics): the runtime suite runs the BASELINE
  enforcement probe first, in-process and without any profile. If the
  baseline probe fails, the environment cannot run the daemon's
  enforcement at all, and the whole runtime module SKIPs. If the
  baseline passes and the hardened daemon's probe fails, that is a
  FAIL, never a skip — a returning PrivateTmp must go red, not silent.
- Condition 2 (behavioural controls): address-family, NoNewPrivileges,
  memory-protection and UMask controls are checked by BEHAVIOUR
  (a refused socket, a refused WX mmap, a 0600 file, a set flag),
  not by ``Seccomp: 2`` or analyzer output alone.
- Condition 3 (SystemCallArchitectures=native): included ONLY after
  live runtime tests passed under it (daemon starts, probe green,
  spawn/STOP work, all behavioural checks identical) — verified
  2026-11-26 during implementation.
- Condition 4 (no pinning of absent isolation): the mount/device
  non-binding is DOCUMENTED, not required. The tests assert only what
  the daemon needs (its devices open, its roots writable); whether
  /dev/ptmx, /dev/kvm or a host mount namespace are reachable is
  recorded, never asserted — a future systemd that starts enforcing
  must not count as a regression.
- Condition 5 (lifecycle): spawn, STOP, timeout kill, crash restart
  and orphan cleanup are all exercised under the final profile.

Runtime tests are host-only: they need a systemd user manager plus
the bus (``systemctl --user``) and a live daemon, so they skip in CI.
The skip is explicit and named, never silent. The daemon runs from a
transient service built from the example file itself (its [Service]
settings are parsed and passed as properties, not hand-copied), with
an isolated XDG_RUNTIME_DIR whose ``bus``/``systemd`` entries are
symlinks to the real user bus — the same isolation pattern ``make
check`` uses per worktree. Roots live in a throwaway directory under
~/Project (the ADR-0013 allowlist requires a subdirectory of
~/Project, ~/Documents or ~/Downloads) and are removed in teardown.
"""

from __future__ import annotations

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from host_bridge import client
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.fs import HostFsService, Root, RootsRegistry
from host_bridge.process import build_process_service

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_FILE = REPO / "deploy" / "slavikai-host-bridge.service.example"
SYSTEMD_VERSION_PIN = "259"

# The active hardening profile: exactly these settings, no more, no less.
EXPECTED_SETTINGS = {
    "NoNewPrivileges": "true",
    "RestrictAddressFamilies": "AF_UNIX",
    "SystemCallFilter": "@system-service",
    "SystemCallArchitectures": "native",
    "MemoryDenyWriteExecute": "true",
    "RestrictRealtime": "true",
    "RestrictSUIDSGID": "true",
    "LockPersonality": "true",
    "UMask": "0077",
}

# Settings that must NOT appear as active settings in the file
# (schema §4.3 + directors' verdict Q4): mount settings break the
# probe and do not apply; capability-dropping settings break startup;
# PrivateIPC/ProtectHostname are silently ignored; DevicePolicy does
# not bind and stays commented-out documentation only.
FORBIDDEN_SETTINGS = {
    "ProtectSystem",
    "ReadWritePaths",
    "PrivateTmp",
    "ProtectKernelTunables",
    "ProtectProc",
    "PrivateDevices",
    "ProtectClock",
    "ProtectKernelLogs",
    "ProtectKernelModules",
    "ProtectControlGroups",
    "PrivateUsers",
    "PrivateNetwork",
    "ProtectHome",
    "RestrictNamespaces",
    "PrivateIPC",
    "ProtectHostname",
    "CapabilityBoundingSet",
    "DevicePolicy",
    "DeviceAllow",
}

# Analyzer checks the profile itself turns green (systemd 259).
# The 6 remaining green checks (NoNewPrivileges, AmbientCapabilities,
# KeyringMode, Delegate, NotifyAccess, PrivateMounts) are baseline
# defaults, not profile effects, so they are not pinned here.
# The "…" entry is literal: systemd prints the exotic-AF check as
# "RestrictAddressFamilies=~…" (U+2026).
EXPECTED_GREEN_CHECKS = {
    "SystemCallFilter=~@swap",
    "SystemCallFilter=~@reboot",
    "SystemCallFilter=~@raw-io",
    "SystemCallFilter=~@obsolete",
    "SystemCallFilter=~@mount",
    "SystemCallFilter=~@module",
    "SystemCallFilter=~@debug",
    "SystemCallFilter=~@cpu-emulation",
    "SystemCallFilter=~@clock",
    "RestrictAddressFamilies=~AF_PACKET",
    "RestrictAddressFamilies=~AF_NETLINK",
    "RestrictAddressFamilies=~…",
    "RestrictAddressFamilies=~AF_(INET|INET6)",
    "LockPersonality=",
    "MemoryDenyWriteExecute=",
    "RestrictRealtime=",
    "RestrictSUIDSGID=",
    "UMask=",
    "SystemCallArchitectures=",
}
EXPECTED_EXPOSURE = 6.7
EXPECTED_GREEN_COUNT = 25
EXPECTED_RED_COUNT = 54

_PRINCIPAL = "t09-test"


def _parse_service_settings() -> dict[str, str]:
    """Active [Service] settings of the example file (comments ignored)."""
    parser = configparser.ConfigParser(strict=False)
    parser.optionxform = str  # type: ignore[assignment]
    parser.read(EXAMPLE_FILE, encoding="utf-8")
    return dict(parser["Service"])


def _analyzer_version() -> str | None:
    """``systemd-analyze`` version (``259``) or None when absent."""
    try:
        out = subprocess.run(
            ["systemd-analyze", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"systemd (\d+)", out.stdout)
    return match.group(1) if match else None


def _analyzer_output(service_file: Path) -> str:
    out = subprocess.run(
        ["systemd-analyze", "security", "--offline=true", str(service_file)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, f"systemd-analyze failed: {out.stderr[:500]}"
    return out.stdout


# --- offline: the file itself ---------------------------------------------


def test_example_file_has_exactly_the_profile() -> None:
    """Parse-guard: exactly the 9 profile settings, none forbidden.

    Catches a well-meant "one more hardening line" (in particular a
    returning mount setting or DevicePolicy) at review time.
    """
    settings = _parse_service_settings()
    profile = {k: v for k, v in settings.items() if k in EXPECTED_SETTINGS}
    assert profile == EXPECTED_SETTINGS, f"profile drift in {EXAMPLE_FILE}: {profile!r}"
    forbidden = sorted(set(settings) & FORBIDDEN_SETTINGS)
    assert not forbidden, f"forbidden settings active in {EXAMPLE_FILE}: {forbidden}"
    assert settings["Type"] == "simple"
    assert settings["Restart"] == "on-failure"
    assert settings["RestartSec"] == "3"


def test_device_policy_stays_commented_documentation() -> None:
    """Q4: DevicePolicy is documented option, not mandatory profile."""
    text = EXAMPLE_FILE.read_text(encoding="utf-8")
    assert "# DevicePolicy=closed" in text
    assert "# DeviceAllow=/dev/null rw" in text
    for line in text.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("DevicePolicy="), line
        assert not stripped.startswith("DeviceAllow="), line


def test_analyzer_version_is_pinned() -> None:
    """Q5: the analyzer gate is pinned to the verified systemd version."""
    version = _analyzer_version()
    if version is None:
        pytest.skip("no systemd-analyze binary in this environment")
    assert version == SYSTEMD_VERSION_PIN, (
        f"profile verified on systemd {SYSTEMD_VERSION_PIN}, "
        f"this host has {version} — re-verify the profile"
    )


def test_analyzer_green_list_and_exposure(tmp_path: Path) -> None:
    """Q5/Q6: the 19 named checks are green, exposure 6.7 MEDIUM.

    The gate is the named list plus exposure < 7.0, not the "SAFE"
    headline (unreachable for a user service). Skipped unless the
    analyzer is the pinned version — another version must re-verify
    the profile instead of silently passing or failing it.
    """
    version = _analyzer_version()
    if version is None:
        pytest.skip("no systemd-analyze binary in this environment")
    if version != SYSTEMD_VERSION_PIN:
        pytest.skip(
            f"systemd-analyze {version}, profile verified on "
            f"{SYSTEMD_VERSION_PIN} — re-verify before gating on it"
        )
    service_file = tmp_path / "t09.service"
    # The analyzer rejects the .example suffix: copy under a *.service name.
    shutil.copy(EXAMPLE_FILE, service_file)
    output = _analyzer_output(service_file)
    green: set[str] = set()
    red = 0
    for line in output.splitlines():
        if line.startswith("✓"):
            green.add(line[2:].split("  ")[0].strip())
        elif line.startswith("✗"):
            red += 1
    missing = sorted(EXPECTED_GREEN_CHECKS - green)
    assert not missing, f"analyzer checks not green: {missing}"
    assert len(green) == EXPECTED_GREEN_COUNT, (
        f"green count drift: {len(green)} != {EXPECTED_GREEN_COUNT}"
    )
    assert red == EXPECTED_RED_COUNT, f"red count drift: {red} != {EXPECTED_RED_COUNT}"
    match = re.search(r"Overall exposure level for .*: ([0-9.]+) (\w+)", output)
    assert match is not None, "no exposure line in analyzer output"
    exposure = float(match.group(1))
    assert exposure == EXPECTED_EXPOSURE, (
        f"exposure recalculated: {exposure} != {EXPECTED_EXPOSURE} "
        "(condition 6 — update the claim and this test together)"
    )
    assert match.group(2) == "MEDIUM"


# --- runtime: shared harness ------------------------------------------------


def _write_daemon_config(config_path: Path, work: Path, docs: Path) -> None:
    """Daemon config: two roots + the profiles the runtime tests use."""
    payload = {
        "roots": [
            {"id": "work", "path": str(work), "mode": "rw"},
            {"id": "docs", "path": str(docs), "mode": "ro"},
        ],
        "process_profiles": [
            {"id": "true", "binary": "/usr/bin/true", "cwd": {"root_id": "work", "path": "."}},
            {
                "id": "echo",
                "binary": "/usr/bin/echo",
                "argv_prefix": ["--"],
                "args": [
                    {
                        "name": "text",
                        "kind": "string",
                        "required": True,
                        "form": "after_dashdash",
                    }
                ],
                "cwd": {"root_id": "work", "path": "."},
            },
            {
                "id": "sleep",
                "binary": "/usr/bin/sleep",
                "args": [{"name": "seconds", "kind": "int", "required": True, "min": 1, "max": 30}],
                "cwd": {"root_id": "work", "path": "."},
                "timeout_s": 35,
            },
            {
                "id": "sleep_timeout",
                "binary": "/usr/bin/sleep",
                "args": [{"name": "seconds", "kind": "int", "required": True, "min": 1, "max": 5}],
                "cwd": {"root_id": "work", "path": "."},
                "timeout_s": 3,
            },
        ],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload), encoding="utf-8")


def _run_systemctl(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _user_manager_available() -> bool:
    """True when a systemd user manager answers on this machine."""
    try:
        out = _run_systemctl("list-units", "--type=scope", "--all", "--plain", "--no-legend")
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0


def _list_scopes() -> list[str]:
    out = _run_systemctl("list-units", "--type=scope", "--all", "--plain", "--no-legend")
    if out.returncode != 0:
        return []
    return [line.split()[0] for line in out.stdout.splitlines() if line.strip()]


def _unit_main_pid(unit: str) -> int | None:
    out = _run_systemctl("show", unit, "-p", "MainPID")
    match = re.search(r"MainPID=(\d+)", out.stdout)
    return int(match.group(1)) if match else None


def _unit_restarts(unit: str) -> int:
    out = _run_systemctl("show", unit, "-p", "NRestarts")
    match = re.search(r"NRestarts=(\d+)", out.stdout)
    return int(match.group(1)) if match else 0


@pytest.fixture(scope="module")
def baseline_probe_ok(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Condition 1, first half: the BASELINE probe, no profile at all.

    The enforcement probe is run in-process against throwaway tmp
    roots. If it fails, this environment cannot run the daemon's
    enforcement in the first place, so the whole runtime module
    SKIPs — the failure is environmental, not a profile regression.
    If it passes, every hardened-probe failure below is a FAIL.
    """
    if not _user_manager_available():
        pytest.skip("no systemd user manager on this machine")
    base = tmp_path_factory.mktemp("baseline")
    home = base / "home"
    work = home / "Project" / "work"
    work.mkdir(parents=True)
    (home / "Documents" / "docs").mkdir(parents=True)
    config = base / "host-bridge.json"
    _write_daemon_config(config, work, home / "Documents" / "docs")
    registry = RootsRegistry(
        roots=(
            Root(id="work", path=work.resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    fs_service = HostFsService(registry, trash_root=base / "Trash")
    svc = build_process_service(registry, fs_service, runtime_dir=base / "run")
    probe = svc.probe()
    svc.shutdown()
    if not probe.available:
        pytest.skip(f"baseline enforcement probe failed: {probe.reason}: {probe.checks}")


@pytest.fixture(scope="module")
def _project_area(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """One throwaway directory under ~/Project for all runtime tests.

    The ADR-0013 allowlist admits only the built-in workspace or a
    specific subdirectory of ~/Project, ~/Documents or ~/Downloads,
    so the daemon's test roots must live there — never under /tmp.
    Removed in teardown.
    """
    area = Path.home() / "Project" / f"slavikai-t09-{uuid.uuid4().hex[:12]}"
    area.mkdir(parents=True)
    try:
        yield area
    finally:
        shutil.rmtree(area, ignore_errors=True)


class _HardenedDaemon:
    """A live daemon under a transient service built from the file."""

    def __init__(
        self, unit: str, socket_path: Path, xdg: Path, work: Path, docs: Path, config: Path
    ) -> None:
        self.unit = unit
        self.socket_path = socket_path
        self.xdg = xdg
        self.work = work
        self.docs = docs
        self.config = config

    @property
    def main_pid(self) -> int | None:
        return _unit_main_pid(self.unit)


def _service_properties(xdg: Path, config: Path) -> list[str]:
    """systemd-run properties from the example file (test overrides
    for WorkingDirectory/ExecStart: the repo, not /opt/slavikai)."""
    settings = _parse_service_settings()
    settings.pop("WorkingDirectory", None)
    settings.pop("ExecStart", None)
    props = [f"--property={key}={value}" for key, value in settings.items()]
    props.append(f"--property=WorkingDirectory={REPO}")
    props.append(f"--property=Environment=XDG_RUNTIME_DIR={xdg}")
    props.append(f"--property=Environment=SLAVIK_HOST_BRIDGE_CONFIG={config}")
    return props


def _start_service(unit: str, xdg: Path, config: Path) -> None:
    out = subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            *_service_properties(xdg, config),
            sys.executable,
            "-m",
            "host_bridge",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert out.returncode == 0, f"systemd-run failed: {out.stdout} {out.stderr[:500]}"


def _stop_service(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", unit],
        capture_output=True,
        text=True,
        timeout=15,
    )
    subprocess.run(
        ["systemctl", "--user", "reset-failed", unit],
        capture_output=True,
        text=True,
        timeout=15,
    )


def _wait_socket_ping(socket_path: Path, *, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if socket_path.exists():
                client.ping(socket_path)
                return
        except Exception as exc:  # noqa: BLE001 - retry until deadline
            last_error = exc
        time.sleep(0.2)
    raise AssertionError(f"daemon socket {socket_path} never answered ping: {last_error}")


@pytest.fixture()
def hardened_daemon(
    baseline_probe_ok: None,
    tmp_path: Path,
    _project_area: Path,
) -> Iterator[_HardenedDaemon]:
    """One live daemon per test, under the profile from the file.

    The XDG_RUNTIME_DIR is isolated per test; its ``bus``/``systemd``
    entries symlink to the real user bus, because ``systemctl --user``
    derives the bus address from XDG_RUNTIME_DIR (the same isolation
    pattern ``make check`` uses per worktree). Without the symlinks
    the daemon's orphan sweep cannot reach the bus and exits 2.
    """
    uid = os.getuid()
    bus = Path(f"/run/user/{uid}/bus")
    if not bus.exists():
        pytest.skip("no user bus socket on this machine")
    run_id = uuid.uuid4().hex[:12]
    work = _project_area / f"work-{run_id}"
    docs = _project_area / f"docs-{run_id}"
    work.mkdir(parents=True)
    docs.mkdir(parents=True)
    config = tmp_path / "host-bridge.json"
    _write_daemon_config(config, work, docs)
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    os.symlink(f"/run/user/{uid}/bus", xdg / "bus")
    os.symlink(f"/run/user/{uid}/systemd", xdg / "systemd")
    unit = f"t09-test-{run_id}"
    socket_path = xdg / "slavikai" / "host-bridge.sock"
    _start_service(unit, xdg, config)
    try:
        _wait_socket_ping(socket_path)
        yield _HardenedDaemon(unit, socket_path, xdg, work, docs, config)
    finally:
        _stop_service(unit)


def _wait_process_state(
    socket_path: Path,
    *,
    principal_id: str,
    run_id: str,
    process_id: str,
    states: set[str],
    timeout_s: float = 15.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        last = client.call(
            socket_path,
            "host.process.poll",
            principal_id=principal_id,
            run_id=run_id,
            params={"process_id": process_id},
        )
        if last.get("state") in states:
            return last
        time.sleep(0.2)
    raise AssertionError(f"process {process_id} never reached {states}: {last}")


def _run_process(
    socket_path: Path, *, profile_id: str, args: list[str], run_id: str
) -> dict[str, object]:
    return client.call(
        socket_path,
        "host.process.run",
        principal_id=_PRINCIPAL,
        run_id=run_id,
        params={"profile_id": profile_id, "args": args},
    )


# --- runtime: the profile does not break the daemon ------------------------


def test_daemon_starts_under_profile_and_probe_is_green(
    hardened_daemon: _HardenedDaemon,
) -> None:
    """Condition 1, second half: baseline passed, so a dark probe FAILS.

    The daemon starts under the profile and answers ping; the
    process capabilities are visible, which only happens when the
    startup enforcement probe passed under this exact profile.
    """
    assert client.ping(hardened_daemon.socket_path) == {"pong": True}
    caps = client.capabilities(hardened_daemon.socket_path)
    for required in (
        "host.process.run",
        "host.process.poll",
        "host.process.kill",
        "host.process.stop",
    ):
        assert required in caps, f"probe dark under profile: {required} missing"
    assert client.version(hardened_daemon.socket_path)


def test_fs_boundary_is_enforced_by_daemon_not_unit(
    hardened_daemon: _HardenedDaemon,
    tmp_path: Path,
) -> None:
    """§6.5: the FS refusal comes from daemon code, not from the unit.

    A write into the configured root succeeds under the profile; a
    write addressed to a root the daemon does not know is rejected
    by the daemon with ``unknown_root``; the same rejection comes
    from an in-process daemon with NO profile at all.
    """
    sock = hardened_daemon.socket_path
    run_id = f"fs-{uuid.uuid4().hex[:8]}"
    result = client.call(
        sock,
        "host.fs.write",
        principal_id=_PRINCIPAL,
        run_id=run_id,
        params={
            "root_id": "work",
            "path": "t09.txt",
            "mode": "create",
            "content": "hello t09\n",
            "commit": True,
        },
        request_id=f"w-{uuid.uuid4().hex}",
    )
    assert result.get("bytes_written") == 10
    assert (hardened_daemon.work / "t09.txt").read_text(encoding="utf-8") == "hello t09\n"
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(
            sock,
            "host.fs.write",
            principal_id=_PRINCIPAL,
            run_id=run_id,
            params={
                "root_id": "nope",
                "path": "t09.txt",
                "mode": "create",
                "content": "x",
                "commit": True,
            },
            request_id=f"w-{uuid.uuid4().hex}",
        )
    assert excinfo.value.code == "unknown_root"
    # Same refusal from a daemon with no profile at all.
    home = tmp_path / "home"
    work = home / "Project" / "work"
    work.mkdir(parents=True)
    docs = home / "Documents" / "docs"
    docs.mkdir(parents=True)
    config = tmp_path / "plain-host-bridge.json"
    _write_daemon_config(config, work, docs)
    registry = RootsRegistry(
        roots=(
            Root(id="work", path=work.resolve(), mode="rw"),
            Root(id="docs", path=docs.resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    fs_service = HostFsService(registry, trash_root=tmp_path / "Trash")
    proc_service = build_process_service(registry, fs_service, runtime_dir=tmp_path / "run")
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=fs_service, process_service=proc_service)
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(client.BridgeClientError) as plain_excinfo:
            client.call(
                socket_path,
                "host.fs.write",
                principal_id=_PRINCIPAL,
                run_id=run_id,
                params={
                    "root_id": "nope",
                    "path": "t09.txt",
                    "mode": "create",
                    "content": "x",
                    "commit": True,
                },
                request_id=f"w-{uuid.uuid4().hex}",
            )
        assert plain_excinfo.value.code == "unknown_root"
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_process_run_works_under_profile(hardened_daemon: _HardenedDaemon) -> None:
    """§6.6: the profile does not break execution (probe tripwire)."""
    sock = hardened_daemon.socket_path
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    started = _run_process(sock, profile_id="echo", args=["hello-t09"], run_id=run_id)
    process_id = str(started["process_id"])
    final = _wait_process_state(
        sock,
        principal_id=_PRINCIPAL,
        run_id=run_id,
        process_id=process_id,
        states={"exited", "failed", "killed", "timeout"},
    )
    assert final["state"] == "exited"
    assert final["exit_code"] == 0
    assert "hello-t09" in str(final.get("stdout", ""))


def test_tree_kill_under_profile(hardened_daemon: _HardenedDaemon) -> None:
    """§6.7: STOP kills the tree and the scope is gone (kill path intact)."""
    sock = hardened_daemon.socket_path
    run_id = f"kill-{uuid.uuid4().hex[:8]}"
    started = _run_process(sock, profile_id="sleep", args=["30"], run_id=run_id)
    process_id = str(started["process_id"])
    _wait_process_state(
        sock,
        principal_id=_PRINCIPAL,
        run_id=run_id,
        process_id=process_id,
        states={"running"},
    )
    scopes_before = [s for s in _list_scopes() if process_id in s]
    assert scopes_before, "sleep scope not listed before STOP"
    stopped = client.call(
        sock,
        "host.process.stop",
        principal_id=_PRINCIPAL,
        run_id=run_id,
        params={"run_id": run_id},
    )
    assert stopped.get("all_final") is True
    final = _wait_process_state(
        sock,
        principal_id=_PRINCIPAL,
        run_id=run_id,
        process_id=process_id,
        states={"killed", "exited", "failed", "timeout"},
    )
    assert final["state"] == "killed"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not [s for s in _list_scopes() if process_id in s]:
            break
        time.sleep(0.2)
    assert not [s for s in _list_scopes() if process_id in s], "scope survived STOP"


def test_timeout_kill_under_profile(hardened_daemon: _HardenedDaemon) -> None:
    """Condition 5: the daemon kills a run past its profile timeout."""
    sock = hardened_daemon.socket_path
    run_id = f"to-{uuid.uuid4().hex[:8]}"
    started = _run_process(sock, profile_id="sleep_timeout", args=["5"], run_id=run_id)
    process_id = str(started["process_id"])
    final = _wait_process_state(
        sock,
        principal_id=_PRINCIPAL,
        run_id=run_id,
        process_id=process_id,
        states={"timeout", "exited", "failed", "killed"},
        timeout_s=15.0,
    )
    assert final["state"] == "timeout", f"expected timeout kill, got {final}"


def test_seccomp_applies(hardened_daemon: _HardenedDaemon) -> None:
    """§6.9: the syscall layer binds (Seccomp: 2 on the live daemon)."""
    pid = hardened_daemon.main_pid
    assert pid is not None, "daemon has no MainPID"
    status = (Path(f"/proc/{pid}/status")).read_text(encoding="utf-8")
    seccomp = next(
        line.split(":", 1)[1].strip() for line in status.splitlines() if line.startswith("Seccomp:")
    )
    assert seccomp == "2", f"seccomp not enforced on daemon {pid}: {seccomp}"


# --- runtime: behavioural controls (condition 2) ----------------------------

_BEHAVIOURAL_HELPER = '''
"""Stdlib-only probe: what does the profile ACTUALLY enforce?

Runs under the profile via a transient service. Prints one
``name:ok:<value>`` or ``name:refused:<errno>`` line per check, then
the namespace facts. The unshare check runs LAST: it moves this
process into a new user namespace, so namespace reads come first.
"""
import ctypes
import mmap
import os
import socket
import sys


def rep(name, fn):
    try:
        print(f"{name}:ok:{fn()}")
    except OSError as e:
        print(f"{name}:refused:{e.errno}")
    except Exception as e:  # noqa: BLE001 - report, do not crash the run
        print(f"{name}:error:{type(e).__name__}:{e}")


def af_inet():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    return "opened"


def mmap_wx():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    addr = libc.mmap(None, 4096, 0x1 | 0x2 | 0x4, 0x22, -1, 0)
    if addr in (None, -1):
        raise OSError(ctypes.get_errno(), "mmap failed")
    libc.munmap(addr, 4096)
    return "mapped"


def umask_file(probe_dir):
    path = os.path.join(probe_dir, "umask-probe.txt")
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    with open(path, "w", encoding="utf-8") as f:
        f.write("x")
    return oct(os.stat(path).st_mode & 0o777)


def nnp():
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("NoNewPrivs:"):
            return line.split(":", 1)[1].strip()
    return "?"


def seccomp():
    for line in open("/proc/self/status", encoding="utf-8"):
        if line.startswith("Seccomp:"):
            return line.split(":", 1)[1].strip()
    return "?"


def dev(path):
    fd = os.open(path, os.O_RDONLY)
    os.close(fd)
    return "opened"


def unshare_combo():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    combo = 0x10000000 | 0x40000000 | 0x08000000  # NEWUSER|NEWNET|NEWIPC
    if libc.unshare(combo) != 0:
        raise OSError(ctypes.get_errno(), "unshare failed")
    return "ok"


probe_dir = sys.argv[1]
rep("af_inet", af_inet)
rep("mmap_wx", mmap_wx)
rep("umask_file", lambda: umask_file(probe_dir))
rep("nnp", nnp)
rep("seccomp", seccomp)
for name in ("null", "zero", "full", "urandom", "random"):
    rep(f"dev_{name}", lambda p=f"/dev/{name}": dev(p))
rep("dev_ptmx", lambda: dev("/dev/ptmx"))
rep("dev_kvm", lambda: dev("/dev/kvm"))
print("mount_ns:" + os.readlink("/proc/self/ns/mnt"))
print("user_ns:" + os.readlink("/proc/self/ns/user"))
rep("unshare_combo", unshare_combo)
'''


def _run_behavioural(tmp_path: Path) -> dict[str, str]:
    """Run the behavioural helper under the profile; parse its lines."""
    if not _user_manager_available():
        pytest.skip("no systemd user manager on this machine")
    helper = tmp_path / "behavioural.py"
    helper.write_text(_BEHAVIOURAL_HELPER, encoding="utf-8")
    probe_dir = tmp_path / "probe"
    probe_dir.mkdir(exist_ok=True)
    settings = _parse_service_settings()
    settings.pop("WorkingDirectory", None)
    settings.pop("ExecStart", None)
    props = [f"--property={key}={value}" for key, value in settings.items()]
    props.append(f"--property=WorkingDirectory={REPO}")
    unit = f"t09-behav-{uuid.uuid4().hex[:12]}"
    out = subprocess.run(
        [
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--pipe",
            *props,
            sys.executable,
            str(helper),
            str(probe_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    _stop_service(unit)
    assert out.returncode == 0, (
        f"behavioural helper failed under the profile: {out.stdout} {out.stderr[:500]}"
    )
    result: dict[str, str] = {}
    for line in out.stdout.splitlines():
        if ":" not in line:
            continue
        key, _, rest = line.partition(":")
        result[key] = rest
    return result


@pytest.fixture(scope="module")
def behavioural(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """One behavioural run shared by the condition-2/4 tests."""
    return _run_behavioural(tmp_path_factory.mktemp("behavioural"))


def test_af_inet_socket_is_refused(behavioural: dict[str, str]) -> None:
    """Condition 2: RestrictAddressFamilies actually refuses AF_INET."""
    assert behavioural["af_inet"].startswith("refused:"), behavioural


def test_writable_executable_mmap_is_refused(behavioural: dict[str, str]) -> None:
    """Condition 2: MemoryDenyWriteExecute actually refuses WX mappings."""
    assert behavioural["mmap_wx"].startswith("refused:"), behavioural


def test_umask_default_is_0600(behavioural: dict[str, str]) -> None:
    """Condition 2: UMask=0077 actually applies to new files."""
    assert behavioural["umask_file"] == "ok:0o600", behavioural


def test_no_new_privileges_flag_is_set(behavioural: dict[str, str]) -> None:
    """Condition 2: NoNewPrivileges actually applies (survives exec)."""
    assert behavioural["nnp"] == "ok:1", behavioural


def test_seccomp_filter_is_active(behavioural: dict[str, str]) -> None:
    """Seccomp: 2 under the profile (necessary, never sufficient alone)."""
    assert behavioural["seccomp"] == "ok:2", behavioural


def test_spawn_helper_unshare_works(behavioural: dict[str, str]) -> None:
    """§6.10 reworked (condition 4): the spawn helper's exact unshare
    works under the profile. This is the functional tripwire for the
    F6 mechanism — a returning mount setting breaks it — without
    pinning which user namespace the daemon lives in."""
    assert behavioural["unshare_combo"] == "ok:ok", behavioural
    assert behavioural["user_ns"].startswith("user:["), behavioural


def test_required_devices_open_documents_rest(behavioural: dict[str, str]) -> None:
    """§6.11 reworked (condition 4): the daemon's devices open.

    Only the devices the daemon actually needs are asserted. Whether
    /dev/ptmx or /dev/kvm (outside the old allowlist) open is
    RECORDED here, never asserted: a future systemd that starts
    enforcing the device filter must not count as a regression.
    """
    for name in ("null", "zero", "full", "urandom", "random"):
        assert behavioural[f"dev_{name}"] == "ok:opened", behavioural
    # Informational only: non-binding on this host (audit F9).
    print(f"dev_ptmx: {behavioural.get('dev_ptmx')}")
    print(f"dev_kvm: {behavioural.get('dev_kvm')}")
    print(f"mount_ns: {behavioural.get('mount_ns')}")


def test_mount_layer_documents_current_behavior(
    hardened_daemon: _HardenedDaemon,
    behavioural: dict[str, str],
) -> None:
    """§6.12 reworked (condition 4): the daemon can do its job.

    A write into the configured root succeeds under the profile —
    the functional requirement. The mount namespace identity is
    recorded, never asserted: the unit must not be required to
    leave the mount view untouched on future systems.
    """
    sock = hardened_daemon.socket_path
    result = client.call(
        sock,
        "host.fs.write",
        principal_id=_PRINCIPAL,
        run_id=f"mnt-{uuid.uuid4().hex[:8]}",
        params={
            "root_id": "work",
            "path": "mount-check.txt",
            "mode": "create",
            "content": "x",
            "commit": True,
        },
        request_id=f"w-{uuid.uuid4().hex}",
    )
    assert result.get("bytes_written") == 1
    print(f"mount_ns: {behavioural.get('mount_ns')}")


# --- runtime: lifecycle under the final profile (condition 5) --------------


def test_restart_from_same_unit_keeps_probe_green(
    hardened_daemon: _HardenedDaemon,
) -> None:
    """§6.8: hardening does not break socket lifecycle/restart."""
    unit = hardened_daemon.unit
    _stop_service(unit)
    _start_service(unit, hardened_daemon.xdg, hardened_daemon.config)
    try:
        _wait_socket_ping(hardened_daemon.socket_path)
        caps = client.capabilities(hardened_daemon.socket_path)
        assert "host.process.run" in caps, "probe dark after restart"
    finally:
        _stop_service(unit)


def test_crash_restart_recovers(hardened_daemon: _HardenedDaemon) -> None:
    """§6.13: kill -9 -> systemd restarts -> socket answers, probe green."""
    pid = hardened_daemon.main_pid
    assert pid is not None, "daemon has no MainPID"
    restarts_before = _unit_restarts(hardened_daemon.unit)
    os.kill(pid, 9)
    _wait_socket_ping(hardened_daemon.socket_path, timeout_s=20.0)
    assert _unit_restarts(hardened_daemon.unit) > restarts_before
    caps = client.capabilities(hardened_daemon.socket_path)
    assert "host.process.run" in caps, "probe dark after crash restart"


def test_orphan_cleanup_on_restart(hardened_daemon: _HardenedDaemon) -> None:
    """Condition 5: a daemon restart sweeps the previous run's scopes."""
    sock = hardened_daemon.socket_path
    run_id = f"orphan-{uuid.uuid4().hex[:8]}"
    started = _run_process(sock, profile_id="sleep", args=["30"], run_id=run_id)
    process_id = str(started["process_id"])
    _wait_process_state(
        sock,
        principal_id=_PRINCIPAL,
        run_id=run_id,
        process_id=process_id,
        states={"running"},
    )
    scopes_before = [s for s in _list_scopes() if process_id in s]
    assert scopes_before, "sleep scope not listed before daemon kill"
    pid = hardened_daemon.main_pid
    assert pid is not None, "daemon has no MainPID"
    os.kill(pid, 9)
    _wait_socket_ping(sock, timeout_s=20.0)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if not [s for s in _list_scopes() if process_id in s]:
            break
        time.sleep(0.2)
    assert not [s for s in _list_scopes() if process_id in s], (
        "orphan scope survived daemon restart"
    )
    caps = client.capabilities(sock)
    assert "host.process.run" in caps, "probe dark after orphan sweep"
