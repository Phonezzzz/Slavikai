"""Approved-profile process API (E03-T05, ADR-0013 §10a).

Unit tests use an in-test fake spawner (no systemd, no enforcement)
for the service logic — profile validation, arg validation,
idempotent launch, poll offsets/retained_from/frame budget, kill
idempotence, caps, retention, and cancellation at the launch
boundary — plus wire tests over a real daemon socket. Integration
tests use the real systemd-run spawner + helper and are skipped
when the enforcement probe fails in this environment (the probe
itself is the fail-closed gate); they cover the enforcement stack
(socket refusal, protected-file denial from inside the process,
tree kill after leader exit, timeout tree kill, env minimality,
binary identity, and live idempotent retry).
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

import host_bridge.process as process_module
from host_bridge import client
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.fs import CancelToken, HostFsService, Root, RootsRegistry
from host_bridge.process import (
    HostProcessService,
    ProbeResult,
    ProcessError,
    SpawnError,
    SpawnHandle,
    load_profiles,
)
from host_bridge.protocol import (
    MAX_FRAME_BYTES,
    BridgeConfigError,
    RequestTimeoutError,
    encode_success,
)

_TRUE = "/usr/bin/true"
_CAT = "/usr/bin/cat"
_GREP = "/usr/bin/grep"
_ECHO = "/usr/bin/echo"
_SLEEP = "/usr/bin/sleep"
_PY = "/usr/bin/python3"

# Leader that forks a long-lived child and exits after 1 s: the
# child must not survive the tree kill (cgroup v2 scope). The
# child is a fork, not a second exec: under the tightened EXECUTE
# grant only the approved binary itself may execute.
_SPAWN_CODE = (
    "import os, time; 'TREEKILL47'; pid = os.fork(); pid == 0 and time.sleep(47); time.sleep(1)"
)
# Prints the marker env var if it leaked into the child, else ABSENT.
_ENV_CODE = (
    "import os; print(os.environ.get('SLAVIK_TEST_MARKER', 'ABSENT')); "
    "print(os.environ.get('HOME', ''))"
)


def _profiles(extra: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    profiles: list[dict[str, object]] = [
        {"id": "true", "binary": _TRUE, "cwd": {"root_id": "work", "path": "."}},
        {
            "id": "echo",
            "binary": _ECHO,
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
            "id": "cat",
            "binary": _CAT,
            "args": [{"name": "file", "kind": "path_in_root", "required": True}],
            "cwd": {"root_id": "work", "path": "."},
        },
        {
            "id": "grep",
            "binary": _GREP,
            "argv_prefix": ["--"],
            "args": [
                {
                    "name": "needle",
                    "kind": "string",
                    "required": True,
                    "form": "after_dashdash",
                },
                {"name": "file", "kind": "path_in_root", "required": True},
            ],
            "cwd": {"root_id": "work", "path": "."},
        },
        {
            "id": "sleep",
            "binary": _SLEEP,
            "args": [{"name": "seconds", "kind": "int", "required": True, "min": 1, "max": 5}],
            "cwd": {"root_id": "work", "path": "."},
            "timeout_s": 10,
        },
        {
            "id": "py_spawn",
            "binary": _PY,
            "argv_prefix": ["-c", _SPAWN_CODE],
            "cwd": {"root_id": "work", "path": "."},
            "timeout_s": 30,
        },
        {
            "id": "py_env",
            "binary": _PY,
            "argv_prefix": ["-c", _ENV_CODE],
            "cwd": {"root_id": "work", "path": "."},
        },
        {
            # Optional slot before a required one: the required slot
            # must still be enforced when it is skipped.
            "id": "optreq",
            "binary": _TRUE,
            "args": [
                {
                    "name": "mode",
                    "kind": "enum",
                    "required": False,
                    "values": ["ok", "no"],
                },
                {"name": "count", "kind": "int", "required": True, "min": 0, "max": 10},
            ],
            "cwd": {"root_id": "work", "path": "."},
        },
    ]
    if extra:
        profiles.extend(extra)
    return profiles


def _write_config(home: Path, profiles: list[dict[str, object]]) -> Path:
    config = home / ".config" / "slavikai" / "host-bridge.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roots": [
            {"id": "work", "path": str(home / "Project" / "work"), "mode": "rw"},
            {"id": "docs", "path": str(home / "Documents" / "docs"), "mode": "ro"},
        ],
        "process_profiles": profiles,
    }
    config.write_text(json.dumps(payload), encoding="utf-8")
    return config


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    fake_home = tmp_path / "home"
    work = fake_home / "Project" / "work"
    work.mkdir(parents=True)
    (fake_home / "Documents" / "docs").mkdir(parents=True)
    (work / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    (work / ".env").write_text("SECRET=do-not-leak\n", encoding="utf-8")
    (work / "sub").mkdir(exist_ok=True)
    (work / "sub" / "data.txt").write_text("data\n", encoding="utf-8")
    return fake_home


@pytest.fixture()
def registry(home: Path) -> RootsRegistry:
    config = _write_config(home, _profiles())
    return RootsRegistry(
        roots=(
            Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )


@pytest.fixture()
def work_root(registry: RootsRegistry) -> Path:
    root = registry.by_id("work")
    assert root is not None
    return root.path


@pytest.fixture()
def fs_service(registry: RootsRegistry, tmp_path: Path) -> HostFsService:
    return HostFsService(registry, trash_root=tmp_path / "Trash")


def _now_iso() -> str:
    """Current UTC time for fake status files.

    The finished-record sweep evicts records older than the 10 min
    retention, so a fixed timestamp goes stale and freshly finalized
    records vanish mid-test once the wall clock passes it.
    """
    return datetime.now(UTC).isoformat()


class FakeSpawner:
    """In-test spawner: a real ``sleep`` stands in for the scope
    leader (so ``leader_done`` is honest), and status/tail files are
    written by hand. No systemd, no Landlock, no seccomp — the unit
    tests cover service logic, not enforcement."""

    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []
        self.units: dict[str, dict[str, object]] = {}
        self.finish_state = "killed"
        self.delay_running = 0.0
        self.signal_calls: list[tuple[str, int]] = []
        self.kill_calls: list[str] = []

    def _status(self, run_dir: Path, state: str, **extra: object) -> dict[str, object]:
        status: dict[str, object] = {
            "version": 1,
            "process_id": run_dir.name,
            "state": state,
            "pid": 1234,
            "helper_pid": 1235,
            "cgroup": "0::/fake",
            "started_at": "2026-11-24T00:00:00+00:00",
            "finished_at": None,
            "exit_code": None,
            "signal": None,
            "error": None,
            "stdout_total_bytes": 0,
            "stderr_total_bytes": 0,
            "stdout_retained_from": 0,
            "stderr_retained_from": 0,
            "timeout_s": 120,
        }
        status.update(extra)
        return status

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spawns.append(spec)
        leader = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        handle = SpawnHandle(popen=leader, unit_name=unit_name, cgroup_path=None, run_dir=run_dir)
        self.units[unit_name] = {"handle": handle, "procs": [str(leader.pid)]}

        def _write_running() -> None:
            if self.delay_running:
                time.sleep(self.delay_running)
            (run_dir / "status.json").write_text(
                json.dumps(self._status(run_dir, "running")), encoding="utf-8"
            )
            (run_dir / "stdout").write_bytes(b"")
            (run_dir / "stderr").write_bytes(b"")

        if self.delay_running:
            threading.Thread(target=_write_running, daemon=True).start()
        else:
            _write_running()
        return handle

    def signal_tree(self, handle: SpawnHandle, sig: int) -> None:
        self.signal_calls.append((handle.unit_name, sig))
        if handle.popen is not None and handle.popen.poll() is None:
            handle.popen.send_signal(sig)

    def kill_all(self, handle: SpawnHandle) -> None:
        self.kill_calls.append(handle.unit_name)
        unit = self.units.get(handle.unit_name)
        if unit is not None:
            unit["procs"] = []
        if handle.popen is not None and handle.popen.poll() is None:
            handle.popen.kill()
        exit_code = 0 if self.finish_state == "exited" else None
        exit_signal = None if self.finish_state == "exited" else 15
        (handle.run_dir / "status.json").write_text(
            json.dumps(
                self._status(
                    handle.run_dir,
                    self.finish_state,
                    finished_at=_now_iso(),
                    exit_code=exit_code,
                    signal=exit_signal,
                )
            ),
            encoding="utf-8",
        )

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        unit = self.units.get(handle.unit_name)
        if unit is None:
            return True
        procs = unit.get("procs")
        return not procs if isinstance(procs, list) else True

    def finish(
        self,
        handle: SpawnHandle,
        *,
        state: str = "exited",
        exit_code: int | None = 0,
        exit_signal: int | None = None,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        unit = self.units.get(handle.unit_name)
        if unit is not None:
            unit["procs"] = []
        if handle.popen is not None and handle.popen.poll() is None:
            handle.popen.kill()
        (handle.run_dir / "stdout").write_bytes(stdout)
        (handle.run_dir / "stderr").write_bytes(stderr)
        (handle.run_dir / "status.json").write_text(
            json.dumps(
                self._status(
                    handle.run_dir,
                    state,
                    finished_at=_now_iso(),
                    exit_code=exit_code,
                    signal=exit_signal,
                    stdout_total_bytes=len(stdout),
                    stderr_total_bytes=len(stderr),
                )
            ),
            encoding="utf-8",
        )

    def set_output(
        self,
        handle: SpawnHandle,
        *,
        stdout: bytes = b"",
        stderr: bytes = b"",
        stdout_total: int | None = None,
        stderr_total: int | None = None,
        stdout_retained_from: int = 0,
        stderr_retained_from: int = 0,
    ) -> None:
        (handle.run_dir / "stdout").write_bytes(stdout)
        (handle.run_dir / "stderr").write_bytes(stderr)
        status = json.loads((handle.run_dir / "status.json").read_text(encoding="utf-8"))
        status["stdout_total_bytes"] = len(stdout) if stdout_total is None else stdout_total
        status["stderr_total_bytes"] = len(stderr) if stderr_total is None else stderr_total
        status["stdout_retained_from"] = stdout_retained_from
        status["stderr_retained_from"] = stderr_retained_from
        (handle.run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")


@pytest.fixture()
def spawner() -> FakeSpawner:
    return FakeSpawner()


@pytest.fixture()
def service(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    spawner: FakeSpawner,
) -> Iterator[HostProcessService]:
    svc = HostProcessService(
        fs_service,
        registry,
        runtime_dir=tmp_path / "run",
        profiles=load_profiles(registry),
        spawner=spawner,  # type: ignore[arg-type]
    )
    # Unit tests drive the service logic, not the enforcement probe.
    svc._probe = ProbeResult(available=True, checks={}, reason="unit-test")
    try:
        yield svc
    finally:
        svc.shutdown()


def _token(deadline_s: float = 5.0) -> CancelToken:
    return CancelToken(deadline=time.monotonic() + deadline_s, request_id="test-req")


def _service_with_spawner(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    spawner: object,
) -> HostProcessService:
    """A service like the ``service`` fixture, but with a custom spawner."""
    svc = HostProcessService(
        fs_service,
        registry,
        runtime_dir=tmp_path / "run",
        profiles=load_profiles(registry),
        spawner=spawner,  # type: ignore[arg-type]
    )
    # Unit tests drive the service logic, not the enforcement probe.
    svc._probe = ProbeResult(available=True, checks={}, reason="unit-test")
    return svc


def _registry_with_exclusions(home: Path, write_excluded: tuple[Path, ...]) -> RootsRegistry:
    config = _write_config(home, _profiles())
    return RootsRegistry(
        roots=(
            Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=write_excluded,
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )


def _run(
    svc: HostProcessService, profile_id: str, request_id: str, **params: object
) -> dict[str, object]:
    return svc.handle(
        "host.process.run",
        {"profile_id": profile_id, **params},
        request_id,
        _token(),
    )


def _poll(svc: HostProcessService, process_id: str, **params: object) -> dict[str, object]:
    return svc.handle(
        "host.process.poll", {"process_id": process_id, **params}, "poll-req", _token()
    )


def _kill(svc: HostProcessService, process_id: str) -> dict[str, object]:
    return svc.handle("host.process.kill", {"process_id": process_id}, "kill-req", _token())


def _wait_state(
    svc: HostProcessService, process_id: str, state: str, *, timeout_s: float = 5.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        last = _poll(svc, process_id)
        if last.get("state") == state:
            return last
        time.sleep(0.05)
    raise AssertionError(f"process {process_id} did not reach {state}: {last}")


# --- profile loading -------------------------------------------------------


def test_load_profiles_ok(registry: RootsRegistry) -> None:
    profiles = load_profiles(registry)
    by_id = {p.id: p for p in profiles}
    assert set(by_id) == {
        "true",
        "echo",
        "cat",
        "grep",
        "sleep",
        "py_spawn",
        "py_env",
        "optreq",
    }
    assert by_id["grep"].args[0].form == "after_dashdash"
    assert by_id["sleep"].args[0].minimum == 1
    assert by_id["sleep"].args[0].maximum == 5
    assert by_id["sleep"].timeout_s == 10


def test_profiles_description_hides_paths(service: HostProcessService) -> None:
    descriptions = service.profiles_description()
    assert {
        "id": "grep",
        "args": [
            {"name": "needle", "kind": "string", "required": True},
            {"name": "file", "kind": "path_in_root", "required": True},
        ],
    } in descriptions
    assert "usr/bin" not in json.dumps(descriptions)
    assert "Project" not in json.dumps(descriptions)


def test_load_profiles_rejects_unknown_field(home: Path) -> None:
    config = _write_config(
        home,
        [{"id": "x", "binary": _TRUE, "cwd": {"root_id": "work", "path": "."}, "nope": 1}],
    )
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    with pytest.raises(BridgeConfigError, match="unknown fields"):
        load_profiles(registry)


def test_load_profiles_rejects_shell_binary(home: Path) -> None:
    config = _write_config(
        home,
        [{"id": "sh", "binary": "/usr/bin/sh", "cwd": {"root_id": "work", "path": "."}}],
    )
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    with pytest.raises(BridgeConfigError, match="shell"):
        load_profiles(registry)


def test_load_profiles_rejects_binary_inside_root(home: Path) -> None:
    tool = home / "Project" / "work" / "tool"
    shutil.copy(_TRUE, tool)
    config = _write_config(
        home,
        [{"id": "tool", "binary": str(tool), "cwd": {"root_id": "work", "path": "."}}],
    )
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    with pytest.raises(BridgeConfigError, match="inside allowed root"):
        load_profiles(registry)


def test_load_profiles_rejects_string_slot_without_data_form(home: Path) -> None:
    config = _write_config(
        home,
        [
            {
                "id": "bad",
                "binary": _ECHO,
                "argv_prefix": ["-n"],
                "args": [{"name": "text", "kind": "string", "required": True}],
                "cwd": {"root_id": "work", "path": "."},
            }
        ],
    )
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    with pytest.raises(BridgeConfigError, match="string requires form"):
        load_profiles(registry)


def test_load_profiles_rejects_forbidden_env(home: Path) -> None:
    config = _write_config(
        home,
        [
            {
                "id": "bad",
                "binary": _TRUE,
                "env": {"LD_PRELOAD": "/tmp/evil.so"},
                "cwd": {"root_id": "work", "path": "."},
            }
        ],
    )
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    with pytest.raises(BridgeConfigError, match="forbidden"):
        load_profiles(registry)


def test_load_profiles_absent_config_means_no_profiles(home: Path) -> None:
    registry = RootsRegistry(
        roots=(Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=home / ".config" / "slavikai" / "host-bridge.json",
        socket_dir=None,
    )
    assert load_profiles(registry) == ()


# --- run / arg validation --------------------------------------------------


def test_exec_grants_cover_binary_and_interp() -> None:
    from host_bridge.process import Grant, _exec_grants

    grants = _exec_grants(Path(_TRUE))
    # _exec_grants resolves symlinks: the helper never grants through
    # one (it skips symlink grant paths), and profile binaries are
    # often symlinks (/usr/bin/true -> gnutrue, /usr/bin/python3 ->
    # python3.14, and the interpreter /lib64/ld-linux-x86-64.so.2).
    true_path = Path(_TRUE).resolve()
    true_st = true_path.lstat()
    assert grants[0] == Grant(true_path, "rx", dev=true_st.st_dev, ino=true_st.st_ino)
    interp_path = Path("/lib64/ld-linux-x86-64.so.2").resolve()
    interp_st = interp_path.lstat()
    assert grants[1] == Grant(interp_path, "rx", dev=interp_st.st_dev, ino=interp_st.st_ino)
    assert len(grants) == 2


def test_run_unknown_profile(service: HostProcessService) -> None:
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "nope", "req-1")
    assert excinfo.value.code == "unknown_profile"


def test_run_invalid_args(service: HostProcessService) -> None:
    # enum/int/path cases, one per slot kind where applicable
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "sleep", "req-int", args=["0"])
    assert excinfo.value.code == "invalid_args"
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "sleep", "req-int2", args=["not-a-number"])
    assert excinfo.value.code == "invalid_args"
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "sleep", "req-int3", args=["1", "2"])
    assert excinfo.value.code == "invalid_args"
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "cat", "req-missing")
    assert excinfo.value.code == "invalid_args"


def test_run_path_arg_uses_fs_codes(service: HostProcessService) -> None:
    # Escape attempt: the FS layer's codes are reused, not invented.
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "cat", "req-escape", args=["../outside.txt"])
    assert excinfo.value.code == "invalid_path"
    # Protected file inside the allowed root: denied before any spawn.
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "cat", "req-secret", args=[".env"])
    assert excinfo.value.code == "protected_path"


def test_run_string_slot_is_positioned_after_dashdash(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    result = _run(service, "grep", "req-grep", args=["-v", "notes.txt"])
    assert result["state"] == "running"
    # A value that looks like a flag stays data: it sits after "--".
    spec = spawner.spawns[-1]
    argv = spec["argv"]
    assert argv == [_GREP, "--", "-v", str(service.registry.by_id("work").path / "notes.txt")]


def test_run_spec_env_is_minimal(
    service: HostProcessService, spawner: FakeSpawner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAVIK_TEST_MARKER", "marker-secret")
    monkeypatch.setenv("HOME", "/home/ki")
    result = _run(service, "true", "req-env")
    assert result["state"] == "running"
    spec = spawner.spawns[-1]
    env = spec["env"]
    assert "SLAVIK_TEST_MARKER" not in env
    assert env["HOME"].endswith("/tmp")
    assert env["PATH"].startswith("/usr")
    assert set(env) <= {"PATH", "LANG", "LC_ALL", "TZ", "HOME", "TMPDIR"}


def test_run_spec_grants_exclude_protected_file(
    service: HostProcessService, spawner: FakeSpawner, work_root: Path
) -> None:
    _run(service, "true", "req-grants")
    spec = spawner.spawns[-1]
    grant_paths = {g["path"] for g in spec["grants"]}
    assert str(work_root / ".env") not in grant_paths
    assert str(work_root / "notes.txt") in grant_paths
    # EXECUTE only for the approved binary + its interpreter:
    # the system dirs are read-only, so no other binary can run.
    # (Grants also carry dev/ino identity for the helper; compare
    # path+access, not the whole dict.)
    grant_pairs = {(g["path"], g["access"]) for g in spec["grants"]}
    assert (str(Path(_TRUE).resolve()), "rx") in grant_pairs
    assert (str(Path("/lib64/ld-linux-x86-64.so.2").resolve()), "rx") in grant_pairs
    assert ("/usr", "rx") not in grant_pairs


def _no_rw_grant_covers(grants: list[dict[str, object]], target: Path) -> None:
    """No ``rw`` grant may be an ancestor of (or equal to) ``target``."""
    for grant in grants:
        if grant["access"] != "rw":
            continue
        grant_path = Path(str(grant["path"]))
        assert not (target == grant_path or target.is_relative_to(grant_path)), (
            f"rw grant {grant_path} covers write-excluded {target}"
        )


def _is_read_only_covered(grants: list[dict[str, object]], target: Path) -> bool:
    """True when ``target`` is covered by an ``r`` grant on itself or
    an ancestor (a clean excluded directory may be granted as one
    read-only rule instead of per-file rules)."""
    for grant in grants:
        if grant["access"] != "r":
            continue
        grant_path = Path(str(grant["path"]))
        if target == grant_path or target.is_relative_to(grant_path):
            return True
    return False


def test_cwd_grants_split_write_excluded_subtree(home: Path, tmp_path: Path) -> None:
    """A configured write_excluded subtree never gets a covering rw grant."""
    keep = home / "Project" / "work" / "keep"
    keep.mkdir()
    (keep / "data.txt").write_text("keep\n", encoding="utf-8")
    (home / "Project" / "work" / "open.txt").write_text("open\n", encoding="utf-8")
    registry = _registry_with_exclusions(home, (keep.resolve(),))
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        _run(svc, "true", "req-excl")
    finally:
        svc.shutdown()
    spec = spawner.spawns[-1]
    grants = spec["grants"]
    _no_rw_grant_covers(grants, keep / "data.txt")
    assert _is_read_only_covered(grants, keep / "data.txt")
    # The rest of the tree is still writable.
    assert any(
        g["access"] == "rw"
        and (home / "Project" / "work" / "open.txt").is_relative_to(Path(str(g["path"])))
        for g in grants
    )


def test_cwd_grants_carry_identity(home: Path, tmp_path: Path) -> None:
    """Every grant the daemon computes is bound to the (dev, ino)
    of the object it saw: the helper refuses a path whose object
    was swapped after the grant was computed (TOCTOU)."""
    spawner = FakeSpawner()
    registry = _registry_with_exclusions(home, ())
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        _run(svc, "true", "req-grant-identity")
    finally:
        svc.shutdown()
    for grant in spawner.spawns[-1]["grants"]:
        path = Path(str(grant["path"]))
        st = path.lstat()
        assert grant["dev"] == st.st_dev
        assert grant["ino"] == st.st_ino


def test_cwd_grant_refuses_protected_cwd(home: Path, tmp_path: Path) -> None:
    """A cwd that classifies as protected ("credentials") gets no
    grant at all: the walk classifies every child, never the cwd
    itself, so without this check one clean child would grant the
    whole protected directory as a single rw rule."""
    creds = home / "Project" / "work" / "credentials"
    creds.mkdir()
    (creds / "token.txt").write_text("token\n", encoding="utf-8")
    config = _write_config(
        home,
        _profiles(
            [
                {
                    "id": "creds",
                    "binary": _TRUE,
                    "cwd": {"root_id": "work", "path": "credentials"},
                }
            ]
        ),
    )
    registry = RootsRegistry(
        roots=(
            Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "creds", "req-creds")
        assert excinfo.value.code == "spawn_failed"
    finally:
        svc.shutdown()


def test_cwd_grants_split_nested_checkout(home: Path, tmp_path: Path) -> None:
    """A SlavikAI checkout nested inside the root is not rw-granted."""
    clone = home / "Project" / "work" / "clone"
    (clone / "host_bridge").mkdir(parents=True)
    (clone / "docs").mkdir()
    (clone / "host_bridge" / "daemon.py").write_text("x\n", encoding="utf-8")
    (clone / "docs" / "runtime_contract_claims.json").write_text("{}\n", encoding="utf-8")
    (clone / "code.py").write_text("code\n", encoding="utf-8")
    registry = _registry_with_exclusions(home, ())
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        _run(svc, "true", "req-clone")
    finally:
        svc.shutdown()
    grants = spawner.spawns[-1]["grants"]
    _no_rw_grant_covers(grants, clone / "code.py")
    assert _is_read_only_covered(grants, clone / "code.py")


def test_cwd_symlink_outside_root_rejected(home: Path, tmp_path: Path) -> None:
    """A cwd that is a symlink to a directory outside the root is refused."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / ".env").write_text("SECRET=x\n", encoding="utf-8")
    (outside / "data.txt").write_text("data\n", encoding="utf-8")
    link = home / "Project" / "work" / "link"
    link.symlink_to(outside, target_is_directory=True)
    config = _write_config(
        home,
        _profiles(
            [
                {
                    "id": "linked",
                    "binary": _TRUE,
                    "cwd": {"root_id": "work", "path": "link"},
                }
            ]
        ),
    )
    registry = RootsRegistry(
        roots=(
            Root(id="work", path=(home / "Project" / "work").resolve(), mode="rw"),
            Root(id="docs", path=(home / "Documents" / "docs").resolve(), mode="ro"),
        ),
        write_excluded=(),
        home=home.resolve(),
        config_path=config,
        socket_dir=None,
    )
    fs = HostFsService(registry, trash_root=tmp_path / "Trash")
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "linked", "req-link")
        assert excinfo.value.code == "spawn_failed"
    finally:
        svc.shutdown()
    assert not spawner.spawns, "no spec may be built for a symlinked cwd"


def test_cwd_grants_skip_bind_mounted_file(home: Path, tmp_path: Path) -> None:
    """A file bind-mounted into the cwd (same device, so st_dev cannot
    see it) gets no grant: the walk checks the real mount topology."""
    work = home / "Project" / "work"
    bound = work / "bound.txt"
    bound.write_text("outside\n", encoding="utf-8")
    (work / "open.txt").write_text("open\n", encoding="utf-8")
    registry = _registry_with_exclusions(home, ())
    fs = HostFsService(
        registry,
        trash_root=tmp_path / "Trash",
        mount_bindings_fn=lambda: [(bound.resolve(), bound.resolve())],
    )
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        _run(svc, "true", "req-bind")
    finally:
        svc.shutdown()
    grants = spawner.spawns[-1]["grants"]
    assert not any(
        bound == Path(str(g["path"])) or bound.is_relative_to(Path(str(g["path"]))) for g in grants
    ), "a bind-mounted file must never be granted"
    assert any(
        g["access"] == "rw" and (work / "open.txt").is_relative_to(Path(str(g["path"])))
        for g in grants
    )


def test_cwd_on_mount_point_rejected(home: Path, tmp_path: Path) -> None:
    """A cwd that is itself a mount point is refused, not granted."""
    work = home / "Project" / "work"
    mnt = work / "mnt"
    mnt.mkdir()
    (mnt / "data.txt").write_text("data\n", encoding="utf-8")
    profiles = _profiles(
        [{"id": "inmount", "binary": _TRUE, "cwd": {"root_id": "work", "path": "mnt"}}]
    )
    config = _write_config(home, profiles)
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
    fs = HostFsService(
        registry,
        trash_root=tmp_path / "Trash",
        mount_bindings_fn=lambda: [(mnt.resolve(), mnt.resolve())],
    )
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "inmount", "req-mnt")
        assert excinfo.value.code == "spawn_failed"
    finally:
        svc.shutdown()
    assert not spawner.spawns, "no spec may be built for a mounted cwd"


def test_run_spec_carries_binary_identity(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    _run(service, "true", "req-identity")
    spec = spawner.spawns[-1]
    canonical = Path(_TRUE).resolve()
    st = canonical.stat()
    assert spec["binary_dev"] == st.st_dev
    assert spec["binary_ino"] == st.st_ino
    assert spec["binary_size"] == st.st_size
    assert spec["binary_mtime_ns"] == st.st_mtime_ns


def test_run_idempotent_retry_returns_same_process_id(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    first = _run(service, "true", "req-same")
    second = _run(service, "true", "req-same")
    assert first["process_id"] == second["process_id"]
    assert len(spawner.spawns) == 1


def test_run_and_poll_finish(service: HostProcessService, spawner: FakeSpawner) -> None:
    result = _run(service, "true", "req-true")
    process_id = result["process_id"]
    assert result["state"] == "running"
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.finish(handle, state="exited", exit_code=0, stdout=b"ok\n")
    final = _wait_state(service, process_id, "exited")
    assert final["exit_code"] == 0
    assert final["stdout"] == "ok\n"
    assert final["elapsed_ms"] >= 0


# --- poll ------------------------------------------------------------------


def test_poll_offsets_and_retained_from(service: HostProcessService, spawner: FakeSpawner) -> None:
    result = _run(service, "true", "req-off")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    # Total 10 bytes, first 4 already dropped: tail holds bytes 4..9.
    spawner.set_output(handle, stdout=b"456789", stdout_total=10, stdout_retained_from=4)
    first = _poll(service, process_id, stdout_offset=0)
    assert first["stdout"] == "456789"
    assert first["stdout_next_offset"] == 10
    assert first["stdout_truncated"] is True
    assert first["stdout_retained_from"] == 4
    second = _poll(service, process_id, stdout_offset=7)
    assert second["stdout"] == "789"
    assert second["stdout_truncated"] is False
    with pytest.raises(ProcessError) as excinfo:
        _poll(service, process_id, stdout_offset=11)
    assert excinfo.value.code == "offset_out_of_range"


def test_poll_frame_budget(service: HostProcessService, spawner: FakeSpawner) -> None:
    result = _run(service, "true", "req-frame")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    # 40 KiB of '"' serializes to ~80 KiB: the poll must shrink it.
    spawner.set_output(handle, stdout=b'"' * 40_000)
    chunk = _poll(service, process_id, max_bytes=32 * 1024)
    assert len(encode_success(chunk, "frame-req")) - 1 <= MAX_FRAME_BYTES
    assert chunk["stdout_next_offset"] < 40_000
    assert chunk["stdout_next_offset"] > 0


def test_poll_utf8_replacement(service: HostProcessService, spawner: FakeSpawner) -> None:
    result = _run(service, "true", "req-utf8")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.set_output(handle, stdout=b"\xff\xfe hello")
    chunk = _poll(service, process_id)
    assert chunk["stdout"] == "\ufffd\ufffd hello"


def test_poll_pairs_tail_with_consistent_counters(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    """A tail newer than its status counters never yields an offset
    past the reported total (the next poll would fail with
    offset_out_of_range), and no bytes are lost between polls."""
    result = _run(service, "true", "req-snap")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    # The helper flushed a 6-byte tail; the status still says 3.
    spawner.set_output(handle, stdout=b"abcdef", stdout_total=3)
    first = _poll(service, process_id, stdout_offset=0)
    assert first["stdout_next_offset"] <= first["stdout_total_bytes"]
    assert first["stdout_next_offset"] > 0
    # The poll handed out above is usable as the next offset, and
    # the two polls together deliver every byte exactly once.
    spawner.set_output(handle, stdout=b"abcdef", stdout_total=6)
    second = _poll(service, process_id, stdout_offset=first["stdout_next_offset"])
    assert second["stdout_next_offset"] == 6
    assert first["stdout"] + second["stdout"] == "abcdef"


def test_poll_rotation_snapshot_is_generation_bound(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a 256 KiB tail rotation the tail keeps its length while
    its content changes, so a length check alone accepts a torn
    snapshot: an old status plus a new tail hands new bytes out at
    old absolute offsets (and repeats them on the next poll). The
    publication generation binds the tail to its counters: a poll
    that reads across a publication retries and answers from the new
    counters instead."""
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    # The reaper thread would consume scripted status reads (and
    # update the record's totals) out of order; park it — this test
    # drives _refresh explicitly. (Stopping it via _stop would make
    # _run treat the spawn as a late launch instead.)
    monkeypatch.setattr(svc, "_reap_once", lambda: None)
    try:
        result = _run(svc, "true", "req-gen")
        process_id = result["process_id"]
        handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
        # Rotated tail: A x 262140 then B x 4 (absolute 262144..262147).
        (handle.run_dir / "stdout").write_bytes(b"A" * 262140 + b"B" * 4)
        (handle.run_dir / "stderr").write_bytes(b"")
        old = {
            "version": 1,
            "state": "running",
            "stdout_total_bytes": 262144,
            "stdout_retained_from": 0,
            "stderr_total_bytes": 0,
            "stderr_retained_from": 0,
            "generation": 1,
        }
        new = dict(old, stdout_total_bytes=262148, stdout_retained_from=4, generation=2)
        script = [old, old, new, new, new, new, new, new]
        calls = {"n": 0}

        def _scripted_status(run_dir: Path) -> dict[str, object] | None:
            index = min(calls["n"], len(script) - 1)
            calls["n"] += 1
            return dict(script[index])

        monkeypatch.setattr(process_module, "_read_status_file", _scripted_status)
        # Absolute offset 262140 is the last four A's; with the torn
        # snapshot (old counters, new tail) it would read BBBB.
        polled = _poll(svc, process_id, stdout_offset=262140, max_bytes=4)
        assert polled["stdout"] == "AAAA"
        assert polled["stdout_next_offset"] == 262144
        assert polled["stdout_total_bytes"] == 262148
        assert polled["stdout_retained_from"] == 4
        # And the new bytes follow exactly once, at their own offsets.
        following = _poll(svc, process_id, stdout_offset=262144, max_bytes=4)
        assert following["stdout"] == "BBBB"
        assert following["stdout_next_offset"] == 262148
    finally:
        svc.shutdown()


def test_poll_frame_budget_keeps_source_byte_offsets(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    """Trimming by the frame budget must not miscount consumed source
    bytes: one 0xff byte decodes to one U+FFFD, not three UTF-8 bytes."""
    result = _run(service, "true", "req-bytes")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.set_output(handle, stdout=b"\xff" * 40_000, stderr=b"\xff" * 40_000)
    first = _poll(service, process_id, max_bytes=32 * 1024)
    assert 0 < first["stdout_next_offset"] < 40_000
    assert first["stdout_next_offset"] == len(first["stdout"])
    assert 0 < first["stderr_next_offset"] < 40_000
    assert first["stderr_next_offset"] == len(first["stderr"])
    # No bytes are skipped: the next chunk continues exactly where
    # the first one stopped, on both streams.
    second = _poll(
        service,
        process_id,
        stdout_offset=first["stdout_next_offset"],
        stderr_offset=first["stderr_next_offset"],
        max_bytes=32 * 1024,
    )
    assert second["stdout"]
    assert second["stdout_next_offset"] == first["stdout_next_offset"] + len(second["stdout"])
    assert second["stderr"]
    assert second["stderr_next_offset"] == first["stderr_next_offset"] + len(second["stderr"])


def test_poll_chunk_eof_is_per_slice(service: HostProcessService, spawner: FakeSpawner) -> None:
    """A max_bytes cut through a multibyte character must not consume
    the incomplete prefix: EOF is a property of the byte slice, not
    of the process state."""
    result = _run(service, "true", "req-slice")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.finish(handle, state="exited", exit_code=0, stdout="a€b".encode())
    _wait_state(service, process_id, "exited")
    first = _poll(service, process_id, max_bytes=3)
    assert first["stdout"] == "a"
    assert first["stdout_next_offset"] == 1
    second = _poll(service, process_id, stdout_offset=first["stdout_next_offset"], max_bytes=3)
    assert second["stdout"] == "€"
    assert second["stdout_next_offset"] == 4
    third = _poll(service, process_id, stdout_offset=second["stdout_next_offset"], max_bytes=3)
    assert third["stdout"] == "b"
    assert third["stdout_next_offset"] == 5
    assert first["stdout"] + second["stdout"] + third["stdout"] == "a€b"


def test_decode_poll_chunk_keeps_suffix_after_invalid_byte() -> None:
    """An invalid byte must not eat a following incomplete sequence:
    the decoder replaces only the bad byte and holds the suffix back
    for the next chunk."""
    decode = process_module._decode_poll_chunk
    assert decode(b"\xff\xe2\x82", at_eof=False) == ("\ufffd", 1)
    assert decode(b"\xe2\x82\xac", at_eof=True) == ("€", 3)
    assert decode(b"\xe2\x82", at_eof=False) == ("", 0)
    assert decode(b"\xe2\x82", at_eof=True) == ("\ufffd", 2)
    assert decode(b"a\xffb", at_eof=False) == ("a\ufffdb", 3)


def test_poll_invalid_byte_before_incomplete_char(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    """Through the service: stdout b"\\xff\\xe2\\x82\\xac" polled
    with max_bytes=3 must give "\\ufffd" then "€" — the correct
    character after the bad byte must survive the chunk boundary,
    not be replaced along with it."""
    result = _run(service, "true", "req-bad-byte")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.finish(handle, state="exited", exit_code=0, stdout=b"\xff\xe2\x82\xac")
    _wait_state(service, process_id, "exited")
    first = _poll(service, process_id, max_bytes=3)
    assert first["stdout"] == "\ufffd"
    assert first["stdout_next_offset"] == 1
    second = _poll(service, process_id, stdout_offset=first["stdout_next_offset"], max_bytes=3)
    assert second["stdout"] == "€"
    assert second["stdout_next_offset"] == 4
    assert first["stdout"] + second["stdout"] == "\ufffd€"


def test_poll_max_bytes_smaller_than_char(
    service: HostProcessService, spawner: FakeSpawner
) -> None:
    """A max_bytes smaller than the next UTF-8 character must not
    spin forever returning '' at offset 0: it is a typed
    too-small-limit error (the FS read precedent)."""
    result = _run(service, "true", "req-small-limit")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.finish(handle, state="exited", exit_code=0, stdout="€".encode())
    _wait_state(service, process_id, "exited")
    with pytest.raises(ProcessError) as excinfo:
        _poll(service, process_id, max_bytes=1)
    assert excinfo.value.code == "invalid_params"
    assert "too small" in str(excinfo.value)


def test_poll_frame_budget_trim_never_stalls(
    service: HostProcessService,
    spawner: FakeSpawner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame budget smaller than one character must not truncate a
    poll to an empty, non-advancing chunk: the trim stops at the last
    readable character instead of returning '' at offset 0."""

    def _tiny_budget(result: dict[str, object], request_id: str) -> bool:
        return (
            len(str(result.get("stdout", "")).encode("utf-8")) <= 1
            and len(str(result.get("stderr", "")).encode("utf-8")) <= 1
        )

    monkeypatch.setattr(process_module, "_frame_fits", _tiny_budget)
    result = _run(service, "true", "req-tiny-budget")
    process_id = result["process_id"]
    handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
    spawner.finish(handle, state="exited", exit_code=0, stdout="€".encode())
    _wait_state(service, process_id, "exited")
    polled = _poll(service, process_id)
    assert polled["stdout"] == "€"
    assert polled["stdout_next_offset"] == 3


def test_poll_unknown_process(service: HostProcessService) -> None:
    with pytest.raises(ProcessError) as excinfo:
        _poll(service, "deadbeef")
    assert excinfo.value.code == "process_not_found"


# --- kill / caps / retention ----------------------------------------------


def test_kill_is_idempotent(service: HostProcessService, spawner: FakeSpawner) -> None:
    result = _run(service, "sleep", "req-kill", args=["5"])
    process_id = result["process_id"]
    killed = _kill(service, process_id)
    assert killed["state"] == "killed"
    again = _kill(service, process_id)
    assert again["state"] == "killed"
    assert again["exit_code"] == killed["exit_code"]
    with pytest.raises(ProcessError) as excinfo:
        _kill(service, "deadbeef")
    assert excinfo.value.code == "process_not_found"


def test_concurrency_cap(service: HostProcessService) -> None:
    ids = [_run(service, "sleep", f"req-cap-{i}", args=["5"])["process_id"] for i in range(4)]
    assert len(ids) == 4
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "sleep", "req-cap-5", args=["5"])
    assert excinfo.value.code == "too_many_processes"
    for process_id in ids:
        _kill(service, process_id)


def test_timeout_finalizes_via_reaper(
    service: HostProcessService,
    spawner: FakeSpawner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(process_module, "KILL_GRACE_S", 0.05)
    spawner.finish_state = "timeout"
    result = _run(service, "sleep", "req-timeout", args=["5"], timeout_s=1)
    process_id = result["process_id"]
    final = _wait_state(service, process_id, "timeout", timeout_s=8.0)
    assert final["state"] == "timeout"
    assert spawner.kill_calls, "the reaper must kill the tree on timeout"


class _StubbornSpawner(FakeSpawner):
    """A spawner whose cgroup never empties: kill_all records the
    call but leaves the unit's procs alone, so the record stays in
    ``killing`` and a repeat kill must not move its deadline."""

    def kill_all(self, handle: SpawnHandle) -> None:
        self.kill_calls.append(handle.unit_name)


def test_repeated_kill_preserves_original_deadline(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repeat kill observes the already-started termination: it
    must not move the original kill deadline, overwrite the reason,
    or re-signal the tree (a descendant ignoring SIGTERM would
    otherwise never reach its cgroup SIGKILL)."""
    monkeypatch.setattr(process_module, "KILL_GRACE_S", 0.3)
    spawner = _StubbornSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        result = _run(svc, "sleep", "req-kill-twice", args=["5"])
        process_id = result["process_id"]
        first = svc.handle("host.process.kill", {"process_id": process_id}, "kill-1", _token(1.0))
        assert first["state"] == "killing"
        record = svc._records[process_id]
        deadline = record.kill_deadline
        assert deadline is not None
        assert record.kill_reason == "killed"
        signals = len(spawner.signal_calls)
        second = svc.handle("host.process.kill", {"process_id": process_id}, "kill-2", _token(1.0))
        assert second["state"] == "killing"
        assert record.kill_deadline == deadline
        assert record.kill_reason == "killed"
        assert len(spawner.signal_calls) == signals
    finally:
        svc.shutdown()


def test_kill_after_timeout_keeps_timeout_reason(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A kill that arrives while a timeout kill is already in
    progress must not overwrite the ``timeout`` reason or move the
    deadline the reaper set."""
    monkeypatch.setattr(process_module, "KILL_GRACE_S", 0.3)
    spawner = _StubbornSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        result = _run(svc, "sleep", "req-kill-timeout", args=["5"], timeout_s=1)
        process_id = result["process_id"]
        record = svc._records[process_id]
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            if record.state == "killing" and record.kill_reason == "timeout":
                break
            time.sleep(0.02)
        assert record.state == "killing"
        assert record.kill_reason == "timeout"
        original_deadline = record.kill_deadline
        assert original_deadline is not None
        signals = len(spawner.signal_calls)
        killed = svc.handle("host.process.kill", {"process_id": process_id}, "kill-3", _token(0.5))
        assert killed["state"] == "killing"
        assert record.kill_reason == "timeout"
        assert record.kill_deadline == original_deadline
        assert len(spawner.signal_calls) == signals
    finally:
        svc.shutdown()


def test_reaper_timeout_after_natural_exit_keeps_exit_code(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that exits 0 just before its timeout deadline must
    still report ``exited`` with that code when the reaper's first
    iteration after the deadline runs: setting ``kill_requested``
    before checking natural completion lets ``_try_finalize`` replace
    the helper's published exit with ``timeout`` and no exit code,
    turning a successful command into an error depending on reaper
    scheduling."""
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    # The background reaper would finalize the scripted exit itself
    # before the deadline is moved; park it and drive the real
    # _reap_once explicitly via the class (the instance attribute is
    # the parked no-op).
    monkeypatch.setattr(svc, "_reap_once", lambda: None)
    try:
        result = _run(svc, "sleep", "req-exit-before-timeout", args=["5"], timeout_s=1)
        process_id = result["process_id"]
        handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
        # The process exits 0 just before its 1 s deadline; the
        # reaper runs 0.15 s after it.
        spawner.finish(handle, state="exited", exit_code=0)
        record = svc._records[process_id]
        record.started_mono = time.monotonic() - 1.15
        HostProcessService._reap_once(svc)
        assert record.final
        assert record.state == "exited"
        assert record.exit_code == 0
        assert not spawner.signal_calls, "no SIGTERM may be sent to a finished tree"
        assert not spawner.kill_calls, "no cgroup kill may run for a finished tree"
    finally:
        svc.shutdown()


class _CrashedHelperSpawner(FakeSpawner):
    """A spawner whose leader exits at once, no status file is ever
    published, and the cgroup stays non-empty until kill_all: the
    helper crashed after the target started, before its first
    status.json."""

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spawns.append(spec)
        leader = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        handle = SpawnHandle(popen=leader, unit_name=unit_name, cgroup_path=None, run_dir=run_dir)
        self.units[unit_name] = {"handle": handle, "procs": [str(leader.pid)]}
        leader.kill()
        leader.wait()
        return handle

    def kill_all(self, handle: SpawnHandle) -> None:
        self.kill_calls.append(handle.unit_name)
        unit = self.units.get(handle.unit_name)
        if unit is not None:
            unit["procs"] = []
        # No status file is ever written: the helper died before its
        # first publication, so the reaper must finalize from the
        # cgroup state alone.


def test_helper_crash_before_first_status_kills_tree(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
) -> None:
    """If the helper dies after the target started but before its
    first status.json, the record must not finalize while the cgroup
    is non-empty: the tree is destroyed first, and the reaper
    finalizes once the cgroup is empty. Finalizing on leader death
    would leave descendants outside every cleanup path."""
    spawner = _CrashedHelperSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "true", "req-crash")
        assert excinfo.value.code == "spawn_failed"
        process_id = svc._by_request_id["req-crash"]
        record = svc._records[process_id]
        assert not record.final
        assert record.state == "killing"
        assert spawner.kill_calls, "the tree was never destroyed"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not record.final:
            svc._reap_once()
            time.sleep(0.02)
        assert record.final
        assert record.state == "killed"
    finally:
        svc.shutdown()


class _SignalTimeoutSpawner(FakeSpawner):
    """A spawner whose SIGTERM send times out (systemctl hangs):
    the launch deadline expires, the signal never lands, and the
    tree must stay under reaper control — not finalize while alive."""

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spawns.append(spec)
        leader = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        handle = SpawnHandle(popen=leader, unit_name=unit_name, cgroup_path=None, run_dir=run_dir)
        self.units[unit_name] = {"handle": handle, "procs": [str(leader.pid)]}
        # No status file is ever written: the launch never confirms.
        return handle

    def signal_tree(self, handle: SpawnHandle, sig: int) -> None:
        self.signal_calls.append((handle.unit_name, sig))
        raise subprocess.TimeoutExpired("systemctl", 5)


def test_signal_timeout_after_spawn_keeps_tree_under_cleanup(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SIGTERM send times out while the spawn deadline
    expires, the record must not finalize while the tree may still
    be alive: a transient signal error must not switch off the
    120 s enforcement. The reaper destroys the tree after the grace
    period and finalizes once the cgroup is empty."""
    monkeypatch.setattr(process_module, "SPAWN_START_TIMEOUT_S", 0.5)
    monkeypatch.setattr(process_module, "KILL_GRACE_S", 0.3)
    spawner = _SignalTimeoutSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "true", "req-signal-timeout")
        assert excinfo.value.code == "spawn_failed"
        process_id = svc._by_request_id["req-signal-timeout"]
        record = svc._records[process_id]
        assert not record.final
        assert record.state == "killing"
        assert spawner.signal_calls, "the SIGTERM attempt never happened"
        assert not spawner.kill_calls, "the tree was destroyed before the grace period"
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not record.final:
            svc._reap_once()
            time.sleep(0.02)
        assert record.final
        assert record.state == "killed"
        assert spawner.kill_calls, "the tree was never destroyed"
    finally:
        svc.shutdown()


def test_finished_record_cap(service: HostProcessService, spawner: FakeSpawner) -> None:
    from host_bridge.process import MAX_FINISHED_RECORDS

    ids: list[str] = []
    for i in range(MAX_FINISHED_RECORDS + 1):
        result = _run(service, "true", f"req-fin-{i}")
        process_id = result["process_id"]
        handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
        spawner.finish(handle, state="exited", exit_code=0)
        # The reaper finalizes asynchronously; wait for it before the
        # next run, or the concurrency cap (4 running) trips first.
        _wait_state(service, process_id, "exited")
        ids.append(process_id)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        service._sweep_finished()
        finished = [r for r in service._records.values() if r.final]
        if len(finished) <= MAX_FINISHED_RECORDS:
            break
        time.sleep(0.05)
    finished = [r for r in service._records.values() if r.final]
    assert len(finished) <= MAX_FINISHED_RECORDS
    assert ids[0] not in service._records  # the oldest was evicted


def test_cancel_at_launch_boundary(
    service: HostProcessService,
    spawner: FakeSpawner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel racing spawn: exactly one process_id, no duplicate spawn."""
    monkeypatch.setattr(process_module, "SPAWN_START_TIMEOUT_S", 0.5)
    spawner.delay_running = 1.0
    token = CancelToken(deadline=time.monotonic() + 0.3, request_id="req-cancel")
    with pytest.raises(RequestTimeoutError):
        service.handle("host.process.run", {"profile_id": "true"}, "req-cancel", token)
    # Exactly one record exists for that request_id, and no second
    # spawn happened even after the reaper settles the stalled launch.
    assert len(spawner.spawns) == 1
    process_id = service._by_request_id["req-cancel"]
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not service._records[process_id].final:
        time.sleep(0.05)
    assert service._records[process_id].final
    # Either outcome is exactly one process for the request: the
    # launch may settle as spawn_failed (stalled) or as killed (the
    # deadline path cancelled it first) — what must never happen is
    # a second process or a second process_id for this request.
    assert service._records[process_id].state in ("spawn_failed", "killed")
    # Retry after the lost response returns the same process_id.
    retry = service.handle("host.process.run", {"profile_id": "true"}, "req-cancel", _token())
    assert retry["process_id"] == process_id
    assert len(spawner.spawns) == 1


class _ExitingLeaderSpawner:
    """Spawn returns a handle whose leader exits at once without ever
    writing a status file: ``_spawn`` finalizes the launch as
    ``spawn_failed`` through its ``leader_done()`` branch — and that
    finalization needs the service lock."""

    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []
        self.units: dict[str, dict[str, object]] = {}
        self.kill_calls: list[str] = []
        self.signal_calls: list[tuple[str, int]] = []

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spawns.append(spec)
        leader = subprocess.Popen(
            ["sleep", "0.5"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        handle = SpawnHandle(popen=leader, unit_name=unit_name, cgroup_path=None, run_dir=run_dir)
        self.units[unit_name] = {"handle": handle, "procs": [str(leader.pid)]}
        return handle

    def signal_tree(self, handle: SpawnHandle, sig: int) -> None:
        self.signal_calls.append((handle.unit_name, sig))

    def kill_all(self, handle: SpawnHandle) -> None:
        self.kill_calls.append(handle.unit_name)
        if handle.popen is not None and handle.popen.poll() is None:
            handle.popen.kill()

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        return True


def test_retry_run_does_not_hold_lock_while_waiting(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
) -> None:
    """A retry that waits for a launch in flight must not hold the
    service lock: the failing launch finalizes through that lock."""
    spawner = _ExitingLeaderSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    errors: list[BaseException] = []
    results: list[dict[str, object]] = []

    def first_run() -> None:
        try:
            _run(svc, "true", "req-lock")
        except BaseException as exc:  # noqa: BLE001 - captured for the test
            errors.append(exc)

    def retry_run() -> None:
        token = CancelToken(deadline=time.monotonic() + 2.0, request_id="req-lock")
        try:
            results.append(
                svc.handle("host.process.run", {"profile_id": "true"}, "req-lock", token)
            )
        except BaseException as exc:  # noqa: BLE001 - captured for the test
            errors.append(exc)

    try:
        first = threading.Thread(target=first_run, daemon=True)
        first.start()
        # The first launch is in flight (its leader exits at 0.5 s);
        # the retry arrives while it is still launching.
        time.sleep(0.1)
        retry = threading.Thread(target=retry_run, daemon=True)
        retry.start()
        retry.join(timeout=1.5)
        assert not retry.is_alive(), "retry is still waiting: it holds the lock _finalize needs"
        assert results and results[0]["state"] == "spawn_failed"
        first.join(timeout=5.0)
        assert errors and isinstance(errors[0], ProcessError)
        assert errors[0].code == "spawn_failed"
    finally:
        svc.shutdown()


class _DelayedSpawner(FakeSpawner):
    """FakeSpawner whose spawn() returns late (after the launch timeout)."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.spawn_delay = delay

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        time.sleep(self.spawn_delay)
        return super().spawn(spec_path, unit_name=unit_name, run_dir=run_dir)


def test_late_spawn_handle_is_destroyed_not_adopted(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handle that arrives after the reaper gave up on the launch is
    destroyed, not adopted into the final record."""
    monkeypatch.setattr(process_module, "SPAWN_START_TIMEOUT_S", 0.2)
    spawner = _DelayedSpawner(delay=0.6)
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "true", "req-late")
        assert excinfo.value.code == "spawn_failed"
        process_id = svc._by_request_id["req-late"]
        record = svc._records[process_id]
        assert record.final
        assert record.state == "spawn_failed"
        assert spawner.kill_calls == [f"slavikai-bridge-{process_id}.scope"]
        handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
        assert handle.popen is None or handle.popen.poll() is not None
        # No resurrection: the record stays spawn_failed.
        time.sleep(0.5)
        assert svc._records[process_id].state == "spawn_failed"
    finally:
        svc.shutdown()
        for unit in spawner.units.values():
            handle = unit["handle"]
            if handle.popen is not None and handle.popen.poll() is None:
                handle.popen.kill()


def test_shutdown_destroys_inflight_launch(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
) -> None:
    """A handle that arrives after shutdown began is destroyed, not
    adopted: no process may report running after the service stopped."""
    spawner = _DelayedSpawner(delay=0.6)
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            _run(svc, "true", "req-shutdown")
        except BaseException as exc:  # noqa: BLE001 - captured for the test
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        time.sleep(0.15)
        svc.shutdown()
        thread.join(timeout=5.0)
        assert not thread.is_alive(), "the launch did not return after shutdown"
        assert errors and isinstance(errors[0], ProcessError)
        assert errors[0].code == "spawn_failed"
        process_id = svc._by_request_id["req-shutdown"]
        record = svc._records[process_id]
        assert record.final
        assert record.state == "spawn_failed"
        assert spawner.kill_calls == [f"slavikai-bridge-{process_id}.scope"]
        handle = spawner.units[f"slavikai-bridge-{process_id}.scope"]["handle"]
        deadline = time.monotonic() + 2.0
        while (
            handle.popen is not None and handle.popen.poll() is None and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert handle.popen is None or handle.popen.poll() is not None
    finally:
        svc.shutdown()
        for unit in spawner.units.values():
            handle = unit["handle"]
            if handle.popen is not None and handle.popen.poll() is None:
                handle.popen.kill()


def test_run_rejects_missing_required_arg_after_optional(
    service: HostProcessService,
) -> None:
    """An optional slot before a required one must not swallow the
    required arg: args=['ok'] leaves the required int slot empty."""
    with pytest.raises(ProcessError) as excinfo:
        _run(service, "optreq", "req-opt-missing", args=["ok"])
    assert excinfo.value.code == "invalid_args"
    result = _run(service, "optreq", "req-opt-ok", args=["ok", "3"])
    assert result["state"] == "running"


class _RaisingSpawner:
    """spawn() always fails before any handle exists."""

    def __init__(self) -> None:
        self.spawns: list[dict[str, object]] = []

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        raise SpawnError("scope creation failed")

    def signal_tree(self, handle: SpawnHandle, sig: int) -> None:
        raise AssertionError("signal_tree must not be called without a handle")

    def kill_all(self, handle: SpawnHandle) -> None:
        raise AssertionError("kill_all must not be called without a handle")

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        return True


def test_failed_spawn_run_dir_is_cleaned(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run directory left by a spawn that failed before the handle
    is still removed when the record is swept."""
    monkeypatch.setattr(process_module, "SPAWN_START_TIMEOUT_S", 0.2)
    monkeypatch.setattr(process_module, "FINISHED_RETENTION_S", 0.0)
    spawner = _RaisingSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        with pytest.raises(ProcessError) as excinfo:
            _run(svc, "true", "req-nodir")
        assert excinfo.value.code == "spawn_failed"
        process_id = svc._by_request_id["req-nodir"]
        run_dir = svc._run_base / process_id
        assert (run_dir / "spec.json").exists()
        assert (run_dir / "tmp").is_dir()
        # The reaper finalizes the stalled launch, then sweeps it.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and process_id in svc._records:
            time.sleep(0.05)
        assert process_id not in svc._records
        assert not run_dir.exists()
    finally:
        svc.shutdown()


def test_early_spawn_error_finalizes_immediately(
    fs_service: HostFsService,
    registry: RootsRegistry,
    tmp_path: Path,
) -> None:
    """A spawn error raised before any process exists (here: the cwd
    vanished) settles the record at once instead of holding an active
    slot until the spawn timeout."""
    work = registry.by_id("work")
    assert work is not None
    shutil.rmtree(work.path)
    spawner = FakeSpawner()
    svc = _service_with_spawner(fs_service, registry, tmp_path, spawner)
    try:
        for i in range(4):
            with pytest.raises(ProcessError) as excinfo:
                _run(svc, "true", f"req-early-{i}")
            assert excinfo.value.code == "spawn_failed"
            record = svc._records[svc._by_request_id[f"req-early-{i}"]]
            assert record.final
            assert record.state == "spawn_failed"
        assert not spawner.spawns, "no process may be created for a deleted cwd"
        # The slots are free again: restore the cwd and run a valid profile.
        work.path.mkdir(parents=True)
        (work.path / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
        result = _run(svc, "true", "req-early-ok")
        assert result["state"] == "running"
    finally:
        svc.shutdown()


# --- wire: capabilities visibility + dispatch ------------------------------


@pytest.fixture()
def bridge(
    tmp_path: Path, fs_service: HostFsService, service: HostProcessService
) -> Iterator[Path]:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=fs_service, process_service=service)
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        yield socket_path
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_capabilities_show_profiles_when_available(bridge: Path) -> None:
    caps = client.capabilities(bridge)
    assert "host.process.run" in caps
    assert "host.process.poll" in caps
    assert "host.process.kill" in caps
    payload = client.call(bridge, "bridge.capabilities")
    profiles = payload["profiles"]
    assert {"id": "true", "args": []} in profiles
    assert "usr/bin" not in json.dumps(profiles)


def test_capabilities_hide_process_when_probe_failed(
    tmp_path: Path, fs_service: HostFsService, registry: RootsRegistry
) -> None:
    svc = HostProcessService(
        fs_service,
        registry,
        runtime_dir=tmp_path / "run2",
        profiles=load_profiles(registry),
        spawner=FakeSpawner(),  # type: ignore[arg-type]
    )
    svc._probe = ProbeResult(available=False, checks={"probe": "failed"}, reason="probe failed")
    try:
        socket_path = tmp_path / "slavikai2" / "host-bridge.sock"
        daemon = HostBridgeDaemon(socket_path, fs_service=fs_service, process_service=svc)
        daemon.start()
        thread = threading.Thread(target=daemon.serve_forever, daemon=True)
        thread.start()
        try:
            caps = client.capabilities(socket_path)
            assert "host.process.run" not in caps
            payload = client.call(socket_path, "bridge.capabilities")
            assert payload["profiles"] == []
            with pytest.raises(client.BridgeClientError) as excinfo:
                client.call(
                    socket_path,
                    "host.process.run",
                    params={"profile_id": "true"},
                )
            assert excinfo.value.code == "process_unavailable"
        finally:
            daemon.shutdown()
            thread.join(timeout=2.0)
    finally:
        svc.shutdown()


def test_wire_run_poll_kill(bridge: Path) -> None:
    run = client.call(
        bridge, "host.process.run", params={"profile_id": "true"}, request_id="wire-1"
    )
    process_id = run["process_id"]
    assert run["state"] == "running"
    again = client.call(
        bridge, "host.process.run", params={"profile_id": "true"}, request_id="wire-1"
    )
    assert again["process_id"] == process_id
    killed = client.call(bridge, "host.process.kill", params={"process_id": process_id})
    assert killed["state"] == "killed"
    polled = client.call(bridge, "host.process.poll", params={"process_id": process_id})
    assert polled["state"] == "killed"


def test_wire_fs_validator_error_keeps_request_id(bridge: Path) -> None:
    """Errors from the shared FS validators keep their request_id, so
    the client reports invalid_params instead of malformed_response."""
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(
            bridge,
            "host.process.kill",
            params={"process_id": 42, "extra": 1},
            request_id="wire-fs-err",
        )
    assert excinfo.value.code == "invalid_params"


# --- supervisor (spawn_helper, direct) -------------------------------------


def _supervisor_spec(
    run_dir: Path, cwd: Path, code: str, *, timeout_s: int = 10
) -> dict[str, object]:
    """A minimal spec for running ``_Supervisor`` directly (no systemd)."""
    from host_bridge.process import _BASE_ENV, _exec_grants, _system_grants

    binary = Path(_PY).resolve()
    st = binary.stat()
    grants = _system_grants() + _exec_grants(binary)
    return {
        "version": 1,
        "process_id": "supervisor-test",
        "binary": str(binary),
        "binary_dev": st.st_dev,
        "binary_ino": st.st_ino,
        "binary_size": st.st_size,
        "binary_mtime_ns": st.st_mtime_ns,
        "argv": [str(binary), "-c", code],
        "env": dict(_BASE_ENV),
        "cwd": str(cwd),
        "timeout_s": timeout_s,
        "run_dir": str(run_dir),
        "grants": [{"path": str(g.path), "access": g.access} for g in grants],
        "outer_uid": os.getuid(),
        "outer_gid": os.getgid(),
    }


def _run_supervisor(spec: dict[str, object], run_dir: Path) -> tuple[int, float]:
    """Run ``_Supervisor`` in the main thread (it installs signal
    handlers); return (exit code, elapsed s)."""
    import signal as _signal

    from host_bridge.spawn_helper import _Supervisor

    supervisor = _Supervisor(spec, run_dir)
    saved = {sig: _signal.getsignal(sig) for sig in (_signal.SIGTERM, _signal.SIGINT)}
    started = time.monotonic()
    try:
        code = supervisor.run()
    finally:
        for sig, handler in saved.items():
            _signal.signal(sig, handler)
    return code, time.monotonic() - started


def test_supervisor_no_busy_loop_when_streams_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child that closes both streams but keeps running must not
    spin the supervisor on wait4 + tail flushes with fsync."""
    from host_bridge.spawn_helper import _Supervisor, landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    spec = _supervisor_spec(
        run_dir, cwd, "import os, time; os.close(1); os.close(2); time.sleep(3)"
    )
    calls: list[int] = []
    original = _Supervisor._flush_tails

    def counting(self: _Supervisor) -> None:
        calls.append(1)
        original(self)

    monkeypatch.setattr(_Supervisor, "_flush_tails", counting)
    code, elapsed = _run_supervisor(spec, run_dir)
    assert code == 0
    assert elapsed >= 2.5, "the supervisor returned before the child finished"
    assert len(calls) < 50, f"supervisor flushed tails {len(calls)} times in {elapsed:.1f} s"


def test_supervisor_keeps_reading_after_leader_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descendant that outlives the leader still gets its output
    read: the supervisor must not close the pipes on wait4(leader)."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    code = (
        "import os, time; pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(0.7)\n"
        "    os.write(1, b'descendant-output\\n')\n"
        "    os._exit(0)\n"
        "os._exit(0)\n"
    )
    spec = _supervisor_spec(run_dir, cwd, code)
    exit_code, elapsed = _run_supervisor(spec, run_dir)
    assert exit_code == 0
    assert elapsed >= 0.6, "the supervisor returned before the descendant wrote"
    assert (run_dir / "stdout").read_bytes() == b"descendant-output\n"


def _grant_identity(path: Path) -> tuple[int, int]:
    st = path.lstat()
    return st.st_dev, st.st_ino


def test_open_grant_fd_rejects_symlink_components(tmp_path: Path) -> None:
    """_open_grant_fd opens each component with O_NOFOLLOW: a symlink
    anywhere in the path fails instead of following it, so a grant
    can never be bound to a swapped-in link target (a symlink to a
    directory under O_NOFOLLOW|O_DIRECTORY raises ENOTDIR, a symlink
    to a file raises ELOOP — both refuse the path)."""
    from host_bridge.spawn_helper import _open_grant_fd

    real = tmp_path / "real"
    (real / "sub").mkdir(parents=True)
    (real / "sub" / "data.txt").write_text("ok\n", encoding="utf-8")
    fd = _open_grant_fd(real / "sub" / "data.txt")
    try:
        st = os.fstat(fd)
        assert (st.st_dev, st.st_ino) == _grant_identity(real / "sub" / "data.txt")
    finally:
        os.close(fd)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError) as excinfo:
        _open_grant_fd(tmp_path / "link" / "data.txt")
    assert excinfo.value.errno in (errno.ELOOP, errno.ENOTDIR)
    (real / "alias").symlink_to(real / "sub" / "data.txt")
    with pytest.raises(OSError) as excinfo:
        _open_grant_fd(real / "alias")
    assert excinfo.value.errno == errno.ELOOP


def test_landlock_grant_ignores_swapped_symlink(tmp_path: Path) -> None:
    """A subdir swapped for a symlink after the grant was computed
    must not hand out the link target: the grant is skipped and the
    outside file stays denied (static symlinks were already skipped;
    this pins the behaviour for the swapped case)."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work = tmp_path / "work"
    sub = work / "sub"
    sub.mkdir(parents=True)
    (sub / "data.txt").write_text("ok\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("SECRET\n", encoding="utf-8")
    dev, ino = _grant_identity(sub)
    spec = _supervisor_spec(
        run_dir, work, "import sys; sys.stdout.write(open('sub/data.txt').read())"
    )
    spec["grants"] = [
        *spec["grants"],
        {"path": str(sub), "access": "rw", "dev": dev, "ino": ino},
    ]
    # Control: the verified directory is granted.
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 0
    assert (run_dir / "stdout").read_bytes() == b"ok\n"
    # Attack: swap the granted subdir for a symlink to the outside
    # tree after the grant (and its identity) was computed.
    shutil.rmtree(sub)
    sub.symlink_to(outside, target_is_directory=True)
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 0
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["exit_code"] == 1
    assert b"SECRET" not in (run_dir / "stdout").read_bytes()


def test_landlock_grant_refuses_replaced_object(tmp_path: Path) -> None:
    """A granted path replaced by a different real directory (same
    path, different inode) fails the whole spawn instead of granting
    the replacement: the grant is bound to the recorded identity."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work = tmp_path / "work"
    sub = work / "sub"
    sub.mkdir(parents=True)
    (sub / "data.txt").write_text("ok\n", encoding="utf-8")
    dev, ino = _grant_identity(sub)
    spec = _supervisor_spec(
        run_dir, work, "import sys; sys.stdout.write(open('sub/data.txt').read())"
    )
    spec["grants"] = [
        *spec["grants"],
        {"path": str(sub), "access": "rw", "dev": dev, "ino": ino},
    ]
    # Swap in a different directory object via rename, not
    # rmtree+mkdir: a fresh mkdir can reuse the freed inode on some
    # filesystems (CI), which makes the replacement identical by
    # (dev, ino) and the grant legitimately accepted. A renamed
    # directory keeps its own inode, so the two objects are
    # guaranteed different.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / "data.txt").write_text("EVIL\n", encoding="utf-8")
    shutil.rmtree(sub)
    os.rename(replacement, sub)
    assert _grant_identity(sub) != (dev, ino)
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 1
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "spawn_failed"
    # A failed spawn never writes a stdout tail: the replacement
    # directory was never granted, so its content cannot leak.
    assert not (run_dir / "stdout").exists() or b"EVIL" not in (run_dir / "stdout").read_bytes()


def test_landlock_mkfifo_denied_even_in_granted_cwd(tmp_path: Path) -> None:
    """mkfifo inside a rw-granted cwd is refused: MAKE_FIFO is in
    the handled mask with no allowing grant. Unlisted handled rights
    are allowed by Landlock by default, so the mask must list it —
    otherwise a profile could create a named pipe in any
    owner-accessible directory."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    spec = _supervisor_spec(
        run_dir,
        work,
        "import os\n"
        "try:\n"
        "    os.mkfifo('newfifo')\n"
        "    print('allowed')\n"
        "except OSError as e:\n"
        "    print('refused' if e.errno in (1, 13) else 'error:%d' % e.errno)\n",
    )
    spec["grants"] = [
        *spec["grants"],
        {"path": str(work), "access": "rw"},
    ]
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 0
    assert (run_dir / "stdout").read_bytes() == b"refused\n"


def test_seccomp_setpriority_foreign_pid_denied(tmp_path: Path) -> None:
    """setpriority on another process is refused by the seccomp ban:
    with no PID namespace and matching kernel credentials, a profile
    could otherwise renice the daemon/server to 19 and leave it
    degraded after the profile ends (lowering nice back needs a
    privileged process)."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    spec = _supervisor_spec(
        run_dir,
        work,
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(2)\n"
        "    os._exit(0)\n"
        "try:\n"
        "    os.setpriority(os.PRIO_PROCESS, pid, 19)\n"
        "    print('allowed')\n"
        "except OSError as e:\n"
        "    print('refused' if e.errno in (1, 13) else 'error:%d' % e.errno)\n"
        "finally:\n"
        "    try:\n"
        "        os.kill(pid, 9)\n"
        "    except OSError:\n"
        "        pass\n"
        "    try:\n"
        "        os.waitpid(pid, 0)\n"
        "    except OSError:\n"
        "        pass\n",
    )
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 0
    assert (run_dir / "stdout").read_bytes() == b"refused\n"


def test_supervisor_status_carries_generation(tmp_path: Path) -> None:
    """Every status publication bumps the generation counter the
    daemon uses to bind a tail to its counters (see
    test_poll_rotation_snapshot_is_generation_bound)."""
    from host_bridge.spawn_helper import landlock_abi

    if landlock_abi() is None:
        pytest.skip("Landlock is unavailable in this environment")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    spec = _supervisor_spec(run_dir, work, "print('hi')")
    code, _ = _run_supervisor(spec, run_dir)
    assert code == 0
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "exited"
    assert isinstance(status.get("generation"), int)
    assert status["generation"] >= 1


# --- integration: real spawner + helper (skipped when the probe fails) ---


def _live_profiles(work: Path, bin_dir: Path) -> list[dict[str, object]]:
    profiles = _profiles()
    profiles.append(
        {
            "id": "cat_abs",
            "binary": _CAT,
            "argv_prefix": [str(work / ".env")],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "tool",
            "binary": str(bin_dir / "tool"),
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "py_rename",
            "binary": _PY,
            "argv_prefix": ["-c", "import os; os.rename('a/file.txt', 'b/file.txt')"],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "py_chmod",
            "binary": _PY,
            "argv_prefix": ["-c", "import os; os.chmod('.env', 0o600)"],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "py_prlimit",
            "binary": _PY,
            "argv_prefix": [
                "-c",
                "import os, resource, time\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(2)\n"
                "    os._exit(0)\n"
                "try:\n"
                "    resource.prlimit(pid, resource.RLIMIT_NOFILE, (0, 0))\n"
                "    print('allowed')\n"
                "except OSError as e:\n"
                "    print('refused' if e.errno in (1, 13) else 'error:%d' % e.errno)\n"
                "finally:\n"
                "    try:\n"
                "        os.kill(pid, 9)\n"
                "    except OSError:\n"
                "        pass\n"
                "    try:\n"
                "        os.waitpid(pid, 0)\n"
                "    except OSError:\n"
                "        pass\n",
            ],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "py_fifo",
            "binary": _PY,
            "argv_prefix": [
                "-c",
                "import os\n"
                "try:\n"
                "    os.mkfifo('newfifo')\n"
                "    print('allowed')\n"
                "except OSError as e:\n"
                "    print('refused' if e.errno in (1, 13) else 'error:%d' % e.errno)\n",
            ],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    profiles.append(
        {
            "id": "py_setpriority",
            "binary": _PY,
            "argv_prefix": [
                "-c",
                "import os, time\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    time.sleep(2)\n"
                "    os._exit(0)\n"
                "try:\n"
                "    os.setpriority(os.PRIO_PROCESS, pid, 19)\n"
                "    print('allowed')\n"
                "except OSError as e:\n"
                "    print('refused' if e.errno in (1, 13) else 'error:%d' % e.errno)\n"
                "finally:\n"
                "    try:\n"
                "        os.kill(pid, 9)\n"
                "    except OSError:\n"
                "        pass\n"
                "    try:\n"
                "        os.waitpid(pid, 0)\n"
                "    except OSError:\n"
                "        pass\n",
            ],
            "cwd": {"root_id": "work", "path": "."},
        }
    )
    return profiles


@pytest.fixture(scope="module")
def live(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[HostProcessService, Path]]:
    base = tmp_path_factory.mktemp("live")
    home = base / "home"
    work = home / "Project" / "work"
    work.mkdir(parents=True)
    (home / "Documents" / "docs").mkdir(parents=True)
    (work / "notes.txt").write_text("hello\nworld\n", encoding="utf-8")
    (work / ".env").write_text("SECRET=do-not-leak\n", encoding="utf-8")
    bin_dir = base / "bin"
    bin_dir.mkdir()
    shutil.copy(_TRUE, bin_dir / "tool")
    config = _write_config(home, _live_profiles(work, bin_dir))
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
    svc = HostProcessService(
        fs_service,
        registry,
        runtime_dir=base / "run",
        profiles=load_profiles(registry),
    )
    probe = svc.probe()
    if not probe.available:
        svc.shutdown()
        pytest.skip(f"enforcement probe failed: {probe.reason}: {probe.checks}")
    try:
        yield svc, work
    finally:
        svc.shutdown()


def _live_wait(
    svc: HostProcessService, process_id: str, state: str, *, timeout_s: float = 15.0
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        last = _poll(svc, process_id)
        if last.get("state") == state:
            return last
        time.sleep(0.1)
    raise AssertionError(f"process {process_id} did not reach {state}: {last}")


def _no_tree_child() -> bool:
    out = subprocess.run(["pgrep", "-f", "TREEKILL47"], capture_output=True, text=True)
    return out.returncode != 0


def test_live_probe_reports_all_layers(live: tuple[HostProcessService, Path]) -> None:
    svc, _ = live
    probe = svc.probe()
    assert probe.available
    assert probe.checks["tcp"] == "refused"
    assert probe.checks["udp"] == "refused"
    assert probe.checks["unix"] == "refused"
    assert probe.checks["protected"] == "refused"
    assert probe.checks["allowed"] == "read"
    assert probe.checks["signal"] == "refused"
    assert probe.checks["metadata"] == "refused"
    assert probe.checks["prlimit"] == "refused"
    assert probe.checks["fifo"] == "refused"
    assert probe.checks["setpriority"] == "refused"
    assert probe.checks["uid"] == 0
    assert probe.checks["placement"] is True
    assert probe.checks["landlock_abi"] == 8


def test_live_run_true_and_cat(live: tuple[HostProcessService, Path]) -> None:
    svc, _ = live
    run = _run(svc, "true", "live-true")
    final = _live_wait(svc, run["process_id"], "exited")
    assert final["exit_code"] == 0
    run = _run(svc, "cat", "live-cat", args=["notes.txt"])
    final = _live_wait(svc, run["process_id"], "exited")
    assert "hello" in final["stdout"]
    assert final["exit_code"] == 0


def test_live_protected_file_denied_from_inside(live: tuple[HostProcessService, Path]) -> None:
    """Owner-mandated: a secret nested in the allowed tree is denied
    from inside the process (Landlock), not just by daemon validation."""
    svc, _ = live
    run = _run(svc, "cat_abs", "live-cat-abs")
    final = _live_wait(svc, run["process_id"], "exited")
    assert "SECRET" not in final["stdout"]
    assert final["exit_code"] != 0
    assert "Permission denied" in final["stderr"] or "Operation not permitted" in final["stderr"]


def test_live_string_slot_is_data(live: tuple[HostProcessService, Path]) -> None:
    svc, _ = live
    run = _run(svc, "grep", "live-grep-flag", args=["-v", "notes.txt"])
    final = _live_wait(svc, run["process_id"], "exited")
    # "-v" after "--" is the pattern (data), not grep's invert flag:
    # no line contains "-v", so grep exits 1 with no output.
    assert final["exit_code"] == 1
    assert final["stdout"] == ""
    run = _run(svc, "grep", "live-grep-ok", args=["hello", "notes.txt"])
    final = _live_wait(svc, run["process_id"], "exited")
    assert final["exit_code"] == 0
    assert "hello" in final["stdout"]


def test_live_env_minimality(live: tuple[HostProcessService, Path]) -> None:
    svc, _ = live
    os.environ["SLAVIK_TEST_MARKER"] = "marker-secret"
    try:
        run = _run(svc, "py_env", "live-env")
        final = _live_wait(svc, run["process_id"], "exited")
    finally:
        del os.environ["SLAVIK_TEST_MARKER"]
    assert "ABSENT" in final["stdout"]
    assert "marker-secret" not in final["stdout"]
    assert "host-bridge-proc" in final["stdout"]  # HOME is the private tmp dir


def test_live_binary_identity(live: tuple[HostProcessService, Path]) -> None:
    svc, work = live
    # The "tool" profile binary was loaded from base/bin/tool; replace
    # the file (new inode, different content) and the helper must
    # refuse it — identity is bound to the open file from load time.
    bin_path = work.parents[2] / "bin" / "tool"
    assert bin_path.exists()
    bin_path.unlink()
    shutil.copy(_SLEEP, bin_path)
    with pytest.raises(ProcessError) as excinfo:
        _run(svc, "tool", "live-tool")
    assert excinfo.value.code == "spawn_failed"


def test_live_binary_identity_in_place_change(
    live: tuple[HostProcessService, Path],
) -> None:
    svc, work = live
    bin_path = work.parents[2] / "bin" / "tool"
    assert bin_path.exists()
    before = bin_path.stat()
    # Rewrite with a different VALID binary in place: the file stays
    # executable, so only the identity check can refuse it.
    data = Path(_SLEEP).read_bytes()
    with open(bin_path, "r+b") as handle:
        handle.truncate(len(data))
        handle.write(data)
    after = bin_path.stat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)
    assert (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns)
    with pytest.raises(ProcessError) as excinfo:
        _run(svc, "tool", "live-tool-inplace")
    assert excinfo.value.code == "spawn_failed"


def test_live_tree_kill_after_leader_exit(live: tuple[HostProcessService, Path]) -> None:
    """Owner-mandated: leader exits, child outlives it; kill leaves NO
    surviving descendants (verified via the cgroup and via pgrep)."""
    svc, _ = live
    run = _run(svc, "py_spawn", "live-tree-kill")
    process_id = run["process_id"]
    # Wait until the leader has exited (1 s) but the tree is alive.
    deadline = time.monotonic() + 10.0
    saw_running_with_child = False
    while time.monotonic() < deadline:
        polled = _poll(svc, process_id)
        if polled["state"] == "running" and not _no_tree_child():
            saw_running_with_child = True
            break
        time.sleep(0.1)
    assert saw_running_with_child, "the spawned sleep child never appeared"
    killed = _kill(svc, process_id)
    assert killed["state"] in ("killed", "killing")
    final = _live_wait(svc, process_id, "killed")
    assert final["state"] == "killed"
    assert _no_tree_child(), "a descendant survived the tree kill"


def test_live_timeout_tree_kill(live: tuple[HostProcessService, Path]) -> None:
    """Owner-mandated (timeout variant): the 120 s cap is shortened
    per run; on expiry the whole tree dies, not just the leader."""
    svc, _ = live
    run = _run(svc, "py_spawn", "live-tree-timeout", timeout_s=3)
    process_id = run["process_id"]
    final = _live_wait(svc, process_id, "timeout", timeout_s=15.0)
    assert final["state"] == "timeout"
    assert _no_tree_child(), "a descendant survived the timeout tree kill"


def test_live_rename_between_writable_subdirs(live: tuple[HostProcessService, Path]) -> None:
    """A rename between two subdirs of one granted rw tree works
    (Landlock REFER); without it the move fails with EXDEV."""
    svc, work = live
    (work / "a").mkdir(exist_ok=True)
    (work / "b").mkdir(exist_ok=True)
    (work / "a" / "file.txt").write_text("moved\n", encoding="utf-8")
    run = _run(svc, "py_rename", "live-rename")
    final = _live_wait(svc, run["process_id"], "exited")
    assert final["exit_code"] == 0, final["stderr"]
    assert (work / "b" / "file.txt").read_text(encoding="utf-8") == "moved\n"
    assert not (work / "a" / "file.txt").exists()


def test_live_metadata_ops_denied(live: tuple[HostProcessService, Path]) -> None:
    """A chmod on a file the profile cannot read is refused by the
    seccomp metadata ban — Landlock does not restrict metadata ops."""
    svc, work = live
    env_path = work / ".env"
    before = env_path.stat().st_mode
    try:
        run = _run(svc, "py_chmod", "live-chmod")
        final = _live_wait(svc, run["process_id"], "exited")
    finally:
        os.chmod(env_path, before)
    assert final["exit_code"] != 0
    assert "Operation not permitted" in final["stderr"]
    assert env_path.stat().st_mode == before


def test_live_prlimit_foreign_pid_denied(live: tuple[HostProcessService, Path]) -> None:
    """A profile that knows another process's PID cannot zero its
    limits: prlimit64 is banned by seccomp (there is no PID namespace,
    and the uid/gid mapping keeps matching kernel credentials)."""
    svc, _ = live
    run = _run(svc, "py_prlimit", "live-prlimit")
    final = _live_wait(svc, run["process_id"], "exited")
    assert "refused" in final["stdout"], final


def test_live_mkfifo_denied(live: tuple[HostProcessService, Path]) -> None:
    """A profile cannot mkfifo even in its writable cwd: MAKE_FIFO
    is handled by Landlock with no allowing grant."""
    svc, _ = live
    run = _run(svc, "py_fifo", "live-fifo")
    final = _live_wait(svc, run["process_id"], "exited")
    assert "refused" in final["stdout"], final


def test_live_setpriority_foreign_pid_denied(live: tuple[HostProcessService, Path]) -> None:
    """A profile cannot renice another process: setpriority is
    banned by seccomp (same family as prlimit64)."""
    svc, _ = live
    run = _run(svc, "py_setpriority", "live-setpriority")
    final = _live_wait(svc, run["process_id"], "exited")
    assert "refused" in final["stdout"], final


def test_live_idempotent_retry(live: tuple[HostProcessService, Path]) -> None:
    svc, _ = live
    first = _run(svc, "sleep", "live-retry", args=["5"])
    second = _run(svc, "sleep", "live-retry", args=["5"])
    assert first["process_id"] == second["process_id"]
    assert sum(1 for pid in svc._records if pid == first["process_id"]) == 1
    _kill(svc, first["process_id"])
