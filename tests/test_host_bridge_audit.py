"""Audit/effect log (E03-T08, ADR-0013 §6 + owner conditions).

Every mutation answer carries an ``effect`` block (kind/seq/
boot_id/at + kind fields + content_sha256, no content); the
daemon assigns a monotonic per-boot seq and frame-budgets the
serialized result+effect BEFORE the publish syscall. Wire tests
run a real daemon socket in a thread and call it through
``host_bridge.client``; service-level tests drive the services
directly with a fake spawner (the E03-T06 pattern).

Red/green: these tests were written against the T08 schema
(rev. 1 §6.1 + rev. 2 + the 7 owner conditions) before the
implementation; the implementation makes them pass.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from host_bridge import client
from host_bridge.client import BridgeClientError, _validate_effect
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.effect import (
    EFFECT_BUILD_FAILED,
    EffectEmitter,
    stop_process_placeholder,
)
from host_bridge.fs import (
    CancelToken,
    FsError,
    HostFsService,
    Root,
    RootsRegistry,
)
from host_bridge.process import (
    HostProcessService,
    ProbeResult,
    ProcessError,
    SpawnError,
    SpawnHandle,
    _Record,
    load_profiles,
)
from host_bridge.protocol import MAX_FRAME_BYTES, BridgeRequest, encode_success

_OWNER = "owner"
_OTHER = "someone-else"
_TRUE = "/usr/bin/true"
_SLEEP = "/usr/bin/sleep"

# Effect keys that may appear, by kind (metadata only: no
# content, patch text, args, env, stdout/stderr — ever).
_EFFECT_KEYS = {
    "seq",
    "boot_id",
    "kind",
    "at",
    "principal_id",
    "run_id",
}


def _profiles() -> list[dict[str, object]]:
    return [
        {"id": "true", "binary": _TRUE, "cwd": {"root_id": "work", "path": "."}},
        {
            "id": "sleep",
            "binary": _SLEEP,
            "args": [{"name": "seconds", "kind": "int", "required": True, "min": 1, "max": 5}],
            "cwd": {"root_id": "work", "path": "."},
            "timeout_s": 10,
        },
    ]


def _write_config(home: Path) -> Path:
    config = home / ".config" / "slavikai" / "host-bridge.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "roots": [
            {"id": "work", "path": str(home / "Project" / "work"), "mode": "rw"},
            {"id": "docs", "path": str(home / "Documents" / "docs"), "mode": "ro"},
        ],
        "process_profiles": _profiles(),
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
    return fake_home


@pytest.fixture()
def registry(home: Path) -> RootsRegistry:
    config = _write_config(home)
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
def fs_service(registry: RootsRegistry, tmp_path: Path) -> HostFsService:
    return HostFsService(registry, trash_root=tmp_path / "Trash")


class FakeSpawner:
    """Minimal in-test spawner (E03-T06 pattern, trimmed): a real
    ``sleep`` stands in for the scope leader (so ``leader_done``
    is honest), and status files are written by hand. No systemd,
    no Landlock, no seccomp."""

    def __init__(self) -> None:
        self.units: dict[str, dict[str, object]] = {}
        self.signal_calls: list[tuple[str, int]] = []
        self.kill_calls: list[str] = []
        self.finish_state = "killed"

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
        leader = subprocess.Popen(
            ["sleep", "30"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        handle = SpawnHandle(popen=leader, unit_name=unit_name, cgroup_path=None, run_dir=run_dir)
        self.units[unit_name] = {"handle": handle, "procs": [str(leader.pid)]}
        (run_dir / "status.json").write_text(
            json.dumps(self._status(run_dir, "running")), encoding="utf-8"
        )
        (run_dir / "stdout").write_bytes(b"")
        (run_dir / "stderr").write_bytes(b"")
        return handle

    def signal_tree(self, handle: SpawnHandle, sig: int, *, timeout_s: float = 5.0) -> None:
        self.signal_calls.append((handle.unit_name, sig))
        if handle.popen is not None and handle.popen.poll() is None:
            handle.popen.send_signal(sig)

    def kill_all(self, handle: SpawnHandle, *, timeout_s: float = 5.0) -> None:
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
                    finished_at=datetime.now(UTC).isoformat(),
                    exit_code=exit_code,
                    signal=exit_signal,
                )
            ),
            encoding="utf-8",
        )

    def list_bridge_scopes(self) -> list[str]:
        return list(self.units)

    def destroy_scope_checked(self, unit_name: str, timeout_s: float = 5.0) -> bool:
        unit = self.units.pop(unit_name, None)
        if unit is not None:
            handle = unit["handle"]
            assert isinstance(handle, SpawnHandle)
            if handle.popen is not None and handle.popen.poll() is None:
                handle.popen.kill()
        return True

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        unit = self.units.get(handle.unit_name)
        if unit is None:
            return True
        procs = unit.get("procs")
        return not procs if isinstance(procs, list) else True


@pytest.fixture()
def spawner() -> FakeSpawner:
    return FakeSpawner()


@pytest.fixture()
def proc_service(
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


@pytest.fixture()
def bridge(
    fs_service: HostFsService,
    proc_service: HostProcessService,
    tmp_path: Path,
) -> Iterator[Path]:
    """A real daemon on a real socket in a thread, with profiles
    and the fake spawner (so run/kill work over the wire)."""
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=fs_service, process_service=proc_service)
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    try:
        yield socket_path
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _token(deadline_s: float = 5.0) -> CancelToken:
    return CancelToken(deadline=time.monotonic() + deadline_s, request_id="test-req")


def _call(
    socket_path: Path,
    capability: str,
    *,
    params: dict[str, object] | None = None,
    request_id: str | None = None,
    principal_id: str | None = _OWNER,
    run_id: str | None = None,
) -> dict[str, object]:
    return client.call(
        socket_path,
        capability,
        params=params,
        request_id=request_id,
        principal_id=principal_id,
        run_id=run_id,
    )


def _assert_effect_shape(effect: object, *, kind: str) -> dict[str, object]:
    assert isinstance(effect, dict)
    assert effect["kind"] == kind
    assert isinstance(effect["seq"], int) and effect["seq"] >= 1
    assert isinstance(effect["boot_id"], str) and effect["boot_id"]
    assert isinstance(effect["at"], str) and effect["at"]
    return effect


def _assert_no_forbidden_effect_keys(effect: dict[str, object], *, kind: str) -> None:
    """Metadata-only: the effect carries fixed fields, never
    content, patch text, args, env or stdout/stderr."""
    forbidden = {"content", "patch", "args", "argv", "env", "stdout", "stderr"}
    assert not (set(effect) & forbidden), (
        f"effect {kind} leaks forbidden keys: {set(effect) & forbidden}"
    )


def _write_commit(
    socket_path: Path, *, content: str, path: str = "audit.txt", request_id: str | None = None
) -> dict[str, object]:
    """Single-call write + commit (the standard small-file flow):
    the commit call carries the content, so ``bytes_written`` is
    the real byte count in both the result and the effect."""
    return _call(
        socket_path,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": path,
            "mode": "create",
            "content": content,
            "commit": True,
        },
        request_id=request_id or f"w-{uuid.uuid4().hex}",
    )


def _write_commit_two_calls(
    socket_path: Path, *, content: str, path: str = "audit2.txt"
) -> dict[str, object]:
    """Chunk, then a content-less commit. The commit call's own
    ``bytes_written`` is 0 (pre-existing: it is len(data) of the
    commit call) — the effect mirrors the result it rides on."""
    chunk = _call(
        socket_path,
        "host.fs.write",
        params={"root_id": "work", "path": path, "mode": "create", "content": content},
        request_id=f"w-{uuid.uuid4().hex}",
    )
    assert "effect" not in chunk  # a chunk is not a mutation yet
    write_id = chunk["write_id"]
    assert isinstance(write_id, str)
    return _call(
        socket_path,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": path,
            "mode": "create",
            "write_id": write_id,
            "commit": True,
        },
        request_id=f"c-{uuid.uuid4().hex}",
    )


# --- fs.write ------------------------------------------------------------


def test_write_commit_carries_effect(bridge: Path) -> None:
    result = _write_commit(bridge, content="audit me\n")
    assert result["committed"] is True
    effect = _assert_effect_shape(result.get("effect"), kind="fs.write")
    assert effect["root_id"] == "work"
    assert effect["path"] == "audit.txt"
    assert effect["size"] == 9
    assert effect["bytes_written"] == 9
    assert effect["committed"] is True
    assert effect["content_sha256"] == hashlib.sha256(b"audit me\n").hexdigest()
    assert effect["principal_id"] == _OWNER
    _assert_no_forbidden_effect_keys(effect, kind="fs.write")
    assert set(effect) == _EFFECT_KEYS | {
        "root_id",
        "path",
        "size",
        "bytes_written",
        "committed",
        "content_sha256",
    }


def test_two_call_commit_effect_mirrors_result(bridge: Path) -> None:
    """The chunked flow: the commit call carries no content, so
    its ``bytes_written`` is 0 — pre-existing result behaviour
    (len(data) of the commit call). The effect mirrors the result
    it rides on: same 0, same size, and the sha256 of the staged
    bytes."""
    result = _write_commit_two_calls(bridge, content="audit me\n")
    assert result["committed"] is True
    assert result["bytes_written"] == 0
    assert result["size"] == 9
    effect = _assert_effect_shape(result.get("effect"), kind="fs.write")
    assert effect["bytes_written"] == 0
    assert effect["size"] == 9
    assert effect["content_sha256"] == hashlib.sha256(b"audit me\n").hexdigest()


def test_write_chunk_creates_parents_carries_effect(bridge: Path, home: Path) -> None:
    """Codex round 2, finding 2: a first chunk inside missing
    parents CREATES those directories before the file is published,
    so the chunk answer must not claim no_effect. The effect names
    the created parents; a later chunk in the same session creates
    nothing and carries no effect; abort does not remove the dirs
    (which is exactly why the creation must be journaled)."""
    chunk = _call(
        bridge,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": "new-parent/sub/file.txt",
            "mode": "create",
            "content": "x",
        },
    )
    assert "effect" in chunk  # red before the fix: no_effect lie
    effect = _assert_effect_shape(chunk.get("effect"), kind="fs.write")
    assert effect["root_id"] == "work"
    assert effect["path"] == "new-parent/sub/file.txt"
    assert effect["size"] == 1
    assert effect["bytes_written"] == 1
    assert effect["committed"] is False
    assert effect["parents_created"] == ["new-parent", "new-parent/sub"]
    assert "content_sha256" not in effect  # nothing published yet
    _assert_no_forbidden_effect_keys(effect, kind="fs.write")
    work = home / "Project" / "work"
    assert (work / "new-parent" / "sub").is_dir()
    write_id = chunk["write_id"]
    assert isinstance(write_id, str)
    second = _call(
        bridge,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": "new-parent/sub/file.txt",
            "mode": "create",
            "write_id": write_id,
            "content": "y",
        },
    )
    assert "effect" not in second  # parents already exist
    aborted = _call(
        bridge,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": "new-parent/sub/file.txt",
            "mode": "create",
            "write_id": write_id,
            "abort": True,
        },
    )
    assert aborted["aborted"] is True
    assert "effect" not in aborted
    assert (work / "new-parent" / "sub").is_dir()  # abort keeps them


def test_write_chunk_existing_parents_has_no_effect(bridge: Path) -> None:
    """The same chunk flow one level down, where the parents
    already exist: nothing is created, so no effect is correct."""
    _write_commit(bridge, content="seed", path="sub/seed.txt")
    chunk = _call(
        bridge,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": "sub/other.txt",
            "mode": "create",
            "content": "x",
        },
    )
    assert "effect" not in chunk


def test_write_chunk_and_abort_have_no_effect(bridge: Path) -> None:
    chunk = _call(
        bridge,
        "host.fs.write",
        params={"root_id": "work", "path": "chunk.txt", "mode": "create", "content": "x"},
    )
    assert "effect" not in chunk
    write_id = chunk["write_id"]
    assert isinstance(write_id, str)
    aborted = _call(
        bridge,
        "host.fs.write",
        params={
            "root_id": "work",
            "path": "chunk.txt",
            "mode": "create",
            "write_id": write_id,
            "abort": True,
        },
    )
    assert aborted["aborted"] is True
    assert "effect" not in aborted


def test_effect_seq_is_monotonic_across_mutations(bridge: Path) -> None:
    first = _write_commit(bridge, content="one", path="one.txt")
    second = _write_commit(bridge, content="two", path="two.txt")
    first_effect = first["effect"]
    second_effect = second["effect"]
    assert isinstance(first_effect, dict) and isinstance(second_effect, dict)
    assert first_effect["boot_id"] == second_effect["boot_id"]
    assert second_effect["seq"] == first_effect["seq"] + 1


def test_effect_frame_budget_holds_for_long_paths(bridge: Path) -> None:
    # A path near the 4096-char limit: the serialized result +
    # effect must still fit the 64 KiB frame (condition 5).
    long_name = "a" * 200 + ".txt"
    result = _write_commit(bridge, content="x", path=long_name)
    assert result["committed"] is True
    rid = "frame-check"
    assert len(encode_success(result, rid)) - 1 <= MAX_FRAME_BYTES


# --- fs.patch / move / delete ----------------------------------------------


def test_patch_applied_carries_effect(bridge: Path) -> None:
    _write_commit(bridge, content="hello\nworld\n", path="patch.txt")
    result = _call(
        bridge,
        "host.fs.patch",
        params={
            "root_id": "work",
            "path": "patch.txt",
            "patch": "@@ -1,2 +1,2 @@\n-hello\n+goodbye\n world\n",
        },
    )
    assert result["applied"] is True
    effect = _assert_effect_shape(result.get("effect"), kind="fs.patch")
    assert effect["root_id"] == "work"
    assert effect["path"] == "patch.txt"
    assert effect["content_sha256"] == hashlib.sha256(b"goodbye\nworld\n").hexdigest()
    _assert_no_forbidden_effect_keys(effect, kind="fs.patch")


def test_patch_dry_run_has_no_effect(bridge: Path) -> None:
    _write_commit(bridge, content="hello\nworld\n", path="dry.txt")
    result = _call(
        bridge,
        "host.fs.patch",
        params={
            "root_id": "work",
            "path": "dry.txt",
            "patch": "@@ -1,2 +1,2 @@\n-hello\n+goodbye\n world\n",
            "dry_run": True,
        },
    )
    assert result["applied"] is False
    assert result["dry_run"] is True
    assert "effect" not in result


def test_move_carries_effect(bridge: Path) -> None:
    _write_commit(bridge, content="move me", path="src.txt")
    result = _call(
        bridge,
        "host.fs.move",
        params={"root_id": "work", "path": "src.txt", "dest_path": "dst.txt"},
    )
    assert result["moved"] is True
    effect = _assert_effect_shape(result.get("effect"), kind="fs.move")
    assert effect["path"] == "src.txt"
    assert effect["dest_path"] == "dst.txt"
    assert effect["dest_root_id"] == "work"
    _assert_no_forbidden_effect_keys(effect, kind="fs.move")


def test_delete_carries_effect(bridge: Path, tmp_path: Path) -> None:
    _write_commit(bridge, content="delete me", path="gone.txt")
    result = _call(
        bridge,
        "host.fs.delete",
        params={"root_id": "work", "path": "gone.txt"},
    )
    assert result["recoverable"] is True
    effect = _assert_effect_shape(result.get("effect"), kind="fs.delete")
    assert effect["path"] == "gone.txt"
    assert effect["trash_name"] == result["trash_name"]
    assert effect["recoverable"] is True
    _assert_no_forbidden_effect_keys(effect, kind="fs.delete")


# --- process.stop (owner condition 1) ---------------------------------------


def test_stop_unknown_run_carries_effect(bridge: Path) -> None:
    result = _call(
        bridge,
        "host.process.stop",
        params={"run_id": "run-never-existed"},
        run_id="run-never-existed",
    )
    assert result["processes"] == []
    assert result["all_final"] is True
    effect = _assert_effect_shape(result.get("effect"), kind="process.stop")
    assert effect["run_id"] == "run-never-existed"
    assert effect["processes"] == []
    assert effect["mark_set"] is True


def test_repeat_stop_emits_a_new_effect_not_a_duplicate(bridge: Path) -> None:
    """Owner condition 1: the repeat STOP re-sets the mark
    timestamp (verified in host_bridge/process.py: _stop_run sets
    _stopped_runs[pair] unconditionally), so it IS a new effect —
    the recorder must never dedupe it by subject."""
    first = _call(bridge, "host.process.stop", params={"run_id": "run-1"}, run_id="run-1")
    second = _call(bridge, "host.process.stop", params={"run_id": "run-1"}, run_id="run-1")
    first_effect = first["effect"]
    second_effect = second["effect"]
    assert isinstance(first_effect, dict) and isinstance(second_effect, dict)
    assert first_effect["boot_id"] == second_effect["boot_id"]
    assert second_effect["seq"] == first_effect["seq"] + 1
    assert second_effect["mark_set"] is True


# --- process.run / kill ------------------------------------------------------


def test_run_and_kill_carry_effects(bridge: Path) -> None:
    run = _call(
        bridge,
        "host.process.run",
        params={"profile_id": "sleep", "args": ["2"]},
        request_id="run-1",
        run_id="run-A",
    )
    assert run["state"] == "running"
    run_effect = _assert_effect_shape(run.get("effect"), kind="process.run")
    assert run_effect["process_id"] == run["process_id"]
    assert run_effect["profile_id"] == "sleep"
    assert run_effect["state"] == "running"
    _assert_no_forbidden_effect_keys(run_effect, kind="process.run")

    process_id = run["process_id"]
    assert isinstance(process_id, str)
    kill = _call(
        bridge, "host.process.kill", params={"process_id": process_id}, request_id="kill-1"
    )
    assert kill["state"] in ("killed", "killing")
    kill_effect = _assert_effect_shape(kill.get("effect"), kind="process.kill")
    assert kill_effect["process_id"] == process_id
    assert kill_effect["seq"] != run_effect["seq"]

    # Repeat kill re-attaches the ORIGINAL effect (same seq): no
    # new mutation, no new effect.
    repeat = _call(
        bridge, "host.process.kill", params={"process_id": process_id}, request_id="kill-2"
    )
    repeat_effect = repeat.get("effect")
    assert isinstance(repeat_effect, dict)
    assert repeat_effect["seq"] == kill_effect["seq"]
    assert repeat_effect["boot_id"] == kill_effect["boot_id"]


def test_run_retry_reattaches_the_original_effect(bridge: Path) -> None:
    first = _call(
        bridge,
        "host.process.run",
        params={"profile_id": "sleep", "args": ["2"]},
        request_id="run-same",
        run_id="run-B",
    )
    second = _call(
        bridge,
        "host.process.run",
        params={"profile_id": "sleep", "args": ["2"]},
        request_id="run-same",
        run_id="run-B",
    )
    assert second["process_id"] == first["process_id"]
    first_effect = first["effect"]
    second_effect = second["effect"]
    assert isinstance(first_effect, dict) and isinstance(second_effect, dict)
    assert second_effect["seq"] == first_effect["seq"]
    assert second_effect["boot_id"] == first_effect["boot_id"]


def test_run_retry_after_kill_reattaches_run_effect(bridge: Path) -> None:
    """Codex round 1, finding 1 (reverse half): a successful kill
    must not overwrite the stored launch effect. A retry of the
    original run with the same request_id re-attaches the RUN
    effect, not the kill effect. Before the fix the repeat run
    answered with the kill effect and the client rejected it as
    malformed_response (kind mismatch)."""
    first = _call(
        bridge,
        "host.process.run",
        params={"profile_id": "sleep", "args": ["2"]},
        request_id="run-kill-retry",
        run_id="run-C",
    )
    assert first["state"] == "running"
    first_effect = _assert_effect_shape(first.get("effect"), kind="process.run")
    process_id = first["process_id"]
    assert isinstance(process_id, str)
    kill = _call(
        bridge, "host.process.kill", params={"process_id": process_id}, request_id="kill-3"
    )
    assert kill["state"] in ("killed", "killing")
    kill_effect = _assert_effect_shape(kill.get("effect"), kind="process.kill")
    assert kill_effect["seq"] != first_effect["seq"]
    # The original run, retried with the same request_id, answers
    # with its own launch effect — never the kill effect.
    retry = _call(
        bridge,
        "host.process.run",
        params={"profile_id": "sleep", "args": ["2"]},
        request_id="run-kill-retry",
        run_id="run-C",
    )
    assert retry["process_id"] == process_id
    retry_effect = _assert_effect_shape(retry.get("effect"), kind="process.run")
    assert retry_effect["seq"] == first_effect["seq"]
    assert retry_effect["boot_id"] == first_effect["boot_id"]


def test_boot_id_rotates_on_restart(
    fs_service: HostFsService, proc_service: HostProcessService, tmp_path: Path
) -> None:
    """A restarted daemon never re-uses a (boot_id, seq) pair."""
    socket_1 = tmp_path / "slavikai-1" / "host-bridge.sock"
    daemon_1 = HostBridgeDaemon(socket_1, fs_service=fs_service, process_service=proc_service)
    daemon_1.start()
    thread_1 = threading.Thread(target=daemon_1.serve_forever, daemon=True)
    thread_1.start()
    try:
        first = _write_commit(socket_1, content="one", path="boot-1.txt")
    finally:
        daemon_1.shutdown()
        thread_1.join(timeout=2.0)
    first_effect = first.get("effect")
    assert isinstance(first_effect, dict)
    # A second daemon instance (new socket, same services) rotates
    # the boot identity at start().
    socket_2 = tmp_path / "slavikai-2" / "host-bridge.sock"
    daemon_2 = HostBridgeDaemon(socket_2, fs_service=fs_service, process_service=proc_service)
    daemon_2.start()
    thread_2 = threading.Thread(target=daemon_2.serve_forever, daemon=True)
    thread_2.start()
    try:
        second = _write_commit(socket_2, content="two", path="boot-2.txt")
    finally:
        daemon_2.shutdown()
        thread_2.join(timeout=2.0)
    second_effect = second["effect"]
    assert isinstance(second_effect, dict)
    assert second_effect["boot_id"] != first_effect["boot_id"]
    assert second_effect["seq"] == 1


def _write_commit_via_services(
    fs_service: HostFsService, path: str, content: str
) -> dict[str, object]:
    request = BridgeRequest(
        version=1,
        request_id=f"svc-{uuid.uuid4().hex}",
        capability="host.fs.write",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": path, "mode": "create", "content": content},
    )
    chunk = fs_service.handle(request, _token())
    write_id = chunk["write_id"]
    assert isinstance(write_id, str)
    commit_request = BridgeRequest(
        version=1,
        request_id=f"svc-{uuid.uuid4().hex}",
        capability="host.fs.write",
        principal_id=_OWNER,
        run_id=None,
        params={
            "root_id": "work",
            "path": path,
            "mode": "create",
            "write_id": write_id,
            "commit": True,
        },
    )
    return fs_service.handle(commit_request, _token())


# --- service-level -----------------------------------------------------------


def test_fs_handle_threads_envelope_attribution(fs_service: HostFsService) -> None:
    """The FS handle receives principal_id/run_id from the
    envelope (not params) and stamps them into the effect."""
    result = _write_commit_via_services(fs_service, "attr.txt", "x")
    effect = _assert_effect_shape(result.get("effect"), kind="fs.write")
    assert effect["principal_id"] == _OWNER


def test_emitter_frame_fits_rejects_oversized() -> None:
    emitter = EffectEmitter()
    projection = emitter.build_projection(
        kind="fs.write",
        principal_id=_OWNER,
        run_id=None,
        fields={
            "root_id": "work",
            "path": "x.txt",
            "size": 1,
            "bytes_written": 1,
            "committed": True,
            "content_sha256": "0" * 64,
        },
    )
    assert emitter.frame_fits({"committed": True, "effect": projection}, "req-1") is True
    huge = {"committed": True, "effect": projection, "pad": "x" * (MAX_FRAME_BYTES + 1)}
    assert emitter.frame_fits(huge, "req-1") is False


def test_emitter_finalize_assigns_seq_and_rotates_boot() -> None:
    emitter = EffectEmitter()
    first_boot = emitter.boot_id
    first = emitter.finalize(
        emitter.build_projection(kind="fs.delete", principal_id=None, run_id=None, fields={})
    )
    second = emitter.finalize(
        emitter.build_projection(kind="fs.delete", principal_id=None, run_id=None, fields={})
    )
    assert first["seq"] == 1
    assert second["seq"] == 2
    assert first["boot_id"] == first_boot == second["boot_id"]
    emitter.note_started()
    assert emitter.boot_id != first_boot
    third = emitter.finalize(
        emitter.build_projection(kind="fs.delete", principal_id=None, run_id=None, fields={})
    )
    assert third["seq"] == 1


def test_stop_projection_fits_frame_for_max_run_id() -> None:
    """The stop projection (64 placeholder entries, 256-char
    run_id) must fit the frame — the pre-mutation check can never
    fail on a legitimate STOP (condition 5)."""
    emitter = EffectEmitter()
    projection = emitter.build_projection(
        kind="process.stop",
        principal_id=_OWNER,
        run_id="r" * 256,
        fields={
            "run_id": "r" * 256,
            "processes": [stop_process_placeholder() for _ in range(64)],
            "all_final": False,
            "mark_set": True,
        },
    )
    result = {
        "run_id": "r" * 256,
        "processes": [stop_process_placeholder() for _ in range(64)],
        "all_final": False,
        "effect": projection,
    }
    assert emitter.frame_fits(result, "req-stop") is True


def test_effect_build_failed_on_write_hash_fault(
    fs_service: HostFsService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-8.1: a hashing failure before the publish is
    effect_build_failed, and the file is never created."""

    def _boom(fd: int, size: int) -> str:
        raise OSError("injected hash failure")

    monkeypatch.setattr("host_bridge.fs._sha256_file_fd", _boom)
    request = BridgeRequest(
        version=1,
        request_id="fault-1",
        capability="host.fs.write",
        principal_id=_OWNER,
        run_id=None,
        params={"root_id": "work", "path": "fault.txt", "mode": "create", "content": "x"},
    )
    chunk = fs_service.handle(request, _token())
    write_id = chunk["write_id"]
    assert isinstance(write_id, str)
    commit = BridgeRequest(
        version=1,
        request_id="fault-2",
        capability="host.fs.write",
        principal_id=_OWNER,
        run_id=None,
        params={
            "root_id": "work",
            "path": "fault.txt",
            "mode": "create",
            "write_id": write_id,
            "commit": True,
        },
    )
    with pytest.raises(FsError) as exc_info:
        fs_service.handle(commit, _token())
    assert exc_info.value.code == EFFECT_BUILD_FAILED
    work = fs_service.registry.by_id("work")
    assert work is not None
    assert not (work.path / "fault.txt").exists()


def test_effect_build_failed_on_patch_hash_fault(
    fs_service: HostFsService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-8.1 (patch): same discipline — the file stays unpatched."""
    _write_commit_via_services(fs_service, "pfault.txt", "hello\nworld\n")

    def _boom(data: bytes) -> str:
        raise RuntimeError("injected hash failure")

    monkeypatch.setattr("host_bridge.fs._sha256_bytes", _boom)
    request = BridgeRequest(
        version=1,
        request_id="fault-3",
        capability="host.fs.patch",
        principal_id=_OWNER,
        run_id=None,
        params={
            "root_id": "work",
            "path": "pfault.txt",
            "patch": "@@ -1,2 +1,2 @@\n-hello\n+goodbye\n world\n",
        },
    )
    with pytest.raises(FsError) as exc_info:
        fs_service.handle(request, _token())
    assert exc_info.value.code == EFFECT_BUILD_FAILED
    work = fs_service.registry.by_id("work")
    assert work is not None
    assert (work.path / "pfault.txt").read_text(encoding="utf-8") == "hello\nworld\n"


def test_write_chunk_parents_frame_budget_fails_before_mkdir(
    fs_service: HostFsService, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex round 2, finding 2 (condition 5): the parents
    effect is frame-budgeted BEFORE _open_dirs creates anything.
    With the frame check forced to fail, the call is
    effect_build_failed and no parent directory appears."""
    monkeypatch.setattr(EffectEmitter, "frame_fits", staticmethod(lambda *args, **kwargs: False))
    request = BridgeRequest(
        version=1,
        request_id="parents-budget-1",
        capability="host.fs.write",
        principal_id=_OWNER,
        run_id=None,
        params={
            "root_id": "work",
            "path": "new-parent/file.txt",
            "mode": "create",
            "content": "x",
        },
    )
    with pytest.raises(FsError) as exc_info:
        fs_service.handle(request, _token())
    assert exc_info.value.code == EFFECT_BUILD_FAILED
    work = fs_service.registry.by_id("work")
    assert work is not None
    assert not (work.path / "new-parent").exists()


def test_kill_final_record_has_no_effect(proc_service: HostProcessService) -> None:
    """A kill of an already-final record requests nothing: the
    final state is returned with no effect block (the recorder
    reads that as no_effect for this call)."""
    record = _Record(
        process_id=uuid.uuid4().hex,
        profile_id="true",
        state="exited",
        run_id="run-1",
        principal_id=_OWNER,
        started_at=datetime.now(UTC).isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        exit_code=0,
        exit_signal=None,
        started_mono=time.monotonic(),
        final=True,
    )
    proc_service._records[record.process_id] = record
    result = proc_service.handle(
        "host.process.kill", {"process_id": record.process_id}, "kill-final", _token()
    )
    assert result["state"] == "exited"
    assert result["exit_code"] == 0
    assert "effect" not in result


def test_kill_final_record_with_run_effect_returns_no_effect(
    proc_service: HostProcessService,
) -> None:
    """Codex round 1, finding 1: the launch effect stored on the
    record must NOT leak into a kill answer. A kill of a process
    that already exited on its own requests nothing — the answer
    carries no effect block (the recorder reads that as no_effect),
    even though the record still holds its run effect. Before the
    fix the kill answered with the run effect and the client
    rejected it as malformed_response (kind mismatch)."""
    record = _Record(
        process_id=uuid.uuid4().hex,
        profile_id="true",
        state="exited",
        run_id="run-1",
        principal_id=_OWNER,
        started_at=datetime.now(UTC).isoformat(),
        finished_at=datetime.now(UTC).isoformat(),
        exit_code=0,
        exit_signal=None,
        started_mono=time.monotonic(),
        final=True,
    )
    record.effect = {
        "seq": 1,
        "boot_id": "b" * 32,
        "kind": "process.run",
        "at": datetime.now(UTC).isoformat(),
        "principal_id": _OWNER,
        "run_id": "run-1",
        "process_id": record.process_id,
        "state": "exited",
    }
    proc_service._records[record.process_id] = record
    result = proc_service.handle(
        "host.process.kill", {"process_id": record.process_id}, "kill-final-run", _token()
    )
    assert result["state"] == "exited"
    assert result["exit_code"] == 0
    assert "effect" not in result


class _BlockingSpawner:
    """Spawner whose spawn() blocks until released, then fails.

    Reproduces Codex round 2, finding 1: a retry arriving while
    the launch is still committing, its wait expiring before the
    spawn completes.
    """

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.signal_calls: list[tuple[str, int]] = []
        self.kill_calls: list[str] = []

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        self.started.set()
        self.release.wait(timeout=30)
        raise SpawnError("injected spawn failure")

    def signal_tree(self, handle: SpawnHandle, sig: int, *, timeout_s: float = 5.0) -> None:
        self.signal_calls.append((handle.unit_name, sig))

    def kill_all(self, handle: SpawnHandle, *, timeout_s: float = 5.0) -> None:
        self.kill_calls.append(handle.unit_name)

    def list_bridge_scopes(self) -> list[str]:
        return []

    def destroy_scope_checked(self, unit_name: str, timeout_s: float = 5.0) -> bool:
        return True

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        return True


def test_run_retry_during_launch_carries_no_effect(
    fs_service: HostFsService, registry: RootsRegistry, tmp_path: Path
) -> None:
    """Codex round 2, finding 1: a retry whose wait on the
    launching record expires must NOT finalize the run effect —
    the spawn may still fail, and a confirmed effect for a process
    that never existed would poison the journal (and resolve an
    unknown for a mutation that never happened). The projection
    stays on the record; after the spawn fails, a further retry
    answers spawn_failed with no effect at all."""
    spawner = _BlockingSpawner()
    svc = HostProcessService(
        fs_service,
        registry,
        runtime_dir=tmp_path / "run-race",
        profiles=load_profiles(registry),
        spawner=spawner,  # type: ignore[arg-type]
    )
    svc._probe = ProbeResult(available=True, checks={}, reason="unit-test")
    try:
        outcome: dict[str, object] = {}
        errors: dict[str, Exception] = {}

        def _first() -> None:
            try:
                outcome["first"] = svc.handle(
                    "host.process.run",
                    {"profile_id": "true"},
                    "run-launch-race",
                    _token(deadline_s=30),
                    principal_id=_OWNER,
                    run_id="run-race",
                )
            except Exception as exc:  # noqa: BLE001 - recorded for the assertions below
                errors["first"] = exc

        thread = threading.Thread(target=_first, daemon=True)
        thread.start()
        assert spawner.started.wait(timeout=5)
        # The retry arrives while the launch is still committing;
        # its short wait expires before the spawn completes.
        retry = svc.handle(
            "host.process.run",
            {"profile_id": "true"},
            "run-launch-race",
            _token(deadline_s=0.5),
            principal_id=_OWNER,
            run_id="run-race",
        )
        assert retry["state"] == "launching"
        assert "effect" not in retry  # red before the fix
        spawner.release.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert "first" not in outcome
        first_error = errors.get("first")
        assert isinstance(first_error, ProcessError)
        assert first_error.code == "spawn_failed"
        # The spawn failed after the retry's wait expired: the
        # answer must be spawn_failed with NO run effect — the
        # stale projection must not survive the finalization.
        final = svc.handle(
            "host.process.run",
            {"profile_id": "true"},
            "run-launch-race",
            _token(),
            principal_id=_OWNER,
            run_id="run-race",
        )
        assert final["state"] == "spawn_failed"
        assert "effect" not in final  # red before the fix
    finally:
        svc.shutdown()


def test_finalize_spawn_failed_clears_effect(proc_service: HostProcessService) -> None:
    """Codex round 2, finding 1 (second half): finalizing a
    record as spawn_failed must drop any effect finalized by an
    earlier retry while the launch was still in flight — a
    spawn_failed answer must never carry a run effect. Other
    final states keep the effect: the process did run."""
    record = _Record(
        process_id=uuid.uuid4().hex,
        profile_id="true",
        state="launching",
        run_id="run-1",
        principal_id=_OWNER,
        started_mono=time.monotonic(),
    )
    record.effect = {
        "seq": 1,
        "boot_id": "b" * 32,
        "kind": "process.run",
        "at": datetime.now(UTC).isoformat(),
        "principal_id": _OWNER,
        "run_id": "run-1",
        "process_id": record.process_id,
        "state": "launching",
    }
    record.effect_projection = dict(record.effect)
    proc_service._finalize(record, "spawn_failed")
    assert record.state == "spawn_failed"
    assert record.final
    assert record.effect is None
    assert record.effect_projection is None
    # Control: a record that actually ran keeps its effect.
    ran = _Record(
        process_id=uuid.uuid4().hex,
        profile_id="true",
        state="running",
        run_id="run-2",
        principal_id=_OWNER,
        started_mono=time.monotonic(),
    )
    ran.effect = {
        "seq": 2,
        "boot_id": "b" * 32,
        "kind": "process.run",
        "at": datetime.now(UTC).isoformat(),
        "principal_id": _OWNER,
        "run_id": "run-2",
        "process_id": ran.process_id,
        "state": "running",
    }
    proc_service._finalize(ran, "exited")
    assert ran.effect is not None
    assert ran.effect["kind"] == "process.run"


def test_client_rejects_malformed_effect() -> None:
    good = {
        "seq": 1,
        "boot_id": "b" * 32,
        "kind": "fs.write",
        "at": "2026-11-25T00:00:00+00:00",
        "principal_id": _OWNER,
        "run_id": None,
    }
    _validate_effect({"effect": good}, "host.fs.write")
    with pytest.raises(BridgeClientError):
        _validate_effect({"effect": {**good, "seq": 0}}, "host.fs.write")
    with pytest.raises(BridgeClientError):
        _validate_effect({"effect": {**good, "kind": "fs.patch"}}, "host.fs.write")
    with pytest.raises(BridgeClientError):
        _validate_effect({"effect": good}, "host.fs.stat")
    with pytest.raises(BridgeClientError):
        _validate_effect({"effect": "not-a-dict"}, "host.fs.write")
