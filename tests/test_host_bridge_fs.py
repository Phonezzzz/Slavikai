"""Bounded host FS API (E03-T04, ADR-0013 §6/§9/§10/§11).

Covers the ``host.fs.*`` capabilities over a real daemon socket plus
unit-level config validation: happy paths per capability, root escape
attempts (``..``, absolute, symlink components), secret denial inside
an allowed root, built-in write-exclusions without config entries,
frame budgeting on serialized JSON, chunked-write idempotence and
concurrent-create atomicity, unfinished-write cleanup,
cancel-before-commit (no late publish after a timeout response),
subtree refusal for move/delete, EXDEV as a typed error, freedesktop
trash with metadata rollback, and denied entries omitted from
list/search.
"""

from __future__ import annotations

import errno
import json
import os
import resource
import socket
import stat
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

import host_bridge.fs as fs_module
from host_bridge import client
from host_bridge.daemon import HostBridgeDaemon
from host_bridge.fs import (
    CancelToken,
    HostFsService,
    Root,
    RootsRegistry,
    load_registry,
)
from host_bridge.protocol import (
    MAX_FRAME_BYTES,
    BridgeConfigError,
    BridgeRequest,
    encode_request,
)


def _start_daemon(daemon: HostBridgeDaemon) -> threading.Thread:
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    fake_home = tmp_path / "home"
    (fake_home / "Project" / "work").mkdir(parents=True)
    (fake_home / "Documents" / "docs").mkdir(parents=True)
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
        config_path=None,
        socket_dir=None,
    )


@pytest.fixture()
def work_root(registry: RootsRegistry) -> Path:
    root = registry.by_id("work")
    assert root is not None
    return root.path


@pytest.fixture()
def service(registry: RootsRegistry, tmp_path: Path) -> HostFsService:
    return HostFsService(registry, trash_root=tmp_path / "Trash")


@pytest.fixture()
def bridge(tmp_path: Path, service: HostFsService) -> Iterator[Path]:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=service)
    thread = _start_daemon(daemon)
    try:
        yield socket_path
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _fs(bridge: Path, capability: str, **params: object) -> dict[str, object]:
    return client.call(bridge, capability, params=params)


def _fs_error(bridge: Path, capability: str, **params: object) -> str:
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(bridge, capability, params=params)
    return excinfo.value.code


def _seed(work_root: Path, rel: str, content: str | bytes) -> Path:
    path = work_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


# --- stat / list / read happy paths ---


def test_stat_file_and_root(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "a.txt", "hello")
    result = _fs(bridge, "host.fs.stat", root_id="work", path="a.txt")
    assert result["type"] == "file"
    assert result["size"] == 5
    assert result["path"] == "a.txt"
    root_stat = _fs(bridge, "host.fs.stat", root_id="work", path=".")
    assert root_stat["type"] == "directory"
    assert root_stat["path"] == "."


def test_stat_missing_is_path_not_found(bridge: Path) -> None:
    assert _fs_error(bridge, "host.fs.stat", root_id="work", path="nope.txt") == "path_not_found"


def test_list_with_cursor(bridge: Path, work_root: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        _seed(work_root, name, "x")
    page1 = _fs(bridge, "host.fs.list", root_id="work", path=".", limit=2)
    assert [e["name"] for e in page1["entries"]] == ["a.txt", "b.txt"]  # type: ignore[index]
    assert page1["truncated"] is True
    page2 = _fs(
        bridge, "host.fs.list", root_id="work", path=".", limit=2, cursor=page1["next_cursor"]
    )
    assert [e["name"] for e in page2["entries"]] == ["c.txt"]  # type: ignore[index]
    assert page2["truncated"] is False


def test_read_chunked_offsets(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "data.txt", "0123456789")
    first = _fs(bridge, "host.fs.read", root_id="work", path="data.txt", offset=0, length=4)
    assert first["content"] == "0123"
    assert first["bytes_read"] == 4
    assert first["truncated"] is True
    rest = _fs(bridge, "host.fs.read", root_id="work", path="data.txt", offset=4, length=100)
    assert rest["content"] == "456789"
    assert rest["truncated"] is False
    assert rest["size"] == 10


def test_read_frame_budget_counts_json_escaping(bridge: Path, work_root: Path) -> None:
    # 40k newlines: each serializes as \n (2 bytes), so a raw 32 KiB
    # chunk would overflow the 64 KiB frame; the daemon must shrink
    # by serialized size and still advance the offset.
    _seed(work_root, "esc.txt", "\n" * 40_000)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(bridge))
        conn.sendall(
            encode_request(
                "host.fs.read",
                request_id="esc-1",
                params={"root_id": "work", "path": "esc.txt", "offset": 0, "length": 32_768},
            )
        )
        frame = conn.makefile("rb").readline(MAX_FRAME_BYTES + 1)
    assert len(frame) - 1 <= MAX_FRAME_BYTES
    payload = json.loads(frame.decode("utf-8"))
    result = payload["result"]
    assert 0 < result["bytes_read"] < 40_000
    assert result["truncated"] is True
    follow = _fs(
        bridge,
        "host.fs.read",
        root_id="work",
        path="esc.txt",
        offset=result["bytes_read"],
        length=32_768,
    )
    assert follow["offset"] == result["bytes_read"]


def test_read_typed_errors(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "bin.dat", b"\x00\x01\x02")
    assert _fs_error(bridge, "host.fs.read", root_id="work", path="bin.dat") == "binary_file"
    _seed(work_root, "bad.txt", b"\xff\xfe")
    assert _fs_error(bridge, "host.fs.read", root_id="work", path="bad.txt") == "invalid_utf8"
    _seed(work_root, "small.txt", "abc")
    assert (
        _fs_error(bridge, "host.fs.read", root_id="work", path="small.txt", offset=99)
        == "offset_out_of_range"
    )
    _seed(work_root, "uni.txt", "éé")
    # offset inside a multibyte character
    assert (
        _fs_error(bridge, "host.fs.read", root_id="work", path="uni.txt", offset=1)
        == "invalid_utf8"
    )
    # length too small to advance past one character
    assert (
        _fs_error(bridge, "host.fs.read", root_id="work", path="uni.txt", offset=0, length=1)
        == "invalid_params"
    )


# --- path safety: escapes, symlinks, secrets, write-exclusions ---


def test_path_escapes_rejected(bridge: Path) -> None:
    for bad in ("../secret", "/etc/passwd", "a/../../x", "~/.ssh/id_rsa"):
        assert _fs_error(bridge, "host.fs.read", root_id="work", path=bad) == "invalid_path", bad


def test_symlink_component_rejected_at_operation_time(bridge: Path, work_root: Path) -> None:
    # The symlink target is INSIDE the root, so lexical classification
    # of the resolved path passes; the fd-relative walk must still
    # refuse traversal (owner review criterion: dir_fd restrictions
    # hold at operation time, not only at validation time).
    _seed(work_root, "real/file.txt", "data")
    (work_root / "link").symlink_to(work_root / "real", target_is_directory=True)
    assert (
        _fs_error(bridge, "host.fs.read", root_id="work", path="link/file.txt")
        == "path_outside_root"
    )


def test_stat_symlink_target_containment(bridge: Path, work_root: Path, tmp_path: Path) -> None:
    _seed(work_root, "real.txt", "data")
    (work_root / "inside-link").symlink_to("real.txt")
    result = _fs(bridge, "host.fs.stat", root_id="work", path="inside-link")
    assert result["type"] == "symlink"
    # Target outside the root but in an ordinary location, so the
    # classification layer passes and the symlink-containment check
    # is what refuses it.
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("outside", encoding="utf-8")
    (work_root / "outside-link").symlink_to(outside)
    assert (
        _fs_error(bridge, "host.fs.stat", root_id="work", path="outside-link")
        == "path_outside_root"
    )


def test_secret_inside_root_denied_and_omitted(bridge: Path, work_root: Path) -> None:
    _seed(work_root, ".env", "SECRET=1")
    _seed(work_root, "ok.txt", "fine")
    assert _fs_error(bridge, "host.fs.read", root_id="work", path=".env") == "protected_path"
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path=".env",
            content="X=2",
            commit=True,
        )
        == "protected_path"
    )
    listing = _fs(bridge, "host.fs.list", root_id="work", path=".")
    assert [e["name"] for e in listing["entries"]] == ["ok.txt"]  # type: ignore[index]


def test_write_excluded_checkout_without_config_entry(bridge: Path, work_root: Path) -> None:
    checkout = work_root / "checkout"
    (checkout / "host_bridge").mkdir(parents=True)
    (checkout / "host_bridge" / "daemon.py").write_text("# daemon", encoding="utf-8")
    (checkout / "docs").mkdir()
    (checkout / "docs" / "runtime_contract_claims.json").write_text("{}", encoding="utf-8")
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="checkout/new.txt",
            content="x",
            commit=True,
        )
        == "write_excluded"
    )


# --- write: create/overwrite, chunking, concurrency, cleanup ---


def test_write_create_overwrite_and_conflict(bridge: Path, work_root: Path) -> None:
    result = _fs(
        bridge, "host.fs.write", root_id="work", path="new.txt", content="hello", commit=True
    )
    assert result["committed"] is True
    assert (work_root / "new.txt").read_text(encoding="utf-8") == "hello"
    assert (
        _fs_error(
            bridge, "host.fs.write", root_id="work", path="new.txt", content="again", commit=True
        )
        == "already_exists"
    )
    _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="new.txt",
        mode="overwrite",
        content="replaced",
        commit=True,
    )
    assert (work_root / "new.txt").read_text(encoding="utf-8") == "replaced"


def test_write_chunked_idempotent_resend(bridge: Path, work_root: Path) -> None:
    begin = _fs(bridge, "host.fs.write", root_id="work", path="chunked.txt", content="abc")
    write_id = begin["write_id"]
    # re-send the same chunk at the same offset: idempotent, no growth
    again = _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="chunked.txt",
        content="abc",
        offset=0,
        write_id=write_id,
    )
    assert again["size"] == 3
    _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="chunked.txt",
        content="def",
        offset=3,
        write_id=write_id,
        commit=True,
    )
    assert (work_root / "chunked.txt").read_text(encoding="utf-8") == "abcdef"


def test_concurrent_create_one_wins(bridge: Path, work_root: Path) -> None:
    first = _fs(bridge, "host.fs.write", root_id="work", path="race.txt", content="winner")
    second = _fs(bridge, "host.fs.write", root_id="work", path="race.txt", content="loser!")
    _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="race.txt",
        content="",
        offset=6,
        write_id=first["write_id"],
        commit=True,
    )
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="race.txt",
            content="",
            offset=6,
            write_id=second["write_id"],
            commit=True,
        )
        == "already_exists"
    )
    assert (work_root / "race.txt").read_text(encoding="utf-8") == "winner"


def test_unfinished_write_cleanup(registry: RootsRegistry, tmp_path: Path, work_root: Path) -> None:
    now = [1000.0]
    service = HostFsService(registry, trash_root=tmp_path / "Trash", clock=lambda: now[0])
    begin = service.handle(
        BridgeRequest(
            version=1,
            request_id="w1",
            capability="host.fs.write",
            params={"root_id": "work", "path": "stale.txt", "content": "partial"},
        ),
        CancelToken(),
    )
    write_id = begin["write_id"]
    temps = [p for p in work_root.iterdir() if ".hb-" in p.name]
    assert len(temps) == 1
    now[0] += 601.0  # past the session TTL
    service.handle(
        BridgeRequest(
            version=1,
            request_id="w2",
            capability="host.fs.write",
            params={"root_id": "work", "path": "other.txt", "content": "x", "commit": True},
        ),
        CancelToken(),
    )
    assert [p for p in work_root.iterdir() if ".hb-" in p.name] == []
    with pytest.raises(Exception) as excinfo:
        service.handle(
            BridgeRequest(
                version=1,
                request_id="w3",
                capability="host.fs.write",
                params={
                    "root_id": "work",
                    "path": "stale.txt",
                    "content": "",
                    "offset": 7,
                    "write_id": write_id,
                    "commit": True,
                },
            ),
            CancelToken(),
        )
    assert getattr(excinfo.value, "code", None) == "invalid_params"


def test_cancel_before_commit_no_late_write(
    tmp_path: Path, registry: RootsRegistry, work_root: Path
) -> None:
    release = threading.Event()
    service = HostFsService(
        registry,
        trash_root=tmp_path / "Trash",
        commit_hook=lambda: release.wait(timeout=5.0),
    )
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(
        socket_path,
        fs_service=service,
        capability_deadlines={"host.fs.write": 0.3},
    )
    thread = _start_daemon(daemon)
    try:
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.call(
                socket_path,
                "host.fs.write",
                params={
                    "root_id": "work",
                    "path": "cancelled.txt",
                    "content": "must never land",
                    "commit": True,
                },
            )
        assert excinfo.value.code == "request_timeout"
        release.set()  # let the abandoned handler reach the commit gate
        time.sleep(0.5)
        assert not (work_root / "cancelled.txt").exists()
        assert client.ping(socket_path) == {"pong": True}
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


# --- patch ---


def test_patch_happy_dry_run_and_errors(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "code.txt", "a\nb\nc\n")
    patch = "@@ -1,3 +1,3 @@\n a\n-b\n+B\n c\n"
    dry = _fs(bridge, "host.fs.patch", root_id="work", path="code.txt", patch=patch, dry_run=True)
    assert dry["applied"] is False
    assert (work_root / "code.txt").read_text(encoding="utf-8") == "a\nb\nc\n"
    done = _fs(bridge, "host.fs.patch", root_id="work", path="code.txt", patch=patch)
    assert done["applied"] is True
    assert (work_root / "code.txt").read_text(encoding="utf-8") == "a\nB\nc\n"
    assert (
        _fs_error(
            bridge,
            "host.fs.patch",
            root_id="work",
            path="code.txt",
            patch="@@ -1,1 +1,1 @@\n-zzz\n+yyy\n",
        )
        == "patch_mismatch"
    )
    assert (
        _fs_error(bridge, "host.fs.patch", root_id="work", path="code.txt", patch="no hunks")
        == "invalid_patch"
    )


# --- move ---


def test_move_happy_and_existing(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "src.txt", "payload")
    _seed(work_root, "dst.txt", "old")
    assert (
        _fs_error(bridge, "host.fs.move", root_id="work", path="src.txt", dest_path="dst.txt")
        == "already_exists"
    )
    _fs(
        bridge,
        "host.fs.move",
        root_id="work",
        path="src.txt",
        dest_path="moved/dst.txt",
    )
    assert not (work_root / "src.txt").exists()
    assert (work_root / "moved" / "dst.txt").read_text(encoding="utf-8") == "payload"


def test_move_dir_with_protected_descendant_refused(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "tree/.env", "SECRET=1")
    _seed(work_root, "tree/ok.txt", "fine")
    assert (
        _fs_error(bridge, "host.fs.move", root_id="work", path="tree", dest_path="tree2")
        == "protected_path"
    )
    assert (work_root / "tree" / "ok.txt").exists()


def test_move_cross_device_is_typed(
    bridge: Path, work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A real second filesystem cannot be used here (any root under
    # /dev or /run is deny-classified by design), so EXDEV is
    # simulated at the rename syscall: the typed error must surface
    # and the source must stay untouched (no copy+delete fallback).
    # overwrite=false goes through the atomic renameat2 helper.
    _seed(work_root, "xdev.txt", "data")

    def exdev_rename(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EXDEV, "simulated cross-device rename")

    monkeypatch.setattr(fs_module, "_rename_noreplace", exdev_rename)
    assert (
        _fs_error(
            bridge,
            "host.fs.move",
            root_id="work",
            path="xdev.txt",
            dest_path="moved.txt",
        )
        == "cross_device_move"
    )
    assert (work_root / "xdev.txt").exists()
    assert not (work_root / "moved.txt").exists()


# --- delete (trash) ---


def test_delete_trash_with_info(bridge: Path, work_root: Path, tmp_path: Path) -> None:
    _seed(work_root, "gone.txt", "bye")
    result = _fs(bridge, "host.fs.delete", root_id="work", path="gone.txt")
    assert result["recoverable"] is True
    assert not (work_root / "gone.txt").exists()
    trash_file = tmp_path / "Trash" / "files" / "gone.txt"
    info_file = tmp_path / "Trash" / "info" / "gone.txt.trashinfo"
    assert trash_file.read_text(encoding="utf-8") == "bye"
    assert "Path=" in info_file.read_text(encoding="utf-8")


def test_delete_dir_requires_recursive(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "dir/inner.txt", "x")
    assert _fs_error(bridge, "host.fs.delete", root_id="work", path="dir") == "is_a_directory"
    _fs(bridge, "host.fs.delete", root_id="work", path="dir", recursive=True)
    assert not (work_root / "dir").exists()


def test_delete_metadata_rollback_on_failure(
    bridge: Path, work_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(work_root, "stuck.txt", "stay")
    real_noreplace = fs_module._rename_noreplace

    def failing_noreplace(*args: object, **kwargs: object) -> None:
        raise OSError(errno.EIO, "simulated rename failure")

    monkeypatch.setattr(fs_module, "_rename_noreplace", failing_noreplace)
    try:
        assert _fs_error(bridge, "host.fs.delete", root_id="work", path="stuck.txt") == "io_error"
    finally:
        monkeypatch.setattr(fs_module, "_rename_noreplace", real_noreplace)
    assert (work_root / "stuck.txt").exists()
    info_dir = tmp_path / "Trash" / "info"
    assert not info_dir.exists() or list(info_dir.iterdir()) == []


# --- search / capabilities / misc ---


def test_search_name_only_omits_denied(bridge: Path, work_root: Path) -> None:
    _seed(work_root, "report.txt", "x")
    _seed(work_root, "sub/report2.txt", "x")
    _seed(work_root, ".env", "SECRET=1")
    result = _fs(bridge, "host.fs.search", root_id="work", query="report")
    paths = sorted(m["path"] for m in result["matches"])  # type: ignore[index]
    assert paths == ["report.txt", "sub/report2.txt"]
    denied = _fs(bridge, "host.fs.search", root_id="work", query="env")
    assert denied["matches"] == []


def test_capabilities_roots_without_paths(bridge: Path) -> None:
    roots = client.roots(bridge)
    assert roots == [
        {"id": "work", "mode": "rw"},
        {"id": "docs", "mode": "ro"},
    ]


def test_read_only_root_and_unknown_root(bridge: Path, home: Path) -> None:
    docs = home / "Documents" / "docs"
    (docs / "note.txt").write_text("read me", encoding="utf-8")
    result = _fs(bridge, "host.fs.read", root_id="docs", path="note.txt")
    assert result["content"] == "read me"
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="docs",
            path="new.txt",
            content="x",
            commit=True,
        )
        == "root_read_only"
    )
    assert _fs_error(bridge, "host.fs.stat", root_id="nope", path=".") == "unknown_root"
    assert _fs_error(bridge, "host.fs.stat", root_id="work", path=".", bogus=1) == "invalid_params"


# --- roots config validation (fail-closed) ---


def _config_env(tmp_path: Path, payload: object) -> dict[str, str]:
    config = tmp_path / "host-bridge.json"
    config.write_text(
        payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8"
    )
    return {"SLAVIK_HOST_BRIDGE_CONFIG": str(config)}


def test_config_absent_gives_builtin_workspace_only(tmp_path: Path, home: Path) -> None:
    env = {"SLAVIK_HOST_BRIDGE_CONFIG": str(tmp_path / "missing.json")}
    registry = load_registry(env=env, home=home)
    assert registry.describe() == [{"id": "workspace", "mode": "rw"}]


def test_config_accepts_valid_roots(tmp_path: Path, home: Path) -> None:
    env = _config_env(
        tmp_path,
        {
            "roots": [
                {"id": "proj", "path": "~/Project/work", "mode": "rw"},
                {"id": "workspace", "path": "~/.local/share/slavikai/host-workspace", "mode": "rw"},
            ]
        },
    )
    registry = load_registry(env=env, home=home)
    assert registry.describe() == [
        {"id": "proj", "mode": "rw"},
        {"id": "workspace", "mode": "rw"},
    ]


def test_config_rejects_outside_allowlist(tmp_path: Path, home: Path) -> None:
    (home / "Other").mkdir()
    env = _config_env(tmp_path, {"roots": [{"id": "bad", "path": "~/Other", "mode": "rw"}]})
    with pytest.raises(BridgeConfigError):
        load_registry(env=env, home=home)


def test_config_rejects_parent_and_never_root_dirs(tmp_path: Path, home: Path) -> None:
    (home / ".config" / "slavikai").mkdir(parents=True)
    for raw_path in ("~/Project", "~/.config/slavikai"):
        env = _config_env(tmp_path, {"roots": [{"id": "bad", "path": raw_path, "mode": "rw"}]})
        with pytest.raises(BridgeConfigError):
            load_registry(env=env, home=home)


def test_config_rejects_overlap_duplicate_and_bad_mode(tmp_path: Path, home: Path) -> None:
    (home / "Project" / "work" / "nested").mkdir(parents=True)
    overlap = _config_env(
        tmp_path,
        {
            "roots": [
                {"id": "a", "path": "~/Project/work", "mode": "rw"},
                {"id": "b", "path": "~/Project/work/nested", "mode": "ro"},
            ]
        },
    )
    with pytest.raises(BridgeConfigError):
        load_registry(env=overlap, home=home)
    duplicate = _config_env(
        tmp_path,
        {
            "roots": [
                {"id": "a", "path": "~/Project/work", "mode": "rw"},
                {"id": "a", "path": "~/Documents/docs", "mode": "ro"},
            ]
        },
    )
    with pytest.raises(BridgeConfigError):
        load_registry(env=duplicate, home=home)
    bad_mode = _config_env(
        tmp_path, {"roots": [{"id": "a", "path": "~/Project/work", "mode": "write"}]}
    )
    with pytest.raises(BridgeConfigError):
        load_registry(env=bad_mode, home=home)


def test_config_rejects_malformed_json(tmp_path: Path, home: Path) -> None:
    env = _config_env(tmp_path, "{not json")
    with pytest.raises(BridgeConfigError):
        load_registry(env=env, home=home)


# --- Codex rework regressions (PR #196 review) ---


def test_staging_symlink_swap_cannot_redirect_write(
    bridge: Path, work_root: Path, tmp_path: Path
) -> None:
    # P1: the staging temp name is learnable via list; swapping it for
    # a symlink to a file outside the root (via move overwrite) must
    # not redirect later chunk writes. The staging file keeps its
    # identity until commit.
    outside = tmp_path / "outside.txt"
    outside.write_text("ORIGINAL", encoding="utf-8")
    begin = _fs(bridge, "host.fs.write", root_id="work", path="victim.txt", content="abc")
    write_id = begin["write_id"]
    temps = [p.name for p in work_root.iterdir() if ".hb-" in p.name]
    assert len(temps) == 1
    (work_root / "evil-link").symlink_to(outside)
    _fs(
        bridge,
        "host.fs.move",
        root_id="work",
        path="evil-link",
        dest_path=temps[0],
        overwrite=True,
    )
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="victim.txt",
            content="XYZ",
            offset=3,
            write_id=write_id,
        )
        == "path_outside_root"
    )
    assert outside.read_text(encoding="utf-8") == "ORIGINAL"
    # Commit must refuse to publish the swapped-in symlink as well.
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="victim.txt",
            content="",
            offset=3,
            write_id=write_id,
            commit=True,
        )
        == "path_outside_root"
    )
    assert outside.read_text(encoding="utf-8") == "ORIGINAL"
    assert not (work_root / "victim.txt").exists()


def test_delete_timeout_during_trashinfo_write_keeps_source(
    tmp_path: Path,
    registry: RootsRegistry,
    work_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # P1: if the .trashinfo write runs past the deadline, the client
    # gets request_timeout and the source must remain in place even
    # after the abandoned worker finishes (cancel-before-commit).
    _seed(work_root, "keep.txt", "stay")
    real_write_all = fs_module._write_all

    def slow_write_all(fd: int, data: bytes) -> None:
        time.sleep(0.6)
        real_write_all(fd, data)

    monkeypatch.setattr(fs_module, "_write_all", slow_write_all)
    service = HostFsService(registry, trash_root=tmp_path / "Trash")
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(
        socket_path,
        fs_service=service,
        capability_deadlines={"host.fs.delete": 0.3},
    )
    thread = _start_daemon(daemon)
    try:
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.call(
                socket_path,
                "host.fs.delete",
                params={"root_id": "work", "path": "keep.txt"},
            )
        assert excinfo.value.code == "request_timeout"
        time.sleep(1.0)  # let the abandoned worker finish the info write
        assert (work_root / "keep.txt").read_text(encoding="utf-8") == "stay"
        info_dir = tmp_path / "Trash" / "info"
        files_dir = tmp_path / "Trash" / "files"
        assert not info_dir.exists() or list(info_dir.iterdir()) == []
        assert not files_dir.exists() or list(files_dir.iterdir()) == []
        assert client.ping(socket_path) == {"pong": True}
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_new_checkout_protected_immediately(bridge: Path, work_root: Path) -> None:
    # P1: a directory checked while it was not a checkout must become
    # write-excluded as soon as checkout markers appear, without a
    # daemon restart (negative results are not cached).
    _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="proj/notes.txt",
        content="plain",
        commit=True,
    )
    checkout = work_root / "proj"
    (checkout / "host_bridge").mkdir()
    (checkout / "host_bridge" / "daemon.py").write_text("# daemon", encoding="utf-8")
    (checkout / "docs").mkdir()
    (checkout / "docs" / "runtime_contract_claims.json").write_text("{}", encoding="utf-8")
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="proj/new.txt",
            content="x",
            commit=True,
        )
        == "write_excluded"
    )
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="proj/notes.txt",
            mode="overwrite",
            content="tampered",
            commit=True,
        )
        == "write_excluded"
    )
    assert (checkout / "notes.txt").read_text(encoding="utf-8") == "plain"


def test_move_noreplace_refuses_concurrent_destination(
    bridge: Path, work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P2: with overwrite=false, a destination created after any
    # existence check but before the rename syscall must survive;
    # the atomic renameat2(RENAME_NOREPLACE) refuses instead.
    _seed(work_root, "src.txt", "payload")
    real_noreplace = fs_module._rename_noreplace

    def racing_noreplace(src: str, src_fd: int, dst: str, dst_fd: int) -> None:
        (work_root / "dst.txt").write_text("concurrent", encoding="utf-8")
        real_noreplace(src, src_fd, dst, dst_fd)

    monkeypatch.setattr(fs_module, "_rename_noreplace", racing_noreplace)
    assert (
        _fs_error(bridge, "host.fs.move", root_id="work", path="src.txt", dest_path="dst.txt")
        == "already_exists"
    )
    assert (work_root / "dst.txt").read_text(encoding="utf-8") == "concurrent"
    assert (work_root / "src.txt").read_text(encoding="utf-8") == "payload"


def test_search_and_subtree_respect_mount_points(
    registry: RootsRegistry, tmp_path: Path, work_root: Path
) -> None:
    # P2: a same-device bind mount inside the root must not be
    # descended by search (names/metadata beyond the tree) and must
    # refuse a subtree move, matching direct list behavior.
    _seed(work_root, "report-top.txt", "x")
    _seed(work_root, "mounted/report-inside.txt", "x")
    _seed(work_root, "tree/mounted2/inner.txt", "x")
    mounted = (work_root / "mounted").resolve()
    mounted2 = (work_root / "tree" / "mounted2").resolve()
    service = HostFsService(
        registry,
        trash_root=tmp_path / "Trash",
        mount_bindings_fn=lambda: [(mounted, mounted), (mounted2, mounted2)],
    )
    result = service.handle(
        BridgeRequest(
            version=1,
            request_id="s1",
            capability="host.fs.search",
            params={"root_id": "work", "query": "report"},
        ),
        CancelToken(),
    )
    paths = sorted(m["path"] for m in result["matches"])  # type: ignore[index]
    assert paths == ["report-top.txt"]
    with pytest.raises(Exception) as excinfo:
        service.handle(
            BridgeRequest(
                version=1,
                request_id="m1",
                capability="host.fs.move",
                params={"root_id": "work", "path": "tree", "dest_path": "tree2"},
            ),
            CancelToken(),
        )
    assert getattr(excinfo.value, "code", None) == "path_outside_root"
    assert (work_root / "tree" / "mounted2" / "inner.txt").exists()


def test_subtree_walk_bounded_fds(registry: RootsRegistry, tmp_path: Path, work_root: Path) -> None:
    # P2: moving a wide tree must not need one open fd per directory;
    # with a tight RLIMIT_NOFILE the old walk died with EMFILE.
    for i in range(150):
        _seed(work_root, f"tree/d{i:03d}/f.txt", "x")
    service = HostFsService(registry, trash_root=tmp_path / "Trash")
    open_fds = len(os.listdir("/proc/self/fd"))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_fds + 12, hard))
    try:
        result = service.handle(
            BridgeRequest(
                version=1,
                request_id="mv1",
                capability="host.fs.move",
                params={"root_id": "work", "path": "tree", "dest_path": "tree2"},
            ),
            CancelToken(),
        )
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert result["moved"] is True
    assert (work_root / "tree2" / "d149" / "f.txt").exists()


def test_patch_preserves_mode(bridge: Path, work_root: Path) -> None:
    # P2: patch publishes a new inode; the original permission bits
    # must be carried over (0755 executable must not become 0600).
    patch = "@@ -1,1 +1,1 @@\n-echo old\n+echo new\n"
    for mode in (0o755, 0o644, 0o700, 0o640):
        script = _seed(work_root, f"script-{mode:o}.sh", "echo old\n")
        script.chmod(mode)
        done = _fs(bridge, "host.fs.patch", root_id="work", path=script.name, patch=patch)
        assert done["applied"] is True
        assert script.read_text(encoding="utf-8") == "echo new\n"
        assert stat.S_IMODE(script.stat().st_mode) == mode


def test_list_long_names_frame_budget_with_cursor(bridge: Path, work_root: Path) -> None:
    # P2: 500 entries with ~192-char names at limit=500 — the frame
    # budget must include the actual next_cursor; the client gets
    # pages, not an internal_error.
    names = [f"f{i:03d}" + "x" * 185 + ".txt" for i in range(500)]
    for name in names:
        _seed(work_root, name, "x")
    collected: list[str] = []
    cursor: object = None
    for _ in range(20):
        params: dict[str, object] = {"root_id": "work", "path": ".", "limit": 500}
        if cursor is not None:
            params["cursor"] = cursor
        page = _fs(bridge, "host.fs.list", **params)
        entries = page["entries"]  # type: ignore[index]
        assert entries, "a page must never be empty while entries remain"
        collected.extend(e["name"] for e in entries)  # type: ignore[index]
        if not page["truncated"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
        assert cursor == entries[-1]["name"]  # type: ignore[index]
    else:
        raise AssertionError("pagination did not terminate")
    assert collected == names


# --- second Codex review on E03-T04 (c018b57 findings) ---


def test_root_inside_checkout_write_excluded(tmp_path: Path, home: Path) -> None:
    # P1: a root configured INSIDE another SlavikAI checkout must not
    # become a write path into that clone: checkout detection walks
    # ancestors of the root, not only from the root down. Reads stay
    # allowed (ADR-0013 §9 write-exclusion, not root rejection).
    clone = home / "Project" / "clone"
    (clone / "host_bridge").mkdir(parents=True)
    (clone / "host_bridge" / "daemon.py").write_text("ORIGINAL", encoding="utf-8")
    (clone / "docs").mkdir()
    (clone / "docs" / "runtime_contract_claims.json").write_text("{}", encoding="utf-8")
    inner_registry = RootsRegistry(
        roots=(Root(id="inner", path=(clone / "host_bridge").resolve(), mode="rw"),),
        write_excluded=(),
        home=home.resolve(),
        config_path=None,
        socket_dir=None,
    )
    service = HostFsService(inner_registry, trash_root=tmp_path / "Trash")
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, fs_service=service)
    thread = _start_daemon(daemon)
    try:
        assert (
            _fs_error(
                socket_path,
                "host.fs.write",
                root_id="inner",
                path="daemon.py",
                mode="overwrite",
                content="tampered",
                commit=True,
            )
            == "write_excluded"
        )
        assert (
            _fs_error(
                socket_path,
                "host.fs.write",
                root_id="inner",
                path="new.txt",
                content="x",
                commit=True,
            )
            == "write_excluded"
        )
        read = _fs(socket_path, "host.fs.read", root_id="inner", path="daemon.py")
        assert read["content"] == "ORIGINAL"
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)
    assert (clone / "host_bridge" / "daemon.py").read_text(encoding="utf-8") == "ORIGINAL"


def test_read_and_patch_fifo_not_a_file_without_hanging(
    service: HostFsService, work_root: Path
) -> None:
    # P2: opening a FIFO without a writer must not block the worker
    # past the deadline; O_NONBLOCK + fstat type check refuses typed.
    os.mkfifo(work_root / "pipe")
    outcomes: dict[str, object] = {}

    def run(capability: str, params: dict[str, object], key: str) -> None:
        try:
            service.handle(
                BridgeRequest(version=1, request_id=key, capability=capability, params=params),
                CancelToken(),
            )
            outcomes[key] = "ok"
        except Exception as exc:  # noqa: BLE001 - test probe records the typed code
            outcomes[key] = getattr(exc, "code", repr(exc))

    threads = [
        threading.Thread(
            target=run,
            args=("host.fs.read", {"root_id": "work", "path": "pipe"}, "read"),
            daemon=True,
        ),
        threading.Thread(
            target=run,
            args=(
                "host.fs.patch",
                {"root_id": "work", "path": "pipe", "patch": "@@ -1,1 +1,1 @@\n-a\n+b\n"},
                "patch",
            ),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(thread.is_alive() for thread in threads), "FIFO open blocked the worker"
    assert outcomes == {"read": "not_a_file", "patch": "not_a_file"}


def test_read_and_patch_final_component_bind_mount_refused(
    registry: RootsRegistry, tmp_path: Path, work_root: Path
) -> None:
    # P2: a regular file bind-mounted directly onto a file inside the
    # root must be refused: the mount check covers the final
    # component, not only parent directories.
    target = _seed(work_root, "bound.txt", "outside-content")
    mounted = target.resolve()
    service = HostFsService(
        registry,
        trash_root=tmp_path / "Trash",
        mount_bindings_fn=lambda: [(mounted, mounted)],
    )
    cases: tuple[tuple[str, dict[str, object]], ...] = (
        ("host.fs.read", {"root_id": "work", "path": "bound.txt"}),
        (
            "host.fs.patch",
            {"root_id": "work", "path": "bound.txt", "patch": "@@ -1,1 +1,1 @@\n-a\n+b\n"},
        ),
    )
    for capability, params in cases:
        with pytest.raises(Exception) as excinfo:
            service.handle(
                BridgeRequest(version=1, request_id="bm", capability=capability, params=params),
                CancelToken(),
            )
        assert getattr(excinfo.value, "code", None) == "path_outside_root"
    assert target.read_text(encoding="utf-8") == "outside-content"


def test_write_session_chunk_commit_serialized(
    bridge: Path, work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P2: chunk and commit on one write_id are serialized: a delayed
    # pwrite must never land after commit published the inode.
    begin = _fs(bridge, "host.fs.write", root_id="work", path="race.txt", content="old")
    write_id = begin["write_id"]
    real_pwrite = os.pwrite
    pwrite_started = threading.Event()

    def slow_pwrite(fd: int, data: bytes, offset: int) -> int:
        if data == b"NEW":
            pwrite_started.set()
            time.sleep(0.5)
        return real_pwrite(fd, data, offset)

    monkeypatch.setattr(os, "pwrite", slow_pwrite)
    chunk_outcome: dict[str, object] = {}

    def chunk() -> None:
        try:
            chunk_outcome["result"] = _fs(
                bridge,
                "host.fs.write",
                root_id="work",
                path="race.txt",
                content="NEW",
                offset=0,
                write_id=write_id,
            )
        except Exception as exc:  # noqa: BLE001 - test probe records the typed code
            chunk_outcome["error"] = getattr(exc, "code", repr(exc))

    thread = threading.Thread(target=chunk, daemon=True)
    thread.start()
    assert pwrite_started.wait(timeout=5.0)
    committed = _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="race.txt",
        content="",
        offset=3,
        write_id=write_id,
        commit=True,
    )
    assert committed["committed"] is True
    at_commit = (work_root / "race.txt").read_text(encoding="utf-8")
    thread.join(timeout=5.0)
    assert not thread.is_alive()
    # No post-publish mutation: the content seen when commit returned
    # is the final content, and the serialized chunk landed first.
    assert (work_root / "race.txt").read_text(encoding="utf-8") == at_commit == "NEW"
    assert "result" in chunk_outcome


def test_delete_trash_name_collision_preserves_existing(
    bridge: Path, work_root: Path, tmp_path: Path
) -> None:
    # P2: an orphaned Trash/files entry without .trashinfo must not be
    # destroyed by a later delete of a same-named file; the publish is
    # atomic no-replace and picks another name on collision.
    files_dir = tmp_path / "Trash" / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "a.txt").write_text("previous", encoding="utf-8")
    _seed(work_root, "a.txt", "new")
    result = _fs(bridge, "host.fs.delete", root_id="work", path="a.txt")
    assert result["recoverable"] is True
    trash_name = str(result["trash_name"])
    assert trash_name != "a.txt"
    assert (files_dir / "a.txt").read_text(encoding="utf-8") == "previous"
    assert (files_dir / trash_name).read_text(encoding="utf-8") == "new"
    assert (tmp_path / "Trash" / "info" / f"{trash_name}.trashinfo").exists()
    assert not (work_root / "a.txt").exists()


def test_write_and_patch_long_names(bridge: Path, work_root: Path) -> None:
    # P2: staging names must not embed the target name: a valid
    # 220-byte file name plus a 41-byte suffix exceeded NAME_MAX and
    # made write/patch fail with ENAMETOOLONG.
    name = "a" * 216 + ".txt"
    assert len(name.encode()) == 220
    _fs(bridge, "host.fs.write", root_id="work", path=name, content="echo old\n", commit=True)
    assert (work_root / name).read_text(encoding="utf-8") == "echo old\n"
    patch = "@@ -1,1 +1,1 @@\n-echo old\n+echo new\n"
    done = _fs(bridge, "host.fs.patch", root_id="work", path=name, patch=patch)
    assert done["applied"] is True
    assert (work_root / name).read_text(encoding="utf-8") == "echo new\n"


def test_delete_timeout_during_trashinfo_create_keeps_source(
    tmp_path: Path,
    registry: RootsRegistry,
    work_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # P1 (strengthened): cancellation while the .trashinfo is being
    # CREATED (not only written) must also keep the source in place
    # after the abandoned worker finishes.
    _seed(work_root, "keep2.txt", "stay")
    real_open = os.open

    def slow_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if isinstance(path, str) and path.endswith(".trashinfo"):
            time.sleep(0.6)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", slow_open)
    service = HostFsService(registry, trash_root=tmp_path / "Trash")
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(
        socket_path,
        fs_service=service,
        capability_deadlines={"host.fs.delete": 0.3},
    )
    thread = _start_daemon(daemon)
    try:
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.call(
                socket_path,
                "host.fs.delete",
                params={"root_id": "work", "path": "keep2.txt"},
            )
        assert excinfo.value.code == "request_timeout"
        time.sleep(1.0)  # let the abandoned worker finish creating metadata
        assert (work_root / "keep2.txt").read_text(encoding="utf-8") == "stay"
        info_dir = tmp_path / "Trash" / "info"
        files_dir = tmp_path / "Trash" / "files"
        assert not info_dir.exists() or list(info_dir.iterdir()) == []
        assert not files_dir.exists() or list(files_dir.iterdir()) == []
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


# --- third Codex review on E03-T04 (03d6eeb findings) ---


def test_move_creating_excluded_destination_descendant_refused(tmp_path: Path, home: Path) -> None:
    # P1: write_excluded may name a path INSIDE the future destination.
    # Checking only the source subtree let a directory move create the
    # excluded file at the destination; descendants must also be
    # checked by their future destination paths.
    work = (home / "Project" / "work").resolve()
    _seed(work, "src/protected", "payload")
    _seed(work, "src/ok.txt", "fine")
    custom = RootsRegistry(
        roots=(Root(id="work", path=work, mode="rw"),),
        write_excluded=((work / "dest" / "protected").resolve(),),
        home=home.resolve(),
        config_path=None,
        socket_dir=None,
    )
    service = HostFsService(custom, trash_root=tmp_path / "Trash")
    with pytest.raises(Exception) as excinfo:
        service.handle(
            BridgeRequest(
                version=1,
                request_id="mv-excluded",
                capability="host.fs.move",
                params={"root_id": "work", "path": "src", "dest_path": "dest"},
            ),
            CancelToken(),
        )
    assert getattr(excinfo.value, "code", None) == "write_excluded"
    assert (work / "src" / "protected").exists()
    assert not (work / "dest" / "protected").exists()


def test_write_partial_pwrite_hard_limit_no_publish(
    bridge: Path, work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P2: os.pwrite may write fewer bytes than requested (file-size
    # limit / disk full). Ignoring the return value published a
    # truncated staging file as success and, on overwrite, destroyed
    # the previous full version. A chunk that cannot be written in
    # full must fail typed before publish.
    _seed(work_root, "keep.txt", "previous-full")
    real_pwrite = os.pwrite

    def limited_pwrite(fd: int, data: bytes, offset: int) -> int:
        if offset >= 4:
            raise OSError(errno.EFBIG, "simulated file size limit")
        return real_pwrite(fd, data[:4], offset)

    monkeypatch.setattr(os, "pwrite", limited_pwrite)
    assert (
        _fs_error(
            bridge,
            "host.fs.write",
            root_id="work",
            path="keep.txt",
            mode="overwrite",
            content="abcdefgh",
            commit=True,
        )
        == "io_error"
    )
    assert (work_root / "keep.txt").read_text(encoding="utf-8") == "previous-full"


def test_write_short_pwrite_loops_until_complete(
    bridge: Path, work_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # P2 (positive side): short-but-completable pwrites must be looped
    # over until the whole chunk is on disk; commit then publishes
    # the full content, not the first short write.
    real_pwrite = os.pwrite

    def short_pwrite(fd: int, data: bytes, offset: int) -> int:
        return real_pwrite(fd, data[:3], offset)

    monkeypatch.setattr(os, "pwrite", short_pwrite)
    result = _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="looped.txt",
        content="abcdefgh",
        commit=True,
    )
    assert result["committed"] is True
    assert result["size"] == 8
    assert (work_root / "looped.txt").read_text(encoding="utf-8") == "abcdefgh"


def test_delete_creates_home_trash_on_fresh_home(registry: RootsRegistry, work_root: Path) -> None:
    # P2: on a fresh home ~/.local/share/Trash does not exist yet.
    # The device must be taken from an existing ancestor and the
    # home trash created when it matches the source's device,
    # instead of failing trash_unavailable on the first ENOENT.
    home_trash = registry.home / ".local" / "share" / "Trash"
    assert not home_trash.exists()
    _seed(work_root, "gone.txt", "bye")
    service = HostFsService(registry, mount_bindings_fn=lambda: [])
    result = service.handle(
        BridgeRequest(
            version=1,
            request_id="del-fresh-home",
            capability="host.fs.delete",
            params={"root_id": "work", "path": "gone.txt"},
        ),
        CancelToken(),
    )
    assert result["recoverable"] is True
    assert (home_trash / "files" / "gone.txt").read_text(encoding="utf-8") == "bye"
    assert (home_trash / "info" / "gone.txt.trashinfo").exists()
    assert not (work_root / "gone.txt").exists()


def test_config_wrong_type_path_rejected(tmp_path: Path, home: Path) -> None:
    # P2: only a truly ABSENT config may fall back to the built-in
    # default registry. An existing directory or dangling symlink at
    # the config path is a broken owner config and must fail closed
    # with BridgeConfigError, not silently start with other powers.
    cfg_dir = tmp_path / "host-bridge.json"
    cfg_dir.mkdir()
    with pytest.raises(BridgeConfigError):
        load_registry(env={"SLAVIK_HOST_BRIDGE_CONFIG": str(cfg_dir)}, home=home)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(BridgeConfigError):
        load_registry(env={"SLAVIK_HOST_BRIDGE_CONFIG": str(dangling)}, home=home)
    absent = load_registry(
        env={"SLAVIK_HOST_BRIDGE_CONFIG": str(tmp_path / "absent.json")}, home=home
    )
    assert absent.describe() == [{"id": "workspace", "mode": "rw"}]


def test_orphan_sweep_never_deletes_user_files(tmp_path: Path, home: Path) -> None:
    # P2: the orphan sweep hard-deleted ANY file matching the
    # staging name pattern older than the TTL — including user
    # files and files in write_excluded paths — during an
    # unrelated write. Only staging files proven to be created by
    # this daemon (identity registry) may be removed by cleanup.
    work = (home / "Project" / "work").resolve()
    user_name = ".checkpoint.hb-" + "a" * 32 + ".tmp"
    excluded_name = ".keep.hb-" + "b" * 32 + ".tmp"
    _seed(work, user_name, "user data")
    _seed(work, excluded_name, "excluded data")
    old = time.time() - 3600
    os.utime(work / user_name, (old, old))
    os.utime(work / excluded_name, (old, old))
    custom = RootsRegistry(
        roots=(Root(id="work", path=work, mode="rw"),),
        write_excluded=((work / excluded_name).resolve(),),
        home=home.resolve(),
        config_path=None,
        socket_dir=None,
    )
    service = HostFsService(custom, trash_root=tmp_path / "Trash")
    # An unrelated committed write in the same directory runs the sweep.
    service.handle(
        BridgeRequest(
            version=1,
            request_id="n1-write",
            capability="host.fs.write",
            params={"root_id": "work", "path": "other.txt", "content": "x", "commit": True},
        ),
        CancelToken(),
    )
    assert (work / user_name).read_text(encoding="utf-8") == "user data"
    assert (work / excluded_name).read_text(encoding="utf-8") == "excluded data"
    # Positive side: a proven-internal orphan (staging file of a
    # session this daemon created, the session since lost) IS swept.
    begin = service.handle(
        BridgeRequest(
            version=1,
            request_id="n1-begin",
            capability="host.fs.write",
            params={"root_id": "work", "path": "lost.txt", "content": "partial"},
        ),
        CancelToken(),
    )
    temps = [
        p for p in work.iterdir() if ".hb-" in p.name and p.name not in (user_name, excluded_name)
    ]
    assert len(temps) == 1
    orphan = temps[0]
    os.utime(orphan, (old, old))
    with service._sessions_lock:
        service._sessions.pop(begin["write_id"], None)
    service.handle(
        BridgeRequest(
            version=1,
            request_id="n1-trigger",
            capability="host.fs.write",
            params={"root_id": "work", "path": "trigger.txt", "content": "y", "commit": True},
        ),
        CancelToken(),
    )
    assert not orphan.exists()
    assert (work / user_name).exists()
    assert (work / excluded_name).exists()


def test_list_pagination_after_cursor_deleted(bridge: Path, work_root: Path) -> None:
    # P2: if the file named by next_cursor was deleted between
    # pages, `started` never became True and the next page
    # wrongly returned entries=[] truncated=false although files
    # remained. Continuation must select names strictly after the
    # cursor without requiring the cursor entry to exist.
    for name in ("a.txt", "b.txt", "c.txt"):
        _seed(work_root, name, "x")
    page1 = _fs(bridge, "host.fs.list", root_id="work", path=".", limit=1)
    assert [e["name"] for e in page1["entries"]] == ["a.txt"]  # type: ignore[index]
    assert page1["next_cursor"] == "a.txt"
    (work_root / "a.txt").unlink()
    page2 = _fs(
        bridge, "host.fs.list", root_id="work", path=".", limit=1, cursor=page1["next_cursor"]
    )
    assert [e["name"] for e in page2["entries"]] == ["b.txt"]  # type: ignore[index]
    assert page2["truncated"] is True
    page3 = _fs(
        bridge, "host.fs.list", root_id="work", path=".", limit=1, cursor=page2["next_cursor"]
    )
    assert [e["name"] for e in page3["entries"]] == ["c.txt"]  # type: ignore[index]
    assert page3["truncated"] is False


def test_delete_long_name_into_trash(bridge: Path, work_root: Path, tmp_path: Path) -> None:
    # P2: a valid 250-byte file name could not be deleted because
    # the .trashinfo twin (name + 10 bytes) exceeded NAME_MAX=255
    # (ENAMETOOLONG). The trash entry name is byte-truncated to
    # fit; the full original path stays recorded in .trashinfo.
    long_name = "x" * 246 + ".txt"  # 250 bytes
    _seed(work_root, long_name, "bye")
    result = _fs(bridge, "host.fs.delete", root_id="work", path=long_name)
    assert result["recoverable"] is True
    assert not (work_root / long_name).exists()
    trash_name = result["trash_name"]
    assert isinstance(trash_name, str)
    assert len(os.fsencode(trash_name)) + len(".trashinfo") <= 255
    trash_file = tmp_path / "Trash" / "files" / trash_name
    info_file = tmp_path / "Trash" / "info" / f"{trash_name}.trashinfo"
    assert trash_file.read_text(encoding="utf-8") == "bye"
    assert long_name in info_file.read_text(encoding="utf-8")


def test_abort_after_staging_substitution_keeps_user_file(bridge: Path, work_root: Path) -> None:
    # N2 (fifth review): _discard_session_temp unlinked by name with
    # no identity check. Moving the staging file away and a user
    # file onto the staging name, then aborting, permanently deleted
    # the substituted user file. Abort must verify st_dev/st_ino and
    # leave foreign files (and the moved staging file) untouched.
    begin = _fs(bridge, "host.fs.write", root_id="work", path="victim.txt", content="abc")
    write_id = begin["write_id"]
    temps = [p.name for p in work_root.iterdir() if ".hb-" in p.name]
    assert len(temps) == 1
    _fs(
        bridge,
        "host.fs.move",
        root_id="work",
        path=temps[0],
        dest_path="moved-staging.txt",
    )
    _seed(work_root, "user.txt", "USER DATA")
    _fs(bridge, "host.fs.move", root_id="work", path="user.txt", dest_path=temps[0])
    result = _fs(
        bridge,
        "host.fs.write",
        root_id="work",
        path="victim.txt",
        write_id=write_id,
        abort=True,
    )
    assert result["aborted"] is True
    assert (work_root / temps[0]).read_text(encoding="utf-8") == "USER DATA"
    assert (work_root / "moved-staging.txt").read_text(encoding="utf-8") == "abc"


def test_ttl_discard_after_staging_substitution_keeps_user_file(
    registry: RootsRegistry, tmp_path: Path, work_root: Path
) -> None:
    # N2, TTL path: session expiry goes through the same discard;
    # a substituted user file must survive the sweep.
    now = [1000.0]
    service = HostFsService(registry, trash_root=tmp_path / "Trash", clock=lambda: now[0])
    service.handle(
        BridgeRequest(
            version=1,
            request_id="w1",
            capability="host.fs.write",
            params={"root_id": "work", "path": "victim.txt", "content": "abc"},
        ),
        CancelToken(),
    )
    temps = [p.name for p in work_root.iterdir() if ".hb-" in p.name]
    assert len(temps) == 1
    service.handle(
        BridgeRequest(
            version=1,
            request_id="m1",
            capability="host.fs.move",
            params={"root_id": "work", "path": temps[0], "dest_path": "moved-staging.txt"},
        ),
        CancelToken(),
    )
    _seed(work_root, "user.txt", "USER DATA")
    service.handle(
        BridgeRequest(
            version=1,
            request_id="m2",
            capability="host.fs.move",
            params={"root_id": "work", "path": "user.txt", "dest_path": temps[0]},
        ),
        CancelToken(),
    )
    now[0] += 601.0  # past the session TTL
    service.handle(
        BridgeRequest(
            version=1,
            request_id="w2",
            capability="host.fs.write",
            params={"root_id": "work", "path": "other.txt", "content": "x", "commit": True},
        ),
        CancelToken(),
    )
    assert (work_root / temps[0]).read_text(encoding="utf-8") == "USER DATA"
    assert (work_root / "moved-staging.txt").read_text(encoding="utf-8") == "abc"


def test_trash_dirs_created_private_under_permissive_umask(
    bridge: Path, work_root: Path, tmp_path: Path
) -> None:
    # N1 (fifth review): trash dirs created via mkdir(parents=True)
    # inherited umask 022 -> 0755, so deleting a 0644 file from a
    # private 0700 directory made it readable by other local users
    # (ADR-0013 §7). Trash dirs must be created 0700.
    _seed(work_root, "del.txt", "bye")
    old_umask = os.umask(0o022)
    try:
        result = _fs(bridge, "host.fs.delete", root_id="work", path="del.txt")
    finally:
        os.umask(old_umask)
    assert result["recoverable"] is True
    trash = tmp_path / "Trash"
    for directory in (trash, trash / "files", trash / "info"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert (trash / "files" / "del.txt").read_text(encoding="utf-8") == "bye"


def test_existing_loose_trash_dir_tightened(bridge: Path, work_root: Path, tmp_path: Path) -> None:
    # N1: an existing trash dir owned by the user but group/other
    # accessible is tightened to 0700 before use (documented policy
    # in _ensure_private_trash_dir), so delete keeps working without
    # widening access.
    trash = tmp_path / "Trash"
    (trash / "files").mkdir(parents=True)
    (trash / "info").mkdir(parents=True)
    for directory in (trash, trash / "files", trash / "info"):
        os.chmod(directory, 0o755)
    _seed(work_root, "del.txt", "bye")
    result = _fs(bridge, "host.fs.delete", root_id="work", path="del.txt")
    assert result["recoverable"] is True
    for directory in (trash, trash / "files", trash / "info"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_home_trash_dirs_created_private(registry: RootsRegistry, work_root: Path) -> None:
    # N1, home branch: ~/.local/share/Trash and its subdirs are
    # created 0700 even under a permissive umask.
    home_trash = registry.home / ".local" / "share" / "Trash"
    _seed(work_root, "gone.txt", "bye")
    service = HostFsService(registry, mount_bindings_fn=lambda: [])
    old_umask = os.umask(0o022)
    try:
        result = service.handle(
            BridgeRequest(
                version=1,
                request_id="del-home-private",
                capability="host.fs.delete",
                params={"root_id": "work", "path": "gone.txt"},
            ),
            CancelToken(),
        )
    finally:
        os.umask(old_umask)
    assert result["recoverable"] is True
    for directory in (home_trash, home_trash / "files", home_trash / "info"):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
