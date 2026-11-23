"""Owner Host Bridge daemon (E03-T02, ADR-0013; wire schema E03-T03).

Lifecycle/auth slice: AF_UNIX socket in a 0700 directory with a 0600
socket file, SO_PEERCRED UID check before any protocol bytes, and the
three ``bridge.*`` capabilities over the versioned E03-T03 schema.
Wrong-UID rejection is simulated with an injected peer-cred reader
(tests run as a single UID); the same-UID path uses the real
SO_PEERCRED. Schema validation itself is covered in
tests/test_host_bridge_protocol.py.
"""

from __future__ import annotations

import json
import os
import socket
import stat
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from host_bridge import client
from host_bridge.__main__ import main
from host_bridge.daemon import (
    BridgeConfigError,
    HostBridgeDaemon,
    PeerCred,
    resolve_socket_path,
)
from host_bridge.protocol import MAX_FRAME_BYTES


def _start_daemon(daemon: HostBridgeDaemon) -> threading.Thread:
    daemon.start()
    thread = threading.Thread(target=daemon.serve_forever, daemon=True)
    thread.start()
    return thread


@pytest.fixture()
def bridge(tmp_path: Path) -> Iterator[Path]:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path)
    thread = _start_daemon(daemon)
    try:
        yield socket_path
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def _raw_roundtrip(socket_path: Path, payload: bytes) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(socket_path))
        conn.sendall(payload)
        return conn.makefile("rb").readline(MAX_FRAME_BYTES + 1)


# --- socket path resolution (fail-closed, no persistent fallback) ---


def test_resolve_socket_path_requires_xdg_runtime_dir() -> None:
    with pytest.raises(BridgeConfigError, match="XDG_RUNTIME_DIR is not set"):
        resolve_socket_path({})


def test_resolve_socket_path_rejects_relative_dir(tmp_path: Path) -> None:
    with pytest.raises(BridgeConfigError, match="must be absolute"):
        resolve_socket_path({"XDG_RUNTIME_DIR": "relative/run"})


def test_resolve_socket_path_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(BridgeConfigError, match="does not exist"):
        resolve_socket_path({"XDG_RUNTIME_DIR": str(tmp_path / "nope")})


def test_resolve_socket_path_builds_bridge_path(tmp_path: Path) -> None:
    assert resolve_socket_path({"XDG_RUNTIME_DIR": str(tmp_path)}) == (
        tmp_path / "slavikai" / "host-bridge.sock"
    )


def test_main_refuses_to_start_without_xdg_runtime_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    assert main() == 2


# --- socket modes: world-writable impossible ---


def test_socket_and_directory_modes_are_closed(bridge: Path) -> None:
    dir_mode = stat.S_IMODE(bridge.parent.stat().st_mode)
    sock_mode = stat.S_IMODE(bridge.stat().st_mode)
    assert dir_mode == 0o700
    assert sock_mode == 0o600
    assert dir_mode & 0o077 == 0
    assert sock_mode & 0o077 == 0


def test_start_refuses_to_replace_regular_file(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    socket_path.parent.mkdir(parents=True)
    socket_path.write_text("not a socket", encoding="utf-8")
    daemon = HostBridgeDaemon(socket_path)
    with pytest.raises(BridgeConfigError, match="non-socket"):
        daemon.start()


def test_start_replaces_stale_socket_file(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    socket_path.parent.mkdir(parents=True)
    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()  # socket file remains, nobody listens
    daemon = HostBridgeDaemon(socket_path)
    thread = _start_daemon(daemon)
    try:
        assert client.ping(socket_path) == {"pong": True}
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_shutdown_removes_socket_file(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path)
    thread = _start_daemon(daemon)
    assert socket_path.exists()
    daemon.shutdown()
    thread.join(timeout=2.0)
    assert not socket_path.exists()


def test_second_daemon_on_live_socket_is_refused(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    first = HostBridgeDaemon(socket_path)
    thread = _start_daemon(first)
    try:
        second = HostBridgeDaemon(socket_path)
        with pytest.raises(BridgeConfigError, match="already running"):
            second.start()
        # the refused attempt must not disturb the running instance
        assert client.ping(socket_path) == {"pong": True}
    finally:
        first.shutdown()
        thread.join(timeout=2.0)
    # after the refused second attempt, the first shutdown still
    # removes its own socket
    assert not socket_path.exists()


def test_shutdown_leaves_replaced_socket_in_place(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    first = HostBridgeDaemon(socket_path)
    first_thread = _start_daemon(first)
    # Simulate another instance taking over the path: the first socket
    # file is unlinked and a second daemon binds a fresh socket there.
    socket_path.unlink()
    second = HostBridgeDaemon(socket_path)
    second_thread = _start_daemon(second)
    try:
        first.shutdown()
        first_thread.join(timeout=2.0)
        # first.shutdown() must not unlink the second instance's socket
        assert socket_path.exists()
        assert client.ping(socket_path) == {"pong": True}
    finally:
        second.shutdown()
        second_thread.join(timeout=2.0)


# --- same-UID capabilities over the real socket ---


def test_ping(bridge: Path) -> None:
    assert client.ping(bridge) == {"pong": True}


def test_version(bridge: Path) -> None:
    assert client.version(bridge) == {"protocol": 1, "bridge": "0.1.0"}


def test_capabilities(bridge: Path) -> None:
    assert client.capabilities(bridge) == [
        "bridge.ping",
        "bridge.version",
        "bridge.capabilities",
    ]


def test_unknown_capability_is_typed_rejection(bridge: Path) -> None:
    with pytest.raises(client.BridgeClientError) as excinfo:
        client.call(bridge, "host.fs.read")
    assert excinfo.value.code == "unknown_capability"


def test_unknown_capability_keeps_connection_usable(bridge: Path) -> None:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5.0)
        conn.connect(str(bridge))
        reader = conn.makefile("rb")
        conn.sendall(b'{"version": 1, "request_id": "r-1", "capability": "host.process.run"}\n')
        rejected = json.loads(reader.readline(MAX_FRAME_BYTES + 1))
        assert rejected["ok"] is False
        assert rejected["error"]["code"] == "unknown_capability"
        assert rejected["request_id"] == "r-1"
        conn.sendall(b'{"version": 1, "request_id": "r-2", "capability": "bridge.ping"}\n')
        accepted = json.loads(reader.readline(MAX_FRAME_BYTES + 1))
        assert accepted == {
            "version": 1,
            "request_id": "r-2",
            "ok": True,
            "result": {"pong": True},
        }


def test_malformed_json_is_typed_rejection(bridge: Path) -> None:
    line = _raw_roundtrip(bridge, b"this is not json\n")
    payload = json.loads(line)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "malformed_request"
    assert payload["request_id"] is None  # nothing extractable to echo


def test_missing_capability_field_is_malformed(bridge: Path) -> None:
    line = _raw_roundtrip(bridge, b'{"version": 1, "request_id": "r-1"}\n')
    payload = json.loads(line)
    assert payload["error"]["code"] == "malformed_request"


def test_oversized_line_is_rejected(bridge: Path) -> None:
    line = _raw_roundtrip(bridge, b"x" * (MAX_FRAME_BYTES + 10) + b"\n")
    payload = json.loads(line)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "frame_too_large"


# --- peer authentication ---


def test_wrong_uid_peer_gets_no_response(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"

    def foreign_peer(conn: socket.socket) -> PeerCred:
        return PeerCred(pid=4242, uid=os.getuid() + 4242, gid=0)

    daemon = HostBridgeDaemon(socket_path, peer_cred_fn=foreign_peer)
    thread = _start_daemon(daemon)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(5.0)
            conn.connect(str(socket_path))
            conn.sendall(b'{"version": 1, "request_id": "r-1", "capability": "bridge.ping"}\n')
            assert conn.recv(4096) == b""  # closed before any protocol bytes
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_expected_uid_override_is_enforced(tmp_path: Path) -> None:
    socket_path = tmp_path / "slavikai" / "host-bridge.sock"
    daemon = HostBridgeDaemon(socket_path, expected_uid=os.getuid() + 4242)
    thread = _start_daemon(daemon)
    try:
        with pytest.raises(client.BridgeClientError) as excinfo:
            client.ping(socket_path)
        assert excinfo.value.code == "connection_closed"
    finally:
        daemon.shutdown()
        thread.join(timeout=2.0)


def test_client_missing_socket_raises_os_error(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        client.ping(tmp_path / "no-such.sock")
