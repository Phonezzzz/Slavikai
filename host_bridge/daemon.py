"""Owner Host Bridge daemon (E03-T02).

AF_UNIX stream daemon implementing the ADR-0013 trust model for the
lifecycle/auth slice only: ``bridge.ping`` / ``bridge.version`` /
``bridge.capabilities``, no dangerous operations. Filesystem and process
capabilities are E03-T04/T05; protocol validation is E03-T03.

Trust-model points enforced here (ADR-0013):
- AF_UNIX only, no TCP listener.
- Socket at ``$XDG_RUNTIME_DIR/slavikai/host-bridge.sock``; the directory
  is mode 0700 and the socket mode 0600, so a world-writable (or
  group-writable) socket is impossible by construction.
- No persistent fallback: without ``XDG_RUNTIME_DIR`` the daemon refuses
  to start (fail-closed).
- Every connection is authenticated with ``SO_PEERCRED`` before any
  protocol bytes are processed; only the expected UID (the owner's UID,
  equal to the daemon's own UID) is accepted.
- Non-root: the daemon runs as the owner's UID under a systemd user
  service and never invokes sudo/pkexec/setuid helpers.
"""

from __future__ import annotations

import os
import socket
import stat
import struct
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from host_bridge.protocol import (
    BRIDGE_VERSION,
    CAPABILITIES,
    CAPABILITY_CAPABILITIES,
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    MAX_FRAME_BYTES,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    MalformedRequestError,
    UnknownCapabilityError,
    decode_request,
    encode_error,
    encode_success,
)

SOCKET_DIR_NAME: Final[str] = "slavikai"
SOCKET_FILE_NAME: Final[str] = "host-bridge.sock"
_DIR_MODE: Final[int] = 0o700
_SOCKET_MODE: Final[int] = 0o600
_ACCEPT_TIMEOUT_S: Final[float] = 0.2


class BridgeConfigError(Exception):
    """The daemon cannot start with the given configuration (fail-closed)."""


@dataclass(frozen=True)
class PeerCred:
    pid: int
    uid: int
    gid: int


def resolve_socket_path(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the bridge socket path from XDG_RUNTIME_DIR (ADR-0013 §1)."""
    source = os.environ if env is None else env
    raw = source.get("XDG_RUNTIME_DIR", "").strip()
    if not raw:
        raise BridgeConfigError(
            "XDG_RUNTIME_DIR is not set; the host bridge has no persistent "
            "fallback and refuses to start"
        )
    runtime_dir = Path(raw)
    if not runtime_dir.is_absolute():
        raise BridgeConfigError(f"XDG_RUNTIME_DIR must be absolute: {raw!r}")
    if not runtime_dir.is_dir():
        raise BridgeConfigError(f"XDG_RUNTIME_DIR does not exist: {raw!r}")
    return runtime_dir / SOCKET_DIR_NAME / SOCKET_FILE_NAME


def read_peer_cred(conn: socket.socket) -> PeerCred:
    """Read SO_PEERCRED (pid, uid, gid) for a connected AF_UNIX socket."""
    if not hasattr(socket, "SO_PEERCRED"):
        raise BridgeConfigError("SO_PEERCRED is unavailable on this platform")
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("III"))
    values = struct.unpack("III", raw)
    return PeerCred(pid=int(values[0]), uid=int(values[1]), gid=int(values[2]))


class HostBridgeDaemon:
    """AF_UNIX daemon serving the bridge.* lifecycle capabilities."""

    def __init__(
        self,
        socket_path: Path,
        *,
        expected_uid: int | None = None,
        peer_cred_fn: Callable[[socket.socket], PeerCred] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._expected_uid = os.getuid() if expected_uid is None else expected_uid
        self._peer_cred_fn = read_peer_cred if peer_cred_fn is None else peer_cred_fn
        self._listener: socket.socket | None = None
        self._stop = threading.Event()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def start(self) -> None:
        """Create the socket directory/file with closed modes and listen."""
        parent = self._socket_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        except OSError as exc:
            raise BridgeConfigError(f"cannot create socket directory {parent}: {exc}") from exc
        if not parent.is_dir():
            raise BridgeConfigError(f"socket directory is not a directory: {parent}")
        os.chmod(parent, _DIR_MODE)
        if self._socket_path.exists() or self._socket_path.is_symlink():
            mode = self._socket_path.lstat().st_mode
            if not stat.S_ISSOCK(mode):
                raise BridgeConfigError(f"refusing to replace non-socket file: {self._socket_path}")
            self._socket_path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o077)
        try:
            listener.bind(str(self._socket_path))
        finally:
            os.umask(old_umask)
        os.chmod(self._socket_path, _SOCKET_MODE)
        listener.listen(16)
        listener.settimeout(_ACCEPT_TIMEOUT_S)
        self._listener = listener

    def serve_forever(self) -> None:
        if self._listener is None:
            raise BridgeConfigError("daemon is not started")
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            thread = threading.Thread(target=self._handle, args=(conn,), daemon=True)
            thread.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        try:
            self._socket_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _close_gracefully(conn: socket.socket) -> None:
        """Close without RST: half-close, drain pending input, then close.

        Closing a socket with unread received data sends RST, which the
        peer sees as a reset instead of a clean EOF. Draining first makes
        the refusal deterministic and observable as EOF.
        """
        try:
            conn.shutdown(socket.SHUT_WR)
        except OSError:
            return
        conn.settimeout(0.5)
        try:
            while conn.recv(4096):
                pass
        except OSError:
            pass

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                peer = self._peer_cred_fn(conn)
            except (BridgeConfigError, OSError):
                self._close_gracefully(conn)
                return
            if peer.uid != self._expected_uid:
                # deterministic refusal before any protocol bytes
                self._close_gracefully(conn)
                return
            reader = conn.makefile("rb")
            while not self._stop.is_set():
                line = reader.readline(MAX_FRAME_BYTES + 1)
                if not line:
                    return
                if len(line) > MAX_FRAME_BYTES:
                    conn.sendall(
                        encode_error(
                            BridgeProtocolError(
                                "frame_too_large", "request line exceeds the frame bound"
                            )
                        )
                    )
                    return
                try:
                    capability = decode_request(line)
                    result = self._dispatch(capability)
                except MalformedRequestError as exc:
                    conn.sendall(encode_error(exc))
                    return
                except UnknownCapabilityError as exc:
                    conn.sendall(encode_error(exc))
                    continue
                conn.sendall(encode_success(result))

    @staticmethod
    def _dispatch(capability: str) -> dict[str, object]:
        if capability == CAPABILITY_PING:
            return {"pong": True}
        if capability == CAPABILITY_VERSION:
            return {"protocol": PROTOCOL_VERSION, "bridge": BRIDGE_VERSION}
        if capability == CAPABILITY_CAPABILITIES:
            return {"capabilities": list(CAPABILITIES)}
        raise UnknownCapabilityError(capability)
