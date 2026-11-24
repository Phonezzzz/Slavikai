"""Owner Host Bridge daemon (E03-T02, protocol validation E03-T03,
bounded host FS API E03-T04).

AF_UNIX stream daemon implementing the ADR-0013 trust model:
``bridge.ping`` / ``bridge.version`` / ``bridge.capabilities`` plus the
bounded ``host.fs.*`` capabilities (enforced daemon-side in
``host_bridge.fs``). Process capabilities are E03-T05.

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
- Single instance: an flock(2) lock file in the socket directory,
  held for the daemon's whole lifetime, refuses a second daemon
  before it can touch the socket; only a provably stale socket
  file (connection refused) is replaced, and shutdown unlinks
  only the socket this instance created. The lock closes the
  bind->listen window a single inode check cannot (E03-T06).
- Protocol validation (E03-T03): every frame is checked against the
  versioned schema in ``host_bridge.protocol`` (frame limit, protocol
  version, request ID, unknown fields); dispatch runs under a
  per-request deadline (``request_timeout``). ``principal_id`` /
  ``run_id`` in a request are attribution data only and are never used
  for authorization (ADR-0013 §4).
"""

from __future__ import annotations

import fcntl
import os
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Final

from host_bridge.fs import CancelToken, HostFsService
from host_bridge.process import HostProcessService, ProcessError, build_process_service
from host_bridge.protocol import (
    BRIDGE_VERSION,
    CAPABILITIES,
    CAPABILITY_CAPABILITIES,
    CAPABILITY_PING,
    CAPABILITY_VERSION,
    FS_CAPABILITIES,
    MAX_FRAME_BYTES,
    PROCESS_CAPABILITIES,
    PROTOCOL_VERSION,
    BridgeProtocolError,
    BridgeRequest,
    FrameTooLargeError,
    MalformedRequestError,
    RequestTimeoutError,
    UnknownCapabilityError,
    capability_deadline,
    decode_request,
    encode_error,
    encode_success,
    read_frame,
)
from host_bridge.protocol import (
    BridgeConfigError as BridgeConfigError,
)

SOCKET_DIR_NAME: Final[str] = "slavikai"
SOCKET_FILE_NAME: Final[str] = "host-bridge.sock"
_DIR_MODE: Final[int] = 0o700
_SOCKET_MODE: Final[int] = 0o600
_ACCEPT_TIMEOUT_S: Final[float] = 0.2
_PROBE_TIMEOUT_S: Final[float] = 2.0

# Per-request deadline (E03-T03): dispatch must finish within this bound
# or the request fails with request_timeout and the connection closes.
DEFAULT_REQUEST_TIMEOUT_S: Final[float] = 5.0


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


def _socket_has_live_listener(socket_path: Path) -> bool:
    """Return True when a live listener accepts connections at socket_path.

    A stale socket file (bound once, listener gone) refuses the
    connection. Any other probe outcome means staleness cannot be
    proven, so startup fails closed instead of stealing the path.
    """
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(_PROBE_TIMEOUT_S)
    try:
        probe.connect(str(socket_path))
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError as exc:
        raise BridgeConfigError(
            f"cannot verify existing socket {socket_path} is stale: {exc}"
        ) from exc
    finally:
        probe.close()
    return True


def read_peer_cred(conn: socket.socket) -> PeerCred:
    """Read SO_PEERCRED (pid, uid, gid) for a connected AF_UNIX socket."""
    if not hasattr(socket, "SO_PEERCRED"):
        raise BridgeConfigError("SO_PEERCRED is unavailable on this platform")
    raw = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("III"))
    values = struct.unpack("III", raw)
    return PeerCred(pid=int(values[0]), uid=int(values[1]), gid=int(values[2]))


class HostBridgeDaemon:
    """AF_UNIX daemon serving bridge.* lifecycle + host.fs.* capabilities."""

    def __init__(
        self,
        socket_path: Path,
        *,
        expected_uid: int | None = None,
        peer_cred_fn: Callable[[socket.socket], PeerCred] | None = None,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
        dispatch_fn: Callable[[BridgeRequest], dict[str, object]] | None = None,
        fs_service: HostFsService | None = None,
        process_service: HostProcessService | None = None,
        capability_deadlines: Mapping[str, float] | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._expected_uid = os.getuid() if expected_uid is None else expected_uid
        self._peer_cred_fn = read_peer_cred if peer_cred_fn is None else peer_cred_fn
        self._request_timeout_s = request_timeout_s
        self._custom_dispatch = dispatch_fn is not None
        self._dispatch_fn: Callable[[BridgeRequest], dict[str, object]] | None = dispatch_fn
        self._capability_deadlines = capability_deadlines
        self._fs_service = (
            fs_service
            if fs_service is not None
            else HostFsService.default(home=Path.home(), socket_dir=socket_path.parent)
        )
        self._process_service = (
            process_service
            if process_service is not None
            else build_process_service(
                self._fs_service.registry,
                self._fs_service,
                runtime_dir=socket_path.parent,
            )
        )
        self._listener: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        # Exclusive-ownership lock file (E03-T06): held from start()
        # until shutdown(); never unlinked (see start()).
        self._lock_file: IO[Any] | None = None
        self._stop = threading.Event()

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def start(self) -> None:
        """Create the socket directory/file with closed modes and listen.

        Startup order is the E03-T06 fail-closed sequence (schema
        §5.3): bind -> identity re-check -> orphan sweep -> probe ->
        serve. Exclusive ownership of the namespace is held for the
        daemon's whole lifetime by an flock(2) lock file, taken
        BEFORE any socket probing: the bind->listen window is closed
        because a second daemon fails at the lock step before it can
        touch the socket, and a single inode check never has to
        carry ownership through the sweep.
        """
        parent = self._socket_path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        except OSError as exc:
            raise BridgeConfigError(f"cannot create socket directory {parent}: {exc}") from exc
        if not parent.is_dir():
            raise BridgeConfigError(f"socket directory is not a directory: {parent}")
        os.chmod(parent, _DIR_MODE)
        # Exclusive ownership (E03-T06, approval condition 1):
        # flock on a lock file in the same directory as the socket,
        # held until shutdown. LOCK_NB fails immediately when
        # another daemon holds it — before any socket probe, so a
        # second daemon can never mistake the first's live socket
        # for stale and unlink it, and the sweep below runs under
        # ownership that lasts the whole sweep. The lock file is
        # never unlinked: unlinking races with a concurrent opener,
        # and a stale file is harmless — flock is released when
        # its holder dies, so the next daemon re-locks the same
        # inode.
        lock_path = parent / "host-bridge.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as exc:
            raise BridgeConfigError(f"cannot open daemon lock file {lock_path}: {exc}") from exc
        lock_file = os.fdopen(lock_fd, "w")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            lock_file.close()
            raise BridgeConfigError(
                f"host bridge daemon already running: {self._socket_path} ({exc})"
            ) from exc
        self._lock_file = lock_file
        listener: socket.socket | None = None
        try:
            if self._socket_path.exists() or self._socket_path.is_symlink():
                mode = self._socket_path.lstat().st_mode
                if not stat.S_ISSOCK(mode):
                    raise BridgeConfigError(
                        f"refusing to replace non-socket file: {self._socket_path}"
                    )
                if _socket_has_live_listener(self._socket_path):
                    raise BridgeConfigError(
                        f"host bridge daemon already running: {self._socket_path}"
                    )
                self._socket_path.unlink(missing_ok=True)
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            old_umask = os.umask(0o077)
            try:
                listener.bind(str(self._socket_path))
            finally:
                os.umask(old_umask)
            os.chmod(self._socket_path, _SOCKET_MODE)
            bound = self._socket_path.stat()
            self._socket_identity = (bound.st_dev, bound.st_ino)
            listener.listen(16)
            listener.settimeout(_ACCEPT_TIMEOUT_S)
            self._listener = listener
            # Identity re-check (E03-T06 §5.3): the socket at the
            # path must still be the one this daemon just bound —
            # if another process replaced it between bind and
            # listen, serving would hand out the wrong socket.
            current = self._socket_path.stat()
            if (current.st_dev, current.st_ino) != self._socket_identity:
                raise BridgeConfigError(f"socket {self._socket_path} was replaced during startup")
            # Orphan sweep (E03-T06): kill scopes and run dirs left
            # by a dead daemon. Any failure is fatal — serving with
            # an unknown tree alive would violate ownership.
            self._process_service.reap_orphans()
        except BaseException:
            if listener is not None:
                listener.close()
            self._listener = None
            self._socket_identity = None
            self._lock_file.close()
            self._lock_file = None
            raise
        # Eager enforcement probe (E03-T05): when profiles exist, verify
        # the whole enforcement stack through the real spawner now, in
        # this daemon's context. A probe failure does not stop the
        # daemon — it hides the process capabilities (fail-closed) and
        # any direct run answers process_unavailable.
        if self._process_service.has_profiles():
            try:
                self._process_service.probe()
            except BaseException:
                pass

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
        self._process_service.shutdown()
        if self._listener is not None:
            self._listener.close()
            self._listener = None
        identity = self._socket_identity
        self._socket_identity = None
        if identity is not None:
            try:
                current = self._socket_path.stat()
            except FileNotFoundError:
                current = None
            # Unlink only the socket this instance created; if the path now
            # refers to another instance's socket, leave it in place.
            if current is not None and (current.st_dev, current.st_ino) == identity:
                self._socket_path.unlink(missing_ok=True)
        # Release the exclusive-ownership lock last (E03-T06): until
        # here the socket above was still ours, and a new daemon
        # starting in between must not be able to take the lock
        # while this instance's socket is still being unlinked.
        lock_file = self._lock_file
        self._lock_file = None
        if lock_file is not None:
            lock_file.close()

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

    @staticmethod
    def _send(conn: socket.socket, payload: bytes) -> bool:
        """Send one response frame; False when the peer is gone."""
        try:
            conn.sendall(payload)
        except OSError:
            return False
        return True

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
            conn.settimeout(self._request_timeout_s)
            reader = conn.makefile("rb")
            while not self._stop.is_set():
                try:
                    frame = read_frame(reader)
                except FrameTooLargeError as exc:
                    self._send(conn, encode_error(exc))
                    return
                except (TimeoutError, OSError):
                    return
                if frame is None:
                    return
                try:
                    request = decode_request(frame)
                    result = self._dispatch_with_deadline(request)
                except MalformedRequestError as exc:
                    # peer does not speak the protocol: answer, then close
                    self._send(conn, encode_error(exc))
                    return
                except RequestTimeoutError as exc:
                    # handler was abandoned; its state is unknown: close
                    self._send(conn, encode_error(exc))
                    return
                except BridgeProtocolError as exc:
                    # semantic rejection (unknown capability/version/field,
                    # invalid request ID): framing is intact, stay usable
                    if not self._send(conn, encode_error(exc)):
                        return
                    continue
                payload = encode_success(result, request.request_id)
                if len(payload) - 1 > MAX_FRAME_BYTES:
                    # Backstop: handlers budget their own output, but a
                    # success frame must never exceed the frame limit.
                    payload = encode_error(
                        BridgeProtocolError(
                            "internal_error",
                            "response exceeds the frame limit",
                            request_id=request.request_id,
                        )
                    )
                if not self._send(conn, payload):
                    return

    def _deadline_for(self, capability: str) -> float:
        """Per-capability deadline (E03-T04): 30 s for tree walks/moves,
        ``request_timeout_s`` otherwise; an explicit override map wins."""
        if self._capability_deadlines is not None:
            return self._capability_deadlines.get(capability, self._request_timeout_s)
        return capability_deadline(capability, default=self._request_timeout_s)

    def _dispatch_with_deadline(self, request: BridgeRequest) -> dict[str, object]:
        """Run dispatch in a worker thread bounded by the request deadline.

        A handler that overruns is abandoned (daemon thread) and the
        request fails with request_timeout; the caller closes the
        connection because the abandoned handler's state is unknown.
        The request's CancelToken is cancelled when the deadline fires:
        FS handlers check it immediately before any publish syscall,
        so a timed-out mutation never commits late (E03-T04).
        """
        deadline = self._deadline_for(request.capability)
        token = CancelToken(deadline=time.monotonic() + deadline, request_id=request.request_id)
        outcome: dict[str, object] = {}

        def run() -> None:
            try:
                if self._custom_dispatch:
                    assert self._dispatch_fn is not None
                    outcome["result"] = self._dispatch_fn(request)
                else:
                    outcome["result"] = self._dispatch(request, token)
            except BaseException as exc:  # re-raised in the connection thread
                outcome["error"] = exc

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(deadline)
        if worker.is_alive():
            token.cancel()
            raise RequestTimeoutError(request_id=request.request_id)
        error = outcome.get("error")
        if isinstance(error, BaseException):
            raise error
        result = outcome.get("result")
        if not isinstance(result, dict):
            raise BridgeProtocolError(
                "internal_error",
                "dispatch returned no result",
                request_id=request.request_id,
            )
        return result

    def _dispatch(self, request: BridgeRequest, token: CancelToken) -> dict[str, object]:
        if request.capability == CAPABILITY_PING:
            return {"pong": True}
        if request.capability == CAPABILITY_VERSION:
            return {"protocol": PROTOCOL_VERSION, "bridge": BRIDGE_VERSION}
        if request.capability == CAPABILITY_CAPABILITIES:
            capabilities = list(CAPABILITIES)
            profiles: list[dict[str, object]] = []
            if self._process_service.is_available():
                profiles = self._process_service.profiles_description()
            else:
                # Process capabilities are hidden, not degraded: when
                # the enforcement probe failed (or no profiles exist),
                # the client must not see them at all.
                capabilities = [c for c in capabilities if c not in PROCESS_CAPABILITIES]
            return {
                "capabilities": capabilities,
                "roots": self._fs_service.roots_description(),
                "profiles": profiles,
            }
        if request.capability in FS_CAPABILITIES:
            return self._fs_service.handle(request, token)
        if request.capability in PROCESS_CAPABILITIES:
            if not self._process_service.has_profiles():
                # No profiles configured: the process API does not
                # exist on this daemon at all — it is unknown, not
                # unavailable. (A daemon without profiles must answer
                # exactly like a daemon that never had the capability.)
                raise UnknownCapabilityError(request.capability, request_id=request.request_id)
            if not self._process_service.is_available():
                # Profiles exist but the enforcement probe failed:
                # the capability is hidden and a direct run is refused
                # with the typed process_unavailable (never degraded).
                raise ProcessError(
                    "process_unavailable",
                    "process execution is unavailable: the enforcement probe failed",
                    request_id=request.request_id,
                )
            return self._process_service.handle(
                request.capability,
                request.params,
                request.request_id,
                token,
                # Envelope attribution (E03-T06): the process
                # service records it as the ownership binding.
                principal_id=request.principal_id,
                run_id=request.run_id,
            )
        raise UnknownCapabilityError(request.capability, request_id=request.request_id)
