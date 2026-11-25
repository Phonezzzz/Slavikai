"""Approved-profile process API for the Owner Host Bridge (E03-T05).

Capabilities ``host.process.run`` / ``host.process.poll`` /
``host.process.kill`` run owner-approved command profiles under the
enforcement stack of ADR-0013 §10a as fixed for T05 by the owner
(2026-11-24): Landlock + ``NEWUSER|NEWNET|NEWIPC`` + ``NO_NEW_PRIVS`` +
a mandatory seccomp socket ban, applied by the trampoline helper
(``host_bridge.spawn_helper``), never by the multithreaded daemon
itself; one cgroup v2 scope per ``process_id`` (created by
``systemd-run --user --scope -p Delegate=yes``) for the tree kill;
a startup probe through the same spawner in the daemon's real
context, whose failure hides the process capabilities and makes a
direct run answer ``process_unavailable``.

Profiles live in the same ``host-bridge.json`` as the FS roots and
use the same fail-closed loader; only the profile *descriptions*
(id + slot names/kinds/required/enum values/int bounds) are exposed
through ``bridge.capabilities`` — never binary paths, argv, env or
the cwd.

Spawn is idempotent per ``request_id``: a retry after a lost
response returns the same ``process_id`` and never starts a second
process. ``running`` is returned only after the helper confirms
both enforcement and a successful ``exec`` (a process that already
finished by the time the run call returns reports its final state
instead — it still ran exactly once). Cancellation and launch are
a single state transition (the record is committed before the
spawn; a cancelled launch still settles to exactly one record).
Poll is instant only, UTF-8 with replacement, absolute offsets from
stream start plus ``retained_from``, chunked by the serialized-JSON
frame budget. The daemon reaper enforces the 120 s process cap and
finalizes a record only after the scope cgroup is empty AND the
scope leader is reaped.
"""

from __future__ import annotations

import codecs
import ctypes
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from host_bridge.effect import (
    AT_PLACEHOLDER,
    EFFECT_BUILD_FAILED,
    LONGEST_STATE,
    MAX_INT_PLACEHOLDER,
    MAX_SIGNAL_PLACEHOLDER,
    STOP_PROJECTION_ENTRIES,
    EffectEmitter,
    stop_process_placeholder,
)
from host_bridge.fs import (
    CancelToken,
    FsError,
    HostFsService,
    RootsRegistry,
    _check_params,
    _int_param,
    _iso,
    _parse_rel_path,
    _str_param,
)
from host_bridge.protocol import (
    BridgeConfigError,
    BridgeProtocolError,
    RequestTimeoutError,
)
from host_bridge.spawn_helper import TAIL_BYTES, landlock_abi

# --- limits (T05 proposal §4 + owner corrections) ---------------------

MAX_RUNNING_PROCESSES = 4
MAX_FINISHED_RECORDS = 32
FINISHED_RETENTION_S = 600.0
MAX_PROCESS_TIMEOUT_S = 120
MAX_ARGS = 64
MAX_ARGS_TOTAL_BYTES = 8 * 1024
MAX_STRING_ARG_CHARS = 1024
MAX_ENV_TOTAL_BYTES = 8 * 1024
MAX_ENV_VALUE_CHARS = 1024
MAX_PROFILE_ENV_VARS = 32
KILL_GRACE_S = 2.0
SPAWN_START_TIMEOUT_S = 10.0
# A stopped-run mark (E03-T06) outlives the run's records: it is
# dropped when the last record bound to the run is swept, but a mark
# for a run that never had records (STOP of an unknown run) has
# nothing pinning it, so it expires after the same bound as finished
# retention instead of living forever.
STOPPED_RUN_MARK_TTL_S = 600.0
# The single shared deadline bound for host.process.stop (E03-T06):
# one deadline governs the whole STOP, not one per process.
STOP_WAIT_MAX_S = 4.0
PROBE_TIMEOUT_S = 20.0
PROBE_TARGET_TIMEOUT_S = 10
GRANT_WALK_MAX_ENTRIES = 20_000
# Grants are now ~one per file + one per directory (directory rules
# carry no file-content rights, so every allowed file needs its own
# rule): the cap tracks the entries cap plus the system/exec/tmp
# grants, so no tree the walk accepts can fail on grant count.
GRANT_WALK_MAX_GRANTS = GRANT_WALK_MAX_ENTRIES + 100
GRANT_WALK_MAX_DEPTH = 128
POLL_MAX_BYTES = 32 * 1024
POLL_DEFAULT_MAX_BYTES = 8 * 1024
# A poll accepts a tail + counters snapshot only when it is
# consistent (see _poll): under a continuously publishing helper a
# few attempts can straddle a publication, so the poll retries with
# a short pause instead of failing on the first torn read — and
# fails typed once the attempts are exhausted, instead of handing
# out bytes that may sit at wrong absolute offsets.
_POLL_SNAPSHOT_ATTEMPTS = 10
_POLL_SNAPSHOT_RETRY_DELAY_S = 0.001

_SLOT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_BINARIES = {"sh", "bash", "dash", "zsh", "fish", "ash"}

# Env vars a profile may never set: they change how the loader or the
# interpreter resolves code and would bypass the approved-binary
# discipline (the owner owns the profile file, but a typo must not
# turn into an escape hatch).
_FORBIDDEN_PROFILE_ENV = {
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "LD_AUDIT",
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
}

# Minimal fixed environment every target process gets. PATH is kept
# only so profile binaries that invoke a fixed argv_prefix (e.g. a
# wrapper that calls a system tool by name) work; nothing from the
# daemon or the model reaches the child.
_BASE_ENV: dict[str, str] = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TZ": "UTC",
}

# System directories the target needs to run at all. Read-only, except
# /dev/null and /dev/full which are read-write by design. /etc is NOT
# granted wholesale: only the loader cache/config files, and only when
# they exist. No EXECUTE here on purpose: a Landlock rule on a
# directory covers its whole subtree, so directory EXECUTE would let
# the target execute ANY binary beneath it. EXECUTE is granted only
# to the approved binary file and its interpreter, by the helper
# (spawn_helper._child_main) — directory traversal needs no EXECUTE
# (verified: nested opens succeed with read-only directory grants).
_SYSTEM_GRANT_PATHS: tuple[tuple[str, str], ...] = (
    ("/usr", "r"),
    ("/lib", "r"),
    ("/lib64", "r"),
    ("/etc/ld.so.cache", "r"),
    ("/etc/ld.so.conf", "r"),
    ("/etc/ld.so.conf.d", "r"),
    ("/dev/null", "rw"),
    ("/dev/zero", "r"),
    ("/dev/full", "rw"),
    ("/dev/urandom", "r"),
)

# Probe program: runs inside the spawned process and reports, as one
# JSON line, whether each layer actually refused what it must refuse.
# argv: [socket_path, protected_path, allowed_path]. It always exits 0
# (the verdict is in the JSON); it sleeps at the end so the daemon can
# verify the cgroup placement while the process is still running.
_PROBE_CODE = r"""
import json, os, socket, sys, time
result = {"tcp": "unknown", "udp": "unknown", "unix": "unknown",
          "protected": "unknown", "allowed": "unknown", "uid": None,
          "signal": "unknown", "metadata": "unknown", "prlimit": "unknown",
          "fifo": "unknown", "setpriority": "unknown", "late": "unknown",
          "scheduler": "unknown", "affinity": "unknown",
          "xattrat": "unknown", "ioprio": "unknown", "ipc": "unknown"}
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("127.0.0.1", 9))
    result["tcp"] = "connected"
except OSError as e:
    result["tcp"] = "refused" if e.errno in (1, 13, 111) else f"error:{e.errno}"
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"x", ("127.0.0.1", 9))
    result["udp"] = "sent"
except OSError as e:
    result["udp"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
try:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(sys.argv[1])
    result["unix"] = "connected"
except OSError as e:
    result["unix"] = "refused" if e.errno in (1, 13, 111) else f"error:{e.errno}"
try:
    with open(sys.argv[2], "r", encoding="utf-8") as fh:
        fh.read()
    result["protected"] = "read"
except OSError as e:
    result["protected"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
try:
    with open(sys.argv[3], "r", encoding="utf-8") as fh:
        result["allowed"] = "read" if fh.read() == "probe-ok" else "wrong-content"
except OSError as e:
    result["allowed"] = f"error:{e.errno}"
try:
    os.kill(os.getppid(), 0)
    result["signal"] = "allowed"
except OSError as e:
    result["signal"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
# Metadata families (chmod/chown/setxattr/utime) are outside Landlock:
# every one of them must be refused by the seccomp ban, or a profile
# could chmod 000 a file it cannot even read.
for _name, _op in (
    ("chmod", lambda: os.chmod(sys.argv[2], 0o600)),
    ("chown", lambda: os.chown(sys.argv[2], 0, -1)),
    ("setxattr", lambda: os.setxattr(sys.argv[2], "user.probe", b"x")),
    ("utime", lambda: os.utime(sys.argv[2], ns=(0, 0))),
):
    try:
        _op()
        result["metadata"] = f"allowed:{_name}"
        break
    except OSError as e:
        if e.errno not in (1, 13):
            result["metadata"] = f"error:{_name}:{e.errno}"
            break
else:
    result["metadata"] = "refused"
# prlimit64 is outside Landlock and signal scope: with no PID
# namespace and matching kernel credentials, the target must not
# change another process's limits. Fork a sacrificial child and
# try prlimit on its pid; the child is killed right after, so a
# missing ban cannot hurt a real process.
import resource
_child = os.fork()
if _child == 0:
    time.sleep(2)
    os._exit(0)
try:
    resource.prlimit(_child, resource.RLIMIT_NOFILE, (0, 0))
    result["prlimit"] = "allowed"
except OSError as e:
    result["prlimit"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
finally:
    try:
        os.kill(_child, 9)
    except OSError:
        pass
    try:
        os.waitpid(_child, 0)
    except OSError:
        pass
# mkfifo is outside Landlock's granted rights: MAKE_FIFO is in the
# handled mask with no allowing grant, so creating a named pipe must
# be refused even inside the writable cwd — otherwise a profile
# could create a rendezvous point in any owner-accessible directory.
try:
    os.mkfifo("probe.fifo")
    result["fifo"] = "allowed"
    try:
        os.unlink("probe.fifo")
    except OSError:
        pass
except OSError as e:
    result["fifo"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
# A file created in the cwd after the grants were computed must stay
# unreadable and unwritable: directory rules carry directory-operation
# rights only, never file-content rights, or a secret the owner drops
# in after the walk would leak to a long-lived profile. Creating the
# file itself must succeed (MAKE_REG is a directory operation); only
# writing to it and reading it back must be refused.
try:
    _fd = os.open("probe-late.txt", os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(_fd, b"late")
    finally:
        os.close(_fd)
    result["late"] = "allowed:write"
except OSError as e:
    if e.errno not in (1, 13):
        result["late"] = f"error:write:{e.errno}"
    else:
        # The write half was refused. The file itself is still
        # created first (MAKE_REG is a directory operation), so a
        # read attempt distinguishes "never created" (the dir-op
        # grant is broken) from "created but unreadable" (correct):
        # ENOENT vs EACCES.
        try:
            with open("probe-late.txt", "r", encoding="utf-8") as _fh:
                _fh.read()
            result["late"] = "allowed:read"
        except FileNotFoundError:
            result["late"] = "error:create:missing"
        except OSError as e2:
            result["late"] = "refused" if e2.errno in (1, 13) else f"error:read:{e2.errno}"
# setpriority is outside Landlock and signal scope, like prlimit64:
# with no PID namespace and matching kernel credentials, the target
# must not renice another process. Fork a sacrificial child and try
# to lower its priority; the child is killed right after, so a
# missing ban cannot hurt a real process.
_child2 = os.fork()
if _child2 == 0:
    time.sleep(2)
    os._exit(0)
try:
    os.setpriority(os.PRIO_PROCESS, _child2, 19)
    result["setpriority"] = "allowed"
except OSError as e:
    result["setpriority"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
finally:
    try:
        os.kill(_child2, 9)
    except OSError:
        pass
    try:
        os.waitpid(_child2, 0)
    except OSError:
        pass
# sched_setscheduler/sched_setattr are outside Landlock and signal
# scope, like setpriority: with no PID namespace and matching kernel
# credentials, the target must not move another process to
# SCHED_IDLE (or any other policy) — under CPU contention the victim
# starves, and the change persists after the profile ends. Fork a
# sacrificial child and try both entry points on it; the child is
# killed right after, so a missing ban cannot hurt a real process.
# sched_setattr has no stdlib wrapper, so it goes through the raw
# syscall (314 on x86_64) with a struct sched_attr.
import ctypes
class _SchedAttr(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint32), ("sched_policy", ctypes.c_uint32),
                ("sched_flags", ctypes.c_uint64), ("sched_nice", ctypes.c_int32),
                ("sched_priority", ctypes.c_uint32), ("sched_runtime", ctypes.c_uint64),
                ("sched_deadline", ctypes.c_uint64), ("sched_period", ctypes.c_uint64)]
_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.syscall.restype = ctypes.c_long
_libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
_child3 = os.fork()
if _child3 == 0:
    time.sleep(2)
    os._exit(0)
try:
    try:
        os.sched_setscheduler(_child3, os.SCHED_IDLE, os.sched_param(0))
        result["scheduler"] = "allowed:setscheduler"
    except OSError as e:
        if e.errno not in (1, 13):
            result["scheduler"] = f"error:setscheduler:{e.errno}"
        else:
            _attr = _SchedAttr()
            _attr.size = ctypes.sizeof(_SchedAttr)
            _attr.sched_policy = 5  # SCHED_IDLE
            _ret = _libc.syscall(314, _child3, ctypes.byref(_attr), 0)
            if _ret == 0:
                result["scheduler"] = "allowed:setattr"
            else:
                _errno = ctypes.get_errno()
                result["scheduler"] = ("refused" if _errno in (1, 13)
                                       else f"error:setattr:{_errno}")
finally:
    try:
        os.kill(_child3, 9)
    except OSError:
        pass
    try:
        os.waitpid(_child3, 0)
    except OSError:
        pass
# sched_setaffinity is outside Landlock and signal scope, like
# setpriority: with no PID namespace and matching kernel
# credentials, the target must not pin another process to a CPU —
# under CPU contention the victim is stuck on a busy core, and the
# change persists after the profile ends. Fork a sacrificial child
# and try to narrow its affinity to one CPU; the child is killed
# right after, so a missing ban cannot hurt a real process.
_child4 = os.fork()
if _child4 == 0:
    time.sleep(2)
    os._exit(0)
try:
    os.sched_setaffinity(_child4, {0})
    result["affinity"] = "allowed"
except OSError as e:
    result["affinity"] = "refused" if e.errno in (1, 13) else f"error:{e.errno}"
finally:
    try:
        os.kill(_child4, 9)
    except OSError:
        pass
    try:
        os.waitpid(_child4, 0)
    except OSError:
        pass
# setxattrat/removexattrat are the at-variants of the classic
# setxattr family: Landlock does not control metadata changes, so
# both must be refused by the seccomp ban, or a profile could set a
# POSIX ACL on a file it cannot read and lock the daemon/server out
# of its own config. Raw syscalls (463/466 on x86_64, no stdlib
# wrapper); ENOSYS means this kernel has no such syscall, so there
# is nothing to exploit. A removexattrat that finds no attribute
# returns ENODATA, which proves the call got through seccomp, so it
# counts as allowed.
_libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t,
                          ctypes.c_int]
def _setxattrat():
    return _libc.syscall(463, -100, sys.argv[2].encode(),
                         b"user.probe", b"x", 1, 0)
def _removexattrat():
    _libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_char_p, ctypes.c_int]
    return _libc.syscall(466, -100, sys.argv[2].encode(), b"user.probe", 0)
for _name, _op in (
    ("setxattrat", _setxattrat),
    ("removexattrat", _removexattrat),
):
    _ret = _op()
    if _ret == 0:
        result["xattrat"] = f"allowed:{_name}"
        break
    _errno = ctypes.get_errno()
    if _errno == 38:  # ENOSYS: no such syscall on this kernel
        result["xattrat"] = "refused"
        break
    if _errno == 61 and _name == "removexattrat":  # ENODATA: call executed
        result["xattrat"] = "allowed:removexattrat"
        break
    if _errno not in (1, 13):
        result["xattrat"] = f"error:{_name}:{_errno}"
        break
else:
    result["xattrat"] = "refused"
# ioprio_set is outside Landlock and signal scope, like
# setpriority: with no PID namespace and matching kernel
# credentials, the target must not move another process to
# IOPRIO_CLASS_IDLE — under disk contention the victim starves,
# and the change persists after the profile ends. Fork a
# sacrificial child and try to idle it; the child is killed right
# after, so a missing ban cannot hurt a real process. Raw syscall
# (251 on x86_64, no stdlib wrapper); ENOSYS means this kernel has
# no such syscall, so there is nothing to exploit.
_libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_child5 = os.fork()
if _child5 == 0:
    time.sleep(2)
    os._exit(0)
try:
    _ret = _libc.syscall(251, 1, _child5, 3 << 13)  # WHO_PROCESS, IOPRIO_CLASS_IDLE
    if _ret == 0:
        result["ioprio"] = "allowed"
    else:
        _errno = ctypes.get_errno()
        if _errno == 38:  # ENOSYS: no such syscall on this kernel
            result["ioprio"] = "refused"
        else:
            result["ioprio"] = "refused" if _errno in (1, 13) else f"error:{_errno}"
finally:
    try:
        os.kill(_child5, 9)
    except OSError:
        pass
    try:
        os.waitpid(_child5, 0)
    except OSError:
        pass
# SysV IPC is outside Landlock and outside the seccomp bans, and
# the uid mapping preserves the owner's kernel UID — without its
# own IPC namespace the target could shmat any SysV segment of
# the owner it knows the id of, even mode 0600, and corrupt it.
# The probe parent creates a segment (IPC_PRIVATE) and passes its
# id as argv[4]; the target must not be able to attach to it at
# all. In a fresh IPC namespace shmat on a foreign id fails
# EINVAL; EACCES/EPERM would also be a refusal.
_libc.shmat.restype = ctypes.c_void_p
_libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
_shmid = int(sys.argv[4])
_addr = _libc.shmat(_shmid, None, 0)
if _addr is None or _addr == 0xFFFFFFFFFFFFFFFF:
    _errno = ctypes.get_errno()
    result["ipc"] = "refused" if _errno in (1, 13, 22) else f"error:{_errno}"
else:
    try:
        _seen = ctypes.string_at(_addr, 16)
        if _seen != b"ipc-probe-secret":
            result["ipc"] = "error:wrong-content"
        else:
            try:
                ctypes.memset(_addr, ord("X"), 1)
                result["ipc"] = "allowed:write"
            except OSError:
                result["ipc"] = "allowed:read"
    finally:
        _libc.shmdt.restype = ctypes.c_int
        _libc.shmdt.argtypes = [ctypes.c_void_p]
        _libc.shmdt(_addr)
result["uid"] = os.getuid()
print("PROBE:" + json.dumps(result), flush=True)
time.sleep(2)
"""


class ProcessError(BridgeProtocolError):
    """Typed process error (``code`` is part of the wire schema)."""


@dataclass(frozen=True)
class Grant:
    path: Path
    access: str  # "r" | "rw" | "rx" | "dir" | "rdir" (dir ops only, no file content)
    # Identity the daemon recorded when it verified the object.
    # The helper opens the path without following symlinks and
    # refuses the whole spawn when the fd is a different object
    # (TOCTOU between grant computation and Landlock install).
    dev: int | None = None
    ino: int | None = None


def _grants_wire(grants: list[Grant]) -> list[dict[str, object]]:
    """Grants as the helper expects them, identity included."""
    return [{"path": str(g.path), "access": g.access, "dev": g.dev, "ino": g.ino} for g in grants]


@dataclass(frozen=True)
class ArgSlot:
    name: str
    kind: str  # "path_in_root" | "enum" | "string" | "int"
    required: bool = False
    values: tuple[str, ...] = ()
    minimum: int | None = None
    maximum: int | None = None
    form: str | None = None  # string only: "after_dashdash" | "flag"
    flag: str | None = None  # string with form="flag": "--opt="

    def describe(self) -> dict[str, object]:
        out: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
        }
        if self.kind == "enum":
            out["values"] = list(self.values)
        if self.kind == "int":
            out["min"] = self.minimum
            out["max"] = self.maximum
        return out


@dataclass(frozen=True)
class ProcessProfile:
    id: str
    binary: Path  # canonical absolute path
    binary_dev: int
    binary_ino: int
    binary_size: int
    binary_mtime_ns: int
    argv_prefix: tuple[str, ...]
    args: tuple[ArgSlot, ...]
    cwd_root_id: str
    cwd_parts: tuple[str, ...]
    env: dict[str, str]
    timeout_s: int = MAX_PROCESS_TIMEOUT_S


def _reject(message: str) -> BridgeConfigError:
    return BridgeConfigError(message)


def _req_str(params: dict[str, object], name: str) -> str:
    """Required string param (``_str_param`` returns ``str | None``)."""
    value = _str_param(params, name, required=True)
    if value is None:
        raise ProcessError("invalid_params", f"params.{name} is required")
    return value


def _check_binding(principal_id: str | None, run_id: str | None) -> None:
    """Envelope attribution used as the ownership binding (E03-T06):
    the same 1..256-char string bound the protocol enforces for
    envelope fields. A direct service caller (not the wire) must
    not smuggle an unbounded binding past that validation."""
    for name, value in (("principal_id", principal_id), ("run_id", run_id)):
        if value is not None and not 1 <= len(value) <= 256:
            raise ProcessError("invalid_params", f"{name} must be a string of 1..256 characters")


def _nonneg_int(params: dict[str, object], name: str, *, default: int) -> int:
    """Non-negative int param with no fixed upper bound (offsets)."""
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProcessError("invalid_params", f"params.{name} must be an integer")
    if value < 0:
        raise ProcessError("invalid_params", f"params.{name} must be >= 0")
    return value


def _decode_poll_chunk(raw: bytes, *, at_eof: bool) -> tuple[str, int]:
    """Decode a poll chunk as UTF-8 with replacement.

    Unlike the FS read path (which rejects invalid UTF-8), poll
    replaces undecodable bytes with U+FFFD — process output is
    untrusted bytes, not a file the model asked to read exactly.
    An incremental decoder holds a trailing incomplete sequence
    back (not consumed) until the stream is finished, so it can
    complete in the next chunk. Crucially, an invalid byte earlier
    in the chunk must not eat that suffix: for b"\\xff\\xe2\\x82"
    the first poll returns "\\ufffd" consuming 1 byte, and the
    next poll (re-read from the tail at the new offset) completes
    "\\u20ac". A hand-rolled split on the first decode error
    cannot do this: the first error is about the invalid byte, so
    it replaces — and consumes — the incomplete suffix as well.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = decoder.decode(raw, final=at_eof)
    pending = decoder.getstate()[0]
    return text, len(raw) - len(pending)


# --- profile loading (fail-closed, same discipline as fs.load_registry)


def _profile_slots(raw: object, *, argv_prefix: tuple[str, ...]) -> tuple[ArgSlot, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise _reject("process profile args must be a list")
    if len(raw) > MAX_ARGS:
        raise _reject("process profile has too many arg slots")
    slots: list[ArgSlot] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise _reject("process profile arg slot must be an object")
        allowed = {"name", "kind", "required", "values", "min", "max", "form", "flag"}
        unknown = set(item) - allowed
        if unknown:
            raise _reject(f"process profile arg slot has unknown fields: {sorted(unknown)}")
        name = item.get("name")
        if not isinstance(name, str) or not _SLOT_NAME_RE.match(name):
            raise _reject("process profile arg slot name is invalid")
        if name in seen:
            raise _reject(f"process profile arg slot {name!r} is duplicated")
        seen.add(name)
        kind = item.get("kind")
        if kind not in ("path_in_root", "enum", "string", "int"):
            raise _reject(f"process profile arg slot {name!r} has unknown kind {kind!r}")
        required = item.get("required", False)
        if not isinstance(required, bool):
            raise _reject(f"process profile arg slot {name!r}: required must be a boolean")
        if kind == "enum":
            values = item.get("values")
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(v, str) and v and "\x00" not in v for v in values)
                or len(set(values)) != len(values)
            ):
                raise _reject(f"process profile arg slot {name!r}: enum values are invalid")
            if any(k in item for k in ("min", "max", "form", "flag")):
                raise _reject(f"process profile arg slot {name!r}: enum takes no min/max/form/flag")
            slots.append(ArgSlot(name=name, kind=kind, required=required, values=tuple(values)))
        elif kind == "int":
            minimum = item.get("min")
            maximum = item.get("max")
            if not isinstance(minimum, int) or isinstance(minimum, bool):
                raise _reject(f"process profile arg slot {name!r}: int min is required")
            if not isinstance(maximum, int) or isinstance(maximum, bool):
                raise _reject(f"process profile arg slot {name!r}: int max is required")
            if minimum > maximum:
                raise _reject(f"process profile arg slot {name!r}: int min > max")
            if any(k in item for k in ("values", "form", "flag")):
                raise _reject(f"process profile arg slot {name!r}: int takes no values/form/flag")
            slots.append(
                ArgSlot(name=name, kind=kind, required=required, minimum=minimum, maximum=maximum)
            )
        elif kind == "string":
            form = item.get("form")
            if form not in ("after_dashdash", "flag"):
                raise _reject(
                    f"process profile arg slot {name!r}: string requires form "
                    "'after_dashdash' or 'flag'"
                )
            if form == "after_dashdash":
                if not argv_prefix or argv_prefix[-1] != "--":
                    raise _reject(
                        f"process profile arg slot {name!r}: after_dashdash requires "
                        "argv_prefix to end with '--'"
                    )
                if "flag" in item:
                    raise _reject(
                        f"process profile arg slot {name!r}: after_dashdash takes no flag"
                    )
                slots.append(ArgSlot(name=name, kind=kind, required=required, form=form))
            else:
                flag = item.get("flag")
                if not isinstance(flag, str) or not flag.startswith("--") or not flag.endswith("="):
                    raise _reject(
                        f"process profile arg slot {name!r}: flag must look like '--opt='"
                    )
                slots.append(ArgSlot(name=name, kind=kind, required=required, form=form, flag=flag))
        else:  # path_in_root
            if any(k in item for k in ("values", "min", "max", "form", "flag")):
                raise _reject(
                    f"process profile arg slot {name!r}: path_in_root takes no "
                    "values/min/max/form/flag"
                )
            slots.append(ArgSlot(name=name, kind=kind, required=required))
    return tuple(slots)


def load_profiles(registry: RootsRegistry) -> tuple[ProcessProfile, ...]:
    """Load ``process_profiles`` from the host-bridge config.

    Fail-closed: any validation error raises ``BridgeConfigError``
    and the daemon refuses to start. An absent config (or no
    ``process_profiles`` key) means no profiles — the process
    capabilities simply stay unavailable.
    """
    config_path = registry.config_path
    if config_path is None or not config_path.exists():
        return ()
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _reject(f"host-bridge config is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise _reject("host-bridge config must be a JSON object")
    raw_profiles = data.get("process_profiles", [])
    if not isinstance(raw_profiles, list):
        raise _reject("process_profiles must be a list")
    profiles: list[ProcessProfile] = []
    seen_ids: set[str] = set()
    for item in raw_profiles:
        if not isinstance(item, dict):
            raise _reject("process profile must be an object")
        allowed = {"id", "binary", "argv_prefix", "args", "cwd", "env", "timeout_s"}
        unknown = set(item) - allowed
        if unknown:
            raise _reject(f"process profile has unknown fields: {sorted(unknown)}")
        profile_id = item.get("id")
        if not isinstance(profile_id, str) or not _SLOT_NAME_RE.match(profile_id):
            raise _reject("process profile id is invalid")
        if profile_id in seen_ids:
            raise _reject(f"process profile id {profile_id!r} is duplicated")
        seen_ids.add(profile_id)
        binary_raw = item.get("binary")
        if not isinstance(binary_raw, str) or not binary_raw.startswith("/"):
            raise _reject(f"process profile {profile_id!r}: binary must be an absolute path")
        binary = Path(binary_raw)
        try:
            canonical = binary.resolve(strict=True)
        except OSError as exc:
            raise _reject(f"process profile {profile_id!r}: binary is not readable: {exc}") from exc
        if canonical.name in _SHELL_BINARIES:
            raise _reject(f"process profile {profile_id!r}: shell binaries are not allowed")
        try:
            st = canonical.stat()
        except OSError as exc:
            raise _reject(f"process profile {profile_id!r}: binary stat failed: {exc}") from exc
        if not stat.S_ISREG(st.st_mode):
            raise _reject(f"process profile {profile_id!r}: binary is not a regular file")
        if st.st_mode & 0o6000:
            raise _reject(f"process profile {profile_id!r}: setuid/setgid binaries are not allowed")
        try:
            with open(canonical, "rb") as handle:
                magic = handle.read(5)
        except OSError as exc:
            raise _reject(f"process profile {profile_id!r}: binary is not readable: {exc}") from exc
        if magic != b"\x7fELF\x02":
            raise _reject(f"process profile {profile_id!r}: binary is not a 64-bit ELF executable")
        for root in registry.roots:
            if canonical.is_relative_to(root.path):
                raise _reject(
                    f"process profile {profile_id!r}: binary is inside allowed root "
                    f"{root.id!r} (write-excluded binaries are not allowed)"
                )
        argv_prefix_raw = item.get("argv_prefix", [])
        if not isinstance(argv_prefix_raw, list) or not all(
            isinstance(v, str) and v and "\x00" not in v for v in argv_prefix_raw
        ):
            raise _reject(f"process profile {profile_id!r}: argv_prefix must be a list of strings")
        if sum(len(v) for v in argv_prefix_raw) > MAX_ARGS_TOTAL_BYTES:
            raise _reject(f"process profile {profile_id!r}: argv_prefix is too long")
        argv_prefix = tuple(argv_prefix_raw)
        slots = _profile_slots(item.get("args"), argv_prefix=argv_prefix)
        cwd_raw = item.get("cwd")
        if not isinstance(cwd_raw, dict):
            raise _reject(f"process profile {profile_id!r}: cwd must be an object")
        cwd_unknown = set(cwd_raw) - {"root_id", "path"}
        if cwd_unknown:
            raise _reject(
                f"process profile {profile_id!r}: cwd has unknown fields: {sorted(cwd_unknown)}"
            )
        root_id = cwd_raw.get("root_id")
        if not isinstance(root_id, str) or registry.by_id(root_id) is None:
            raise _reject(f"process profile {profile_id!r}: cwd root_id is unknown")
        cwd_path = cwd_raw.get("path", ".")
        if not isinstance(cwd_path, str):
            raise _reject(f"process profile {profile_id!r}: cwd path must be a string")
        try:
            cwd_parts = _parse_rel_path(cwd_path, allow_root=True)
        except FsError as exc:
            raise _reject(f"process profile {profile_id!r}: cwd path is invalid: {exc}") from exc
        env_raw = item.get("env", {})
        if not isinstance(env_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
        ):
            raise _reject(f"process profile {profile_id!r}: env must be a string map")
        if len(env_raw) > MAX_PROFILE_ENV_VARS:
            raise _reject(f"process profile {profile_id!r}: too many env vars")
        if any(not _ENV_KEY_RE.match(k) for k in env_raw):
            raise _reject(f"process profile {profile_id!r}: env key is invalid")
        if any(k in _FORBIDDEN_PROFILE_ENV for k in env_raw):
            raise _reject(f"process profile {profile_id!r}: env sets a forbidden variable")
        if any(len(v) > MAX_ENV_VALUE_CHARS or "\x00" in v for v in env_raw.values()):
            raise _reject(f"process profile {profile_id!r}: env value is invalid")
        if sum(len(k) + len(v) for k, v in env_raw.items()) > MAX_ENV_TOTAL_BYTES:
            raise _reject(f"process profile {profile_id!r}: env is too large")
        timeout_s = item.get("timeout_s", MAX_PROCESS_TIMEOUT_S)
        if not isinstance(timeout_s, int) or isinstance(timeout_s, bool):
            raise _reject(f"process profile {profile_id!r}: timeout_s must be an integer")
        if not 1 <= timeout_s <= MAX_PROCESS_TIMEOUT_S:
            raise _reject(
                f"process profile {profile_id!r}: timeout_s must be 1..{MAX_PROCESS_TIMEOUT_S}"
            )
        profiles.append(
            ProcessProfile(
                id=profile_id,
                binary=canonical,
                binary_dev=st.st_dev,
                binary_ino=st.st_ino,
                binary_size=st.st_size,
                binary_mtime_ns=st.st_mtime_ns,
                argv_prefix=argv_prefix,
                args=slots,
                cwd_root_id=root_id,
                cwd_parts=cwd_parts,
                env=dict(env_raw),
                timeout_s=timeout_s,
            )
        )
    return tuple(profiles)


# --- Landlock grant walk -------------------------------------------------


def _system_grants() -> list[Grant]:
    grants: list[Grant] = []
    for path, access in _SYSTEM_GRANT_PATHS:
        target = Path(path)
        if target.exists():
            try:
                st = target.lstat()
            except OSError:
                continue
            grants.append(Grant(target, access, dev=st.st_dev, ino=st.st_ino))
    return grants


def _read_interp_path(binary: Path) -> str | None:
    """PT_INTERP path of a 64-bit ELF binary, or None if static.

    The loader already requires a 64-bit ELF. A malformed header
    fails the spawn (fail closed): EXECUTE coverage cannot be
    proven without a parsed interpreter.
    """
    with open(binary, "rb") as handle:
        header = handle.read(64)
        if len(header) < 64 or header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1:
            raise ProcessError("spawn_failed", "binary is not a 64-bit little-endian ELF")
        (phoff,) = struct.unpack_from("<Q", header, 32)
        (phentsize, phnum) = struct.unpack_from("<HH", header, 54)
        if phentsize < 56 or phnum > 256:
            raise ProcessError("spawn_failed", "binary has a bad program header table")
        for index in range(phnum):
            handle.seek(phoff + index * phentsize)
            phdr = handle.read(56)
            if len(phdr) < 56:
                raise ProcessError("spawn_failed", "binary has a short program header")
            (p_type,) = struct.unpack_from("<I", phdr, 0)
            if p_type != 3:  # PT_INTERP
                continue
            (p_offset,) = struct.unpack_from("<Q", phdr, 8)
            (p_filesz,) = struct.unpack_from("<Q", phdr, 32)
            if p_filesz > 4096:
                raise ProcessError("spawn_failed", "binary interpreter path is too long")
            handle.seek(p_offset)
            raw = handle.read(p_filesz)
            interp = raw.split(b"\x00", 1)[0].decode("utf-8")
            if not interp.startswith("/"):
                raise ProcessError("spawn_failed", "binary interpreter is not absolute")
            return interp
    return None


def _exec_grants(binary: Path) -> list[Grant]:
    """EXECUTE grants for exactly the approved binary + its interpreter."""
    # Resolve first: the helper never grants through a symlink (it
    # skips symlink grant paths), and profile binaries are often
    # symlinks (/usr/bin/python3 -> python3.14, uutils multicall).
    binary = binary.resolve(strict=True)
    binary_st = binary.lstat()
    grants = [Grant(binary, "rx", dev=binary_st.st_dev, ino=binary_st.st_ino)]
    interp = _read_interp_path(binary)
    if interp is not None:
        # Resolve too: the interpreter path itself is usually a
        # symlink (/lib64/ld-linux-x86-64.so.2 -> ld-2.x.so),
        # and the helper never grants through a symlink.
        interp_path = Path(interp).resolve(strict=True)
        interp_st = interp_path.lstat()
        grants.append(Grant(interp_path, "rx", dev=interp_st.st_dev, ino=interp_st.st_ino))
    return grants


def _compute_cwd_grants(
    fs_service: HostFsService,
    registry: RootsRegistry,
    profile: ProcessProfile,
    *,
    token: CancelToken,
) -> list[Grant]:
    """Grant only explicitly allowed parts of the profile cwd tree.

    Every directory gets a directory-operations-only rule (``dir``
    for a writable tree, ``rdir`` for a read-only one) and every
    allowed file gets its own file rule: a Landlock path-beneath rule
    on a directory covers every file created beneath it later, so a
    full ``rw`` rule on a directory would let a long-lived profile
    read a secret file the owner drops in after the walk (the
    directory's (dev, ino) does not change and there is no
    re-classification at runtime). Directory rules therefore never
    carry file-content rights, while the directory itself still
    keeps its own operations (list, create, remove, rename), so a
    protected file nested inside an allowed tree stays denied and
    an approved command can still listdir('.') and move files
    between granted directories. Symlinks are never granted or
    followed (Landlock checks the resolved target, so a granted
    symlink would hand out whatever it points to). Write-exclusion
    is per path: a subtree the FS API refuses to write (config
    ``write_excluded`` or a SlavikAI checkout) is granted read-only,
    never ``rw`` (ADR-0013 §9). Landlock rules union within one
    ruleset: a mutating (``dir``) grant on a directory also covers
    every protected subtree beneath it, because the protected
    directory's own read-only rule does not cancel the ancestor's
    REMOVE_FILE/REMOVE_DIR/MAKE_* bits — so before this fix a profile
    could delete a file inside a write-excluded subdirectory or
    create entries inside an excluded checkout even though the FS API
    returns ``write_excluded`` for the same path. A directory that is
    an ancestor of a protected subtree therefore gets the
    non-mutating ``rdir`` grant instead; only directories with no
    protected descendants keep ``dir``. Each allowed directory keeps
    its own ``dir`` grant, so legitimate create/remove/rename inside
    allowed siblings still works, and listdir/traversal still works
    everywhere (``rdir`` carries READ_DIR + REFER). A downgraded
    directory loses create/remove/rename on its own entries: when the
    cwd itself is such an ancestor, even an O_CREAT-only create there
    is refused without leaving an empty entry (before, the empty
    entry appeared and only the open/read failed).

    Verified live on the host (not by reasoning alone): with
    directory-operations-only rules a profile can listdir('.') and
    read/write its granted files, cannot read or write a file
    created under the directory after the rules were installed, and
    cannot read an ungranted file that was there all along. Two
    limits follow from Landlock rule semantics and are accepted,
    not worked around: a new file created directly in the tree can
    be created (empty) but not written or read — new content goes
    in $TMPDIR (a full rw grant) and is hard-linked into the tree,
    which keeps it writable via the tmp path and visible to the
    owner in the tree; and an atomic tmp+rename update of a granted
    file lands but the new inode is not covered by the inode-bound
    file rule, so the updated file is unreadable/unwritable to the
    profile afterwards (in-place writes to granted files still
    work). A third limit, also by Landlock design (no deny rules):
    a protected FILE directly inside an allowed directory can still
    be unlinked or renamed over through that directory's own
    operations — only whole protected subtrees are fully isolated by
    the ancestor downgrade, not individual files nested in an allowed
    directory. The design that would remove these limits — an isolated
    filesystem view (mount namespace + bind mounts of only the
    granted files) — is a design decision for the owner, not part
    of this ticket.
    """
    root = registry.by_id(profile.cwd_root_id)
    if root is None:
        raise ProcessError("spawn_failed", "profile cwd root is unknown")
    cwd = root.path.joinpath(*profile.cwd_parts)
    access = "rw" if root.mode == "rw" else "r"
    entries_seen = 0
    grants: list[Grant] = []
    try:
        mount_points = fs_service._mount_points()
    except FsError as exc:
        raise ProcessError("spawn_failed", f"mount topology unavailable: {exc}") from exc
    # The cwd path itself must not cross a mount: a bind mount below
    # the root would hand the target a foreign tree (the FS API
    # refuses nested mount traversal the same way). The root itself
    # is the anchor, so only components below it are checked.
    lexical = root.path
    for part in profile.cwd_parts:
        lexical = lexical / part
        if lexical in mount_points:
            raise ProcessError("spawn_failed", "cwd crosses a nested mount")

    def walk(directory: Path, depth: int, access: str) -> tuple[list[Grant], bool]:
        """Walk one directory; return (grants, has_protected_subtree).

        ``has_protected_subtree`` is True when a protected subtree
        sits beneath this directory: a write-excluded directory, a
        directory ``_classify()`` denies, or a descendant of either.
        Mount points and symlinks do not count: Landlock rules do not
        cross a mount boundary (the mounted tree is a different
        dentry tree) and a symlink's target is checked on its resolved
        path, so an ancestor's mutating bits cannot reach inside them.
        A protected FILE does not count either: its parent's own
        directory grant is what would let a profile unlink it (see the
        docstring) — only whole subtrees are isolated by downgrading
        ancestors.
        """
        nonlocal entries_seen
        if depth > GRANT_WALK_MAX_DEPTH:
            raise ProcessError("spawn_failed", "grant walk too deep")
        # A directory that is itself write-excluded downgrades its
        # whole subtree to read-only: the FS API refuses mutations
        # there, so no "rw" grant may cover it.
        if access == "rw" and fs_service.is_write_excluded(root, directory):
            access = "r"
        try:
            st = directory.lstat()
        except OSError as exc:
            raise ProcessError("spawn_failed", f"cwd is not readable: {exc}") from exc
        if not stat.S_ISDIR(st.st_mode):
            # lstat, not isdir: a symlinked cwd must be rejected, not
            # followed into a directory outside the root.
            raise ProcessError("spawn_failed", "cwd is not a directory")
        base_dev = st.st_dev
        child_grants: list[Grant] = []
        has_protected = False
        # Consume the iterator entry by entry: materializing it with
        # list(it) would load the whole directory into memory before
        # the entry limit or the cancellation token is ever checked,
        # so a huge directory could exhaust the daemon's memory while
        # a cancelled request keeps enumerating.
        try:
            with os.scandir(directory) as it:
                for entry in it:
                    entries_seen += 1
                    if entries_seen > GRANT_WALK_MAX_ENTRIES:
                        raise ProcessError("spawn_failed", "grant walk too large")
                    token.check()
                    try:
                        entry_st = entry.stat(follow_symlinks=False)
                    except OSError:
                        continue  # unreadable entry: never granted
                    if entry.is_symlink():
                        continue  # never granted, never followed
                    entry_path = directory / entry.name
                    if entry_path in mount_points:
                        # Bind mount (even same-device, so st_dev cannot see
                        # it): never granted, never descended — the FS API
                        # refuses nested mounts the same way, including the
                        # final file component.
                        continue
                    if entry_st.st_dev != base_dev:
                        continue  # nested mount: not granted, not descended
                    classification = fs_service._classify(entry_path, mutation=False)
                    if classification != "normal":
                        # Protected: never granted. A protected directory is
                        # still a protected subtree beneath this one, so the
                        # ancestor downgrade applies to it.
                        if entry.is_dir(follow_symlinks=False):
                            has_protected = True
                        continue
                    # Write-exclusion is per path: a writable parent may hold
                    # an excluded child, whose subtree is read-only.
                    entry_access = access
                    if access == "rw" and fs_service.is_write_excluded(root, entry_path):
                        entry_access = "r"
                    if entry.is_dir(follow_symlinks=False):
                        sub_grants, sub_protected = walk(entry_path, depth + 1, entry_access)
                        child_grants.extend(sub_grants)
                        if sub_protected or (entry_access == "r" and access == "rw"):
                            # A write-excluded child is itself a protected
                            # subtree beneath this directory.
                            has_protected = True
                    else:
                        child_grants.append(
                            Grant(
                                entry_path,
                                entry_access,
                                dev=entry_st.st_dev,
                                ino=entry_st.st_ino,
                            )
                        )
        except OSError as exc:
            raise ProcessError("spawn_failed", f"cwd is not readable: {exc}") from exc
        # The directory itself gets directory-operation rights only
        # ("dir" for a writable tree, "rdir" for a read-only one):
        # never file-content rights, or a file created beneath it
        # after the walk would inherit them. And "dir" only when no
        # protected subtree sits beneath it: Landlock rules union
        # within one ruleset, so a mutating grant on an ancestor
        # would hand REMOVE_FILE/REMOVE_DIR/MAKE_* to the protected
        # subtree too — its own read-only rule does not cancel the
        # ancestor's bits. Such an ancestor gets the non-mutating
        # "rdir" grant instead (list + traversal); each allowed
        # directory keeps its own "dir" grant.
        dir_access = "dir" if access == "rw" and not has_protected else "rdir"
        return (
            [Grant(directory, dir_access, dev=st.st_dev, ino=st.st_ino)] + child_grants,
            has_protected,
        )

    # The walk classifies every child, but never the cwd itself: a
    # profile with cwd="credentials" would get one rw grant over the
    # whole directory even though _classify(credentials) is "deny"
    # (the children classify "normal" one by one, so the parent is
    # granted as a single clean rule). The cwd itself must classify
    # "normal" before it can be granted at all.
    try:
        cwd_classification = fs_service._classify(cwd, mutation=False)
    except FsError as exc:
        raise ProcessError("spawn_failed", f"cwd cannot be classified: {exc}") from exc
    if cwd_classification != "normal":
        raise ProcessError("spawn_failed", "cwd is protected by host policy")
    cwd_grants, _ = walk(cwd, 0, access)
    grants.extend(cwd_grants)
    if len(grants) > GRANT_WALK_MAX_GRANTS:
        raise ProcessError("spawn_failed", "too many Landlock grants")
    return grants


# --- spawner -------------------------------------------------------------


class SpawnError(Exception):
    """The systemd-run scope could not be created."""


@dataclass
class SpawnHandle:
    popen: subprocess.Popen[bytes] | None
    unit_name: str
    cgroup_path: Path | None
    run_dir: Path

    def leader_done(self) -> bool:
        if self.popen is None:
            return True
        return self.popen.poll() is not None


class SystemdRunSpawner:
    """Start the helper in a per-process systemd scope (a cgroup v2).

    The scope IS the process's cgroup: systemd places the helper in
    it before exec, so placement precedes the target by construction.
    The scope is created with ``Delegate=yes`` so the bridge owns the
    subtree (``cgroup.procs`` / ``cgroup.kill`` are writable by the
    owner, not by the target — the target cannot see them through
    Landlock, which grants it no access to /sys).
    """

    def spawn(self, spec_path: Path, *, unit_name: str, run_dir: Path) -> SpawnHandle:
        cmd = [
            "systemd-run",
            "--user",
            "--scope",
            f"--unit={unit_name}",
            "-p",
            "Delegate=yes",
            sys.executable,
            "-m",
            "host_bridge.spawn_helper",
            str(spec_path),
        ]
        try:
            popen = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            raise SpawnError(f"systemd-run failed to start: {exc}") from exc
        cgroup: Path | None = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            cgroup = self._cgroup_of(unit_name)
            if cgroup is not None:
                break
            if popen.poll() is not None:
                break
            time.sleep(0.05)
        if cgroup is None:
            # The scope path could not be resolved: systemd-run may
            # have failed, but _cgroup_of() also returns None when
            # systemctl itself errors or times out — and then the
            # helper may already be running, possibly with a target
            # that already forked descendants. Killing only the
            # launcher leaves second-generation descendants alive
            # (PDEATHSIG does not cover the whole tree), and _run()
            # finalizes the record right after this raise, so the
            # tree would keep running with no enforcement at all
            # (ADR-0013 §10a). Destroy the tree by unit name and
            # confirm the scope is gone before reporting the
            # failed spawn.
            self._destroy_scope_by_unit(unit_name)
            try:
                popen.wait(timeout=2)
            except subprocess.TimeoutExpired:
                popen.kill()
            raise SpawnError("systemd-run scope did not appear")
        return SpawnHandle(popen=popen, unit_name=unit_name, cgroup_path=cgroup, run_dir=run_dir)

    @staticmethod
    def _unit_gone(unit_name: str) -> bool:
        """True when the scope is finished: systemd no longer knows
        the unit, or it is inactive (a scope deactivates only once
        its cgroup is empty)."""
        try:
            out = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit_name,
                    "-p",
                    "LoadState,ActiveState",
                ],
                capture_output=True,
                timeout=5,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        fields = dict(line.split("=", 1) for line in out.stdout.splitlines() if "=" in line)
        return fields.get("LoadState") == "not-found" or fields.get("ActiveState") == "inactive"

    def _destroy_scope_by_unit(self, unit_name: str, timeout_s: float = 5.0) -> None:
        """SIGKILL everything in the scope by unit name and wait
        until the unit is gone. Best effort: the spawn fails either
        way, but a leaked tree would keep running unenforced, so
        the wait is bounded instead of skipped."""
        try:
            subprocess.run(
                ["systemctl", "--user", "kill", "--signal=SIGKILL", unit_name],
                capture_output=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._unit_gone(unit_name):
                return
            time.sleep(0.05)

    @staticmethod
    def list_bridge_scopes() -> list[str]:
        """Unit names of live ``slavikai-bridge-*.scope`` units.

        Used by the E03-T06 startup orphan sweep. Fail-closed:
        any failure that could hide a live scope raises
        BridgeConfigError — except the case where no systemd user
        manager exists at all (no systemctl binary, or the bus is
        unreachable because the system was not booted with systemd):
        then no scope can exist, so the sweep is vacuously empty.
        A broken-but-present manager must never be mistaken for
        an absent one: only two precise signatures count as "no
        manager" — "not booted with systemd", and a bus-connect
        failure whose errno proves the bus socket itself does not
        exist ("No such file or directory"). Any other bus failure
        ("Permission denied", "Connection refused", a transient
        error) means a manager may be alive behind an unreachable
        bus, so the sweep refuses to start.
        """
        try:
            out = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "list-units",
                    "--type=scope",
                    "--all",
                    "--plain",
                    "--no-legend",
                ],
                capture_output=True,
                timeout=5,
                text=True,
            )
        except FileNotFoundError:
            return []  # no systemctl binary: no user manager, no scopes
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise BridgeConfigError(f"orphan sweep cannot list scopes: {exc}") from exc
        if out.returncode != 0:
            err = out.stderr or ""
            if "System has not been booted with systemd" in err:
                return []  # no systemd at all: no scopes can exist
            if "Failed to connect to bus" in err and "No such file or directory" in err:
                # The bus socket itself does not exist: no manager
                # is listening at the address the whole stack
                # derives from XDG_RUNTIME_DIR, so no scope can
                # exist. Any other bus failure leaves the question
                # open and fails closed below.
                return []
            raise BridgeConfigError(
                "orphan sweep cannot list scopes: systemctl exited "
                f"{out.returncode}: {err.strip()[:200]}"
            )
        units: list[str] = []
        for line in out.stdout.splitlines():
            if not line.strip():
                continue
            name = line.split()[0]
            if "." not in name:
                raise BridgeConfigError(f"orphan sweep cannot parse systemctl output: {line!r}")
            if name.startswith("slavikai-bridge-") and name.endswith(".scope"):
                units.append(name)
        return units

    def destroy_scope_checked(self, unit_name: str, timeout_s: float = 5.0) -> bool:
        """Destroy a scope by unit name and verify it is gone.

        Unlike ``_destroy_scope_by_unit`` (best effort, silent on
        timeout), this is the checked variant the E03-T06 orphan
        sweep needs: False means the tree may still be alive and
        the sweep must fail closed instead of serving with it.
        """
        self._destroy_scope_by_unit(unit_name, timeout_s=timeout_s)
        return self._unit_gone(unit_name)

    @staticmethod
    def _cgroup_of(unit_name: str) -> Path | None:
        try:
            out = subprocess.run(
                ["systemctl", "--user", "show", unit_name, "-p", "ControlGroup"],
                capture_output=True,
                timeout=5,
                text=True,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in out.stdout.splitlines():
            if line.startswith("ControlGroup="):
                value = line.split("=", 1)[1].strip()
                if value:
                    return Path("/sys/fs/cgroup") / value.lstrip("/")
        return None

    def signal_tree(self, handle: SpawnHandle, sig: int, *, timeout_s: float = 5.0) -> None:
        if handle.popen is None:
            return
        name = signal.Signals(sig).name
        subprocess.run(
            ["systemctl", "--user", "kill", f"--signal={name}", handle.unit_name],
            capture_output=True,
            timeout=timeout_s,
        )

    def kill_all(self, handle: SpawnHandle, *, timeout_s: float = 5.0) -> None:
        """SIGKILL the whole cgroup subtree (recursive, in-kernel)."""
        if handle.cgroup_path is not None:
            try:
                (handle.cgroup_path / "cgroup.kill").write_text("1", encoding="utf-8")
                return
            except OSError:
                pass
        self.signal_tree(handle, signal.SIGKILL, timeout_s=timeout_s)

    def cgroup_empty(self, handle: SpawnHandle) -> bool:
        """True when no process remains in the scope's cgroup."""
        if handle.cgroup_path is None:
            return handle.leader_done()
        try:
            procs = (handle.cgroup_path / "cgroup.procs").read_text(encoding="utf-8")
        except OSError:
            # The cgroup is gone: the scope (and everything in it) is
            # finished; systemctl agrees when the unit is gone too.
            return True
        return not procs.strip()


# --- probe ----------------------------------------------------------------


@dataclass
class ProbeResult:
    available: bool
    checks: dict[str, object]
    reason: str


def scope_namespace_for(runtime_dir: Path) -> str:
    """Namespace token binding bridge scopes to one daemon's socket
    directory: the first 12 hex chars of the SHA-256 of the canonical
    socket directory. Stable across restarts of the same daemon (same
    socket dir -> same namespace, so crash recovery still finds its
    own orphans) and distinct across socket dirs (a second daemon
    with another socket directory never sweeps the first's scopes:
    the flock guards one socket directory only, while the systemd
    user manager is shared per UID)."""
    return hashlib.sha256(str(Path(runtime_dir).resolve()).encode("utf-8")).hexdigest()[:12]


def scope_unit_name(namespace: str, process_id: str) -> str:
    """Scope unit name for one process of one daemon namespace."""
    return f"slavikai-bridge-{namespace}-{process_id}.scope"


def _process_id_from_unit(unit_name: str) -> str | None:
    """The process_id encoded in a bridge scope unit name
    (``slavikai-bridge-<namespace>-<process_id>.scope``), or None
    when the name does not follow that shape (e.g. the probe unit,
    or a scope of another daemon's namespace).
    Used by the E03-T06 orphan sweep to skip scopes that gained
    a live record while the sweep was running."""
    prefix, suffix = "slavikai-bridge-", ".scope"
    if not unit_name.startswith(prefix) or not unit_name.endswith(suffix):
        return None
    middle = unit_name[len(prefix) : -len(suffix)]
    namespace, _, process_id = middle.partition("-")
    if len(namespace) != 12 or not all(c in "0123456789abcdef" for c in namespace):
        return None
    if not process_id or process_id == "probe":
        return None
    return process_id


def _read_status_file(run_dir: Path) -> dict[str, object] | None:
    try:
        data = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _status_int(status: dict[str, object], key: str) -> int:
    """An integer counter from a status file (0 when absent/malformed)."""
    value = status.get(key)
    return value if isinstance(value, int) else 0


def _status_str(status: dict[str, object], key: str) -> str:
    """A string field from a status file ("" when absent/malformed)."""
    value = status.get(key)
    return value if isinstance(value, str) else ""


def _tails_match_hashes(record: _Record, stdout_tail: bytes, stderr_tail: bytes) -> bool:
    """True when the tails read are the ones the status describes.

    The helper publishes the SHA-256 of each tail's exact bytes
    alongside its counters, so a tail flushed between the two
    status reads (generation still old, counters still old) is
    caught even though the generation and the lengths agree. A
    status published before the hash fields existed carries no
    digests: the generation + length checks are all that binds it.
    """
    if not record.stdout_tail_sha256 or not record.stderr_tail_sha256:
        return True
    return (
        hashlib.sha256(stdout_tail).hexdigest() == record.stdout_tail_sha256
        and hashlib.sha256(stderr_tail).hexdigest() == record.stderr_tail_sha256
    )


def _snapshot_is_torn(record: _Record, stdout_tail: bytes, stderr_tail: bytes) -> bool:
    """True when the last-read snapshot cannot be a prefix of the
    stream the status describes, so falling through with it would
    hand bytes out at wrong absolute offsets.

    A tail longer than the counters is pure growth — and safe to
    clamp — only while it is below the rotation cap: once the
    stream rotated, the tail is always exactly the cap, so a
    cap-sized tail longer than the counters may already have
    dropped its front bytes. A tail shorter than the counters is
    an older file than the status: safe only before any rotation
    (retained_from == 0), because after a rotation the tail is
    always full and a short file cannot be the window the status
    describes. Same length with a digest mismatch is the flush
    window itself: the tail was replaced between the two status
    reads while the generation and the counters were still old.
    """
    for tail, total, retained, digest in (
        (
            stdout_tail,
            record.stdout_total,
            record.stdout_retained_from,
            record.stdout_tail_sha256,
        ),
        (
            stderr_tail,
            record.stderr_total,
            record.stderr_retained_from,
            record.stderr_tail_sha256,
        ),
    ):
        expected = total - retained
        if len(tail) == expected:
            if digest and hashlib.sha256(tail).hexdigest() != digest:
                return True
        elif len(tail) > expected:
            if len(tail) >= TAIL_BYTES:
                return True
        elif retained > 0:
            return True
    return False


def run_enforcement_probe(
    spawner: SystemdRunSpawner,
    *,
    runtime_dir: Path,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> ProbeResult:
    """Run the enforcement probe through the real spawner + helper.

    Verifies, in the daemon's real context, that a spawned process
    cannot open a TCP/UDP/pathname socket, cannot read a protected
    file nested inside its allowed tree, cannot change file metadata
    (chmod/chown/setxattr/utime — outside Landlock, closed by the
    seccomp ban), cannot change another process's limits via
    prlimit64 (outside Landlock and signal scope, closed by the
    seccomp ban), cannot renice another process via setpriority
    (same family, same ban), cannot move another process to a
    different scheduling policy via sched_setscheduler/sched_setattr
    (same family, same ban), cannot pin another process to a CPU
    via sched_setaffinity (same family, same ban), cannot create a named pipe via mkfifo
    (MAKE_FIFO is handled by Landlock with no allowing grant), cannot
    read or write a file created in its cwd after the grants were
    installed (directory rules carry directory-operation rights only,
    never file-content rights), CAN
    read an allowed file, cannot attach to a SysV shared-memory
    segment of the owner (own IPC namespace), runs with the mapped
    uid, cannot signal
    processes outside its Landlock domain, and is placed in its own
    scope cgroup.
    Any failure makes the process API unavailable (fail-closed).
    """
    probe_dir: Path | None = None
    try:
        runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        probe_dir = Path(tempfile.mkdtemp(prefix="probe-", dir=runtime_dir))
        probe_dir.chmod(0o700)
        (probe_dir / "ok.txt").write_text("probe-ok", encoding="utf-8")
        (probe_dir / ".env").write_text("probe-secret", encoding="utf-8")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(probe_dir / "probe.sock"))
        listener.listen(1)
        listener.settimeout(1.0)
        # An external SysV segment for the ipc check: IPC_PRIVATE
        # so no key can collide with a stale segment, 0600 like an
        # owner's secret segment. Created here (host IPC namespace),
        # its id is passed to the probe child, which must not be
        # able to attach to it at all. Removed in the finally below.
        probe_shmid: int | None = None
        _shm_libc = ctypes.CDLL("libc.so.6", use_errno=True)
        try:
            _shm_libc.shmget.restype = ctypes.c_int
            _shm_libc.shmget.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
            probe_shmid = _shm_libc.shmget(0, 64, 0o1000 | 0o600)
            if probe_shmid < 0:
                return ProbeResult(
                    available=False,
                    checks={"ipc": f"unavailable:{ctypes.get_errno()}"},
                    reason="probe SysV segment unavailable",
                )
            _shm_libc.shmat.restype = ctypes.c_void_p
            _shm_libc.shmat.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
            _shm_addr = _shm_libc.shmat(probe_shmid, None, 0)
            if _shm_addr is None or _shm_addr == 0xFFFFFFFFFFFFFFFF:
                return ProbeResult(
                    available=False,
                    checks={"ipc": f"unavailable:{ctypes.get_errno()}"},
                    reason="probe SysV segment unavailable",
                )
            ctypes.memmove(_shm_addr, b"ipc-probe-secret", 16)
            _shm_libc.shmdt.restype = ctypes.c_int
            _shm_libc.shmdt.argtypes = [ctypes.c_void_p]
            _shm_libc.shmdt(_shm_addr)
            try:
                st = os.stat("/usr/bin/python3")
            except OSError as exc:
                return ProbeResult(
                    available=False,
                    checks={"probe_binary": f"unavailable:{exc}"},
                    reason="probe binary is not available",
                )
            # The probe dir is the cwd (the helper chdirs before
            # Landlock is installed, so no grant is needed for it).
            # Grants cover only the allowed file individually, plus
            # directory-operation rights on the probe dir itself (the
            # "late" check needs to create a file there) — never a
            # file-content rule on the whole probe dir, which would
            # cover the protected .env too.
            ok_path = probe_dir / "ok.txt"
            ok_st = ok_path.lstat()
            probe_st = probe_dir.lstat()
            probe_spec = {
                "version": 1,
                "process_id": "probe",
                "binary": "/usr/bin/python3",
                "binary_dev": st.st_dev,
                "binary_ino": st.st_ino,
                "binary_size": st.st_size,
                "binary_mtime_ns": st.st_mtime_ns,
                "argv": [
                    "python3",
                    "-c",
                    _PROBE_CODE,
                    str(probe_dir / "probe.sock"),
                    str(probe_dir / ".env"),
                    str(probe_dir / "ok.txt"),
                    str(probe_shmid),
                ],
                "env": dict(_BASE_ENV),
                "cwd": str(probe_dir),
                "timeout_s": PROBE_TARGET_TIMEOUT_S,
                "run_dir": str(probe_dir),
                "grants": _grants_wire(_system_grants() + _exec_grants(Path("/usr/bin/python3")))
                + _grants_wire(
                    [
                        Grant(ok_path, "r", dev=ok_st.st_dev, ino=ok_st.st_ino),
                        Grant(probe_dir, "dir", dev=probe_st.st_dev, ino=probe_st.st_ino),
                    ]
                ),
                "outer_uid": os.getuid(),
                "outer_gid": os.getgid(),
            }
            spec_path = probe_dir / "spec.json"
            spec_path.write_text(json.dumps(probe_spec), encoding="utf-8")
            spec_path.chmod(0o600)
            # The probe scope carries this daemon's namespace like
            # any other scope: two daemons probing at once must not
            # share one unit name.
            probe_namespace = scope_namespace_for(runtime_dir)
            try:
                handle = spawner.spawn(
                    spec_path,
                    unit_name=f"slavikai-bridge-{probe_namespace}-probe.scope",
                    run_dir=probe_dir,
                )
            except SpawnError as exc:
                return ProbeResult(
                    available=False,
                    checks={"spawn": f"failed:{exc}"},
                    reason="probe spawn failed",
                )
            deadline = time.monotonic() + timeout_s
            verdict: dict[str, object] | None = None
            placement_ok = False
            status: dict[str, object] | None = None
            while time.monotonic() < deadline:
                status = _read_status_file(probe_dir)
                if status is not None and status.get("state") == "running":
                    cgroup = str(status.get("cgroup") or "")
                    if f"slavikai-bridge-{probe_namespace}-probe.scope" in cgroup:
                        placement_ok = True
                try:
                    stdout = (probe_dir / "stdout").read_bytes()
                except OSError:
                    stdout = b""
                for line in stdout.splitlines():
                    if line.startswith(b"PROBE:"):
                        try:
                            verdict = json.loads(line[len(b"PROBE:") :].decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            verdict = None
                if (
                    verdict is not None
                    and status is not None
                    and status.get("state")
                    in (
                        "exited",
                        "killed",
                        "timeout",
                    )
                ):
                    break
                if status is not None and status.get("state") == "spawn_failed":
                    break
                if handle.leader_done() and verdict is not None:
                    break
                time.sleep(0.1)
            if handle.popen is not None:
                try:
                    handle.popen.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            spawner.kill_all(handle)
            if verdict is None:
                return ProbeResult(
                    available=False,
                    checks={"probe": "no verdict", "placement": placement_ok},
                    reason="probe produced no verdict",
                )
            checks: dict[str, object] = {
                "tcp": verdict.get("tcp"),
                "udp": verdict.get("udp"),
                "unix": verdict.get("unix"),
                "protected": verdict.get("protected"),
                "allowed": verdict.get("allowed"),
                "uid": verdict.get("uid"),
                "signal": verdict.get("signal"),
                "metadata": verdict.get("metadata"),
                "prlimit": verdict.get("prlimit"),
                "fifo": verdict.get("fifo"),
                "late": verdict.get("late"),
                "setpriority": verdict.get("setpriority"),
                "scheduler": verdict.get("scheduler"),
                "affinity": verdict.get("affinity"),
                "xattrat": verdict.get("xattrat"),
                "ioprio": verdict.get("ioprio"),
                "ipc": verdict.get("ipc"),
                "placement": placement_ok,
                "landlock_abi": landlock_abi(),
            }
            ok = (
                verdict.get("tcp") == "refused"
                and verdict.get("udp") == "refused"
                and verdict.get("unix") == "refused"
                and verdict.get("protected") == "refused"
                and verdict.get("allowed") == "read"
                and verdict.get("uid") == 0
                and verdict.get("signal") == "refused"
                and verdict.get("metadata") == "refused"
                and verdict.get("prlimit") == "refused"
                and verdict.get("fifo") == "refused"
                and verdict.get("late") == "refused"
                and verdict.get("setpriority") == "refused"
                and verdict.get("scheduler") == "refused"
                and verdict.get("affinity") == "refused"
                and verdict.get("xattrat") == "refused"
                and verdict.get("ioprio") == "refused"
                and verdict.get("ipc") == "refused"
                and placement_ok
            )
            if not ok:
                failed = [k for k, v in checks.items() if v not in ("refused", "read", 0, True, 8)]
                return ProbeResult(
                    available=False,
                    checks=checks,
                    reason=f"probe checks failed: {failed}",
                )
            return ProbeResult(available=True, checks=checks, reason="ok")
        finally:
            listener.close()
            if probe_shmid is not None:
                _shm_libc.shmctl.restype = ctypes.c_int
                _shm_libc.shmctl.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                _shm_libc.shmctl(probe_shmid, 0, None)  # IPC_RMID
    except Exception as exc:  # probe failure is "unavailable", never a crash
        return ProbeResult(
            available=False,
            checks={"probe": f"error:{exc!r}"},
            reason="probe raised",
        )
    finally:
        if probe_dir is not None:
            shutil.rmtree(probe_dir, ignore_errors=True)


# --- service ---------------------------------------------------------------


def _frame_fits(result: dict[str, object], request_id: str) -> bool:
    return HostFsService._frame_fits(result, request_id)


@dataclass
class _Record:
    process_id: str
    profile_id: str
    state: str  # launching | running | killing | exited | killed | timeout | spawn_failed
    handle: SpawnHandle | None = None
    run_dir: Path | None = None
    # Ownership binding (E03-T06): the (principal_id, run_id) pair
    # carried by the request envelope at commit time. Fixed at run
    # time, never rebindable (I2). None means the process is unbound:
    # host.process.stop never touches it.
    run_id: str | None = None
    principal_id: str | None = None
    # Set at the spawn adoption point when the run was stopped while
    # the spawn was in flight: the tree was destroyed before
    # promotion, so the record finalizes as spawn_failed (never
    # "killed") once the cgroup is confirmed empty — unless the
    # command had already executed (see spawn_result): handle
    # receipt is not an exec barrier (the spawner resolves the
    # cgroup after the helper already exec'd), so "the process
    # never ran" does not follow from a late handle alone.
    spawn_destroyed: bool = False
    # Snapshot of a confirmed execution result, taken at the spawn
    # adoption point BEFORE the destroy: (state, exit_code,
    # signal, finished_at) from the helper's status file when it
    # already showed a terminal executed state (exited / killed /
    # timeout). None when the helper had published no such result
    # yet — then the record finalizes as spawn_failed.
    spawn_result: tuple[str, object, object, object] | None = None
    started_at: str = ""
    finished_at: str | None = None
    exit_code: int | None = None
    exit_signal: int | None = None
    started_mono: float = 0.0
    timeout_s: int = MAX_PROCESS_TIMEOUT_S
    spawn_started: bool = False  # spawner.spawn() returned a handle
    kill_requested: bool = False
    kill_reason: str | None = None
    kill_deadline: float = 0.0
    stdout_total: int = 0
    stderr_total: int = 0
    stdout_retained_from: int = 0
    stderr_retained_from: int = 0
    # Publication generation of the last status read: the tail and
    # its counters are one snapshot only when the generation did not
    # change across the read (see _poll).
    status_generation: int = 0
    # SHA-256 digests of the tails the last status read describes
    # ("" when the status predates the hash fields).
    stdout_tail_sha256: str = ""
    stderr_tail_sha256: str = ""
    settled: threading.Event = field(default_factory=threading.Event)
    final: bool = False
    # Confirmed effect of this process's launch (E03-T08): stored
    # here so a retry with the same request_id re-attaches the
    # ORIGINAL effect (same (boot_id, seq)) instead of emitting a
    # second one — the recorder dedupes by emission identity.
    # The kill effect lives in ``kill_effect`` below, never here.
    effect: dict[str, object] | None = None
    # Confirmed effect of this process's kill (E03-T08, Codex
    # round 1, finding 1): stored SEPARATELY from the launch
    # effect. One shared slot made a kill answer re-attach the
    # run effect (and a run retry re-attach the kill effect), and
    # the client rejected both as malformed_response (kind
    # mismatch against the capability).
    kill_effect: dict[str, object] | None = None
    # Pre-kill effect projection (E03-T08, owner condition 5,
    # Codex round 6, finding 2): frame-budgeted before the
    # kill_requested flip; finalized into ``kill_effect`` by
    # ``_answer`` once the kill's outcome is known. Survives on
    # the record so a retry after a lost response (e.g. the
    # signal send raising TimeoutExpired) still emits the effect
    # for the original kill instead of leaving the journal with
    # an unresolvable unknown. Cleared on finalize, and dropped
    # when the record finalizes as spawn_failed (the process never
    # ran — same discipline as ``effect_projection``).
    kill_projection: dict[str, object] | None = None
    # Pre-spawn effect projection (E03-T08, owner condition 5):
    # frame-budgeted before the spawn; finalized into ``effect`` by
    # ``_run_result`` once the record's state is known. Survives on
    # the record so a retry after a lost response still emits the
    # effect for the original spawn instead of leaving the journal
    # with an unresolvable unknown.
    effect_projection: dict[str, object] | None = None


@dataclass(frozen=True)
class _PollSnapshot:
    """Immutable counters + tails of one verified poll snapshot.

    Everything ``_poll`` answers with comes from here, never from
    the shared ``_Record``: a concurrent ``_refresh`` (another poll,
    the reaper) may rewrite the record's counters after the snapshot
    was verified, and answering from the record would then slice the
    verified tails with newer counters — new bytes at old absolute
    offsets. The snapshot is captured before the check runs, so a
    refresh landing during the check cannot move the values the
    answer is built from; the generation + digest checks then prove
    no refresh landed between the capture and the check.
    """

    process_id: str
    state: str
    exit_code: int | None
    exit_signal: int | None
    started_at: str
    finished_at: str | None
    stdout_total: int
    stdout_retained_from: int
    stderr_total: int
    stderr_retained_from: int
    stdout_tail: bytes
    stderr_tail: bytes
    final: bool


def _capture_poll_snapshot(
    record: _Record, stdout_tail: bytes, stderr_tail: bytes
) -> _PollSnapshot:
    """Freeze the record's poll-visible fields with the tails read."""
    return _PollSnapshot(
        process_id=record.process_id,
        state=record.state,
        exit_code=record.exit_code,
        exit_signal=record.exit_signal,
        started_at=record.started_at,
        finished_at=record.finished_at,
        stdout_total=record.stdout_total,
        stdout_retained_from=record.stdout_retained_from,
        stderr_total=record.stderr_total,
        stderr_retained_from=record.stderr_retained_from,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        final=record.final,
    )


class HostProcessService:
    """Implements host.process.run/poll/kill."""

    def __init__(
        self,
        fs_service: HostFsService,
        registry: RootsRegistry,
        *,
        runtime_dir: Path,
        profiles: tuple[ProcessProfile, ...] = (),
        spawner: SystemdRunSpawner | None = None,
        effect_emitter: EffectEmitter | None = None,
    ) -> None:
        self.fs_service = fs_service
        self.registry = registry
        self.runtime_dir = runtime_dir
        # Effect emission (E03-T08): the daemon binds its single
        # emitter here; a standalone service uses its own.
        self._effect_emitter = effect_emitter if effect_emitter is not None else EffectEmitter()
        self._run_base = runtime_dir / "host-bridge-proc"
        self._run_base.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Scope namespace (E03-T06 rework 1): scopes this daemon
        # creates — and the orphan sweep destroys — carry the hash
        # of the socket directory, so a second daemon with another
        # socket directory (same user manager) never loses its
        # trees to this daemon's startup sweep.
        self._scope_namespace = scope_namespace_for(runtime_dir)
        self._profiles = {p.id: p for p in profiles}
        self._spawner = spawner or SystemdRunSpawner()
        self._records: dict[str, _Record] = {}
        self._by_request_id: dict[str, str] = {}
        # Binding recorded with each request_id (E03-T06): a retry
        # with the same request_id but a different (principal_id,
        # run_id) pair is rejected with binding_mismatch instead of
        # silently handing run B a handle to run A's process.
        # Pruned together with _by_request_id in _sweep_finished.
        self._by_request_id_binding: dict[str, tuple[str | None, str | None]] = {}
        # Runs stopped by host.process.stop (E03-T06): pair ->
        # monotonic timestamp of the stop. A run commits only when
        # its pair is absent here (see _run); the mark is dropped
        # when the last record bound to the run is swept, or when
        # it expires (STOPPED_RUN_MARK_TTL_S) with no records left.
        self._stopped_runs: dict[tuple[str | None, str | None], float] = {}
        self._lock = threading.Lock()
        self._probe: ProbeResult | None = None
        self._probe_lock = threading.Lock()
        self._stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_loop, name="host-process-reaper", daemon=True
        )
        self._reaper.start()

    # -- availability / probe ------------------------------------------------

    def profiles_description(self) -> list[dict[str, object]]:
        return [
            {"id": p.id, "args": [slot.describe() for slot in p.args]}
            for p in self._profiles.values()
        ]

    def bind_effect_emitter(self, emitter: EffectEmitter) -> None:
        """Share the daemon's single emitter (E03-T08): one
        ``(boot_id, seq)`` sequence across both services."""
        self._effect_emitter = emitter

    @property
    def effect_emitter(self) -> EffectEmitter:
        """The emitter this service stamps mutation effects with."""
        return self._effect_emitter

    def has_profiles(self) -> bool:
        return bool(self._profiles)

    def probe(self) -> ProbeResult:
        with self._probe_lock:
            if self._probe is not None:
                return self._probe
            self._probe = run_enforcement_probe(self._spawner, runtime_dir=self.runtime_dir)
            return self._probe

    def is_available(self) -> bool:
        if not self._profiles:
            return False
        return self.probe().available

    def list_processes(
        self, *, run_id: str | None = None, principal_id: str | None = None
    ) -> list[dict[str, object]]:
        """Read-only snapshot of the process records (E03-T07).

        The only way diagnostics reads the registry (owner
        clarification 2): a frozen copy of the wire-visible fields,
        sorted by ``started_at`` (then ``process_id`` for a stable
        order). No refresh, no finalization, no mutation of any
        kind — a record whose reaper has not finalized it yet is
        reported with its last published state.

        With ``run_id`` given, only the records bound to the
        (principal_id, run_id) ownership pair are returned; an
        unknown run yields an empty list. Without it — all records.
        No argv/env/cwd/stdout/stderr/scope names ever leave the
        service: the snapshot carries exactly the fields the wire
        schema allows.
        """
        with self._lock:
            entries: list[dict[str, object]] = [
                {
                    "process_id": record.process_id,
                    "profile_id": record.profile_id,
                    "state": record.state,
                    "run_id": record.run_id,
                    "started_at": record.started_at,
                    "finished_at": record.finished_at,
                    "exit_code": record.exit_code,
                    "signal": record.exit_signal,
                }
                for record in self._records.values()
                if run_id is None or (record.principal_id, record.run_id) == (principal_id, run_id)
            ]
        entries.sort(key=lambda entry: (str(entry["started_at"]), str(entry["process_id"])))
        return entries

    # -- dispatch ------------------------------------------------------------

    def handle(
        self,
        capability: str,
        params: dict[str, object],
        request_id: str,
        token: CancelToken,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Dispatch one process capability.

        ``principal_id`` / ``run_id`` come from the request envelope
        (E03-T06): they are recorded on the record as the ownership
        binding, never used for authorization (ADR-0013 §4).
        """
        try:
            if capability == "host.process.run":
                return self._run(
                    params, request_id, token, principal_id=principal_id, run_id=run_id
                )
            if capability == "host.process.poll":
                return self._poll(params, request_id, token)
            if capability == "host.process.kill":
                return self._kill(
                    params, request_id, token, principal_id=principal_id, run_id=run_id
                )
            if capability == "host.process.stop":
                return self._stop_run(
                    params, request_id, token, principal_id=principal_id, run_id=run_id
                )
        except ProcessError as exc:
            # The client matches error responses by request_id (same
            # pattern as HostFsService.handle): without it the client
            # reports malformed_response instead of the typed code.
            if exc.request_id is None:
                exc.request_id = request_id
            raise
        except FsError as exc:
            # The shared validators (_check_params, _str_param,
            # _int_param) raise FsError, not ProcessError: convert and
            # keep the correlation, or the client sees
            # malformed_response instead of invalid_params.
            raise ProcessError(exc.code, str(exc), request_id=request_id) from exc
        raise ProcessError("invalid_params", f"unknown process capability {capability}")

    # -- run -----------------------------------------------------------------

    def _validate_args(self, profile: ProcessProfile, raw: object, token: CancelToken) -> list[str]:
        if raw is None:
            raw = []
        if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
            raise ProcessError("invalid_args", "args must be a list of strings")
        if len(raw) > MAX_ARGS:
            raise ProcessError("invalid_args", "too many args")
        if sum(len(v) for v in raw) > MAX_ARGS_TOTAL_BYTES:
            raise ProcessError("invalid_args", "args are too long")
        # Positional binding: a value fills the first unfilled slot,
        # so a required slot after an optional one is missing even
        # when the arg count matches the number of required slots.
        for index, slot in enumerate(profile.args):
            if slot.required and index >= len(raw):
                raise ProcessError("invalid_args", f"missing required arg {slot.name!r}")
        if len(raw) > len(profile.args):
            raise ProcessError("invalid_args", "too many args for this profile")
        out: list[str] = []
        for index, value in enumerate(raw):
            slot = profile.args[index]
            if "\x00" in value:
                raise ProcessError("invalid_args", f"arg {slot.name!r} contains NUL")
            if slot.kind == "path_in_root":
                try:
                    target = self.fs_service._resolve(
                        profile.cwd_root_id, value, mutation=False, allow_root=False
                    )
                except FsError as exc:
                    raise ProcessError(exc.code, str(exc)) from exc
                out.append(str(target.lexical))
            elif slot.kind == "enum":
                if value not in slot.values:
                    raise ProcessError(
                        "invalid_args", f"arg {slot.name!r} is not one of {list(slot.values)}"
                    )
                out.append(value)
            elif slot.kind == "string":
                if len(value) > MAX_STRING_ARG_CHARS:
                    raise ProcessError("invalid_args", f"arg {slot.name!r} is too long")
                if slot.form == "flag":
                    out.append(f"{slot.flag}{value}")
                else:
                    out.append(value)
            else:  # int
                try:
                    number = int(value, 10)
                except ValueError:
                    raise ProcessError(
                        "invalid_args", f"arg {slot.name!r} is not an integer"
                    ) from None
                if str(number) != value:
                    raise ProcessError("invalid_args", f"arg {slot.name!r} is not a plain integer")
                if slot.minimum is not None and number < slot.minimum:
                    raise ProcessError("invalid_args", f"arg {slot.name!r} is below the minimum")
                if slot.maximum is not None and number > slot.maximum:
                    raise ProcessError("invalid_args", f"arg {slot.name!r} is above the maximum")
                out.append(str(number))
        return out

    def _run(
        self,
        params: dict[str, object],
        request_id: str,
        token: CancelToken,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        _check_params(params, frozenset({"profile_id", "args", "timeout_s"}))
        _check_binding(principal_id, run_id)
        profile_id = _req_str(params, "profile_id")
        profile = self._profiles.get(profile_id)
        if profile is None:
            raise ProcessError("unknown_profile", f"unknown profile {profile_id!r}")
        if not self.is_available():
            raise ProcessError(
                "process_unavailable",
                "the enforcement probe failed; process execution is unavailable",
            )
        args = self._validate_args(profile, params.get("args"), token)
        timeout_s = _int_param(
            params, "timeout_s", default=profile.timeout_s, minimum=1, maximum=MAX_PROCESS_TIMEOUT_S
        )
        if timeout_s > profile.timeout_s:
            raise ProcessError(
                "invalid_args", f"timeout_s exceeds the profile timeout {profile.timeout_s}"
            )
        # Commit the record first: cancellation and launch are one
        # state transition, and a retry with the same request_id finds
        # this exact record instead of spawning a second process.
        # The (principal_id, run_id) binding is fixed at commit time
        # (E03-T06, I2): a retry with the same request_id but a
        # different binding is rejected, never silently rebound.
        binding = (principal_id, run_id)
        existing: _Record | None = None
        record: _Record | None = None
        with self._lock:
            token.check()
            recorded_binding = self._by_request_id_binding.get(request_id)
            if recorded_binding is not None and recorded_binding != binding:
                raise ProcessError(
                    "binding_mismatch",
                    f"request_id {request_id!r} was already bound to "
                    f"principal {recorded_binding[0]!r} run "
                    f"{recorded_binding[1]!r}; retry carries a different binding",
                )
            self._prune_stopped_marks_locked(time.monotonic())
            existing_id = self._by_request_id.get(request_id)
            if existing_id is not None:
                existing = self._records.get(existing_id)
            if existing is None:
                if binding in self._stopped_runs:
                    # The run was stopped before this commit: fail
                    # fast with a typed error instead of creating a
                    # process that then has to be killed (I7).
                    raise ProcessError("run_stopped", f"run {run_id!r} was already stopped")
                running = sum(
                    1
                    for record in self._records.values()
                    if record.state in ("launching", "running", "killing")
                )
                if running >= MAX_RUNNING_PROCESSES:
                    raise ProcessError("too_many_processes", "too many processes are running")
                process_id = uuid.uuid4().hex
                record = _Record(
                    process_id=process_id,
                    profile_id=profile_id,
                    state="launching",
                    timeout_s=timeout_s,
                    started_mono=time.monotonic(),
                    run_id=run_id,
                    principal_id=principal_id,
                )
                self._records[process_id] = record
                self._by_request_id[request_id] = process_id
                self._by_request_id_binding[request_id] = binding
        if existing is not None:
            # Wait outside the lock: _finalize needs it to settle the
            # original run, and the reaper needs it for every other
            # process's timeout enforcement.
            return self._run_result(existing, wait_settled=True, token=token)
        assert record is not None
        # E03-T08 (owner condition 5): frame-budget the result +
        # effect BEFORE the spawn — the spawn is the mutation, so
        # an overflow must fail here with effect_build_failed,
        # before any tree exists. The state/started_at are not
        # known yet, so the projection uses their longest possible
        # values; _run_result finalizes it with the real ones.
        # The projection is kept on the record so a retry after a
        # lost response still emits the effect for the original
        # spawn (owner condition 2). On a frame overflow the
        # commit is rolled back: the record never existed.
        projection = self._effect_emitter.build_projection(
            kind="process.run",
            principal_id=principal_id,
            run_id=run_id,
            fields={
                "process_id": record.process_id,
                "profile_id": profile_id,
                "state": LONGEST_STATE,
                "started_at": AT_PLACEHOLDER,
            },
        )
        projected_result: dict[str, object] = {
            "process_id": record.process_id,
            "profile_id": profile_id,
            "state": LONGEST_STATE,
            "started_at": AT_PLACEHOLDER,
            "run_id": run_id,
            "effect": projection,
        }
        if not self._effect_emitter.frame_fits(projected_result, request_id):
            with self._lock:
                self._records.pop(record.process_id, None)
                self._by_request_id.pop(request_id, None)
                self._by_request_id_binding.pop(request_id, None)
            raise ProcessError(EFFECT_BUILD_FAILED, "effect does not fit the response frame budget")
        record.effect_projection = projection
        try:
            self._spawn(record, profile, args, token)
        except ProcessError:
            # A failure before the spawner returned means no process
            # was ever created: settle the record now instead of
            # holding an active slot until the spawn timeout. A
            # failure after the spawn started (or with unknown
            # outcome) keeps the reaper/cleanup path: the tree may
            # exist and must be killed, not just forgotten.
            if not record.spawn_started:
                self._finalize(record, "spawn_failed")
            raise
        except RequestTimeoutError:
            # The request deadline expired while the launch was still
            # preparing (grants walk, before the spawner returned a
            # handle): no tree can exist yet, so settle the record
            # now instead of leaving it to a reaper that cannot
            # finalize a handle-less record — a record stopped
            # mid-launch would otherwise stay "killing" forever,
            # leak its slot, and no repeat STOP would ever report
            # all_final. When the spawn DID start, the record keeps
            # the reaper path: the tree may exist and must be
            # killed, not forgotten.
            if not record.spawn_started:
                self._finalize(record, "spawn_failed")
            raise
        except BaseException as exc:
            # A failure before the spawner returned means no process
            # was ever created: settle the record now. A failure
            # after the spawn started (or with unknown outcome) must
            # NOT finalize: the tree may still be alive, and every
            # cleanup path skips final records (_reap_once, _kill,
            # shutdown) — finalizing here would leave the tree
            # outside the 120 s enforcement. A transient signal-send
            # error (e.g. systemctl timing out on SIGTERM while the
            # spawn deadline expires) is exactly such a failure: the
            # handle exists, the tree may be alive. Hand the record
            # to the reaper instead: it is already in "killing"
            # state (or put it there, without moving an existing
            # kill deadline), which destroys the tree after the
            # grace period and finalizes once the cgroup is empty.
            if not record.spawn_started:
                self._finalize(record, "spawn_failed")
            elif not record.final:
                if not record.kill_requested:
                    record.kill_requested = True
                    record.kill_reason = "cancelled"
                    record.state = "killing"
                    record.kill_deadline = time.monotonic() + KILL_GRACE_S
                record.settled.set()
            raise ProcessError("spawn_failed", f"spawn failed: {exc}") from exc
        return self._run_result(record, wait_settled=True, token=token)

    def _run_result(
        self, record: _Record, *, wait_settled: bool, token: CancelToken
    ) -> dict[str, object]:
        if wait_settled and not record.settled.is_set():
            remaining = token.remaining_s
            if remaining <= 0:
                token.check()
            record.settled.wait(timeout=remaining if remaining != float("inf") else None)
        # E03-T08: finalize the pre-spawn projection with the
        # record's state at answer time, so the effect mirrors the
        # result it rides on. A record that finalized as
        # spawn_failed never ran (or its spawn was destroyed before
        # promotion): no effect — the recorder classifies the
        # effect-less spawn_failed answer as unknown, fail-closed.
        # (Codex round 2, finding 1): a launch still in flight
        # must NOT finalize the projection — the spawn may still
        # fail, and a confirmed effect for a process that never
        # existed would poison the journal (and resolve an unknown
        # for a mutation that never happened). Finalize only once
        # the spawn is known to have started (a tree exists) or
        # the record has left "launching"; the projection stays on
        # the record for a later retry, and this answer carries no
        # effect (unknown outcome).
        # E03-T08 (Codex round 3, finding 1): the check, the
        # finalization and the store must be atomic against a
        # concurrent retry and _finalize — all three take this
        # lock for the effect/projection slots. Without it, two
        # _run_result calls both pass `effect is None` and
        # finalize the same projection: one spawn consumes two
        # daemon seqs (and finalize mutates the projection in
        # place, so the first answer's seq is rewritten by the
        # second finalize). The settled-wait above stays outside
        # the lock: _finalize needs it to settle the record.
        with self._lock:
            if record.effect is None and record.effect_projection is not None:
                if record.state == "spawn_failed":
                    record.effect_projection = None
                elif record.spawn_started or record.state != "launching":
                    record.effect = self._effect_emitter.finalize(
                        record.effect_projection,
                        fields={"state": record.state, "started_at": record.started_at},
                    )
                    record.effect_projection = None
            effect = record.effect
        result: dict[str, object] = {
            "process_id": record.process_id,
            "profile_id": record.profile_id,
            "state": record.state,
            "started_at": record.started_at,
            # The bound run id, or null when unbound (E03-T06):
            # additive only, no existing field changes.
            "run_id": record.run_id,
        }
        if effect is not None:
            result["effect"] = effect
        return result

    def _spawn(
        self, record: _Record, profile: ProcessProfile, args: list[str], token: CancelToken
    ) -> None:
        token.check()
        root = self.registry.by_id(profile.cwd_root_id)
        if root is None:
            raise ProcessError("spawn_failed", "profile cwd root is unknown")
        cwd = root.path.joinpath(*profile.cwd_parts)
        run_dir = self._run_base / record.process_id
        run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Remember the directory on the record: a spawn that fails
        # before a handle exists must still be swept.
        record.run_dir = run_dir
        tmp_dir = run_dir / "tmp"
        tmp_dir.mkdir(mode=0o700, exist_ok=True)
        grants = _system_grants()
        grants.extend(_compute_cwd_grants(self.fs_service, self.registry, profile, token=token))
        tmp_st = tmp_dir.lstat()
        grants.append(Grant(tmp_dir, "rw", dev=tmp_st.st_dev, ino=tmp_st.st_ino))
        grants.extend(_exec_grants(profile.binary))
        env = dict(_BASE_ENV)
        env["HOME"] = str(tmp_dir)
        env["TMPDIR"] = str(tmp_dir)
        env.update(profile.env)
        argv = [str(profile.binary)] + list(profile.argv_prefix) + args
        spec = {
            "version": 1,
            "process_id": record.process_id,
            "binary": str(profile.binary),
            "binary_dev": profile.binary_dev,
            "binary_ino": profile.binary_ino,
            "binary_size": profile.binary_size,
            "binary_mtime_ns": profile.binary_mtime_ns,
            "argv": argv,
            "env": env,
            "cwd": str(cwd),
            "timeout_s": record.timeout_s,
            "run_dir": str(run_dir),
            "grants": _grants_wire(grants),
            "outer_uid": os.getuid(),
            "outer_gid": os.getgid(),
        }
        spec_path = run_dir / "spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        spec_path.chmod(0o600)
        try:
            handle = self._spawner.spawn(
                spec_path,
                unit_name=scope_unit_name(self._scope_namespace, record.process_id),
                run_dir=run_dir,
            )
        except SpawnError as exc:
            raise ProcessError("spawn_failed", str(exc)) from exc
        # The spawner returned: a tree may now exist, so every later
        # failure keeps the reaper/cleanup path instead of an
        # immediate finalize.
        record.spawn_started = True
        stopped_destroy = False
        with self._lock:
            if record.final or self._stop.is_set():
                # The reaper gave up on this launch while spawn() was
                # still in flight, or the service is stopping: the
                # tree exists now and nobody owns it. Destroy it
                # before returning, or it would run outside the
                # active-process accounting and without daemon
                # timeout enforcement — or after shutdown.
                late = True
            elif (record.principal_id, record.run_id) in self._stopped_runs:
                # The run was stopped while the spawn was in flight:
                # the tree exists now and must be destroyed before
                # promotion, or the run would gain a "running"
                # process after STOP (I7). Give the record its
                # handle so _try_finalize can confirm the cgroup is
                # empty before finalizing it — usually spawn_failed
                # (the process never ran), not "killed". The
                # exception is a command that already executed
                # before the handle arrived: see the spawn_result
                # snapshot below. Never finalize here: all_final
                # :true must mean the tree is confirmed dead, and
                # kill_all alone does not prove that.
                record.handle = handle
                record.spawn_destroyed = True
                stopped_destroy = True
                late = False
            else:
                record.handle = handle
                late = False
        if late:
            self._spawner.kill_all(handle)
            if self._stop.is_set():
                # No reaper is left to settle this record (shutdown
                # already joined it), so finalize it here instead of
                # leaving it launching forever.
                self._finalize(record, "spawn_failed")
            raise ProcessError("spawn_failed", "spawn did not confirm in time")
        if stopped_destroy:
            # Snapshot a confirmed execution result BEFORE the
            # destroy: the helper may already have run the command
            # and published its final state while spawn() was still
            # resolving the cgroup — handle receipt is not an exec
            # barrier. Without the snapshot the destroy erases the
            # evidence and _try_finalize would rewrite the real
            # outcome to spawn_failed.
            status = _read_status_file(run_dir)
            if status is not None:
                snapshot_state = str(status.get("state"))
                if snapshot_state in ("exited", "killed", "timeout"):
                    record.spawn_result = (
                        snapshot_state,
                        status.get("exit_code"),
                        status.get("signal"),
                        status.get("finished_at"),
                    )
            self._spawner.kill_all(handle)
            raise ProcessError("spawn_failed", "run was stopped during spawn")
        # Wait for the helper to confirm enforcement + exec, bounded by
        # the request deadline (the token) and the spawn timeout.
        deadline = min(
            time.monotonic() + SPAWN_START_TIMEOUT_S, time.monotonic() + token.remaining_s
        )
        while time.monotonic() < deadline:
            if self._stop.is_set():
                # The service is stopping while the launch is still
                # confirming: never promote a dying launch to
                # "running" after shutdown.
                if record.handle is not None:
                    self._spawner.kill_all(record.handle)
                self._finalize(record, "spawn_failed")
                raise ProcessError("spawn_failed", "service is shutting down")
            token.check()
            status = _read_status_file(run_dir)
            if status is not None:
                state = status.get("state")
                if state == "running":
                    record.started_at = str(status.get("started_at") or "")
                    record.started_mono = time.monotonic()
                    if record.state == "launching":
                        record.state = "running"
                    record.settled.set()
                    return
                if state == "spawn_failed":
                    self._finalize(record, "spawn_failed")
                    raise ProcessError("spawn_failed", str(status.get("error") or "spawn failed"))
                if state in ("exited", "killed", "timeout"):
                    # A fast process already finished: it ran exactly
                    # once; report its final state, not a spawn failure.
                    # But if a descendant is still alive in the cgroup
                    # the tree is not finished: report running and let
                    # the reaper finalize once the cgroup is empty.
                    self._refresh(record)
                    if record.final:
                        return
                    record.started_at = str(status.get("started_at") or "")
                    record.started_mono = time.monotonic()
                    if record.state == "launching":
                        record.state = "running"
                    record.settled.set()
                    return
            if handle.leader_done():
                # The leader died before publishing a status. If the
                # cgroup is already empty, no tree survives: finalize
                # spawn_failed at once (a retry waiting on this
                # launch settles with it). If a descendant is still
                # alive in the cgroup, PDEATHSIG protects only the
                # direct child — finalizing here would leave the tree
                # outside every cleanup path (_reap_once skips final
                # records, _kill and shutdown do not touch them).
                # Hand the record to the reaper instead: destroy the
                # tree now, and let it finalize once the cgroup is
                # confirmed empty. The spawn itself still fails for
                # the caller either way.
                if self._spawner.cgroup_empty(handle):
                    self._finalize(record, "spawn_failed")
                elif not record.final:
                    record.kill_requested = True
                    record.kill_reason = "cancelled"
                    record.state = "killing"
                    record.kill_deadline = time.monotonic() + KILL_GRACE_S
                    self._spawner.kill_all(handle)
                raise ProcessError("spawn_failed", "leader exited before start")
            time.sleep(0.05)
        # The deadline hit: cancel the launch first (the kill is
        # initiated whatever happens next), then surface the timeout.
        # The reaper may have finalized the record meanwhile — never
        # mutate a final record's state.
        if not record.final:
            record.kill_requested = True
            record.kill_reason = "cancelled"
            record.state = "killing"
            record.kill_deadline = time.monotonic() + KILL_GRACE_S
            if record.handle is not None:
                self._spawner.signal_tree(record.handle, signal.SIGTERM)
            record.settled.set()
        token.check()
        raise ProcessError("spawn_failed", "spawn did not confirm in time")

    # -- poll ----------------------------------------------------------------

    def _refresh(self, record: _Record) -> None:
        """Refresh totals/state from the helper's status file (best effort).

        A final state in the status file does NOT finalize the record
        by itself: finalization happens only through ``_try_finalize``,
        which additionally requires the scope cgroup to be empty AND
        the scope leader reaped — a leader that exited while a
        descendant is still alive in the cgroup is not finished.
        """
        if record.handle is None:
            return
        status = _read_status_file(record.handle.run_dir)
        if status is None:
            return
        record.stdout_total = _status_int(status, "stdout_total_bytes")
        record.stderr_total = _status_int(status, "stderr_total_bytes")
        record.stdout_retained_from = _status_int(status, "stdout_retained_from")
        record.stderr_retained_from = _status_int(status, "stderr_retained_from")
        record.status_generation = _status_int(status, "generation")
        record.stdout_tail_sha256 = _status_str(status, "stdout_tail_sha256")
        record.stderr_tail_sha256 = _status_str(status, "stderr_tail_sha256")
        if record.started_at == "" and status.get("started_at"):
            record.started_at = str(status.get("started_at"))
        if not record.final:
            state = status.get("state")
            if state in ("exited", "killed", "timeout", "spawn_failed"):
                self._try_finalize(record)

    @staticmethod
    def _tail_bytes(run_dir: Path, name: str) -> bytes:
        try:
            return (run_dir / name).read_bytes()
        except OSError:
            return b""

    def _poll(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(
            params, frozenset({"process_id", "stdout_offset", "stderr_offset", "max_bytes"})
        )
        process_id = _req_str(params, "process_id")
        record = self._records.get(process_id)
        if record is None:
            raise ProcessError("process_not_found", f"unknown process {process_id!r}")
        stdout_offset = _nonneg_int(params, "stdout_offset", default=0)
        stderr_offset = _nonneg_int(params, "stderr_offset", default=0)
        max_bytes = _int_param(
            params, "max_bytes", default=POLL_DEFAULT_MAX_BYTES, minimum=1, maximum=POLL_MAX_BYTES
        )
        token.check()
        self._refresh(record)
        if record.handle is None:
            raise ProcessError("process_not_found", f"unknown process {process_id!r}")
        run_dir = record.handle.run_dir
        # The tail and its counters must be one consistent snapshot:
        # the helper publishes the tail before the status, so a tail
        # newer than the status is possible. A length check alone
        # cannot prove consistency: after a 256 KiB tail rotation the
        # tail keeps its length while its content changes, so an old
        # status plus a new tail passes the length check and hands
        # new bytes out at old absolute offsets (and repeats them on
        # the next poll). The binding is the publication generation
        # the helper bumps on every status write, plus the SHA-256 of
        # each tail's exact bytes the status carries: read status,
        # read the tails, read the status again — the snapshot is
        # consistent only when the generation did not change AND
        # total - retained_from == len(tail) AND the tails hash to
        # the published digests for each stream. The hash closes
        # the remaining window the generation cannot see: both
        # status reads landing between _flush_tails() and
        # _write_status(), with the generation and the counters
        # still old while the tail file is already new. The growth
        # case is still clamped below, so a stale status can never
        # publish an offset beyond the bytes actually read. When no
        # consistent snapshot appears within the attempts, the poll
        # falls through with the last read values only when that
        # snapshot cannot be a torn rotation (a small tail newer
        # than its counters is pure growth and clamps safely); a
        # snapshot that _snapshot_is_torn identifies as torn fails
        # typed instead of returning bytes that may sit at wrong
        # absolute offsets. And the answer is built only from the
        # frozen _PollSnapshot below, never from the shared record:
        # a concurrent _refresh (another poll, the reaper) may
        # rewrite the record's counters after the check, and
        # slicing the verified tails with newer counters would hand
        # new bytes out at old absolute offsets without any
        # consistency error.
        consistent = False
        snap: _PollSnapshot | None = None
        for attempt in range(_POLL_SNAPSHOT_ATTEMPTS):
            self._refresh(record)
            generation = record.status_generation
            stdout_tail = self._tail_bytes(run_dir, "stdout")
            stderr_tail = self._tail_bytes(run_dir, "stderr")
            self._refresh(record)
            # Freeze the counters with the tails BEFORE the check:
            # a refresh landing during the check must not move the
            # values the answer is built from. The generation +
            # digest checks then prove no refresh landed between
            # this capture and the check — one that did makes the
            # generation or the digests disagree, and the attempt
            # is retried with a fresh capture.
            snap = _capture_poll_snapshot(record, stdout_tail, stderr_tail)
            if (
                record.status_generation == generation
                and snap.stdout_total - snap.stdout_retained_from == len(stdout_tail)
                and snap.stderr_total - snap.stderr_retained_from == len(stderr_tail)
                and _tails_match_hashes(record, stdout_tail, stderr_tail)
            ):
                consistent = True
                break
            if attempt + 1 < _POLL_SNAPSHOT_ATTEMPTS:
                time.sleep(_POLL_SNAPSHOT_RETRY_DELAY_S)
        if snap is None:  # pragma: no cover - attempts is always >= 1
            raise ProcessError(
                "snapshot_inconsistent",
                "could not read a consistent tail + counters snapshot",
            )
        if not consistent and _snapshot_is_torn(record, stdout_tail, stderr_tail):
            raise ProcessError(
                "snapshot_inconsistent",
                "could not read a consistent tail + counters snapshot",
            )
        stdout_total = max(snap.stdout_total, snap.stdout_retained_from + len(snap.stdout_tail))
        stderr_total = max(snap.stderr_total, snap.stderr_retained_from + len(snap.stderr_tail))
        if stdout_offset > stdout_total or stderr_offset > stderr_total:
            raise ProcessError("offset_out_of_range", "offset is beyond the stream end")
        if snap.started_at:
            start_ts = datetime.fromisoformat(snap.started_at).timestamp()
            end_ts = (
                datetime.fromisoformat(snap.finished_at).timestamp()
                if snap.finished_at
                else time.time()
            )
            elapsed_ms = int(round((end_ts - start_ts) * 1000))
        else:
            elapsed_ms = 0

        def decode_slice(raw: bytes, start: int, total: int) -> tuple[str, int]:
            # EOF is a property of this byte slice, not of the
            # process: a trailing incomplete sequence is held back
            # (not consumed) unless the slice reaches the actual end
            # of the stream — otherwise a max_bytes/frame-budget cut
            # through a multibyte character corrupts the output.
            eof = snap.final and start + len(raw) >= total
            return _decode_poll_chunk(raw, at_eof=eof)

        def chunk(
            offset: int, total: int, retained_from: int, tail: bytes
        ) -> tuple[str, bytes, int, int, bool]:
            start = max(offset, retained_from)
            truncated = offset < retained_from
            raw = tail[start - retained_from : start - retained_from + max_bytes]
            text, consumed = decode_slice(raw, start, total)
            if raw and consumed == 0:
                if start - retained_from + len(raw) >= len(tail) and not snap.final:
                    # The next character is not fully published yet:
                    # the slice ends where the tail ends, so its
                    # continuation may still arrive. Answer an empty
                    # chunk at the same offset instead of blaming
                    # max_bytes — the limit cut nothing, the bytes
                    # simply are not there yet. A final stream never
                    # takes this branch: its trailing incomplete
                    # character is a replacement, not a wait.
                    return "", raw, start, start, truncated
                # FS read precedent: a limit smaller than the next
                # character must fail loudly instead of returning
                # zero bytes forever (the poll would never advance).
                raise ProcessError(
                    "invalid_params", "max_bytes is too small for the next character"
                )
            return text, raw, start, start + consumed, truncated

        def trim_chunk(raw: bytes, start: int, total: int) -> tuple[str, bytes, int]:
            """Halve a chunk for the frame budget, never below one
            full character and never stuck: halving into a bare
            multibyte prefix extends to the first complete character,
            so the poll still advances."""
            shrunk = raw[: max(1, len(raw) // 2)]
            text, consumed = decode_slice(shrunk, start, total)
            if shrunk and consumed == 0:
                lead = shrunk[0]
                n = 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4
                shrunk = raw[:n]
                text, consumed = decode_slice(shrunk, start, total)
            return text, shrunk, start + consumed

        stdout_text, stdout_raw, stdout_start, stdout_next, stdout_truncated = chunk(
            stdout_offset, stdout_total, snap.stdout_retained_from, snap.stdout_tail
        )
        stderr_text, stderr_raw, stderr_start, stderr_next, stderr_truncated = chunk(
            stderr_offset, stderr_total, snap.stderr_retained_from, snap.stderr_tail
        )
        result: dict[str, object] = {
            "process_id": snap.process_id,
            "state": snap.state,
            "exit_code": snap.exit_code,
            "signal": snap.exit_signal,
            "started_at": snap.started_at,
            "finished_at": snap.finished_at,
            "elapsed_ms": elapsed_ms,
            # The bound run id, or null when unbound (E03-T06):
            # additive only, no existing field changes.
            "run_id": record.run_id,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_next_offset": stdout_next,
            "stderr_next_offset": stderr_next,
            "stdout_total_bytes": stdout_total,
            "stderr_total_bytes": stderr_total,
            "stdout_retained_from": snap.stdout_retained_from,
            "stderr_retained_from": snap.stderr_retained_from,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }
        # Chunk by the serialized-JSON frame budget (same discipline as
        # the FS read path): shrink the longer chunk until it fits.
        # One character is the floor: if a trim cannot shrink either
        # chunk, stop instead of looping on the same bytes forever.
        while not _frame_fits(result, request_id):
            before = (len(stdout_raw), len(stderr_raw))
            if stdout_text and len(stdout_text) >= len(stderr_text):
                stdout_text, stdout_raw, stdout_next = trim_chunk(
                    stdout_raw, stdout_start, stdout_total
                )
            elif stderr_text:
                stderr_text, stderr_raw, stderr_next = trim_chunk(
                    stderr_raw, stderr_start, stderr_total
                )
            else:
                break
            result["stdout"] = stdout_text
            result["stderr"] = stderr_text
            result["stdout_next_offset"] = stdout_next
            result["stderr_next_offset"] = stderr_next
            if (len(stdout_raw), len(stderr_raw)) == before:
                break
        return result

    # -- kill ----------------------------------------------------------------

    def _kill(
        self,
        params: dict[str, object],
        request_id: str,
        token: CancelToken,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        _check_params(params, frozenset({"process_id"}))
        process_id = _req_str(params, "process_id")
        record = self._records.get(process_id)
        if record is None:
            raise ProcessError("process_not_found", f"unknown process {process_id!r}")
        projection: dict[str, object] | None = None

        def _answer(state: str, exit_code: object, exit_signal: object) -> dict[str, object]:
            """Kill result with the kill effect attached when this
            call (or an earlier one, via the stored kill effect or
            the stored kill projection) requested the kill. A final
            record with no stored kill effect died on its own (or
            was never kill-requested): no effect — the recorder
            reads that as no_effect for this call. The launch
            effect is never returned here: it lives in
            ``record.effect`` and only ``_run_result`` answers
            with it (Codex round 1, finding 1)."""
            result: dict[str, object] = {
                "process_id": record.process_id,
                "state": state,
                "exit_code": exit_code,
                "signal": exit_signal,
            }
            # E03-T08 (Codex round 6, finding 2): finalize the kill
            # effect exactly once, under the lock (the run
            # finalization pattern, round 3): a first kill that died
            # in signal_tree leaves its projection on the record;
            # the retry finalizes THAT projection, so the journal
            # gets the confirmed effect for the original kill
            # instead of an unresolvable unknown. The stored
            # projection is cleared on finalize — a later retry
            # re-attaches the stored effect, never a second one.
            # The lock also keeps two concurrent repeat kills from
            # finalizing the same stored projection twice (the
            # in-place finalize would consume two daemon seqs and
            # rewrite the first answer's seq).
            # E03-T08 (Codex round 7): the stored effect is checked
            # FIRST, and only the STORED projection is ever
            # finalized — never this call's local one. The round-6
            # code finalized the local projection whenever the
            # caller's first-kill flag was set, but that flag is stale
            # by the time _answer() runs: a retry may already have
            # finalized the shared projection while this call was
            # still inside signal_tree(), and the original call then
            # finalized the same dict a second time (seq N+1,
            # rewriting the retry answer's seq in place). Checking
            # the stored slots also honours the spawn_failed
            # discipline: _finalize clears the stored projection,
            # and a local variable must not resurrect it.
            with self._lock:
                if record.kill_effect is not None:
                    result["effect"] = record.kill_effect
                elif record.kill_projection is not None:
                    effect = self._effect_emitter.finalize(
                        record.kill_projection,
                        fields={
                            "process_id": record.process_id,
                            "state": state,
                            "exit_code": exit_code,
                            "signal": exit_signal,
                        },
                    )
                    record.kill_effect = effect
                    record.kill_projection = None
                    result["effect"] = effect
            return result

        if record.final:
            return _answer(record.state, record.exit_code, record.exit_signal)
        # The tree may already have finished on its own (leader and
        # all descendants gone) without a poll having finalized the
        # record: a kill then is a no-op and the true final state is
        # the natural one, not "killed".
        self._try_finalize(record)
        if record.final:
            return _answer(record.state, record.exit_code, record.exit_signal)
        token.check()
        if record.state == "launching":
            # The launch is still committing: wait briefly for it to
            # settle (or the request deadline), then kill what exists.
            remaining = token.remaining_s
            wait_s = 2.0 if remaining == float("inf") else max(0.0, min(remaining, 2.0))
            record.settled.wait(timeout=wait_s)
            if record.final:
                return _answer(record.state, record.exit_code, record.exit_signal)
        # A repeat kill observes the termination already in progress:
        # it must not move the original deadline, must not re-signal
        # the tree, and must not overwrite a "timeout" reason with
        # "killed" — otherwise kills more frequent than the grace
        # period postpone the cgroup SIGKILL indefinitely.
        if not record.kill_requested:
            # E03-T08 (owner condition 5): frame-budget the result
            # + effect BEFORE flipping kill_requested — the flip is
            # the mutation, so an overflow fails here with
            # effect_build_failed, before the tree is signalled.
            # The state/exit_code/signal are not known yet, so the
            # projection uses their longest possible values.
            projection = self._effect_emitter.build_projection(
                kind="process.kill",
                principal_id=principal_id,
                run_id=run_id,
                fields={
                    "process_id": record.process_id,
                    "state": LONGEST_STATE,
                    "exit_code": MAX_INT_PLACEHOLDER,
                    "signal": MAX_SIGNAL_PLACEHOLDER,
                },
            )
            projected_result: dict[str, object] = {
                "process_id": record.process_id,
                "state": LONGEST_STATE,
                "exit_code": MAX_INT_PLACEHOLDER,
                "signal": MAX_SIGNAL_PLACEHOLDER,
                "effect": projection,
            }
            if not self._effect_emitter.frame_fits(projected_result, request_id):
                raise ProcessError(
                    EFFECT_BUILD_FAILED, "effect does not fit the response frame budget"
                )
            # E03-T08 (Codex round 6, finding 2): keep the
            # projection on the record BEFORE the flip and the
            # signal — signal_tree() may raise TimeoutExpired
            # after sending SIGTERM, and then this local variable
            # is lost with the call. The stored projection lets a
            # retry finalize the original kill effect in _answer.
            record.kill_projection = projection
            record.kill_requested = True
            record.kill_reason = "killed"
            record.state = "killing"
            record.kill_deadline = time.monotonic() + KILL_GRACE_S
            if record.handle is not None:
                self._spawner.signal_tree(record.handle, signal.SIGTERM)
        deadline = time.monotonic() + min(token.remaining_s, 4.0)
        while time.monotonic() < deadline:
            if self._try_finalize(record):
                return _answer(record.state, record.exit_code, record.exit_signal)
            if time.monotonic() >= record.kill_deadline:
                if record.handle is not None:
                    self._spawner.kill_all(record.handle)
            time.sleep(0.05)
        return _answer("killing", None, None)

    # -- stop (E03-T06) ------------------------------------------------------

    def _stop_run(
        self,
        params: dict[str, object],
        request_id: str,
        token: CancelToken,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        """Kill every non-final process bound to one run (E03-T06).

        The ownership key is the pair (envelope principal_id,
        params run_id); the principal never comes from params.
        An unknown run is a successful empty answer, never an
        error. The whole STOP runs against ONE shared deadline
        (``min(token.remaining_s, 4.0)`` s), fixed BEFORE any
        signalling: signalling and the wait share it, so a slow
        systemctl cannot stretch STOP past the deadline — a
        signal (or kill) whose budget is gone is skipped and the
        reaper completes that tree. ``all_final`` is true only
        when every listed process reached a final state within
        the window — a tree that is still dying (or a spawn
        still in flight) reports ``all_final: false`` and the
        reaper completes it.
        """
        _check_params(params, frozenset({"run_id"}))
        _check_binding(principal_id, None)
        target_run_id = _req_str(params, "run_id")
        if not 1 <= len(target_run_id) <= 256:
            raise ProcessError(
                "invalid_params", "params.run_id must be a string of 1..256 characters"
            )
        pair = (principal_id, target_run_id)
        # E03-T08 (owner condition 1, verified against the code):
        # ``_stop_run`` sets ``_stopped_runs[pair] = now``
        # UNCONDITIONALLY — a repeat STOP re-sets the mark timestamp
        # and so extends the mark's TTL/admission effect. The mark
        # update IS a mutation, so every successful STOP emits a
        # fresh effect (``mark_set: true``); the daemon, not the
        # recorder, decides whether a new mutation happened.
        # E03-T08 (owner condition 5): the mark assignment is the
        # mutation, so the result + effect are frame-budgeted BEFORE
        # it. The projection uses STOP_PROJECTION_ENTRIES placeholder
        # entries, which bounds any target count the mark-time
        # re-read can produce (at most ~36 records can exist in the
        # daemon: 32 finished + 4 live).
        projection = self._effect_emitter.build_projection(
            kind="process.stop",
            principal_id=principal_id,
            run_id=target_run_id,
            fields={
                "run_id": target_run_id,
                "processes": [stop_process_placeholder() for _ in range(STOP_PROJECTION_ENTRIES)],
                "all_final": False,
                "mark_set": True,
            },
        )
        projected_result: dict[str, object] = {
            "run_id": target_run_id,
            "processes": [stop_process_placeholder() for _ in range(STOP_PROJECTION_ENTRIES)],
            "all_final": False,
            "effect": projection,
        }
        if not self._effect_emitter.frame_fits(projected_result, request_id):
            raise ProcessError(EFFECT_BUILD_FAILED, "effect does not fit the response frame budget")
        now = time.monotonic()
        with self._lock:
            self._stopped_runs[pair] = now
            targets = [
                record
                for record in self._records.values()
                if (record.principal_id, record.run_id) == pair
            ]
        # Natural completion first, as _kill() does: a process that
        # already finished on its own (cgroup empty, leader reaped)
        # keeps its published exit code instead of being rewritten
        # to "killed" because the reaper had not finalized it yet.
        # Runs outside the lock: _try_finalize finalizes through it.
        for record in targets:
            if not record.final:
                self._try_finalize(record)
        # Mark every target before signalling any of them, and signal
        # each independently: a SIGTERM failure (systemctl timeout or
        # OSError) for one process must not leave the rest of the run
        # running. The signal is best-effort — the grace deadline +
        # kill_all in the wait loop (and the reaper) still destroy
        # the tree.
        to_signal: list[SpawnHandle] = []
        with self._lock:
            for record in targets:
                if record.final:
                    continue
                # A repeat stop observes the termination already in
                # progress: it must not move the original deadline,
                # must not re-signal the tree, and must not
                # overwrite an existing kill reason (e.g. "timeout")
                # with "killed" — the same discipline as _kill.
                if not record.kill_requested:
                    record.kill_requested = True
                    record.kill_reason = "killed"
                    record.state = "killing"
                    record.kill_deadline = now + KILL_GRACE_S
                    if record.handle is not None:
                        to_signal.append(record.handle)
        # The shared STOP deadline is fixed BEFORE signalling:
        # each signal_tree() may wait up to 5 s on a slow
        # systemd, so computing it after the signals would let
        # the signalling alone exhaust the daemon's request
        # deadline and turn STOP into a request_timeout instead
        # of the specified all_final:false answer.
        deadline = time.monotonic() + min(token.remaining_s, STOP_WAIT_MAX_S)
        for handle in to_signal:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # No budget left for this signal: the reaper
                # still destroys the tree (grace deadline +
                # kill_all), and the answer reports all_final
                # false for it.
                continue
            try:
                self._spawner.signal_tree(handle, signal.SIGTERM, timeout_s=remaining)
            except (OSError, subprocess.TimeoutExpired):
                continue
        while time.monotonic() < deadline:
            if all(record.final for record in targets):
                break
            for record in targets:
                if record.final:
                    continue
                self._try_finalize(record)
                if (
                    not record.final
                    and record.kill_requested
                    and record.handle is not None
                    and time.monotonic() >= record.kill_deadline
                ):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # Out of STOP budget: stop waiting and
                        # let the reaper finish the destroy.
                        break
                    try:
                        self._spawner.kill_all(record.handle, timeout_s=remaining)
                    except (OSError, subprocess.TimeoutExpired):
                        # The reaper retries the destroy; a failing
                        # kill must not abort the STOP answer.
                        pass
            if all(record.final for record in targets):
                break
            time.sleep(0.05)
        # E03-T08: the STOP succeeded (the mark was set and the
        # targets signalled): assign the effect its real
        # (boot_id, seq) with the real process entries now.
        processes = [
            {
                "process_id": record.process_id,
                "state": record.state,
                "exit_code": record.exit_code,
                "signal": record.exit_signal,
            }
            for record in targets
        ]
        stop_result: dict[str, object] = {
            "run_id": target_run_id,
            "processes": processes,
            "all_final": all(record.final for record in targets),
        }
        stop_result["effect"] = self._effect_emitter.finalize(
            projection,
            fields={
                "run_id": target_run_id,
                "processes": processes,
                "all_final": stop_result["all_final"],
                "mark_set": True,
            },
        )
        return stop_result

    # -- reaper / finalization ------------------------------------------------

    def _try_finalize(self, record: _Record) -> bool:
        """Finalize once the cgroup is empty AND the leader is reaped."""
        if record.handle is None:
            return False
        if not self._spawner.cgroup_empty(record.handle):
            return False
        if not record.handle.leader_done():
            return False
        status = _read_status_file(record.handle.run_dir)
        status_state = str(status.get("state")) if status is not None else ""
        # The status fields, read once: mypy cannot narrow ``status``
        # from ``status_state`` alone.
        status_exit_code: object = status.get("exit_code") if status is not None else None
        status_signal: object = status.get("signal") if status is not None else None
        status_finished_at: object = status.get("finished_at") if status is not None else None
        if record.spawn_destroyed:
            # The tree was destroyed at the spawn adoption point
            # because the run was stopped mid-spawn (E03-T06): the
            # process usually never ran, so its final state is
            # spawn_failed — but only once the cgroup is confirmed
            # empty, which the gates above just proved. This branch
            # precedes the kill_requested branch on purpose: the
            # record was marked "killing" by the stop scan, and
            # without it the never-started process would finalize
            # "killed". The exception is a command that had already
            # executed before the handle arrived (the snapshot, or
            # a final "exited" the helper published between the
            # snapshot and the destroy): its published result is
            # preserved instead of being rewritten to spawn_failed.
            # The destroy itself cannot fabricate an "exited" —
            # the helper publishes it only on genuine completion.
            if record.spawn_result is not None:
                result_state, result_code, result_signal, result_finished_at = record.spawn_result
                self._finalize(
                    record,
                    result_state,
                    exit_code=result_code,
                    exit_signal=result_signal,
                    finished_at=result_finished_at,
                )
                return True
            if status_state == "exited":
                self._finalize(
                    record,
                    "exited",
                    exit_code=status_exit_code,
                    exit_signal=status_signal,
                    finished_at=status_finished_at,
                )
                return True
            self._finalize(record, "spawn_failed")
            return True
        if status_state == "spawn_failed":
            # The process never started; a kill requested afterwards
            # does not rewrite that.
            self._finalize(
                record,
                "spawn_failed",
                exit_code=status_exit_code,
                exit_signal=status_signal,
                finished_at=status_finished_at,
            )
            return True
        if record.kill_requested:
            # A kill was requested (user kill or timeout): the tree's
            # end is the kill, even if the leader already exited on
            # its own and the helper recorded "exited" for it. The
            # status file describes the leader, not the tree.
            state = "timeout" if record.kill_reason == "timeout" else "killed"
            exit_code: object = None
            exit_signal: object = None
            finished_at: object = None
            if status_state in ("killed", "timeout"):
                # The helper recorded the kill itself (exact signal).
                exit_code = status_exit_code
                exit_signal = status_signal
                finished_at = status_finished_at
            self._finalize(
                record,
                state,
                exit_code=exit_code,
                exit_signal=exit_signal,
                finished_at=finished_at,
            )
            return True
        if status is not None and status_state in ("exited", "killed", "timeout"):
            self._finalize(
                record,
                status_state,
                exit_code=status_exit_code,
                exit_signal=status_signal,
                finished_at=status_finished_at,
            )
            return True
        # No status file (helper died with the tree): the tree ended
        # on its own.
        self._finalize(record, "exited")
        return True

    def _finalize(
        self,
        record: _Record,
        state: str,
        *,
        exit_code: object = None,
        exit_signal: object = None,
        finished_at: object = None,
    ) -> None:
        with self._lock:
            if record.final:
                return
            record.state = state
            record.finished_at = (
                str(finished_at) if isinstance(finished_at, str) else _iso(time.time())
            )
            record.exit_code = exit_code if isinstance(exit_code, int) else None
            record.exit_signal = exit_signal if isinstance(exit_signal, int) else None
            record.final = True
            record.settled.set()
            # E03-T08 (Codex round 2, finding 1): a spawn_failed
            # answer must never carry a run effect — the process
            # never ran (or was destroyed before promotion, which
            # the round-1 decision reads as unknown, not a
            # confirmed mutation). Drop any effect an earlier retry
            # finalized while the launch was still in flight, and
            # the projection with it. Other final states keep the
            # effect: the process did run. The same discipline
            # applies to an unfinalized kill projection (Codex
            # round 6, finding 2): a kill effect for a process
            # that never ran would be a confirmed mutation that
            # never happened.
            if state == "spawn_failed":
                record.effect = None
                record.effect_projection = None
                record.kill_projection = None

    def _reap_loop(self) -> None:
        while not self._stop.wait(0.2):
            try:
                self._reap_once()
            except BaseException:
                continue

    def _reap_once(self) -> None:
        now = time.monotonic()
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.final:
                continue
            if record.state == "running" and now >= record.started_mono + record.timeout_s:
                # The tree may already have finished on its own
                # (leader and all descendants gone) without a poll
                # having finalized the record: a timeout kill then is
                # a no-op and the true final state is the natural one,
                # not "timeout" — check natural completion first, as
                # _kill() does, or a successful command turns into an
                # error depending on reaper scheduling.
                self._refresh(record)
                if self._try_finalize(record):
                    continue
                record.kill_requested = True
                record.kill_reason = "timeout"
                record.state = "killing"
                record.kill_deadline = now + KILL_GRACE_S
                if record.handle is not None:
                    self._spawner.signal_tree(record.handle, signal.SIGTERM)
            if record.state == "killing" and now >= record.kill_deadline:
                if record.handle is not None:
                    self._spawner.kill_all(record.handle)
            if record.state in ("running", "killing"):
                self._refresh(record)
                self._try_finalize(record)
            elif record.state == "launching":
                if now - record.started_mono > SPAWN_START_TIMEOUT_S:
                    if record.handle is not None:
                        self._spawner.kill_all(record.handle)
                    self._finalize(record, "spawn_failed")
        self._sweep_finished()

    def _prune_stopped_marks_locked(self, now_mono: float) -> None:
        """Drop stopped-run marks that can no longer refuse a commit
        (E03-T06). Caller must hold ``self._lock``.

        A mark is kept while a record bound to the run still
        exists — a late retry of that run must still see
        ``run_stopped``. Once the last record is gone, the mark
        lives only until STOPPED_RUN_MARK_TTL_S so a STOP of an
        unknown run (no records at all) does not pin the run
        forever: after expiry the run may be started again.
        """
        live_pairs = {(record.principal_id, record.run_id) for record in self._records.values()}
        for pair, stopped_mono in list(self._stopped_runs.items()):
            if pair not in live_pairs and now_mono - stopped_mono > STOPPED_RUN_MARK_TTL_S:
                del self._stopped_runs[pair]

    def _sweep_finished(self) -> None:
        now = time.time()
        with self._lock:
            finished = [r for r in self._records.values() if r.final]
            finished.sort(
                key=lambda r: datetime.fromisoformat(r.finished_at).timestamp()
                if r.finished_at
                else 0.0
            )
            evict: list[_Record] = []
            for record in finished:
                finished_at = (
                    datetime.fromisoformat(record.finished_at).timestamp()
                    if record.finished_at
                    else 0.0
                )
                if now - finished_at > FINISHED_RETENTION_S:
                    evict.append(record)
            over = len(finished) - len(evict) - MAX_FINISHED_RECORDS
            if over > 0:
                evict.extend(finished[:over])
            for record in evict:
                self._records.pop(record.process_id, None)
                self._by_request_id = {
                    k: v for k, v in self._by_request_id.items() if v != record.process_id
                }
            # Evicted request_ids must not keep their binding
            # alive: rebuild the binding map from the surviving
            # request ids (cheap — at most MAX_FINISHED_RECORDS
            # entries plus live ones).
            self._by_request_id_binding = {
                k: v for k, v in self._by_request_id_binding.items() if k in self._by_request_id
            }
            self._prune_stopped_marks_locked(time.monotonic())
        for record in evict:
            run_dir = record.run_dir
            if run_dir is None and record.handle is not None:
                run_dir = record.handle.run_dir
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)

    def reap_orphans(self) -> None:
        """Kill leftover bridge scopes and stale run dirs (E03-T06).

        Called once by the daemon during startup, after exclusive
        ownership of the socket namespace is established and
        before the daemon serves requests. No adoption: any scope
        or run dir found here belongs to a dead daemon, and the
        process that owned it is gone. Only scopes carrying this
        daemon's namespace are destroyed: the systemd user manager
        is shared per UID, so a scope of another (live) daemon
        with a different socket directory must never be touched.
        Fail-closed: any error raises BridgeConfigError and the
        daemon refuses to start.
        """
        scopes = self._spawner.list_bridge_scopes()
        own_prefix = f"slavikai-bridge-{self._scope_namespace}-"
        for unit_name in scopes:
            if not unit_name.startswith(own_prefix):
                # Another daemon's namespace (same user manager, a
                # different socket directory): never ours to destroy.
                continue
            process_id = _process_id_from_unit(unit_name)
            with self._lock:
                if process_id is not None and process_id in self._records:
                    # A live record appeared for this scope between
                    # the list and the destroy: a concurrent start
                    # raced us. Leave it to its owner — destroying
                    # it would kill a process of a live daemon.
                    continue
            if not self._spawner.destroy_scope_checked(unit_name):
                raise BridgeConfigError(
                    f"orphan sweep failed: scope {unit_name} still present after destroy"
                )
        try:
            run_dirs = list(self._run_base.glob("*"))
        except OSError as exc:
            raise BridgeConfigError(f"orphan sweep cannot list run dirs: {exc}") from exc
        with self._lock:
            live_dirs = {
                record.handle.run_dir
                for record in self._records.values()
                if record.handle is not None
            }
        for run_dir in run_dirs:
            if run_dir in live_dirs:
                continue
            try:
                shutil.rmtree(run_dir)
            except OSError as exc:
                raise BridgeConfigError(
                    f"orphan sweep cannot remove stale run dir {run_dir}: {exc}"
                ) from exc

    def shutdown(self) -> None:
        self._stop.set()
        self._reaper.join(timeout=2.0)
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if not record.final and record.handle is not None:
                self._spawner.kill_all(record.handle)
        # A final best-effort finalize so no record is left "running".
        for record in records:
            if not record.final:
                self._try_finalize(record)


def build_process_service(
    registry: RootsRegistry,
    fs_service: HostFsService,
    *,
    runtime_dir: Path,
    spawner: SystemdRunSpawner | None = None,
    effect_emitter: EffectEmitter | None = None,
) -> HostProcessService:
    """Build the process service from the same registry as the FS roots."""
    profiles = load_profiles(registry)
    return HostProcessService(
        fs_service,
        registry,
        runtime_dir=runtime_dir,
        profiles=profiles,
        spawner=spawner,
        effect_emitter=effect_emitter,
    )


__all__ = [
    "ArgSlot",
    "Grant",
    "HostProcessService",
    "ProcessError",
    "ProcessProfile",
    "ProbeResult",
    "SpawnError",
    "SpawnHandle",
    "SystemdRunSpawner",
    "build_process_service",
    "load_profiles",
    "run_enforcement_probe",
    "scope_namespace_for",
    "scope_unit_name",
]
