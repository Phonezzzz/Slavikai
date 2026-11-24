"""Approved-profile process API for the Owner Host Bridge (E03-T05).

Capabilities ``host.process.run`` / ``host.process.poll`` /
``host.process.kill`` run owner-approved command profiles under the
enforcement stack of ADR-0013 §10a as fixed for T05 by the owner
(2026-11-24): Landlock + ``NEWUSER|NEWNET`` + ``NO_NEW_PRIVS`` +
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
from host_bridge.spawn_helper import landlock_abi

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
PROBE_TIMEOUT_S = 20.0
PROBE_TARGET_TIMEOUT_S = 10
GRANT_WALK_MAX_ENTRIES = 20_000
GRANT_WALK_MAX_GRANTS = 5_000
GRANT_WALK_MAX_DEPTH = 128
POLL_MAX_BYTES = 32 * 1024
POLL_DEFAULT_MAX_BYTES = 8 * 1024

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
          "signal": "unknown", "metadata": "unknown", "prlimit": "unknown"}
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
result["uid"] = os.getuid()
print("PROBE:" + json.dumps(result), flush=True)
time.sleep(2)
"""


class ProcessError(BridgeProtocolError):
    """Typed process error (``code`` is part of the wire schema)."""


@dataclass(frozen=True)
class Grant:
    path: Path
    access: str  # "r" | "rw" | "rx"
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
    A trailing incomplete sequence is held back (not consumed) until
    the stream is finished, so it can complete in the next chunk.
    """
    data = raw
    while data:
        try:
            return data.decode("utf-8"), len(data)
        except UnicodeDecodeError as exc:
            if not at_eof and exc.reason == "unexpected end of data" and exc.end == len(data):
                data = data[: exc.start]
                continue
            return data.decode("utf-8", errors="replace"), len(data)
    return "", 0


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

    A directory whose whole scanned subtree is clean (no protected
    entries, no symlinks, no nested mounts) is granted as one rule;
    a dirty directory is expanded into per-entry grants for its clean
    children, so a secret nested inside an allowed tree is never
    covered by a parent grant. Symlinks are never granted or followed
    (Landlock checks the resolved target, so a granted symlink would
    hand out whatever it points to). Write-exclusion is per path:
    a subtree the FS API refuses to write (config ``write_excluded``
    or a SlavikAI checkout) is granted read-only, never ``rw``, and
    a parent is never granted as one ``rw`` rule over a weaker child
    (ADR-0013 §9).
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
        local: list[Grant] = []
        dirty = False
        try:
            with os.scandir(directory) as it:
                children = list(it)
        except OSError as exc:
            raise ProcessError("spawn_failed", f"cwd is not readable: {exc}") from exc
        for entry in children:
            entries_seen += 1
            if entries_seen > GRANT_WALK_MAX_ENTRIES:
                raise ProcessError("spawn_failed", "grant walk too large")
            token.check()
            try:
                entry_st = entry.stat(follow_symlinks=False)
            except OSError:
                dirty = True
                continue
            if entry.is_symlink():
                continue  # never granted, never followed
            entry_path = directory / entry.name
            if entry_path in mount_points:
                # Bind mount (even same-device, so st_dev cannot see
                # it): never granted, never descended — the FS API
                # refuses nested mounts the same way, including the
                # final file component.
                dirty = True
                continue
            if entry_st.st_dev != base_dev:
                dirty = True  # nested mount: not granted, not descended
                continue
            classification = fs_service._classify(entry_path, mutation=False)
            if classification != "normal":
                dirty = True
                continue
            # Write-exclusion is per path: a writable parent may hold
            # an excluded child, whose subtree is read-only.
            entry_access = access
            if access == "rw" and fs_service.is_write_excluded(root, entry_path):
                entry_access = "r"
            if entry.is_dir(follow_symlinks=False):
                sub, sub_dirty = walk(entry_path, depth + 1, entry_access)
                if sub_dirty:
                    local.extend(sub)
                    dirty = True
                elif entry_access != access:
                    # Clean but weaker than the parent: one grant for
                    # the parent at its access would cover this subtree
                    # with write — expand to per-entry grants instead.
                    local.append(
                        Grant(
                            entry_path,
                            entry_access,
                            dev=entry_st.st_dev,
                            ino=entry_st.st_ino,
                        )
                    )
                    dirty = True
                else:
                    local.append(
                        Grant(
                            entry_path,
                            entry_access,
                            dev=entry_st.st_dev,
                            ino=entry_st.st_ino,
                        )
                    )
            else:
                local.append(
                    Grant(
                        entry_path,
                        entry_access,
                        dev=entry_st.st_dev,
                        ino=entry_st.st_ino,
                    )
                )
                if entry_access != access:
                    dirty = True
        if not dirty:
            return [Grant(directory, access, dev=st.st_dev, ino=st.st_ino)], False
        return local, True

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
            # systemd-run exited (or the unit never appeared): the spawn
            # cannot be placed in a cgroup, so it is not a spawn at all.
            try:
                popen.wait(timeout=2)
            except subprocess.TimeoutExpired:
                popen.kill()
            raise SpawnError("systemd-run scope did not appear")
        return SpawnHandle(popen=popen, unit_name=unit_name, cgroup_path=cgroup, run_dir=run_dir)

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

    def signal_tree(self, handle: SpawnHandle, sig: int) -> None:
        if handle.popen is None:
            return
        name = signal.Signals(sig).name
        subprocess.run(
            ["systemctl", "--user", "kill", f"--signal={name}", handle.unit_name],
            capture_output=True,
            timeout=5,
        )

    def kill_all(self, handle: SpawnHandle) -> None:
        """SIGKILL the whole cgroup subtree (recursive, in-kernel)."""
        if handle.cgroup_path is not None:
            try:
                (handle.cgroup_path / "cgroup.kill").write_text("1", encoding="utf-8")
                return
            except OSError:
                pass
        self.signal_tree(handle, signal.SIGKILL)

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
    seccomp ban), CAN read an allowed file, runs with the mapped uid,
    cannot signal processes outside its Landlock domain, and is
    placed in its own scope cgroup.
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
        try:
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
            # Grants cover only the allowed file individually — never
            # the whole probe dir, which would cover the protected
            # .env too.
            ok_path = probe_dir / "ok.txt"
            ok_st = ok_path.lstat()
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
                ],
                "env": dict(_BASE_ENV),
                "cwd": str(probe_dir),
                "timeout_s": PROBE_TARGET_TIMEOUT_S,
                "run_dir": str(probe_dir),
                "grants": _grants_wire(_system_grants() + _exec_grants(Path("/usr/bin/python3")))
                + _grants_wire([Grant(ok_path, "r", dev=ok_st.st_dev, ino=ok_st.st_ino)]),
                "outer_uid": os.getuid(),
                "outer_gid": os.getgid(),
            }
            spec_path = probe_dir / "spec.json"
            spec_path.write_text(json.dumps(probe_spec), encoding="utf-8")
            spec_path.chmod(0o600)
            try:
                handle = spawner.spawn(
                    spec_path,
                    unit_name="slavikai-bridge-probe.scope",
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
                    if "slavikai-bridge-probe.scope" in cgroup:
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
    settled: threading.Event = field(default_factory=threading.Event)
    final: bool = False


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
    ) -> None:
        self.fs_service = fs_service
        self.registry = registry
        self.runtime_dir = runtime_dir
        self._run_base = runtime_dir / "host-bridge-proc"
        self._run_base.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._profiles = {p.id: p for p in profiles}
        self._spawner = spawner or SystemdRunSpawner()
        self._records: dict[str, _Record] = {}
        self._by_request_id: dict[str, str] = {}
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

    # -- dispatch ------------------------------------------------------------

    def handle(
        self, capability: str, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        try:
            if capability == "host.process.run":
                return self._run(params, request_id, token)
            if capability == "host.process.poll":
                return self._poll(params, request_id, token)
            if capability == "host.process.kill":
                return self._kill(params, request_id, token)
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
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"profile_id", "args", "timeout_s"}))
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
        existing: _Record | None = None
        record: _Record | None = None
        with self._lock:
            token.check()
            existing_id = self._by_request_id.get(request_id)
            if existing_id is not None:
                existing = self._records.get(existing_id)
            if existing is None:
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
                )
                self._records[process_id] = record
                self._by_request_id[request_id] = process_id
        if existing is not None:
            # Wait outside the lock: _finalize needs it to settle the
            # original run, and the reaper needs it for every other
            # process's timeout enforcement.
            return self._run_result(existing, wait_settled=True, token=token)
        assert record is not None
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
            raise
        except BaseException as exc:
            self._finalize(record, "spawn_failed")
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
        return {
            "process_id": record.process_id,
            "profile_id": record.profile_id,
            "state": record.state,
            "started_at": record.started_at,
        }

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
                unit_name=f"slavikai-bridge-{record.process_id}.scope",
                run_dir=run_dir,
            )
        except SpawnError as exc:
            raise ProcessError("spawn_failed", str(exc)) from exc
        # The spawner returned: a tree may now exist, so every later
        # failure keeps the reaper/cleanup path instead of an
        # immediate finalize.
        record.spawn_started = True
        with self._lock:
            if record.final or self._stop.is_set():
                # The reaper gave up on this launch while spawn() was
                # still in flight, or the service is stopping: the
                # tree exists now and nobody owns it. Destroy it
                # before returning, or it would run outside the
                # active-process accounting and without daemon
                # timeout enforcement — or after shutdown.
                late = True
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
                self._finalize(record, "spawn_failed")
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
        # newer than the status is possible, and after a 256 KiB tail
        # rotation a stale retained_from misaligns every offset. In a
        # consistent snapshot total - retained_from == len(tail) for
        # each stream; when it does not hold, re-read status + tails
        # (the helper writes both on every flush, so the fresh
        # counters cover the tail). The growth case is still clamped
        # below, so a stale status can never publish an offset beyond
        # the bytes actually read.
        for _ in range(5):
            stdout_tail = self._tail_bytes(run_dir, "stdout")
            stderr_tail = self._tail_bytes(run_dir, "stderr")
            if record.stdout_total - record.stdout_retained_from == len(
                stdout_tail
            ) and record.stderr_total - record.stderr_retained_from == len(stderr_tail):
                break
            self._refresh(record)
        stdout_total = max(record.stdout_total, record.stdout_retained_from + len(stdout_tail))
        stderr_total = max(record.stderr_total, record.stderr_retained_from + len(stderr_tail))
        if stdout_offset > stdout_total or stderr_offset > stderr_total:
            raise ProcessError("offset_out_of_range", "offset is beyond the stream end")
        if record.started_at:
            start_ts = datetime.fromisoformat(record.started_at).timestamp()
            end_ts = (
                datetime.fromisoformat(record.finished_at).timestamp()
                if record.finished_at
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
            eof = record.final and start + len(raw) >= total
            return _decode_poll_chunk(raw, at_eof=eof)

        def chunk(
            offset: int, total: int, retained_from: int, tail: bytes
        ) -> tuple[str, bytes, int, int, bool]:
            start = max(offset, retained_from)
            truncated = offset < retained_from
            raw = tail[start - retained_from : start - retained_from + max_bytes]
            text, consumed = decode_slice(raw, start, total)
            if raw and consumed == 0:
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
            stdout_offset, stdout_total, record.stdout_retained_from, stdout_tail
        )
        stderr_text, stderr_raw, stderr_start, stderr_next, stderr_truncated = chunk(
            stderr_offset, stderr_total, record.stderr_retained_from, stderr_tail
        )
        result: dict[str, object] = {
            "process_id": record.process_id,
            "state": record.state,
            "exit_code": record.exit_code,
            "signal": record.exit_signal,
            "started_at": record.started_at,
            "finished_at": record.finished_at,
            "elapsed_ms": elapsed_ms,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "stdout_next_offset": stdout_next,
            "stderr_next_offset": stderr_next,
            "stdout_total_bytes": stdout_total,
            "stderr_total_bytes": stderr_total,
            "stdout_retained_from": record.stdout_retained_from,
            "stderr_retained_from": record.stderr_retained_from,
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
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"process_id"}))
        process_id = _req_str(params, "process_id")
        record = self._records.get(process_id)
        if record is None:
            raise ProcessError("process_not_found", f"unknown process {process_id!r}")
        if record.final:
            return {
                "process_id": record.process_id,
                "state": record.state,
                "exit_code": record.exit_code,
                "signal": record.exit_signal,
            }
        # The tree may already have finished on its own (leader and
        # all descendants gone) without a poll having finalized the
        # record: a kill then is a no-op and the true final state is
        # the natural one, not "killed".
        self._try_finalize(record)
        if record.final:
            return {
                "process_id": record.process_id,
                "state": record.state,
                "exit_code": record.exit_code,
                "signal": record.exit_signal,
            }
        token.check()
        if record.state == "launching":
            # The launch is still committing: wait briefly for it to
            # settle (or the request deadline), then kill what exists.
            remaining = token.remaining_s
            wait_s = 2.0 if remaining == float("inf") else max(0.0, min(remaining, 2.0))
            record.settled.wait(timeout=wait_s)
            if record.final:
                return {
                    "process_id": record.process_id,
                    "state": record.state,
                    "exit_code": record.exit_code,
                    "signal": record.exit_signal,
                }
        # A repeat kill observes the termination already in progress:
        # it must not move the original deadline, must not re-signal
        # the tree, and must not overwrite a "timeout" reason with
        # "killed" — otherwise kills more frequent than the grace
        # period postpone the cgroup SIGKILL indefinitely.
        if not record.kill_requested:
            record.kill_requested = True
            record.kill_reason = "killed"
            record.state = "killing"
            record.kill_deadline = time.monotonic() + KILL_GRACE_S
            if record.handle is not None:
                self._spawner.signal_tree(record.handle, signal.SIGTERM)
        deadline = time.monotonic() + min(token.remaining_s, 4.0)
        while time.monotonic() < deadline:
            if self._try_finalize(record):
                return {
                    "process_id": record.process_id,
                    "state": record.state,
                    "exit_code": record.exit_code,
                    "signal": record.exit_signal,
                }
            if time.monotonic() >= record.kill_deadline:
                if record.handle is not None:
                    self._spawner.kill_all(record.handle)
            time.sleep(0.05)
        return {
            "process_id": record.process_id,
            "state": "killing",
            "exit_code": None,
            "signal": None,
        }

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
        for record in evict:
            run_dir = record.run_dir
            if run_dir is None and record.handle is not None:
                run_dir = record.handle.run_dir
            if run_dir is not None:
                shutil.rmtree(run_dir, ignore_errors=True)

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
) -> HostProcessService:
    """Build the process service from the same registry as the FS roots."""
    profiles = load_profiles(registry)
    return HostProcessService(
        fs_service, registry, runtime_dir=runtime_dir, profiles=profiles, spawner=spawner
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
]
