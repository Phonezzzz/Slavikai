"""Spawn trampoline + supervisor for ``host.process.run`` (E03-T05).

Launched by the daemon (via the systemd-run scope spawner) as::

    python -m host_bridge.spawn_helper <spec-path>

The helper is a plain stdlib program: it must not import any
``host_bridge`` module (it runs in a fresh interpreter whose only
guaranteed import path is its own file location) and it must not
depend on the daemon being alive after the spawn.

Flow (all enforcement happens in the forked child, never via
``preexec_fn`` from the daemon's threads — the daemon is
multithreaded and ``preexec_fn`` is documented-unsafe there):

1. Read the JSON spec (binary, argv, env, cwd, Landlock grants,
   timeout) and unlink it immediately.
2. Fork the target child. The child unshares ``NEWUSER|NEWNET|NEWIPC``,
   signals readiness on a pipe, and waits for the parent to write
   its ``uid_map``/``gid_map`` (a process cannot write its own map
   on this host: AppArmor's ``unprivileged_userns`` profile denies
   every capability, so the parent — still unconfined in the initial
   namespace — writes the maps, which is the working pattern).
3. The child then applies, in order: ``NO_NEW_PRIVS``, Landlock FS
   rules (only the explicitly granted parts of the tree, plus the
   specific approved binary with ``EXECUTE``), a seccomp socket ban
   (``socket``/``socketpair``/``connect``/``bind``/``listen``/
   ``accept``/``accept4`` and the rest of the socket family, plus
   ``io_uring`` and the mount/ptrace families) and a seccomp metadata
   ban (the ``chmod``/``chown``/``setxattr``/``utime`` families, which
   Landlock does not restrict), ``PDEATHSIG``,
   its own process group, and finally ``execve`` of
   ``/proc/self/fd/<fd>`` on the already open binary fd — identity is
   bound to the open file, so a binary swapped after the check cannot
   be executed. (``execveat`` with ``AT_EMPTY_PATH`` is not used: it
   breaks uutils multicall binaries on this host, which see the fd
   number as the utility name.)
4. The parent (this helper) supervises: it reads the child's
   stdout/stderr through pipes into bounded tail buffers (256 KiB
   per stream, with total counters and ``retained_from``), enforces
   the runtime timeout itself (the daemon reaper is the primary
   enforcer; the helper is the backstop if the daemon dies), and
   writes ``status.json`` (atomic tmp+rename) with the exact exit
   code/signal from ``waitpid``. On SIGTERM it kills the child tree
   (its process group) and records the final state. The scope
   cgroup kill is the daemon's job, not this helper's: the
   supervisor runs inside the scope cgroup, so writing
   ``cgroup.kill`` here would SIGKILL itself before it can report.

The daemon learns the truth from ``status.json`` + the tail files in
the run directory; the helper's own exit code is only a liveness
signal (``systemd-run`` propagates exit codes but not signals).
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import select
import signal
import struct
import sys
import time
from pathlib import Path

# The helper is launched as ``python -m host_bridge.spawn_helper`` from
# any working directory; make the repository importable for the (few)
# stdlib-only needs below. The helper imports no host_bridge module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- syscalls / constants (x86_64; Landlock ABI 8 verified on the host) ---

_SYS_landlock_create_ruleset = 444
_SYS_landlock_add_rule = 445
_SYS_landlock_restrict_self = 446
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_CREATE_RULESET_VERSION = 1
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWNET = 0x40000000
_CLONE_NEWIPC = 0x08000000
_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_PDEATHSIG = 1
_PR_SET_DUMPABLE = 4

# Landlock FS access rights (ABI 8).
_LL_EXECUTE = 1
_LL_WRITE_FILE = 2
_LL_READ_FILE = 4
_LL_READ_DIR = 8
_LL_REMOVE_DIR = 16
_LL_REMOVE_FILE = 32
_LL_MAKE_DIR = 128
_LL_MAKE_REG = 256
# Node-creation rights that are never granted (no profile needs to
# create a device node, a named pipe, or a UNIX socket object):
# they must still be listed in handled_access_fs, or Landlock
# allows them by default (unlisted handled rights are allowed,
# not denied — /usr/include/linux/landlock.h).
_LL_MAKE_CHAR = 64
_LL_MAKE_SOCK = 512
_LL_MAKE_FIFO = 1024
_LL_MAKE_BLOCK = 2048
_LL_MAKE_SYM = 4096
_LL_REFER = 8192
_LL_TRUNCATE = 16384

# Landlock scoping (ABI 6+; the host reports ABI 8): a restricted
# process may only signal processes in its own Landlock domain, so
# the daemon, the server and other owner processes are unreachable
# by kill() from a profile (ADR-0013 §10).
_LL_SCOPE_SIGNAL = 1 << 1

_ACCESS_BITS: dict[str, int] = {
    # Read-only: read files and list directories.
    "r": _LL_READ_FILE | _LL_READ_DIR,
    # Read + write: create/remove/rename entries, write and truncate.
    # REFER is required for rename/link between different directories
    # of one tree: without it the kernel refuses the move with EXDEV
    # even when both directories are fully writable.
    "rw": (
        _LL_READ_FILE
        | _LL_WRITE_FILE
        | _LL_READ_DIR
        | _LL_REMOVE_DIR
        | _LL_REMOVE_FILE
        | _LL_MAKE_DIR
        | _LL_MAKE_REG
        | _LL_MAKE_SYM
        | _LL_REFER
        | _LL_TRUNCATE
    ),
    # Read + execute (system binary/library trees only).
    "rx": _LL_READ_FILE | _LL_READ_DIR | _LL_EXECUTE,
    # Directory operations only: list/create/remove/rename entries,
    # but NO file-content rights (READ_FILE/WRITE_FILE/TRUNCATE).
    # A path-beneath rule on a directory covers every file created
    # beneath it later, so a full "rw" rule on a directory would also
    # cover a secret file the owner drops in after the walk (a
    # long-lived profile would gain credential material _classify()
    # would deny). Directory rules therefore never carry file-content
    # bits; each allowed file gets its own rule instead, and the
    # directory keeps only the operations on its own entries (this
    # is what lets a profile listdir('.'), create entries and rename
    # between granted directories while a protected file nested
    # inside stays denied).
    "dir": (
        _LL_READ_DIR
        | _LL_REMOVE_DIR
        | _LL_REMOVE_FILE
        | _LL_MAKE_DIR
        | _LL_MAKE_REG
        | _LL_MAKE_SYM
        | _LL_REFER
    ),
    # Read-only directory operations: list + traversal. REFER lets a
    # rename/link name this directory as a party, but REFER alone
    # enables no mutation — a rename or link still needs
    # REMOVE_FILE/MAKE_* on the directories involved, which "rdir"
    # does not carry. Ancestors of protected subtrees get "rdir"
    # instead of "dir": Landlock rules union, so a mutating grant on
    # an ancestor would leak into the protected subtree.
    "rdir": _LL_READ_DIR | _LL_REFER,
}

# Access bits that apply to a single file (or device node). The
# kernel rejects a path-beneath rule whose mask contains
# directory-only rights (READ_DIR, REMOVE_*, MAKE_*) for a
# non-directory — landlock_add_rule fails with EINVAL.
_FILE_BITS = _LL_EXECUTE | _LL_WRITE_FILE | _LL_READ_FILE | _LL_TRUNCATE

# Seccomp actions (libseccomp).
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000

# Syscalls denied with EPERM for the target process: the whole socket
# family (no socket of any kind can be created), io_uring (an alternate
# path to socket operations), the mount family (no new mounts inside
# the user namespace), and ptrace/process_vm (no inspection of other
# processes). Landlock does NOT restrict the metadata families
# (chmod/chown/setxattr/utime — see /usr/include/linux/landlock.h),
# so they are denied here too: without this a profile could chmod 000
# the daemon config or source code it cannot even read. That ban
# covers the at-variants as well: setxattrat/removexattrat are new
# syscalls (kernel 6.10+) for the same metadata change, and the
# classic setxattr ban does not close them — a profile could
# otherwise set a POSIX ACL on a file it cannot read and lock the
# daemon/server out of its own config. prlimit64 is
# denied for the same reason Landlock's signal scope exists: with no
# PID namespace and matching kernel credentials, a profile that knows
# the daemon/server PID could zero its RLIMIT_NOFILE and deny it files
# and connections. setpriority is banned for the same reason: a
# profile could otherwise renice the daemon/server to 19 and leave
# it degraded after the profile ends (nice can only be lowered back
# by a privileged process). sched_setscheduler/sched_setattr are
# banned for the same reason: a profile could otherwise move the
# daemon/server to SCHED_IDLE and leave it starved under CPU
# contention after the profile ends. sched_setaffinity is banned for
# the same reason: a profile could otherwise pin the daemon/server
# to a busy CPU and leave it degraded after the profile ends.
# ioprio_set is banned for the same reason: a profile could
# otherwise move the daemon/server to IOPRIO_CLASS_IDLE and leave
# it starved under disk contention after the profile ends. The
# startup probe verifies the metadata, prlimit, setpriority,
# scheduler, affinity and ioprio bans live. Landlock
# + the empty network
# namespace are the
# other layers; the seccomp bans are mandatory (owner correction
# 2026-11-24).
_DENIED_SYSCALLS: tuple[str, ...] = (
    "socket",
    "socketpair",
    "connect",
    "bind",
    "listen",
    "accept",
    "accept4",
    "shutdown",
    "sendto",
    "recvfrom",
    "sendmsg",
    "recvmsg",
    "io_uring_setup",
    "io_uring_enter",
    "io_uring_register",
    "mount",
    "umount2",
    "pivot_root",
    "chroot",
    "ptrace",
    "process_vm_readv",
    "process_vm_writev",
    "prlimit64",
    "setpriority",
    "sched_setscheduler",
    "sched_setattr",
    "sched_setaffinity",
    "ioprio_set",
    "bpf",
    "open_by_handle_at",
    "name_to_handle_at",
    # Metadata families (not restricted by Landlock):
    "chmod",
    "fchmod",
    "fchmodat",
    "fchmodat2",
    "chown",
    "fchown",
    "lchown",
    "fchownat",
    "setxattr",
    "lsetxattr",
    "fsetxattr",
    "setxattrat",
    "removexattr",
    "lremovexattr",
    "fremovexattr",
    "removexattrat",
    "utime",
    "utimes",
    "futimesat",
    "utimensat",
)

TAIL_BYTES = 256 * 1024
_READ_CHUNK = 64 * 1024
_HANDSHAKE_TIMEOUT_S = 10.0
_SPAWN_FAILED_EXIT = 127

_libc = ctypes.CDLL("libc.so.6", use_errno=True)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint32),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


def landlock_abi() -> int | None:
    """Query the Landlock ABI version; ``None`` when unsupported."""
    result = _libc.syscall(_SYS_landlock_create_ruleset, None, 0, _LANDLOCK_CREATE_RULESET_VERSION)
    if result < 0:
        return None
    return int(result)


def _errno_name(code: int) -> str:
    return errno.errorcode.get(code, f"E{code}")


class _Tail:
    """Bounded tail buffer with absolute byte accounting."""

    def __init__(self, cap: int = TAIL_BYTES) -> None:
        self.cap = cap
        self.buf = bytearray()
        self.total = 0
        self.retained_from = 0

    def append(self, data: bytes) -> None:
        if not data:
            return
        self.total += len(data)
        self.buf += data
        if len(self.buf) > self.cap:
            dropped = len(self.buf) - self.cap
            del self.buf[:dropped]
            self.retained_from += dropped

    def bytes(self) -> bytes:
        return bytes(self.buf)


def _write_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via tmp+rename (no torn reads)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _iso_now() -> str:
    from datetime import datetime

    return datetime.now().astimezone().isoformat()


def _read_cgroup() -> str:
    try:
        return Path("/proc/self/cgroup").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _load_seccomp() -> object:
    """Build the socket-ban filter with libseccomp; None if unavailable."""
    try:
        lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError:
        return None
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = lib.seccomp_init(_SCMP_ACT_ALLOW)
    if not ctx:
        return None
    try:
        for name in _DENIED_SYSCALLS:
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                continue
            if lib.seccomp_rule_add(ctx, _SCMP_ACT_ERRNO | errno.EPERM, number, 0) != 0:
                lib.seccomp_release(ctx)
                return None
        if lib.seccomp_load(ctx) != 0:
            lib.seccomp_release(ctx)
            return None
    finally:
        lib.seccomp_release(ctx)
    return lib


def _open_grant_fd(path: Path) -> int:
    """Open ``path`` without following any symlink component.

    Each component is opened individually with O_NOFOLLOW, so a
    symlink swapped in between the daemon's grant computation and
    this open fails (ELOOP for a file link, ENOTDIR for a directory
    link under O_DIRECTORY) instead of handing out the link target
    — the check and the open are never split on a mutable path.
    Returns the fd for the final component (caller closes).
    """
    if not path.is_absolute():
        raise OSError(errno.EINVAL, "grant path is not absolute")
    parts = [part for part in path.parts if part != "/"]
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for index, part in enumerate(parts):
            last = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
            if not last:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=fd)
            try:
                os.close(fd)
            except OSError:
                pass
            fd = child
        return fd
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def _apply_landlock(grants: list[dict[str, object]]) -> tuple[int, str]:
    """Install the Landlock FS ruleset.

    Every grant is bound to a safely opened descriptor: the path is
    opened component-by-component with O_NOFOLLOW, the fd's identity
    (dev, ino) is checked against the (dev, ino) the daemon recorded
    when it verified the object, and only then is the rule added from
    the fd. A replaced object fails the whole spawn (EACCES) instead
    of silently running with less access; a vanished or symlinked
    target is skipped (no rule, no access).

    Returns (0, "") on success, else (errno, failing grant path).
    """
    attr = _RulesetAttr()
    attr.handled_access_fs = (
        _LL_EXECUTE
        | _LL_WRITE_FILE
        | _LL_READ_FILE
        | _LL_READ_DIR
        | _LL_REMOVE_DIR
        | _LL_REMOVE_FILE
        | _LL_MAKE_DIR
        | _LL_MAKE_REG
        | _LL_MAKE_SYM
        | _LL_REFER
        | _LL_TRUNCATE
        # Node creation is handled but never granted by any rule:
        # a profile cannot mkfifo/mknod anywhere, inside an
        # allowed root or outside it.
        | _LL_MAKE_CHAR
        | _LL_MAKE_SOCK
        | _LL_MAKE_FIFO
        | _LL_MAKE_BLOCK
    )
    attr.handled_access_net = 0
    # Signal scope: a restricted process may only signal processes in
    # its own domain. The supervisor and daemon are unrestricted, so
    # they can still signal the tree; the tree cannot signal them.
    attr.scoped = _LL_SCOPE_SIGNAL
    ruleset = _libc.syscall(
        _SYS_landlock_create_ruleset, ctypes.byref(attr), ctypes.sizeof(attr), 0
    )
    if ruleset < 0:
        return ctypes.get_errno(), "create_ruleset"
    try:
        for grant in grants:
            raw_path = grant.get("path")
            access = grant.get("access")
            if not isinstance(raw_path, str) or access not in _ACCESS_BITS:
                continue
            path = Path(raw_path)
            try:
                fd = _open_grant_fd(path)
            except OSError:
                continue  # target vanished or is a symlink: no access
            try:
                st = os.fstat(fd)
                expected_dev = grant.get("dev")
                expected_ino = grant.get("ino")
                if (
                    isinstance(expected_dev, int)
                    and isinstance(expected_ino, int)
                    and (st.st_dev, st.st_ino) != (expected_dev, expected_ino)
                ):
                    # The object at this path is not the one the
                    # daemon verified: replaced between grant
                    # computation and this open. Fail the spawn
                    # rather than running with a silently different
                    # sandbox.
                    return errno.EACCES, raw_path
                mode = st.st_mode & 0o170000
                if mode == 0o040000:
                    # Directory: the full access set (incl. listing and
                    # creating/removing entries beneath it).
                    allowed = _ACCESS_BITS[str(access)]
                elif mode in (0o100000, 0o020000, 0o060000):
                    # Regular file or device node: directory-only rights
                    # would make the kernel refuse the rule with EINVAL.
                    allowed = _ACCESS_BITS[str(access)] & _FILE_BITS
                else:
                    continue  # fifo, socket, ...: never granted
                rule = _PathBeneathAttr(allowed_access=allowed, parent_fd=fd)
                if (
                    _libc.syscall(
                        _SYS_landlock_add_rule,
                        ruleset,
                        _LANDLOCK_RULE_PATH_BENEATH,
                        ctypes.byref(rule),
                        0,
                    )
                    != 0
                ):
                    return ctypes.get_errno(), raw_path
            finally:
                os.close(fd)
        if _libc.syscall(_SYS_landlock_restrict_self, ruleset, 0) != 0:
            return ctypes.get_errno(), "restrict_self"
    finally:
        os.close(ruleset)
    return 0, ""


def _close_fds_except(keep: set[int]) -> None:
    """Close every open fd except 0/1/2 and ``keep`` (single-threaded)."""
    try:
        fds = [int(name) for name in os.listdir("/proc/self/fd")]
    except OSError:
        fds = list(range(3, 1024))
    for fd in fds:
        if fd not in keep and fd > 2:
            try:
                os.close(fd)
            except OSError:
                pass


def _write_errno(fd: int, code: int, message: str = "") -> None:
    """Write a 4-byte errno plus an optional short diagnostic message.

    The parent reads up to 204 bytes: the errno plus the message that
    names the failing step (e.g. the Landlock grant path), so a
    ``spawn_failed`` status can say *what* failed, not just the errno.
    """
    try:
        os.write(fd, struct.pack("<i", code) + message.encode("utf-8")[:200])
    except OSError:
        pass


def _child_main(spec: dict[str, object], pipes: dict[str, int]) -> None:
    """Target child: namespaces, Landlock, seccomp, then execve."""
    out_w = pipes["out_w"]
    err_w = pipes["err_w"]
    exec_w = pipes["exec_w"]
    ready_w = pipes["ready_w"]
    go_r = pipes["go_r"]
    try:
        _close_fds_except({out_w, err_w, exec_w, ready_w, go_r})
        # stdin is /dev/null; stdout/stderr are the helper's pipes.
        devnull = os.open("/dev/null", os.O_RDONLY)
        os.dup2(devnull, 0)
        os.dup2(out_w, 1)
        os.dup2(err_w, 2)
        if devnull > 2:
            os.close(devnull)
        # NEWIPC as well as NEWUSER|NEWNET: SysV IPC is outside
        # Landlock and outside the seccomp bans, and the uid mapping
        # preserves the owner's kernel UID — without its own IPC
        # namespace the target could shmat/shmctl any SysV segment
        # of the owner it knows the id of, even mode 0600. In a
        # fresh IPC namespace the owner's segments do not exist
        # (shmat -> EINVAL, shmget by key -> ENOENT); the target can
        # still create its own segments, which die with the
        # namespace. Verified unprivileged on this host: one
        # unshare(NEWUSER|NEWNET|NEWIPC) call creates all three.
        if _libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNET | _CLONE_NEWIPC) != 0:
            _write_errno(exec_w, ctypes.get_errno(), "unshare")
            os._exit(_SPAWN_FAILED_EXIT)
        os.write(ready_w, b"R")
        os.close(ready_w)
        # Wait for the parent to write uid_map/gid_map (10 s bound).
        deadline = time.monotonic() + _HANDSHAKE_TIMEOUT_S
        granted = False
        while time.monotonic() < deadline:
            ready, _, _ = select.select([go_r], [], [], 0.2)
            if ready:
                granted = os.read(go_r, 1) == b"G"
                break
        os.close(go_r)
        if not granted:
            _write_errno(exec_w, errno.ETIMEDOUT, "handshake")
            os._exit(_SPAWN_FAILED_EXIT)
        cwd = spec.get("cwd")
        if not isinstance(cwd, str) or not cwd:
            _write_errno(exec_w, errno.EINVAL, "cwd")
            os._exit(_SPAWN_FAILED_EXIT)
        try:
            os.chdir(cwd)
        except OSError as exc:
            _write_errno(exec_w, exc.errno or errno.EIO, "chdir")
            os._exit(_SPAWN_FAILED_EXIT)
        if _libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _write_errno(exec_w, ctypes.get_errno(), "no_new_privs")
            os._exit(_SPAWN_FAILED_EXIT)
        grants = spec.get("grants")
        if not isinstance(grants, list):
            _write_errno(exec_w, errno.EINVAL, "grants")
            os._exit(_SPAWN_FAILED_EXIT)
        landlock_err, landlock_where = _apply_landlock(grants)
        if landlock_err:
            _write_errno(exec_w, landlock_err, f"landlock:{landlock_where}")
            os._exit(_SPAWN_FAILED_EXIT)
        if _load_seccomp() is None:
            _write_errno(exec_w, errno.ENOSYS, "seccomp")
            os._exit(_SPAWN_FAILED_EXIT)
        _libc.prctl(_PR_SET_DUMPABLE, 0, 0, 0, 0)
        _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
        try:
            os.setpgid(0, 0)
        except OSError:
            pass  # cgroup kill is the tree-kill mechanism; groups are extra
        binary = spec.get("binary")
        argv = spec.get("argv")
        env = spec.get("env")
        if (
            not isinstance(binary, str)
            or not isinstance(argv, list)
            or not all(isinstance(item, str) for item in argv)
            or not isinstance(env, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())
            or not argv
        ):
            _write_errno(exec_w, errno.EINVAL, "argv")
            os._exit(_SPAWN_FAILED_EXIT)
        try:
            binary_fd = os.open(binary, os.O_RDONLY | os.O_CLOEXEC)
        except OSError as exc:
            _write_errno(exec_w, exc.errno or errno.EIO, "binary_open")
            os._exit(_SPAWN_FAILED_EXIT)
        try:
            st = os.fstat(binary_fd)
            if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != (
                spec.get("binary_dev"),
                spec.get("binary_ino"),
                spec.get("binary_size"),
                spec.get("binary_mtime_ns"),
            ):
                _write_errno(exec_w, errno.EACCES, "binary_identity")
                os._exit(_SPAWN_FAILED_EXIT)
            # Identity-bound exec through the verified open file:
            # /proc/self/fd/<fd> names exactly that file, so a path
            # swapped after the check cannot redirect the exec.
            # (execveat with AT_EMPTY_PATH is NOT used: it breaks
            # uutils multicall binaries on this host — they see the
            # fd number as the utility name and refuse with
            # "Security violation".)
            try:
                os.execve(f"/proc/self/fd/{binary_fd}", argv, env)
            except OSError as exc:
                _write_errno(exec_w, exc.errno or errno.EIO, "execve")
        finally:
            os.close(binary_fd)
        os._exit(_SPAWN_FAILED_EXIT)
    except BaseException:
        try:
            _write_errno(exec_w, errno.EIO)
        except BaseException:
            pass
        os._exit(_SPAWN_FAILED_EXIT)


class _Supervisor:
    """Parent side: id-map handshake, exec confirmation, supervision."""

    def __init__(self, spec: dict[str, object], run_dir: Path) -> None:
        self.spec = spec
        self.run_dir = run_dir
        self.process_id = str(spec.get("process_id", ""))
        raw_timeout = spec.get("timeout_s")
        # Typed as int: the daemon validates the profile timeout, and
        # an absent/invalid value falls back to the 120 s cap.
        self.timeout_s: int = 120
        if isinstance(raw_timeout, int) and raw_timeout > 0:
            self.timeout_s = raw_timeout
        self.status_path = run_dir / "status.json"
        self.stdout_path = run_dir / "stdout"
        self.stderr_path = run_dir / "stderr"
        self.child_pid: int | None = None
        self.helper_pid = os.getpid()
        self.started_at = ""
        self.started_mono = 0.0
        self.stdout_tail = _Tail()
        self.stderr_tail = _Tail()
        self.terminate_requested = False
        # Publication generation: bumped on every status write, so
        # the daemon can tell a tail + counters snapshot is torn.
        # The tail is written before the status, and a tail rotation
        # keeps the tail length constant — so a length check alone
        # cannot distinguish a new tail from the counters that
        # describe it. A generation that changed between two status
        # reads means a publication happened in between, and the
        # snapshot must be re-read. That still leaves one window:
        # both status reads can land between _flush_tails() and
        # _write_status(), with the generation unchanged and the
        # counters still old while the tail file is already new.
        # The status therefore also carries the SHA-256 of each
        # tail's exact bytes at publication time, so the daemon can
        # verify the tail it read is the one the counters describe.
        self.generation = 0

    def _status(self, state: str, **extra: object) -> dict[str, object]:
        status: dict[str, object] = {
            "version": 1,
            "process_id": self.process_id,
            "state": state,
            "generation": self.generation,
            "pid": self.child_pid,
            "helper_pid": self.helper_pid,
            "cgroup": _read_cgroup(),
            "started_at": self.started_at,
            "finished_at": extra.get("finished_at"),
            "exit_code": extra.get("exit_code"),
            "signal": extra.get("signal"),
            "error": extra.get("error"),
            "stdout_total_bytes": self.stdout_tail.total,
            "stderr_total_bytes": self.stderr_tail.total,
            "stdout_retained_from": self.stdout_tail.retained_from,
            "stderr_retained_from": self.stderr_tail.retained_from,
            "stdout_tail_sha256": hashlib.sha256(self.stdout_tail.bytes()).hexdigest(),
            "stderr_tail_sha256": hashlib.sha256(self.stderr_tail.bytes()).hexdigest(),
            "timeout_s": self.timeout_s,
        }
        return status

    def _write_status(self, state: str, **extra: object) -> None:
        # Bump first: the status that describes a just-flushed tail
        # must carry a newer generation than any status read before
        # the flush, or the daemon cannot tell the two apart.
        self.generation += 1
        try:
            _write_atomic(
                self.status_path,
                json.dumps(self._status(state, **extra)).encode("utf-8"),
            )
        except OSError:
            pass

    def _flush_tails(self) -> None:
        try:
            _write_atomic(self.stdout_path, self.stdout_tail.bytes())
        except OSError:
            pass
        try:
            _write_atomic(self.stderr_path, self.stderr_tail.bytes())
        except OSError:
            pass

    def _kill_tree(self, *, sigkill: bool) -> None:
        # Process group only — never the scope cgroup: this
        # supervisor runs inside the scope cgroup, so a
        # cgroup.kill write here would SIGKILL itself before it
        # can report. A descendant that escaped the group is the
        # daemon's problem: it cgroup.kills from outside the
        # scope (SystemdRunSpawner.kill_all).
        if self.child_pid is not None:
            try:
                os.killpg(self.child_pid, signal.SIGKILL if sigkill else signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
            try:
                os.kill(self.child_pid, signal.SIGKILL if sigkill else signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass

    def _on_sigterm(self, signum: int, frame: object) -> None:
        self.terminate_requested = True

    def run(self) -> int:
        outer_uid = self.spec.get("outer_uid")
        outer_gid = self.spec.get("outer_gid")
        if not isinstance(outer_uid, int) or not isinstance(outer_gid, int):
            self._write_status("spawn_failed", error="invalid_spec")
            return 1
        out_r, out_w = os.pipe()
        err_r, err_w = os.pipe()
        exec_r, exec_w = os.pipe()
        ready_r, ready_w = os.pipe()
        go_r, go_w = os.pipe()
        os.set_blocking(exec_w, False)
        pid = os.fork()
        if pid == 0:
            for fd in (out_r, err_r, exec_r, ready_r, go_w):
                try:
                    os.close(fd)
                except OSError:
                    pass
            _child_main(
                self.spec,
                {
                    "out_w": out_w,
                    "err_w": err_w,
                    "exec_w": exec_w,
                    "ready_w": ready_w,
                    "go_r": go_r,
                },
            )
            os._exit(_SPAWN_FAILED_EXIT)
        self.child_pid = pid
        for fd in (out_w, err_w, exec_w, ready_w, go_r):
            try:
                os.close(fd)
            except OSError:
                pass
        os.set_blocking(out_r, False)
        os.set_blocking(err_r, False)
        os.set_blocking(exec_r, False)
        signal.signal(signal.SIGTERM, self._on_sigterm)
        signal.signal(signal.SIGINT, self._on_sigterm)
        try:
            return self._supervise(pid, out_r, err_r, exec_r, ready_r, go_w)
        finally:
            for fd in (out_r, err_r, exec_r, ready_r, go_w):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _supervise(
        self,
        pid: int,
        out_r: int,
        err_r: int,
        exec_r: int,
        ready_r: int,
        go_w: int,
    ) -> int:
        # 1. Handshake: wait for the child's namespace readiness.
        deadline = time.monotonic() + _HANDSHAKE_TIMEOUT_S
        ready = False
        while time.monotonic() < deadline:
            readable, _, _ = select.select([ready_r], [], [], 0.2)
            if readable:
                ready = os.read(ready_r, 1) == b"R"
                break
            if self._child_gone(pid):
                break
        os.close(ready_r)
        if not ready:
            self._kill_tree(sigkill=True)
            self._reap(pid)
            self._write_status("spawn_failed", error="handshake_timeout")
            return 1
        # 2. Write the id maps (parent side: the child cannot write its
        # own map on this host — AppArmor's unprivileged_userns profile
        # denies every capability after the namespace is created).
        try:
            with open(f"/proc/{pid}/setgroups", "w", encoding="utf-8") as handle:
                handle.write("deny")
            with open(f"/proc/{pid}/uid_map", "w", encoding="utf-8") as handle:
                handle.write(f"0 {self.spec['outer_uid']} 1")
            with open(f"/proc/{pid}/gid_map", "w", encoding="utf-8") as handle:
                handle.write(f"0 {self.spec['outer_gid']} 1")
        except OSError as exc:
            self._kill_tree(sigkill=True)
            self._reap(pid)
            self._write_status("spawn_failed", error=f"id_map_failed:{_errno_name(exc.errno or 0)}")
            return 1
        try:
            os.write(go_w, b"G")
        except OSError:
            self._kill_tree(sigkill=True)
            self._reap(pid)
            self._write_status("spawn_failed", error="handshake_failed")
            return 1
        os.close(go_w)
        # 3. Exec confirmation: EOF on the CLOEXEC exec pipe means the
        # binary executed; 4+ bytes carry the failure errno and a short
        # diagnostic message naming the failing step.
        exec_deadline = time.monotonic() + _HANDSHAKE_TIMEOUT_S
        exec_ok = False
        exec_errno = 0
        exec_message = ""
        while time.monotonic() < exec_deadline:
            readable, _, _ = select.select([exec_r], [], [], 0.2)
            if readable:
                chunk = os.read(exec_r, 204)
                if chunk == b"":
                    exec_ok = True
                elif len(chunk) >= 4:
                    exec_errno = struct.unpack("<i", chunk[:4])[0]
                    exec_message = chunk[4:].decode("utf-8", errors="replace")
                break
            if self._child_gone(pid):
                break
        os.close(exec_r)
        if not exec_ok:
            self._kill_tree(sigkill=True)
            self._reap(pid)
            detail = (
                f"exec_failed:{_errno_name(exec_errno)}:{exec_message}"
                if exec_message
                else f"exec_failed:{_errno_name(exec_errno)}"
                if exec_errno
                else "exec_failed"
            )
            self._write_status("spawn_failed", error=detail)
            return 1
        self.started_at = _iso_now()
        self.started_mono = time.monotonic()
        self._write_status("running")
        # 4. Supervision: drain pipes into bounded tails, enforce the
        # runtime timeout, handle SIGTERM as a graceful tree kill.
        # The leader's exit does NOT end supervision: descendants may
        # still hold the pipes and write after it, and the daemon
        # counts the tree as running until the cgroup is empty. The
        # leader's exit status is saved and the loop keeps draining
        # until both streams reach EOF.
        exit_code: int | None = None
        exit_signal: int | None = None
        final_state = "exited"
        out_open = True
        err_open = True
        leader_status: tuple[int | None, int | None] | None = None
        while True:
            if self.terminate_requested:
                self.terminate_requested = False
                # A SIGTERM that arrives after the runtime timeout
                # is the daemon's timeout kill, not a user kill.
                if time.monotonic() >= self.started_mono + self.timeout_s:
                    final_state = "timeout"
                else:
                    final_state = "killed"
                self._kill_tree(sigkill=False)
                time.sleep(2.0)
                self._kill_tree(sigkill=True)
            if time.monotonic() >= self.started_mono + self.timeout_s:
                final_state = "timeout"
                self._kill_tree(sigkill=True)
            fds: list[int] = []
            if out_open:
                fds.append(out_r)
            if err_open:
                fds.append(err_r)
            if fds:
                readable, _, _ = select.select(fds, [], [], 0.2)
                for fd in readable:
                    try:
                        chunk = os.read(fd, _READ_CHUNK)
                    except OSError:
                        chunk = b""
                    if chunk:
                        if fd == out_r:
                            self.stdout_tail.append(chunk)
                        else:
                            self.stderr_tail.append(chunk)
                    else:
                        if fd == out_r:
                            out_open = False
                        else:
                            err_open = False
            # Publish the tail and its counters together, on every
            # iteration: the daemon reads them as one snapshot, and a
            # status that lags the tail misaligns poll offsets (and
            # loses bytes after a tail rotation). The tail is written
            # first, so the status never describes bytes the tail
            # does not yet contain.
            self._flush_tails()
            if leader_status is None:
                # The leader is reaped at most once: a second wait4
                # would raise ChildProcessError.
                status = self._reap(pid)
                if status is not None:
                    leader_status = status
                    exit_code, exit_signal = status
            if leader_status is not None and not out_open and not err_open:
                # Both streams are at EOF: no descendant can write any
                # more. Drain once more (a last chunk may have arrived
                # with the EOF), publish the final status, and exit.
                for fd, tail in ((out_r, self.stdout_tail), (err_r, self.stderr_tail)):
                    while True:
                        try:
                            chunk = os.read(fd, _READ_CHUNK)
                        except OSError:
                            chunk = b""
                        if not chunk:
                            break
                        tail.append(chunk)
                self._flush_tails()
                self._write_status(
                    final_state,
                    finished_at=_iso_now(),
                    exit_code=exit_code,
                    signal=exit_signal,
                )
                return 0
            self._write_status("running")
            if not fds:
                if leader_status is None and self._child_gone(pid):
                    # Both pipes drained and the child is gone without
                    # a waitpid hit (should not happen); reap once more.
                    status = self._reap(pid)
                    if status is not None:
                        leader_status = status
                        exit_code, exit_signal = status
                        # Fall through: the EOF check above finalizes
                        # on the next iteration.
                        continue
                # Both streams closed but the tree is alive (a child
                # that closed stdout/stderr and keeps running): no fd
                # will ever become readable — sleep instead of spinning
                # on wait4 + tail fsyncs.
                time.sleep(0.2)

    @staticmethod
    def _child_gone(pid: int) -> bool:
        done, status, _ = os.wait4(pid, os.WNOHANG)
        if done == 0:
            return False
        # wait4 reaped the child; stash the status for the reap call.
        _ReapedCache.store(pid, status)
        return True

    @staticmethod
    def _reap(pid: int) -> tuple[int | None, int | None] | None:
        cached = _ReapedCache.take(pid)
        if cached is not None:
            return cached
        done, status, _ = os.wait4(pid, os.WNOHANG)
        if done == 0:
            return None
        return _decode_wait_status(status)


class _ReapedCache:
    """One-slot cache: _child_gone reaps via wait4, _reap consumes it."""

    _store: dict[int, tuple[int | None, int | None]] = {}

    @classmethod
    def store(cls, pid: int, status: int) -> None:
        cls._store[pid] = _decode_wait_status(status)

    @classmethod
    def take(cls, pid: int) -> tuple[int | None, int | None] | None:
        return cls._store.pop(pid, None)


def _decode_wait_status(status: int) -> tuple[int | None, int | None]:
    """(exit_code, signal) from a waitpid status word."""
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status), None
    if os.WIFSIGNALED(status):
        return None, os.WTERMSIG(status)
    return None, None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m host_bridge.spawn_helper <spec-path>", file=sys.stderr)
        return 2
    spec_path = Path(argv[1])
    run_dir = spec_path.parent
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # The daemon waits on status.json; write a failure marker so it
        # does not hang until the spawn timeout.
        try:
            _write_atomic(
                run_dir / "status.json",
                json.dumps(
                    {"version": 1, "state": "spawn_failed", "error": f"bad_spec:{exc}"}
                ).encode("utf-8"),
            )
        except OSError:
            pass
        return 1
    if not isinstance(spec, dict) or spec.get("version") != 1:
        try:
            _write_atomic(
                run_dir / "status.json",
                json.dumps(
                    {"version": 1, "state": "spawn_failed", "error": "bad_spec:version"}
                ).encode("utf-8"),
            )
        except OSError:
            pass
        return 1
    try:
        spec_path.unlink()
    except OSError:
        pass
    try:
        supervisor = _Supervisor(spec, run_dir)
    except BaseException:
        return 1
    try:
        return supervisor.run()
    except BaseException as exc:  # never leave the daemon hanging
        try:
            _write_atomic(
                run_dir / "status.json",
                json.dumps(
                    {"version": 1, "state": "spawn_failed", "error": f"helper:{exc!r}"}
                ).encode("utf-8"),
            )
        except OSError:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
