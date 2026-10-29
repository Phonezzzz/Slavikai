"""Фиксированный mount owner внутри существующего Podman rootless namespace.

Не исполняет workspace code. EOF означает потерю owner: mounts сохраняются,
новый owner обязан отказать; только explicit close пустого пула снимает mounts.
"""

from __future__ import annotations

import ctypes
import fcntl
import json
import os
import stat
import sys
from pathlib import Path


def _private(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o077:
        raise ValueError("staging directory is not private and owned")
    return info


def _require_namespace(uid: int, gid: int, host_mount: str, host_user: str) -> None:
    if (
        os.geteuid() != 0
        or os.readlink("/proc/self/ns/mnt") == host_mount
        or os.readlink("/proc/self/ns/user") == host_user
    ):
        raise ValueError("dedicated rootless namespaces required")
    for name, expected in (("uid_map", uid), ("gid_map", gid)):
        lines = Path("/proc/self/" + name).read_text().splitlines()
        if not lines or lines[0].split() != ["0", str(expected), "1"]:
            raise ValueError("rootless namespace identity differs")


def _drop_capabilities(libc: ctypes.CDLL) -> None:
    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [
            ("effective", ctypes.c_uint32),
            ("permitted", ctypes.c_uint32),
            ("inheritable", ctypes.c_uint32),
        ]

    for capability in range(64):
        if libc.prctl(24, capability, 0, 0, 0) != 0 and ctypes.get_errno() != 22:
            raise OSError(ctypes.get_errno(), "capability bounding drop failed")
    header = Header(0x20080522, 0)
    data = (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        raise OSError(ctypes.get_errno(), "capability drop failed")
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(4, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "collector process protection unavailable")


def _start_time() -> str:
    return Path("/proc/self/stat").read_text().rpartition(")")[2].split()[19]


def run(
    root: Path,
    slot_bytes: int,
    slot_inodes: int,
    count: int,
    uid: int,
    gid: int,
    host_mount: str,
    host_user: str,
    nonce: str,
) -> None:
    _require_namespace(uid, gid, host_mount, host_user)
    if (
        any(type(value) is not int or value <= 0 for value in (slot_bytes, slot_inodes, count))
        or slot_bytes % os.sysconf("SC_PAGE_SIZE")
        or not root.is_absolute()
        or root.resolve() != root
    ):
        raise ValueError("invalid bounded staging configuration")
    _private(root)
    lease = os.open(root / ".lease", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        info = os.fstat(lease)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0
            or info.st_nlink != 1
            or info.st_mode & 0o077
        ):
            raise ValueError("invalid staging lease")
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed = {".lease", *(f"slot-{index}" for index in range(count))}
        if any(path.name not in allowed for path in root.iterdir()):
            raise ValueError("staging namespace requires reconciliation")
        slots = [root / f"slot-{index}" for index in range(count)]
        for slot in slots:
            if not slot.exists():
                slot.mkdir(mode=0o700)
            _private(slot)
            if os.path.ismount(slot) or next(slot.iterdir(), None) is not None:
                raise ValueError("previous staging owner is unknown; refusing reuse")
        libc = ctypes.CDLL(None, use_errno=True)
        mounted: list[Path] = []
        child: int | None = None
        release: int | None = None
        initialized = False
        close_empty = False
        close_requested = False
        try:
            for slot in slots:
                options = f"size={slot_bytes},nr_inodes={slot_inodes},mode=0700".encode()
                if libc.mount(b"tmpfs", os.fsencode(slot), b"tmpfs", 6, options) != 0:
                    raise OSError(ctypes.get_errno(), "bounded staging mount failed")
                mounted.append(slot)
                capacity = os.statvfs(slot)
                if (
                    capacity.f_blocks * capacity.f_frsize != slot_bytes
                    or capacity.f_files != slot_inodes
                ):
                    raise ValueError("actual staging limits differ")
            read_end, release = os.pipe()
            child = os.fork()
            if child == 0:
                os.close(release)
                try:
                    _drop_capabilities(libc)
                    print(
                        json.dumps(
                            {
                                "nonce": nonce,
                                "pid": os.getpid(),
                                "start": _start_time(),
                                "mount_namespace": os.readlink("/proc/self/ns/mnt"),
                                "user_namespace": os.readlink("/proc/self/ns/user"),
                            }
                        ),
                        flush=True,
                    )
                    os.read(read_end, 1)
                    os._exit(0)
                except BaseException:
                    os._exit(1)
            os.close(read_end)
            initialized = True
            command = sys.stdin.buffer.readline(16)
            close_requested = command == b"close\n"
            close_empty = close_requested and all(
                next(slot.iterdir(), None) is None for slot in slots
            )
        finally:
            if release is not None:
                os.close(release)
            if child is not None:
                _, status = os.waitpid(child, 0)
                if os.waitstatus_to_exitcode(status) != 0:
                    close_empty = False
            # Только собственные mounts до readiness либо explicit empty close.
            # Потеря owner/retained outputs никогда не заменяется lazy unmount.
            if not initialized or close_empty:
                for slot in reversed(mounted):
                    if libc.umount2(os.fsencode(slot), 0) != 0:
                        raise OSError(ctypes.get_errno(), "bounded staging unmount failed")
            elif close_requested:
                raise ValueError("staging owner cannot confirm empty cleanup")
    finally:
        os.close(lease)


if __name__ == "__main__":
    if len(sys.argv) != 10:
        raise SystemExit("invalid staging helper invocation")
    run(
        Path(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        sys.argv[7],
        sys.argv[8],
        sys.argv[9],
    )
