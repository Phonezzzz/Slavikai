"""Cross-process active-writer lease for one browser profile directory.

One lease per profile (E02-T03): two runs — two ``BrowserService``
instances in one process, or two processes — must never drive one
Chromium ``user_data_dir`` at the same time. The lease is a kernel-held
``flock`` on a dedicated lease file next to the profile directory, the
same mechanism ``ManagedWorkspacePool`` already uses for workspace
namespaces (``core/managed_workspace.py``): strict ``fstat`` qualification
of the lease file, ``LOCK_EX | LOCK_NB``, a live holder reported as
``profile_busy``.

The lease file persists on disk, but the lock itself does not survive a
dead holder: the kernel releases ``flock`` when the holder process dies,
however it dies, so a crashed holder never blocks the profile forever.
A later run re-acquires the flock only after health validation of the
profile directory (it exists, is a directory, is writable); the previous
holder pid is reported back as ``recovered_from_pid`` so recovery is
observable instead of silent. The lease file is never unlinked while
held — unlinking a locked file lets two processes hold "the" lock on
different inodes.

Nothing here touches browser internals: Chromium's own ``SingletonLock``
is Chromium's job to validate and clean on startup.
"""

from __future__ import annotations

import fcntl
import json
import os
import stat
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

_LEASE_RECORD_LIMIT = 4096


class BrowserProfileBusyError(RuntimeError):
    """The profile is held by another live run: ``profile_busy``."""


class BrowserProfileLeaseError(RuntimeError):
    """The lease cannot be taken for a reason other than a live holder."""


class BrowserProfileLease:
    """A held profile lease: the open fd owns the ``flock``.

    Closing the fd releases the lock — the kernel does the same when the
    holder process dies. ``release`` is idempotent and best-effort.
    """

    def __init__(
        self,
        *,
        lease_path: Path,
        profile_dir: Path,
        holder_pid: int,
        recovered_from_pid: int | None,
        fd: int,
    ) -> None:
        self.lease_path = lease_path
        self.profile_dir = profile_dir
        self.holder_pid = holder_pid
        self.recovered_from_pid = recovered_from_pid
        self._fd = fd

    @property
    def closed(self) -> bool:
        return self._fd < 0

    def release(self) -> None:
        fd, self._fd = self._fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> BrowserProfileLease:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def acquire_profile_lease(
    lease_path: Path,
    *,
    profile_dir: Path,
    holder: str,
) -> BrowserProfileLease:
    """Take the active-writer lease for one profile directory.

    ``lease_path`` is the dedicated lease file (a sibling of the profile
    directory, never inside it); ``profile_dir`` is health-validated after
    the lock is taken. Raises :class:`BrowserProfileBusyError` when
    another live run holds the profile, and
    :class:`BrowserProfileLeaseError` when the lease file is unqualified
    or the profile directory is unhealthy. The caller must
    :meth:`BrowserProfileLease.release` the returned lease.
    """
    if sys.platform != "linux":
        raise BrowserProfileLeaseError("browser profile lease requires a Linux host")
    if not holder.strip():
        raise ValueError("lease holder must not be blank")
    lease_path = Path(lease_path)
    profile_dir = Path(profile_dir)
    previous_pid = _read_holder_pid(lease_path)
    try:
        lease_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BrowserProfileLeaseError(
            f"cannot create browser profile lease directory {lease_path.parent}: {exc}"
        ) from exc
    try:
        fd = os.open(
            lease_path,
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
        )
    except OSError as exc:
        raise BrowserProfileLeaseError(
            f"cannot open browser profile lease {lease_path}: {exc}"
        ) from exc
    try:
        _qualify_lease_file(fd, lease_path)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BrowserProfileBusyError(
                "profile_busy: browser profile is held by another run"
                + (
                    f" (pid {previous_pid})"
                    if previous_pid is not None
                    else " (holder pid unknown)"
                )
            ) from exc
        _validate_profile_dir(profile_dir)
        _write_holder_record(
            fd,
            {
                "pid": os.getpid(),
                "holder": holder,
                "started_at": datetime.now(UTC).isoformat(),
            },
        )
    except (BrowserProfileBusyError, BrowserProfileLeaseError):
        os.close(fd)
        raise
    except OSError as exc:
        os.close(fd)
        raise BrowserProfileLeaseError(
            f"cannot take browser profile lease {lease_path}: {exc}"
        ) from exc
    recovered = previous_pid if previous_pid not in (None, os.getpid()) else None
    return BrowserProfileLease(
        lease_path=lease_path,
        profile_dir=profile_dir,
        holder_pid=os.getpid(),
        recovered_from_pid=recovered,
        fd=fd,
    )


def _qualify_lease_file(fd: int, lease_path: Path) -> None:
    """Refuse a lease file that is not a private regular file of this user."""
    info = os.fstat(fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_nlink != 1
        or info.st_mode & 0o077
    ):
        raise BrowserProfileLeaseError(f"browser profile lease is unqualified: {lease_path}")


def _validate_profile_dir(profile_dir: Path) -> None:
    """Health validation before a profile is reused after a previous holder.

    The directory must exist (it is created when missing), be a real
    directory, and be writable. Anything else is an explicit refusal:
    a broken profile is never silently adopted.
    """
    if profile_dir.exists() and not profile_dir.is_dir():
        raise BrowserProfileLeaseError(
            f"browser profile directory is not a directory: {profile_dir}"
        )
    try:
        profile_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BrowserProfileLeaseError(
            f"cannot create browser profile directory {profile_dir}: {exc}"
        ) from exc
    if not os.access(profile_dir, os.W_OK):
        raise BrowserProfileLeaseError(f"browser profile directory is not writable: {profile_dir}")


def _write_holder_record(fd: int, record: dict[str, object]) -> None:
    """Replace the lease file contents with the current holder record.

    The record is informational only — the ``flock`` is the lock — so a
    torn record is tolerated by readers; it only identifies the holder.
    """
    payload = json.dumps(record, sort_keys=True).encode()
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _read_holder_pid(lease_path: Path) -> int | None:
    """Best-effort pid of the previous holder, or ``None`` if unknown."""
    try:
        fd = os.open(lease_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        raw = os.read(fd, _LEASE_RECORD_LIMIT)
    except OSError:
        return None
    finally:
        os.close(fd)
    try:
        record = json.loads(raw.decode())
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(record, dict):
        return None
    pid = record.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    return pid
