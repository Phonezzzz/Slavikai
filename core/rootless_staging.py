from __future__ import annotations

import hashlib
import json
import os
import selectors
import shutil
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.managed_workspace import ManagedWorkspace, WorkspaceSnapshot, WorkspaceStage
from shared.workspace_binding import WorkspaceConflict, WorkspaceUnavailable


@dataclass(frozen=True)
class StagingLimits:
    """Явные kernel limits конечного набора ephemeral execution slots."""

    slot_bytes: int
    slot_inodes: int
    slots: int

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value <= 0
            for value in (self.slot_bytes, self.slot_inodes, self.slots)
        ):
            raise ValueError("staging limits must be explicit positive integers")
        if self.slot_bytes % os.sysconf("SC_PAGE_SIZE"):
            raise ValueError("staging byte limit must be page aligned")

    @property
    def total_bytes(self) -> int:
        return self.slot_bytes * self.slots

    @property
    def total_inodes(self) -> int:
        return self.slot_inodes * self.slots


if TYPE_CHECKING:
    from core.rootless_podman import RootlessPodmanProfile


def _file_binding(path: Path, owner: int) -> tuple[int, ...]:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != owner
        or info.st_nlink != 1
        or info.st_mode & 0o022
    ):
        raise WorkspaceUnavailable("staging helper file is unqualified")
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _private_directory(path: Path, uid: int) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != uid or info.st_mode & 0o077:
        raise WorkspaceUnavailable("staging namespace is not private and owned")
    return info.st_dev, info.st_ino


def _read_ready(process: subprocess.Popen[bytes]) -> object:
    if process.stdout is None:
        raise WorkspaceUnavailable("staging helper output unavailable")
    buffer = bytearray()
    deadline = time.monotonic() + 15
    fd = process.stdout.fileno()
    os.set_blocking(fd, False)
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        while b"\n" not in buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkspaceUnavailable("staging helper readiness timed out")
            if not selector.select(remaining):
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                raise WorkspaceUnavailable("staging helper readiness unavailable")
            buffer.extend(chunk)
            if len(buffer) > 4096:
                raise WorkspaceUnavailable("staging helper metadata exceeds limit")
    if buffer.count(b"\n") != 1 or not buffer.endswith(b"\n"):
        raise WorkspaceUnavailable("invalid staging helper metadata")
    value: object = json.loads(buffer)
    return value


@dataclass(frozen=True)
class BoundedStage:
    stage: WorkspaceStage
    pool: RootlessStagePool
    engine_root: Path
    operation_root: Path
    operation_identity: tuple[int, int]


class RootlessStagePool:
    """Конечные kernel-bounded slots, а не размер после исполнения.

    Retained stage занимает слот; один principal не получает второй слот до
    reconciliation. Краткий allocation lock не сериализует execution разных owners.
    """

    def __init__(self, profile: RootlessPodmanProfile, limits: StagingLimits) -> None:
        self.profile = profile
        self.limits = limits
        self.root = profile.state_root / "bounded-staging"
        self._lock = threading.Lock()
        self._closed = False
        self._process: subprocess.Popen[bytes] | None = None
        self._view: Path | None = None
        self._pid: int | None = None
        self._start: str | None = None
        self._mount_namespace: str | None = None
        self._user_namespace: str | None = None
        self._slots: tuple[tuple[Path, tuple[int, int]], ...] = ()
        self._files: tuple[tuple[Path, int, tuple[int, ...]], ...] = ()
        self._allocations: dict[Path, BoundedStage | None] = {}
        self._revision: str | None = None

    @classmethod
    def open(cls, profile: RootlessPodmanProfile, limits: StagingLimits) -> RootlessStagePool:
        pool = cls(profile, limits)
        try:
            pool._start_helper()
            return pool
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            pool._shutdown(empty=False)
            raise WorkspaceUnavailable("bounded rootless staging unavailable") from exc

    @property
    def execution_revision(self) -> str:
        if self._revision is None:
            raise WorkspaceUnavailable("bounded staging is not qualified")
        return self._revision

    def _start_helper(self) -> None:
        self.profile.check_current()
        _private_directory(self.profile.state_root, self.profile.uid)
        try:
            self.root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        _private_directory(self.root, self.profile.uid)
        python = Path("/usr/bin/python3").resolve()
        helper = Path(__file__).with_name("rootless_stage_helper.py").resolve()
        self._files = tuple(
            (path, uid, _file_binding(path, uid))
            for path, uid in ((python, 0), (helper, self.profile.uid))
        )
        nonce = uuid.uuid4().hex
        self._process = subprocess.Popen(
            self.profile.argv()
            + [
                "unshare",
                str(python),
                "-I",
                str(helper),
                str(self.root),
                str(self.limits.slot_bytes),
                str(self.limits.slot_inodes),
                str(self.limits.slots),
                str(self.profile.uid),
                str(self.profile.gid),
                os.readlink("/proc/self/ns/mnt"),
                os.readlink("/proc/self/ns/user"),
                nonce,
            ],
            env=self.profile.environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
        ready = _read_ready(self._process)
        if not isinstance(ready, dict) or ready.get("nonce") != nonce:
            raise WorkspaceUnavailable("staging helper identity unavailable")
        pid, start = ready.get("pid"), ready.get("start")
        mount, user = ready.get("mount_namespace"), ready.get("user_namespace")
        if (
            type(pid) is not int
            or pid <= 0
            or not isinstance(start, str)
            or not start.isdecimal()
            or not isinstance(mount, str)
            or not isinstance(user, str)
            or mount == os.readlink("/proc/self/ns/mnt")
            or user == os.readlink("/proc/self/ns/user")
        ):
            raise WorkspaceUnavailable("staging helper namespace unavailable")
        self._pid, self._start = pid, start
        self._mount_namespace, self._user_namespace = mount, user
        self._view = Path(f"/proc/{pid}/root") / self.root.relative_to("/")
        self._slots = tuple(
            (
                self._view / f"slot-{index}",
                _private_directory(self._view / f"slot-{index}", self.profile.uid),
            )
            for index in range(self.limits.slots)
        )
        self.check_current()
        self._revision = hashlib.sha256(
            repr(
                (
                    "bounded-rootless-slots-v1",
                    self.profile.revision,
                    self.limits,
                    nonce,
                    pid,
                    start,
                    mount,
                    user,
                    self._slots,
                    self._files,
                )
            ).encode()
        ).hexdigest()

    def check_current(self) -> None:
        self.profile.check_current()
        process = self._process
        if process is None or process.poll() is not None or self._pid is None:
            raise WorkspaceUnavailable("staging helper is unavailable")
        for path, uid, expected in self._files:
            if _file_binding(path, uid) != expected:
                raise WorkspaceUnavailable("staging helper binding changed")
        proc = Path(f"/proc/{self._pid}")
        if (
            (proc / "stat").read_text().rpartition(")")[2].split()[19] != self._start
            or os.readlink(proc / "ns/mnt") != self._mount_namespace
            or os.readlink(proc / "ns/user") != self._user_namespace
        ):
            raise WorkspaceUnavailable("staging helper process identity changed")
        status = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in (proc / "status").read_text().splitlines()
            if ":" in line
        }
        if any(status.get(name) != "0000000000000000" for name in ("CapEff", "CapPrm")) or (
            status.get("NoNewPrivs") != "1"
        ):
            raise WorkspaceUnavailable("staging collector capabilities differ")
        for slot, expected in self._slots:
            if _private_directory(slot, self.profile.uid) != expected:
                raise WorkspaceUnavailable("staging slot identity changed")
            capacity = os.statvfs(slot)
            if (
                capacity.f_blocks * capacity.f_frsize != self.limits.slot_bytes
                or capacity.f_files != self.limits.slot_inodes
            ):
                raise WorkspaceUnavailable("actual staging capacity differs")

    def prepare(self, workspace: ManagedWorkspace, snapshot: WorkspaceSnapshot) -> BoundedStage:
        principal = hashlib.sha256(snapshot.binding.principal_id.encode()).hexdigest()
        prefix = "operation-" + principal + "-"
        with self._lock:
            if self._closed:
                raise WorkspaceUnavailable("staging pool is closed")
            self.check_current()
            if snapshot.binding.environment_revision != self.execution_revision:
                raise WorkspaceConflict("workspace staging environment binding changed")
            available: Path | None = None
            for slot, _ in self._slots:
                entries = list(slot.iterdir())
                if any(entry.name.startswith(prefix) for entry in entries):
                    raise WorkspaceConflict("principal has active or retained staging")
                if not entries:
                    available = slot if available is None else available
            if available is None:
                raise WorkspaceUnavailable("bounded staging slots exhausted")
            operation = available / (prefix + uuid.uuid4().hex)
            operation.mkdir(mode=0o700)
            identity = _private_directory(operation, self.profile.uid)
            self._allocations[operation] = None
        # Долгое копирование не держит allocation lock других principals.
        stage = workspace.stage(snapshot, storage=operation)
        assert self._view is not None
        bounded = BoundedStage(
            stage, self, self.root / stage.root.relative_to(self._view), operation, identity
        )
        with self._lock:
            self._allocations[operation] = bounded
        self.check_stage(bounded)
        return bounded

    def check_stage(self, bounded: BoundedStage) -> None:
        self.check_current()
        with self._lock:
            if self._allocations.get(bounded.operation_root) is not bounded:
                raise WorkspaceUnavailable("bounded staging ownership unavailable")
        if (
            bounded.pool is not self
            or bounded.stage.binding.environment_revision != self.execution_revision
            or _private_directory(bounded.operation_root, self.profile.uid)
            != bounded.operation_identity
            or _private_directory(bounded.stage.root, self.profile.uid) != bounded.stage.identity
            or self._view is None
            or bounded.engine_root != self.root / bounded.stage.root.relative_to(self._view)
        ):
            raise WorkspaceUnavailable("bounded staging binding changed")

    def release(self, bounded: BoundedStage) -> None:
        self.check_stage(bounded)
        if bounded.stage._state != "published":
            raise WorkspaceUnavailable("unpublished staging must be retained")
        if not shutil.rmtree.avoids_symlink_attacks:
            raise WorkspaceUnavailable("safe bounded staging cleanup unavailable")
        with self._lock:
            if self._allocations.get(bounded.operation_root) is not bounded:
                raise WorkspaceUnavailable("bounded staging owner changed")
            _private_directory(bounded.operation_root, self.profile.uid)
            fd = os.open(
                bounded.operation_root.parent,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                shutil.rmtree(bounded.operation_root.name, dir_fd=fd)
            finally:
                os.close(fd)
            del self._allocations[bounded.operation_root]
            close = self._closed and not self._allocations
        if close:
            self._shutdown(empty=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            active_or_retained = bool(self._allocations)
        # Живые operations удерживают bounded mount и lease, как workspace owner.
        if not active_or_retained:
            self._shutdown(empty=True)

    def _shutdown(self, *, empty: bool) -> None:
        process = self._process
        if process is None:
            return
        delivered = False
        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.write(b"close\n" if empty else b"abandon\n")
                process.stdin.close()
                delivered = True
            except OSError:
                pass  # EOF сохраняет mounts и запрещает последующий reuse.
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            raise WorkspaceUnavailable("staging owner shutdown unknown") from None
        finally:
            for pipe in (process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        if process.returncode != 0 or (empty and not delivered):
            raise WorkspaceUnavailable("staging owner cleanup unavailable")
        self._process = None
