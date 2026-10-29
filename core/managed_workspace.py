from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import stat
import sys
import threading
import uuid
from _thread import LockType
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import BinaryIO, Literal

from shared.workspace_binding import WorkspaceBinding, WorkspaceConflict, WorkspaceUnavailable


@dataclass(frozen=True)
class WorkspaceLimits:
    file_bytes: int
    total_bytes: int
    entries: int
    depth: int

    def __post_init__(self) -> None:
        for value in (self.file_bytes, self.total_bytes, self.entries, self.depth):
            if type(value) is not int or value <= 0:
                raise ValueError("workspace limits must be positive integers")


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    content: bytes
    executable: bool


@dataclass(frozen=True)
class WorkspaceSnapshot:
    binding: WorkspaceBinding
    files: tuple[WorkspaceFile, ...]
    directories: tuple[str, ...]
    fingerprints: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class WorkspaceStage:
    root: Path
    binding: WorkspaceBinding
    identity: tuple[int, int]
    _state: Literal["fresh", "running", "settled", "unknown", "published"] = field(
        default="fresh", init=False, repr=False, compare=False
    )
    _lock: LockType = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def begin_process(self) -> None:
        with self._lock:
            if self._state != "fresh":
                raise WorkspaceUnavailable("workspace stage already used")
            object.__setattr__(self, "_state", "running")

    def mark_settled(self) -> None:
        with self._lock:
            if self._state != "running":
                raise WorkspaceUnavailable("workspace process outcome is not settled")
            object.__setattr__(self, "_state", "settled")

    def mark_unknown(self) -> None:
        with self._lock:
            if self._state == "published":
                raise WorkspaceUnavailable("workspace stage already published")
            object.__setattr__(self, "_state", "unknown")

    @contextmanager
    def publication(self) -> Iterator[None]:
        with self._lock:
            if self._state == "published":
                raise WorkspaceConflict("workspace revision already published")
            if self._state not in {"fresh", "settled"}:
                raise WorkspaceUnavailable("workspace stage processes have not settled")
            yield
            object.__setattr__(self, "_state", "published")


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _directory(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)


def _private_directory(info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise WorkspaceUnavailable("workspace directory must be private and owned")


class ManagedWorkspace:
    """Один владелец private root и его кратких операций; не execution runtime.

    Только доверенный server owner создаёт этот объект. Произвольный выбранный
    путь не становится managed workspace. Bindings не принимаются из tool args.
    """

    def __init__(
        self,
        root: Path,
        *,
        principal_id: str,
        identity: tuple[int, int],
        parent_identity: tuple[int, int],
    ) -> None:
        self.root = root
        self._principal_id = principal_id
        self._identity = identity
        self._parent_identity = parent_identity
        self._workspace_id = uuid.uuid4().hex
        self._revision = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._namespace_lock_file: BinaryIO | None = None

    @classmethod
    def create(cls, parent: Path, *, principal_id: str) -> ManagedWorkspace:
        if not principal_id.strip():
            raise WorkspaceUnavailable("workspace principal is required")
        parent = parent.absolute()
        parent_fd = _directory(parent)
        try:
            parent_info = os.fstat(parent_fd)
            _private_directory(parent_info)
            name = f"workspace-{uuid.uuid4().hex}"
            os.mkdir(name, mode=0o700, dir_fd=parent_fd)
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            return cls(
                parent / name,
                principal_id=principal_id,
                identity=(info.st_dev, info.st_ino),
                parent_identity=(parent_info.st_dev, parent_info.st_ino),
            )
        finally:
            os.close(parent_fd)

    def _open_root(self) -> int:
        try:
            fd = _directory(self.root)
        except OSError as exc:
            raise WorkspaceConflict("workspace root unavailable") from exc
        try:
            info = os.fstat(fd)
            _private_directory(info)
            if (info.st_dev, info.st_ino) != self._identity:
                raise WorkspaceConflict("workspace root identity changed")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def bind(self, *, principal_id: str, session_id: str) -> WorkspaceBinding:
        with self._lock:
            if principal_id != self._principal_id or not session_id.strip():
                raise WorkspaceUnavailable("workspace owner/session mismatch")
            fd = self._open_root()
            os.close(fd)
            return WorkspaceBinding(self._workspace_id, principal_id, session_id, self._revision)

    def _validate(self, binding: WorkspaceBinding) -> None:
        if (
            binding.workspace_id != self._workspace_id
            or binding.principal_id != self._principal_id
            or not binding.session_id.strip()
        ):
            raise WorkspaceUnavailable("workspace owner/session mismatch")
        if binding.revision != self._revision:
            raise WorkspaceConflict("workspace revision changed")

    @contextmanager
    def mutation(self, binding: WorkspaceBinding) -> Iterator[Path]:
        with self._lock:
            self._validate(binding)
            fd = self._open_root()
            try:
                yield self.root
            finally:
                # Ошибка writer не доказывает отсутствие частичного effect.
                self._revision = uuid.uuid4().hex
                os.close(fd)

    @contextmanager
    def inspection(self, binding: WorkspaceBinding) -> Iterator[Path]:
        with self._lock:
            self._validate(binding)
            fd = self._open_root()
            try:
                yield self.root
            finally:
                os.close(fd)

    def capture(self, binding: WorkspaceBinding, *, limits: WorkspaceLimits) -> WorkspaceSnapshot:
        with self._lock:
            self._validate(binding)
            fd = self._open_root()
            try:
                first = _capture_tree(fd, binding=binding, limits=limits)
                second = _capture_tree(fd, binding=binding, limits=limits)
                if first != second:
                    raise WorkspaceConflict("workspace changed during capture")
                return first
            finally:
                os.close(fd)

    def _open_parent(self) -> int:
        fd = _directory(self.root.parent)
        try:
            info = os.fstat(fd)
            _private_directory(info)
            if (info.st_dev, info.st_ino) != self._parent_identity:
                raise WorkspaceConflict("workspace parent identity changed")
            return fd
        except BaseException:
            os.close(fd)
            raise

    def stage(self, snapshot: WorkspaceSnapshot, *, storage: Path | None = None) -> WorkspaceStage:
        with self._lock:
            self._validate(snapshot.binding)
            parent = self.root.parent if storage is None else storage
            parent_fd = self._open_parent() if storage is None else _directory(parent)
            try:
                _private_directory(os.fstat(parent_fd))
                name = f"stage-{uuid.uuid4().hex}"
                os.mkdir(name, mode=0o700, dir_fd=parent_fd)
                fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=parent_fd,
                )
                try:
                    for path in snapshot.directories:
                        _relative_path(path)
                        os.mkdir(path, mode=0o700, dir_fd=fd)
                    for file in snapshot.files:
                        _relative_path(file.path)
                        target = os.open(
                            file.path,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                            mode=0o700 if file.executable else 0o600,
                            dir_fd=fd,
                        )
                        try:
                            remaining = memoryview(file.content)
                            while remaining:
                                written = os.write(target, remaining)
                                if written <= 0:
                                    raise OSError("workspace stage write made no progress")
                                remaining = remaining[written:]
                        finally:
                            os.close(target)
                    info = os.fstat(fd)
                    return WorkspaceStage(
                        parent / name, snapshot.binding, (info.st_dev, info.st_ino)
                    )
                finally:
                    os.close(fd)
            finally:
                os.close(parent_fd)

    def discard_retired(self, snapshot: WorkspaceSnapshot, stage: WorkspaceStage) -> None:
        """Cleanup только собственного прежнего дерева после успешной публикации."""
        with self._lock, stage._lock:
            if (
                stage._state != "published"
                or stage.binding != snapshot.binding
                or stage.binding.workspace_id != self._workspace_id
                or stage.root.parent != self.root.parent
                or stage.root == self.root
            ):
                raise WorkspaceUnavailable("retired workspace cleanup binding mismatch")
            original = next((value for path, value in snapshot.fingerprints if path == ""), None)
            if original is None:
                raise WorkspaceUnavailable("retired workspace identity unavailable")
            parent_fd = self._open_parent()
            try:
                info = os.stat(stage.root.name, dir_fd=parent_fd, follow_symlinks=False)
                if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != original[:2]:
                    raise WorkspaceConflict("retired workspace identity changed")
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise WorkspaceUnavailable("safe workspace cleanup unavailable")
                shutil.rmtree(stage.root.name, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)

    def _discard_fresh(self, stage: WorkspaceStage) -> None:
        if (
            stage._state != "fresh"
            or stage.binding.workspace_id != self._workspace_id
            or stage.root.parent != self.root.parent
            or stage.root == self.root
            or not shutil.rmtree.avoids_symlink_attacks
        ):
            raise WorkspaceUnavailable("unpublished host staging cleanup unavailable")
        parent_fd = self._open_parent()
        try:
            info = os.stat(stage.root.name, dir_fd=parent_fd, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != stage.identity:
                raise WorkspaceConflict("unpublished host staging identity changed")
            shutil.rmtree(stage.root.name, dir_fd=parent_fd)
        finally:
            os.close(parent_fd)

    def publish_bounded(
        self, snapshot: WorkspaceSnapshot, stage: WorkspaceStage, *, limits: WorkspaceLimits
    ) -> Path | None:
        """После settlement переносит только validated bytes в trusted host stage.

        Worker никогда не получает writable host stage. Копирование ограничено
        capture limits, затем используется тот же owner/CAS/atomic exchange.
        Возвращает путь старого host tree только при cleanup warning после commit.
        """
        with self._lock, stage.publication():
            if stage._state != "settled":
                raise WorkspaceUnavailable("bounded stage processes have not settled")
            self._validate(snapshot.binding)
            if stage.binding != snapshot.binding:
                raise WorkspaceUnavailable("bounded output binding mismatch")
            fd = _directory(stage.root)
            try:
                info = os.fstat(fd)
                _private_directory(info)
                if (info.st_dev, info.st_ino) != stage.identity:
                    raise WorkspaceConflict("bounded output identity changed")
                output = _capture_tree(fd, binding=stage.binding, limits=limits)
                if _capture_tree(fd, binding=stage.binding, limits=limits) != output:
                    raise WorkspaceConflict("bounded output is still changing")
            finally:
                os.close(fd)
            host_stage = self.stage(output)
            try:
                self.publish(snapshot, host_stage, limits=limits)
            except BaseException:
                self._discard_fresh(host_stage)
                raise
            try:
                self.discard_retired(snapshot, host_stage)
            except (OSError, WorkspaceUnavailable):
                return host_stage.root
            return None

    def publish(
        self,
        snapshot: WorkspaceSnapshot,
        stage: WorkspaceStage,
        *,
        limits: WorkspaceLimits,
    ) -> WorkspaceBinding:
        """Collector вызывает только после settlement всех stage processes.

        Под owner lock атомарно меняется целое дерево. Старое дерево остаётся по
        stage.root до cleanup collector; никакого check-then-copy по отдельным файлам.
        Неконтролируемые host roots этой authority не получают.
        """
        with self._lock, stage.publication():
            self._validate(snapshot.binding)
            if stage.binding != snapshot.binding or stage.root.parent != self.root.parent:
                raise WorkspaceUnavailable("workspace stage binding mismatch")
            if stage.root == self.root:
                raise WorkspaceUnavailable("workspace stage aliases source")
            if self.capture(snapshot.binding, limits=limits) != snapshot:
                raise WorkspaceConflict("workspace source changed before publication")
            parent_fd = self._open_parent()
            try:
                fd = _directory(stage.root)
                try:
                    info = os.fstat(fd)
                    _private_directory(info)
                    if (info.st_dev, info.st_ino) != stage.identity:
                        raise WorkspaceConflict("workspace stage identity changed")
                    output = _capture_tree(fd, binding=stage.binding, limits=limits)
                    if _capture_tree(fd, binding=stage.binding, limits=limits) != output:
                        raise WorkspaceConflict("workspace output is still changing")
                    _exchange(parent_fd, self.root.name, stage.root.name)
                    self._identity = stage.identity
                    self._revision = uuid.uuid4().hex
                    return WorkspaceBinding(
                        self._workspace_id,
                        stage.binding.principal_id,
                        stage.binding.session_id,
                        self._revision,
                        stage.binding.environment_revision,
                    )
                finally:
                    os.close(fd)
            finally:
                os.close(parent_fd)


def _relative_path(value: str) -> None:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or value == ".":
        raise WorkspaceUnavailable("invalid workspace snapshot path")


def _exchange(parent_fd: int, first: str, second: str) -> None:
    # Linux RENAME_EXCHANGE: unsupported kernel/filesystem => refusal, never copy fallback.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        rename = libc.renameat2
    except AttributeError as exc:
        raise WorkspaceUnavailable("atomic workspace exchange unavailable") from exc
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(parent_fd, os.fsencode(first), parent_fd, os.fsencode(second), 2) != 0:
        raise OSError(ctypes.get_errno(), "atomic workspace exchange failed")


def _capture_tree(
    root_fd: int, *, binding: WorkspaceBinding, limits: WorkspaceLimits
) -> WorkspaceSnapshot:
    files: list[WorkspaceFile] = []
    directories: list[str] = []
    fingerprints: list[tuple[str, tuple[int, ...]]] = []
    total_bytes = 0
    entries = 0

    def walk(fd: int, relative: str, depth: int) -> None:
        nonlocal entries, total_bytes
        if depth > limits.depth:
            raise WorkspaceUnavailable("workspace depth limit")
        before = os.fstat(fd)
        if before.st_uid != os.geteuid():
            raise WorkspaceUnavailable("workspace directory owner mismatch")
        names: list[str] = []
        with os.scandir(fd) as iterator:
            for entry in iterator:
                entries += 1
                if entries > limits.entries:
                    raise WorkspaceUnavailable("workspace entry limit")
                names.append(entry.name)
        for name in sorted(names):
            path = f"{relative}/{name}" if relative else name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if info.st_uid != os.geteuid():
                raise WorkspaceUnavailable("workspace inode owner mismatch")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=fd,
                )
                try:
                    if _fingerprint(os.fstat(child)) != _fingerprint(info):
                        raise WorkspaceConflict("workspace directory changed")
                    directories.append(path)
                    walk(child, path, depth + 1)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
                    dir_fd=fd,
                )
                try:
                    if _fingerprint(os.fstat(child)) != _fingerprint(info):
                        raise WorkspaceConflict("workspace inode changed")
                    if info.st_size > limits.file_bytes:
                        raise WorkspaceUnavailable("workspace file byte limit")
                    content = bytearray()
                    while chunk := os.read(child, min(65536, limits.file_bytes + 1 - len(content))):
                        content.extend(chunk)
                        total_bytes += len(chunk)
                        if len(content) > limits.file_bytes or total_bytes > limits.total_bytes:
                            raise WorkspaceUnavailable("workspace byte limit")
                    if _fingerprint(os.fstat(child)) != _fingerprint(info):
                        raise WorkspaceConflict("workspace inode changed during capture")
                    files.append(WorkspaceFile(path, bytes(content), bool(info.st_mode & 0o111)))
                    fingerprints.append((path, _fingerprint(info)))
                finally:
                    os.close(child)
            else:
                raise WorkspaceUnavailable("workspace inode is linked or special")
        if _fingerprint(os.fstat(fd)) != _fingerprint(before):
            raise WorkspaceConflict("workspace directory changed during capture")
        fingerprints.append((relative, _fingerprint(before)))

    walk(root_fd, "", 0)
    return WorkspaceSnapshot(binding, tuple(files), tuple(directories), tuple(fingerprints))


@dataclass(frozen=True)
class WorkspaceAccess:
    """Scoped server dependency; разные чаты одного owner используют один workspace."""

    workspace: ManagedWorkspace
    principal_id: str
    session_id: str
    environment_revision: str | None = None

    def bind(self) -> WorkspaceBinding:
        return replace(
            self.workspace.bind(principal_id=self.principal_id, session_id=self.session_id),
            environment_revision=self.environment_revision,
        )


class ManagedWorkspacePool:
    """Доверенный app owner отдельных principal roots, без отдельной DB/schema.

    Namespace задаётся явно для backend. Existing обычные directories не
    импортируются. Restart сохраняет bytes того же principal, но выдаёт новую
    runtime identity/revision и не восстанавливает старые approvals/runs.
    """

    def __init__(self, root: Path, *, protected_root: Path | None = None) -> None:
        if ".." in root.parts:
            raise WorkspaceUnavailable("managed workspace parent must be canonical")
        self.root = root.absolute()
        self.protected_root = (protected_root or root).absolute()
        if not self.root.is_relative_to(self.protected_root):
            raise WorkspaceUnavailable("workspace namespace must be inside protected state")
        self._owners: dict[str, ManagedWorkspace] = {}
        self._lock = threading.Lock()
        self._closed = False

    def owns(self, principal_id: str, owner: ManagedWorkspace) -> bool:
        with self._lock:
            return not self._closed and self._owners.get(principal_id) is owner

    def close(self) -> None:
        with self._lock:
            self._closed = True
            # Живой Gateway/executor удерживает owner и его flock до settlement.
            self._owners.clear()

    def for_principal(self, principal_id: str) -> ManagedWorkspace:
        if sys.platform != "linux":
            raise WorkspaceUnavailable("managed execution requires a qualified Linux host")
        import fcntl

        if not principal_id or principal_id != principal_id.strip():
            raise WorkspaceUnavailable("canonical workspace principal is required")
        with self._lock:
            if self._closed:
                raise WorkspaceUnavailable("managed workspace pool is closed")
            owner = self._owners.get(principal_id)
            if owner is None:
                parent_fd = _directory(self.root.parent)
                try:
                    _private_directory(os.fstat(parent_fd))
                    try:
                        os.mkdir(self.root.name, mode=0o700, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                    pool_fd = os.open(
                        self.root.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=parent_fd,
                    )
                    try:
                        pool_info = os.fstat(pool_fd)
                        _private_directory(pool_info)
                        name = "workspace-" + hashlib.sha256(principal_id.encode()).hexdigest()
                        lease_fd = os.open(
                            ".lock-" + name,
                            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
                            mode=0o600,
                            dir_fd=pool_fd,
                        )
                        try:
                            lease_info = os.fstat(lease_fd)
                            if (
                                not stat.S_ISREG(lease_info.st_mode)
                                or lease_info.st_uid != os.geteuid()
                                or lease_info.st_nlink != 1
                                or lease_info.st_mode & 0o077
                            ):
                                raise WorkspaceUnavailable(
                                    "workspace namespace lease is unqualified"
                                )
                            try:
                                fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError as exc:
                                raise WorkspaceUnavailable(
                                    "workspace already has another runtime owner"
                                ) from exc
                            try:
                                os.mkdir(name, mode=0o700, dir_fd=pool_fd)
                            except FileExistsError:
                                pass
                            child = os.open(
                                name,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=pool_fd,
                            )
                            try:
                                info = os.fstat(child)
                                _private_directory(info)
                                owner = ManagedWorkspace(
                                    self.root / name,
                                    principal_id=principal_id,
                                    identity=(info.st_dev, info.st_ino),
                                    parent_identity=(pool_info.st_dev, pool_info.st_ino),
                                )
                                owner._namespace_lock_file = os.fdopen(lease_fd, "rb")
                                lease_fd = -1
                                self._owners[principal_id] = owner
                            finally:
                                os.close(child)
                        finally:
                            if lease_fd >= 0:
                                os.close(lease_fd)
                    finally:
                        os.close(pool_fd)
                finally:
                    os.close(parent_fd)
        # Registry lock не удерживается во время чужой workspace operation.
        fd = owner._open_root()
        os.close(fd)
        return owner

    def resolve_root(self, principal_id: str, selected: Path | None) -> Path:
        if self._closed:
            raise WorkspaceUnavailable("managed workspace pool is closed")
        if not principal_id or principal_id != principal_id.strip():
            raise WorkspaceUnavailable("canonical workspace principal is required")
        if selected is None:
            return self.for_principal(principal_id).root
        selected = selected.resolve()
        parent = self.root.resolve()
        expected = parent / ("workspace-" + hashlib.sha256(principal_id.encode()).hexdigest())
        if selected == expected:
            return self.for_principal(principal_id).root
        protected = self.protected_root.resolve()
        if selected.is_relative_to(protected) or protected.is_relative_to(selected):
            raise WorkspaceUnavailable("workspace root overlaps protected managed namespace")
        # File-only legacy inspection не становится process admission.
        return selected
