"""Bounded host filesystem API for the Owner Host Bridge (E03-T04).

Implements the ``host.fs.*`` capabilities of ADR-0013 §6 on top of the
E03-T03 wire schema (``host_bridge.protocol``). The wire schema for
every capability's ``params``/result was fixed by the E03-T04 schema
proposal with owner corrections (2026-11-23); this module is the
daemon-side enforcement of that contract.

Trust-model points enforced here (ADR-0013 §5, §9, §10, §11):

- **Root-id addressing.** The model addresses files as ``root_id`` +
  relative ``path``; absolute paths are not expressible. Roots come
  from owner configuration only (``~/.config/slavikai/host-bridge.json``,
  fail-closed) or the single built-in workspace root. The config
  validator enforces the ADR §9 allowlist after canonicalization:
  only the built-in workspace and specific subdirectories of
  ``~/Project``, ``~/Documents``, ``~/Downloads`` may become roots.
- **Secret denial over the allowlist.** Every addressed path is
  classified with ``DesktopPathSecurity``; anything but ``normal``
  (including ``ask`` — the bridge never asks, ``/etc`` is a hard deny)
  is rejected with ``protected_path``.
- **Write-exclusions.** SlavikAI checkouts (the daemon's own tree plus
  any structurally detected checkout), the bridge config/launch files,
  the policy store (inside checkouts) and runtime state
  (``~/.local/state/slavikai``) are write-excluded even inside an
  allowed root, even when the config forgets to list them.
- **fd-relative operations (TOCTOU).** After lexical checks, every
  filesystem operation walks directory components with
  ``os.open(..., O_NOFOLLOW, dir_fd=...)`` starting from an fd pinned
  on the root. Symlink traversal is forbidden in v1 (``ELOOP`` →
  ``path_outside_root``) and nested mounts are rejected (``st_dev``
  change or a mount-point component). Containment is therefore
  enforced at operation time by the pinned fd chain, not only by
  string checks at validation time.
- **Frame budget on serialized JSON.** Every response — read chunks,
  list/search pages, capabilities — is shrunk until its serialized
  frame fits ``MAX_FRAME_BYTES`` (JSON escaping counts). A read chunk
  either advances the byte offset or fails with a typed error.
- **Chunked atomic writes.** ``create``/``overwrite`` write chunks to
  a temp file in the target directory (idempotent per offset via
  ``pwrite``, serialized per session under a lock so a delayed
  chunk can never mutate an already-published inode) and publish
  with a single commit: ``os.link`` for
  ``create`` (atomic absence check, no exists()+replace race),
  ``os.rename`` for ``overwrite``. Unfinished writes are size-capped
  and expire after a TTL. The staging file's identity
  (``st_dev``+``st_ino``) is recorded at creation; every later open
  uses ``O_NOFOLLOW`` and re-verifies the identity, so swapping the
  staging name for a symlink (or another file) cannot redirect
  writes outside the root.
- **Cancel-before-commit.** Mutations check the request's
  ``CancelToken`` immediately before the single publish syscall
  (including the trash rename in delete, after the ``.trashinfo``
  write). A cancelled/timed-out mutation never publishes late; if
  the publish already happened when the response is lost, the
  outcome is unknown to the client (documented in
  ``host_bridge.client``) and must not be auto-retried.
- **Move** is a single ``renameat`` (``renameat2`` with
  ``RENAME_NOREPLACE`` when ``overwrite=false``, so a destination
  created concurrently is never destroyed); ``EXDEV`` is a typed
  error (no copy+delete fallback). **Delete** is trash-only with
  freedesktop ``.trashinfo`` created exclusively before the move; no
  trash on the source's mount → ``trash_unavailable`` and the source
  is untouched.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import re
import stat as stat_mod
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final
from urllib.parse import quote

from core.desktop_security import DesktopPathSecurity
from host_bridge.protocol import (
    CAPABILITY_FS_DELETE,
    CAPABILITY_FS_LIST,
    CAPABILITY_FS_MOVE,
    CAPABILITY_FS_PATCH,
    CAPABILITY_FS_READ,
    CAPABILITY_FS_SEARCH,
    CAPABILITY_FS_STAT,
    CAPABILITY_FS_WRITE,
    MAX_FRAME_BYTES,
    BridgeConfigError,
    BridgeProtocolError,
    BridgeRequest,
    RequestTimeoutError,
    UnknownCapabilityError,
    encode_success,
)
from shared.patch import apply_unified_patch, validate_patch_contract
from shared.sandbox import read_mount_bindings

# Content budget per read/write chunk. The frame limit is 64 KiB; the
# read path additionally shrinks by serialized size, so this is an
# upper bound on raw bytes per call, not a guarantee of frame fit.
MAX_FS_CHUNK_BYTES: Final[int] = 32 * 1024
# Patch reads the whole file daemon-side; parity with the in-process
# MAX_DESKTOP_READ_BYTES bound.
MAX_PATCH_FILE_BYTES: Final[int] = 1_048_576
# Availability bound (ADR-0013 §7): one write session may not grow a
# temp file past this size.
MAX_WRITE_BYTES: Final[int] = 64 * 1024 * 1024
WRITE_SESSION_TTL_S: Final[float] = 600.0

DEFAULT_LIST_LIMIT: Final[int] = 200
MAX_LIST_LIMIT: Final[int] = 500
DEFAULT_SEARCH_LIMIT: Final[int] = 50
MAX_SEARCH_LIMIT: Final[int] = 200
MAX_SEARCH_QUERY_CHARS: Final[int] = 256
MAX_SEARCH_VISITED: Final[int] = 200_000
MAX_SEARCH_COLLECTED: Final[int] = 10_000
MAX_SUBTREE_ENTRIES: Final[int] = 50_000

CONFIG_ENV_VAR: Final[str] = "SLAVIK_HOST_BRIDGE_CONFIG"
CONFIG_FILE_NAME: Final[str] = "host-bridge.json"
BUILTIN_WORKSPACE_ID: Final[str] = "workspace"

_ROOT_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9_-]{1,64}$")
_DRIVE_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z]:")
_TEMP_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^\.(?:.*\.)?hb-[0-9a-f]{32}\.tmp$")


def _staging_name() -> str:
    """Short unique staging name.

    The target name is deliberately not embedded: with
    NAME_MAX=255 a long target name plus a suffix would exceed
    the limit and make valid files unwritable (ENAMETOOLONG).
    """
    return f".hb-{uuid.uuid4().hex}.tmp"


_DIR_OPEN_FLAGS: Final[int] = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class CancelToken:
    """Cooperative cancellation for one dispatched request.

    The daemon cancels the token when the per-request deadline fires.
    Handlers check it before any publish syscall and periodically in
    long walks, so a cancelled mutation never commits late.
    """

    def __init__(self, *, deadline: float | None = None, request_id: str | None = None) -> None:
        self._event = threading.Event()
        self._deadline = deadline
        self.request_id = request_id

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        if self._event.is_set():
            return True
        return self._deadline is not None and time.monotonic() >= self._deadline

    def check(self) -> None:
        if self.cancelled:
            raise RequestTimeoutError(request_id=self.request_id)


class FsError(BridgeProtocolError):
    """Typed filesystem failure; ``code`` is the wire error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message)


def _invalid_params(message: str) -> FsError:
    return FsError("invalid_params", message)


def _io_error(exc: OSError) -> FsError:
    name = errno.errorcode.get(exc.errno or 0, "EIO")
    return FsError("io_error", f"filesystem operation failed: {name}")


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


# renameat2 syscall number (x86_64) and RENAME_NOREPLACE flag. The
# bridge daemon runs on the owner's Linux host; if the syscall is
# unavailable the move fails typed instead of silently degrading to a
# racy stat+rename.
_SYS_RENAMEAT2: Final[int] = 316
_RENAME_NOREPLACE: Final[int] = 1


def _rename_noreplace(src: str, src_dir_fd: int, dst: str, dst_dir_fd: int) -> None:
    """Atomically rename without replacing an existing destination.

    Raises ``OSError`` with ``EEXIST`` when the destination exists at
    syscall time (a destination created concurrently is never
    destroyed), ``EXDEV`` across filesystems, etc.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(
        ctypes.c_long(_SYS_RENAMEAT2),
        ctypes.c_int(src_dir_fd),
        ctypes.c_char_p(os.fsencode(src)),
        ctypes.c_int(dst_dir_fd),
        ctypes.c_char_p(os.fsencode(dst)),
        ctypes.c_uint(_RENAME_NOREPLACE),
    )
    if result != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


@dataclass(frozen=True)
class Root:
    """One owner-approved filesystem root."""

    id: str
    path: Path  # canonical absolute path
    mode: str  # "ro" | "rw"
    builtin_workspace: bool = False


@dataclass(frozen=True)
class RootsRegistry:
    """Validated roots + write-exclusions the daemon enforces."""

    roots: tuple[Root, ...]
    write_excluded: tuple[Path, ...]  # canonical prefixes from config
    home: Path
    config_path: Path | None = None
    socket_dir: Path | None = None

    def by_id(self, root_id: str) -> Root | None:
        for root in self.roots:
            if root.id == root_id:
                return root
        return None

    def describe(self) -> list[dict[str, object]]:
        """Root ids + modes for ``bridge.capabilities`` (no host paths)."""
        return [{"id": root.id, "mode": root.mode} for root in self.roots]


def builtin_workspace_path(home: Path) -> Path:
    return home / ".local" / "share" / "slavikai" / "host-workspace"


def _expand_home(raw: str, home: Path) -> Path:
    if raw == "~":
        return home
    if raw.startswith("~/"):
        return home / raw[2:]
    return Path(raw)


def _canonical(raw: str, home: Path) -> Path:
    return _expand_home(raw, home).resolve()


def _never_root_prefixes(home: Path) -> tuple[Path, ...]:
    return (
        home / ".ssh",
        home / ".gnupg",
        home / ".config",
        home / ".local" / "state" / "slavikai",
        Path("/etc"),
        Path("/usr"),
        Path("/root"),
    )


def _validate_root_path(canonical: Path, *, home: Path, builtin: Path) -> None:
    """Enforce the ADR §9 root allowlist on a canonicalized root path."""
    if canonical == Path("/") or canonical == home:
        raise BridgeConfigError(f"root must not be the filesystem or home root: {canonical}")
    for prefix in _never_root_prefixes(home):
        if canonical == prefix or canonical.is_relative_to(prefix):
            raise BridgeConfigError(f"root is inside a never-root prefix: {canonical}")
    if canonical == builtin:
        return
    for parent_name in ("Project", "Documents", "Downloads"):
        parent = home / parent_name
        if canonical != parent and canonical.is_relative_to(parent):
            return
    raise BridgeConfigError(
        "root is outside the ADR-0013 allowlist (built-in workspace or a "
        f"specific subdirectory of ~/Project, ~/Documents, ~/Downloads): {canonical}"
    )


def resolve_config_path(env: Mapping[str, str] | None = None, *, home: Path | None = None) -> Path:
    """Resolve the host-bridge config path (env override, else XDG config)."""
    source = os.environ if env is None else env
    override = source.get(CONFIG_ENV_VAR, "").strip()
    if override:
        base_home = (home or Path.home()).expanduser()
        return _expand_home(override, base_home)
    xdg = source.get("XDG_CONFIG_HOME", "").strip()
    if xdg:
        base = _expand_home(xdg, (home or Path.home()).expanduser())
    else:
        base = (home or Path.home()).expanduser() / ".config"
    return base / "slavikai" / CONFIG_FILE_NAME


def default_registry(*, home: Path | None = None, socket_dir: Path | None = None) -> RootsRegistry:
    """Registry with only the built-in workspace root (no config read)."""
    resolved_home = (home or Path.home()).expanduser().resolve()
    workspace = builtin_workspace_path(resolved_home)
    return RootsRegistry(
        roots=(Root(id=BUILTIN_WORKSPACE_ID, path=workspace, mode="rw", builtin_workspace=True),),
        write_excluded=(),
        home=resolved_home,
        config_path=None,
        socket_dir=socket_dir,
    )


def load_registry(
    env: Mapping[str, str] | None = None,
    *,
    home: Path | None = None,
    socket_dir: Path | None = None,
) -> RootsRegistry:
    """Load and validate the roots config; absent file → built-in default.

    Fail-closed: any parse or validation error raises
    ``BridgeConfigError`` and the daemon refuses to start.
    """
    resolved_home = (home or Path.home()).expanduser().resolve()
    config_path = resolve_config_path(env, home=resolved_home)
    if not config_path.is_file():
        registry = default_registry(home=resolved_home, socket_dir=socket_dir)
        return RootsRegistry(
            roots=registry.roots,
            write_excluded=registry.write_excluded,
            home=registry.home,
            config_path=config_path,
            socket_dir=socket_dir,
        )
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeConfigError(f"cannot read host bridge config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BridgeConfigError(f"host bridge config must be a JSON object: {config_path}")
    unknown = sorted(set(payload) - {"roots", "write_excluded"})
    if unknown:
        raise BridgeConfigError(f"host bridge config has unknown field(s): {', '.join(unknown)}")
    raw_roots = payload.get("roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise BridgeConfigError("host bridge config must list at least one root")
    builtin = builtin_workspace_path(resolved_home)
    roots: list[Root] = []
    seen_ids: set[str] = set()
    for entry in raw_roots:
        if not isinstance(entry, dict) or set(entry) - {"id", "path", "mode"}:
            raise BridgeConfigError(f"invalid root entry in host bridge config: {entry!r}")
        root_id = entry.get("id")
        raw_path = entry.get("path")
        mode = entry.get("mode")
        if not isinstance(root_id, str) or not _ROOT_ID_RE.match(root_id):
            raise BridgeConfigError(f"invalid root id in host bridge config: {root_id!r}")
        if root_id in seen_ids:
            raise BridgeConfigError(f"duplicate root id in host bridge config: {root_id!r}")
        seen_ids.add(root_id)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise BridgeConfigError(f"root {root_id!r} has no path")
        if mode not in ("ro", "rw"):
            raise BridgeConfigError(f"root {root_id!r} has invalid mode: {mode!r}")
        canonical = _canonical(raw_path.strip(), resolved_home)
        _validate_root_path(canonical, home=resolved_home, builtin=builtin)
        is_builtin = canonical == builtin
        if is_builtin:
            canonical.mkdir(parents=True, exist_ok=True)
        if not canonical.is_dir():
            raise BridgeConfigError(f"root {root_id!r} is not an existing directory: {canonical}")
        if config_path.resolve().is_relative_to(canonical):
            raise BridgeConfigError(f"root {root_id!r} contains the bridge config file")
        if socket_dir is not None and socket_dir.resolve().is_relative_to(canonical):
            raise BridgeConfigError(f"root {root_id!r} contains the bridge socket directory")
        roots.append(Root(id=root_id, path=canonical, mode=mode, builtin_workspace=is_builtin))
    for i, first in enumerate(roots):
        for second in roots[i + 1 :]:
            if first.path.is_relative_to(second.path) or second.path.is_relative_to(first.path):
                raise BridgeConfigError(
                    f"roots {first.id!r} and {second.id!r} overlap; nested roots are rejected"
                )
    raw_excluded = payload.get("write_excluded", [])
    if not isinstance(raw_excluded, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_excluded
    ):
        raise BridgeConfigError("host bridge config write_excluded must be a list of paths")
    write_excluded = tuple(_canonical(item.strip(), resolved_home) for item in raw_excluded)
    return RootsRegistry(
        roots=tuple(roots),
        write_excluded=write_excluded,
        home=resolved_home,
        config_path=config_path,
        socket_dir=socket_dir,
    )


# --- per-capability params validation (T03 validates the envelope only) ---


def _check_params(params: dict[str, object], allowed: frozenset[str]) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        shown = ", ".join(name[:64] for name in unknown[:8])
        raise _invalid_params(f"unknown params field(s): {shown}")


def _str_param(params: dict[str, object], name: str, *, required: bool = True) -> str | None:
    value = params.get(name)
    if value is None:
        if required:
            raise _invalid_params(f"params.{name} is required")
        return None
    if not isinstance(value, str):
        raise _invalid_params(f"params.{name} must be a string")
    return value


def _int_param(
    params: dict[str, object],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = params.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_params(f"params.{name} must be an integer")
    if not minimum <= value <= maximum:
        raise _invalid_params(f"params.{name} must be in {minimum}..{maximum}")
    return value


def _bool_param(params: dict[str, object], name: str, *, default: bool) -> bool:
    value = params.get(name, default)
    if not isinstance(value, bool):
        raise _invalid_params(f"params.{name} must be a boolean")
    return value


def _parse_rel_path(raw: str, *, allow_root: bool) -> tuple[str, ...]:
    """Validate a model-supplied relative path; return its components.

    ``()`` means the root itself (only when ``allow_root``). Absolute
    paths, ``~``, drive/UNC forms, NUL, empty components and ``..``
    are rejected before any filesystem touch.
    """
    if not raw or "\x00" in raw:
        raise FsError("invalid_path", "path is empty or contains NUL")
    if raw == ".":
        if allow_root:
            return ()
        raise FsError("invalid_path", "path must name an entry below the root")
    if raw.startswith(("/", "~", "\\")) or _DRIVE_RE.match(raw):
        raise FsError("invalid_path", "path must be relative to the root")
    parts = raw.split("/")
    for part in parts:
        if part in ("", ".", ".."):
            raise FsError("invalid_path", f"path contains a forbidden component: {part!r}")
    return tuple(parts)


@dataclass(frozen=True)
class _Target:
    root: Root
    parts: tuple[str, ...]  # () = root itself
    lexical: Path  # canonical lexical absolute path (root.path / parts)
    display: str  # "." or "a/b/c" (echoed in results/errors)


@dataclass
class _WriteSession:
    write_id: str
    root_id: str
    dir_parts: tuple[str, ...]
    target_name: str
    temp_name: str
    mode: str  # "create" | "overwrite"
    size: int = 0
    last_touch: float = 0.0
    # Identity of the staging file recorded at creation. Every later
    # open re-verifies it (with O_NOFOLLOW), so replacing the staging
    # name with a symlink or another file cannot redirect the write.
    temp_dev: int = 0
    temp_ino: int = 0
    # Serializes chunk/commit/abort for this session so a delayed
    # pwrite can never land after commit published the inode.
    lock: threading.Lock = field(default_factory=threading.Lock)
    finished: bool = False


def _type_name(mode: int) -> str:
    if stat_mod.S_ISREG(mode):
        return "file"
    if stat_mod.S_ISDIR(mode):
        return "directory"
    if stat_mod.S_ISLNK(mode):
        return "symlink"
    return "other"


def _iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime).astimezone().isoformat()


def _entry_dict(name: str, st: os.stat_result) -> dict[str, object]:
    return {
        "name": name,
        "type": _type_name(st.st_mode),
        "size": st.st_size,
        "modified_at": _iso(st.st_mtime),
    }


class HostFsService:
    """Daemon-side ``host.fs.*`` capability handlers (E03-T04)."""

    def __init__(
        self,
        registry: RootsRegistry,
        *,
        security: DesktopPathSecurity | None = None,
        trash_root: Path | None = None,
        mount_bindings_fn: Callable[[], list[tuple[Path, Path]]] = read_mount_bindings,
        commit_hook: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._registry = registry
        self._security = security or DesktopPathSecurity(home=registry.home)
        self._trash_root = trash_root
        self._mount_bindings_fn = mount_bindings_fn
        self._commit_hook = commit_hook
        self._clock = clock
        self._sessions: dict[str, _WriteSession] = {}
        self._sessions_lock = threading.Lock()
        self._checkout_cache: dict[Path, bool] = {}
        self._builtin_excluded = self._compute_builtin_exclusions()
        self._handlers: dict[
            str, Callable[[dict[str, object], str, CancelToken], dict[str, object]]
        ] = {
            CAPABILITY_FS_STAT: self._stat,
            CAPABILITY_FS_LIST: self._list,
            CAPABILITY_FS_READ: self._read,
            CAPABILITY_FS_WRITE: self._write,
            CAPABILITY_FS_PATCH: self._patch,
            CAPABILITY_FS_MOVE: self._move,
            CAPABILITY_FS_DELETE: self._delete,
            CAPABILITY_FS_SEARCH: self._search,
        }

    @classmethod
    def default(cls, *, home: Path | None = None, socket_dir: Path | None = None) -> HostFsService:
        """Service with only the built-in workspace root (no config read)."""
        return cls(default_registry(home=home, socket_dir=socket_dir))

    def roots_description(self) -> list[dict[str, object]]:
        return self._registry.describe()

    def handle(self, request: BridgeRequest, token: CancelToken) -> dict[str, object]:
        handler = self._handlers.get(request.capability)
        if handler is None:
            raise UnknownCapabilityError(request.capability, request_id=request.request_id)
        try:
            return handler(request.params, request.request_id, token)
        except FsError as exc:
            if exc.request_id is None:
                exc.request_id = request.request_id
            raise
        except OSError as exc:
            raise self._map_os_error(exc, request.request_id) from exc

    # --- built-in write exclusions & checkout detection ---

    def _compute_builtin_exclusions(self) -> tuple[Path, ...]:
        home = self._registry.home
        excluded: list[Path] = [
            home / ".local" / "state" / "slavikai",  # runtime state (ADR §9)
            home / ".config" / "slavikai",  # bridge config dir (launch files)
            home / ".config" / "systemd" / "user" / "slavikai-host-bridge.service",
            Path(__file__).resolve().parents[1],  # the daemon's own checkout tree
        ]
        if self._registry.config_path is not None:
            excluded.append(self._registry.config_path.resolve())
        if self._registry.socket_dir is not None:
            excluded.append(self._registry.socket_dir.resolve())
        return tuple(dict.fromkeys(excluded))

    def _is_slavikai_checkout(self, directory: Path) -> bool:
        # Only positive results are cached: a directory that was not a
        # checkout at first check may become one later (clone/restore)
        # and must be write-excluded immediately, without a daemon
        # restart. A cached positive stays excluded (fail-closed).
        if self._checkout_cache.get(directory):
            return True
        result = (directory / "host_bridge" / "daemon.py").is_file() and (
            directory / "docs" / "runtime_contract_claims.json"
        ).is_file()
        if result:
            self._checkout_cache[directory] = True
        return result

    def _check_write_excluded(self, target: _Target) -> None:
        lexical = target.lexical
        for prefix in (*self._registry.write_excluded, *self._builtin_excluded):
            if lexical == prefix or lexical.is_relative_to(prefix):
                raise FsError("write_excluded", f"path is write-excluded: {target.display}")
        # Structural checkout detection: any directory on the path
        # from the filesystem root down to the target that looks like
        # a SlavikAI checkout write-excludes its whole subtree, even
        # when the config forgot to list it (owner correction
        # 2026-11-23). Ancestors of the root are checked too: a root
        # configured inside another checkout (e.g.
        # ~/Project/clone/host_bridge) must not become a write path
        # into that clone. Per ADR-0013 §9 checkouts are
        # write-excluded even inside an allowed root, so the root is
        # excluded for mutations here rather than rejected at load
        # time (reads stay allowed).
        candidates = list(reversed(target.root.path.parents))
        current = target.root.path
        candidates.append(current)
        for part in target.parts:
            current = current / part
            candidates.append(current)
        for candidate in candidates:
            if self._is_slavikai_checkout(candidate):
                raise FsError(
                    "write_excluded",
                    f"path is inside a SlavikAI checkout: {target.display}",
                )

    # --- lexical resolution & classification ---

    def _classify(
        self, lexical: Path, *, mutation: bool, includes_descendants: bool = False
    ) -> str:
        try:
            resolved = self._security.resolve(
                str(lexical),
                mutation=mutation,
                includes_descendants=includes_descendants,
            )
        except (ValueError, RuntimeError) as exc:
            raise FsError("path_outside_root", "path cannot be resolved safely") from exc
        return resolved.protection

    def _require_normal(
        self, lexical: Path, *, mutation: bool, includes_descendants: bool = False
    ) -> None:
        if (
            self._classify(lexical, mutation=mutation, includes_descendants=includes_descendants)
            != "normal"
        ):
            raise FsError("protected_path", "path is protected by host policy")

    def _resolve(
        self,
        root_id_raw: object,
        path_raw: object,
        *,
        mutation: bool,
        allow_root: bool,
    ) -> _Target:
        if not isinstance(root_id_raw, str) or not root_id_raw:
            raise _invalid_params("params.root_id must be a non-empty string")
        root = self._registry.by_id(root_id_raw)
        if root is None:
            raise FsError("unknown_root", f"unknown root: {root_id_raw[:64]!r}")
        if mutation and root.mode != "rw":
            raise FsError("root_read_only", f"root {root.id!r} is read-only")
        if not isinstance(path_raw, str):
            raise _invalid_params("params.path must be a string")
        parts = _parse_rel_path(path_raw, allow_root=allow_root)
        lexical = root.path.joinpath(*parts) if parts else root.path
        if lexical != root.path and not lexical.is_relative_to(root.path):
            raise FsError("path_outside_root", "path escapes the root")
        display = "/".join(parts) if parts else "."
        target = _Target(root=root, parts=parts, lexical=lexical, display=display)
        if mutation:
            self._check_write_excluded(target)
        self._require_normal(lexical, mutation=mutation)
        return target

    # --- fd-relative walking (operation-time enforcement) ---

    def _mount_points(self) -> frozenset[Path]:
        try:
            return frozenset(mount for mount, _ in self._mount_bindings_fn())
        except (OSError, ValueError) as exc:
            raise FsError("io_error", "mount topology unavailable") from exc

    def _open_root(self, root: Root) -> int:
        if root.builtin_workspace:
            root.path.mkdir(parents=True, exist_ok=True)
        fd = os.open("/", _DIR_OPEN_FLAGS)
        try:
            for part in root.path.parts[1:]:
                nfd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=fd)
                os.close(fd)
                fd = nfd
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _open_dirs(self, root: Root, parts: Sequence[str], *, create: bool = False) -> int:
        """Walk directory components below ``root``; return the final dir fd.

        Every component is opened relative to the previously pinned
        directory fd with ``O_NOFOLLOW``: symlink traversal fails with
        ``ELOOP`` and nested mounts are rejected (``st_dev`` change or
        a mount-point component). Containment therefore holds at
        operation time, not only at validation time.
        """
        fd = self._open_root(root)
        try:
            root_dev = os.fstat(fd).st_dev
            mount_points = self._mount_points()
            lexical = root.path
            for part in parts:
                lexical = lexical / part
                if create:
                    try:
                        os.mkdir(part, 0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                try:
                    nfd = os.open(part, _DIR_OPEN_FLAGS, dir_fd=fd)
                except OSError as exc:
                    # O_DIRECTORY|O_NOFOLLOW on a symlink reports ENOTDIR
                    # (not ELOOP) on Linux; name the actual refusal so a
                    # symlink escape is not masked as "not a directory".
                    if exc.errno == errno.ENOTDIR:
                        try:
                            st = os.stat(part, dir_fd=fd, follow_symlinks=False)
                        except OSError:
                            st = None
                        if st is not None and stat_mod.S_ISLNK(st.st_mode):
                            raise FsError(
                                "path_outside_root",
                                "symlink traversal is forbidden in v1",
                            ) from exc
                    raise
                st = os.fstat(nfd)
                if st.st_dev != root_dev or lexical in mount_points:
                    os.close(nfd)
                    raise FsError("path_outside_root", "nested mount traversal is forbidden in v1")
                os.close(fd)
                fd = nfd
        except BaseException:
            os.close(fd)
            raise
        return fd

    def _open_parent(self, target: _Target, *, create: bool = False) -> tuple[int, str]:
        if not target.parts:
            raise FsError("invalid_path", "path must name an entry below the root")
        fd = self._open_dirs(target.root, target.parts[:-1], create=create)
        return fd, target.parts[-1]

    @staticmethod
    def _map_os_error(exc: OSError, request_id: str | None = None) -> FsError:
        err = exc.errno
        if err == errno.ENOENT:
            error = FsError("path_not_found", "path does not exist")
        elif err == errno.ENOTDIR:
            error = FsError("not_a_directory", "path component is not a directory")
        elif err == errno.EISDIR:
            error = FsError("is_a_directory", "path is a directory")
        elif err in (errno.EEXIST, errno.ENOTEMPTY):
            error = FsError("already_exists", "target already exists")
        elif err == errno.ELOOP:
            error = FsError("path_outside_root", "symlink traversal is forbidden in v1")
        elif err == errno.EXDEV:
            error = FsError("cross_device_move", "move across filesystems is not supported")
        else:
            error = _io_error(exc)
        error.request_id = request_id
        return error

    def _check_symlink_target(self, target: _Target, link: str) -> None:
        if os.path.isabs(link):
            dest = Path(os.path.normpath(link))
        else:
            dest = Path(os.path.normpath(str(target.lexical.parent / link)))
        if dest != target.root.path and not dest.is_relative_to(target.root.path):
            raise FsError("path_outside_root", "symlink target escapes the root")

    # --- frame budgeting on serialized JSON ---

    @staticmethod
    def _frame_fits(result: dict[str, object], request_id: str) -> bool:
        return len(encode_success(result, request_id)) - 1 <= MAX_FRAME_BYTES

    # --- host.fs.stat ---

    def _stat(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"root_id", "path"}))
        target = self._resolve(
            params.get("root_id"), params.get("path"), mutation=False, allow_root=True
        )
        if not target.parts:
            fd = self._open_root(target.root)
            try:
                st = os.fstat(fd)
            finally:
                os.close(fd)
            return {
                "root_id": target.root.id,
                "path": target.display,
                "type": "directory",
                "size": st.st_size,
                "modified_at": _iso(st.st_mtime),
            }
        pfd, name = self._open_parent(target)
        try:
            st = os.stat(name, dir_fd=pfd, follow_symlinks=False)
            if stat_mod.S_ISLNK(st.st_mode):
                self._check_symlink_target(target, os.readlink(name, dir_fd=pfd))
        finally:
            os.close(pfd)
        return {
            "root_id": target.root.id,
            "path": target.display,
            "type": _type_name(st.st_mode),
            "size": st.st_size,
            "modified_at": _iso(st.st_mtime),
        }

    # --- host.fs.list ---

    def _list(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"root_id", "path", "limit", "cursor"}))
        target = self._resolve(
            params.get("root_id"),
            params.get("path", "."),
            mutation=False,
            allow_root=True,
        )
        limit = _int_param(
            params, "limit", default=DEFAULT_LIST_LIMIT, minimum=1, maximum=MAX_LIST_LIMIT
        )
        cursor = _str_param(params, "cursor", required=False)
        fd = self._open_dirs(target.root, target.parts)
        try:
            names = sorted(os.listdir(fd))
            entries: list[dict[str, object]] = []
            truncated = False
            started = cursor is None
            for name in names:
                if not started:
                    if name == cursor:
                        started = True
                    continue
                if len(entries) >= limit:
                    truncated = True
                    break
                token.check()
                if self._classify(target.lexical / name, mutation=False) != "normal":
                    continue  # denied entries are omitted (no existence oracle)
                try:
                    st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue  # vanished mid-listing
                entries.append(_entry_dict(name, st))
        finally:
            os.close(fd)
        result: dict[str, object] = {
            "root_id": target.root.id,
            "path": target.display,
            "entries": entries,
            "truncated": truncated,
            "next_cursor": None,
        }

        def _sync_cursor() -> None:
            # The cursor resumes after the last kept entry. It is part
            # of the serialized frame, so it must be set BEFORE the
            # size check and recomputed whenever entries are trimmed.
            result["next_cursor"] = entries[-1]["name"] if result["truncated"] and entries else None

        _sync_cursor()
        # Frame budget: drop trailing entries until the serialized
        # frame (including the actual cursor) fits.
        while entries and not self._frame_fits(result, request_id):
            entries.pop()
            result["truncated"] = True
            _sync_cursor()
        return result

    # --- host.fs.read ---

    def _read(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"root_id", "path", "offset", "length"}))
        target = self._resolve(
            params.get("root_id"), params.get("path"), mutation=False, allow_root=False
        )
        offset = _int_param(params, "offset", default=0, minimum=0, maximum=2**63 - 1)
        length = _int_param(
            params, "length", default=MAX_FS_CHUNK_BYTES, minimum=1, maximum=MAX_FS_CHUNK_BYTES
        )
        if target.lexical in self._mount_points():
            raise FsError("path_outside_root", "nested mount traversal is forbidden in v1")
        pfd, name = self._open_parent(target)
        try:
            # O_NONBLOCK: opening a FIFO without a writer must not
            # block the worker past the deadline; the fstat type
            # check below refuses non-regular files typed.
            fd = os.open(
                name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=pfd
            )
        finally:
            os.close(pfd)
        try:
            st = os.fstat(fd)
            if stat_mod.S_ISDIR(st.st_mode):
                raise FsError("is_a_directory", f"path is a directory: {target.display}")
            if not stat_mod.S_ISREG(st.st_mode):
                raise FsError("not_a_file", f"path is not a regular file: {target.display}")
            size = st.st_size
            if offset > size:
                raise FsError("offset_out_of_range", "offset is beyond end of file")
            raw = os.pread(fd, length, offset) if offset < size else b""
        finally:
            os.close(fd)
        if b"\x00" in raw:
            raise FsError("binary_file", "binary file cannot be returned as text")
        text, consumed = self._decode_chunk(raw, at_eof=offset + len(raw) >= size)
        if raw and consumed == 0:
            # Data exists but no full character fits in `length` bytes:
            # returning zero bytes would loop the client forever.
            raise _invalid_params("length is too small for the next character")
        result: dict[str, object] = {
            "root_id": target.root.id,
            "path": target.display,
            "content": text,
            "offset": offset,
            "bytes_read": consumed,
            "size": size,
            "truncated": offset + consumed < size,
            "encoding": "utf-8",
        }
        # Frame budget on serialized JSON (escaping counts): shrink
        # the content by whole characters until the frame fits. One
        # character always fits, so progress is guaranteed.
        while text and not self._frame_fits(result, request_id):
            text = text[: max(1, len(text) // 2)] if len(text) > 1 else ""
            consumed = len(text.encode("utf-8"))
            result["content"] = text
            result["bytes_read"] = consumed
            result["truncated"] = offset + consumed < size
        return result

    @staticmethod
    def _decode_chunk(raw: bytes, *, at_eof: bool) -> tuple[str, int]:
        """Decode a chunk, holding back a split trailing codepoint.

        Returns (text, bytes_consumed). A chunk that is not valid
        UTF-8 (including an offset inside a multibyte character)
        raises ``invalid_utf8``.
        """
        data = raw
        while data:
            try:
                return data.decode("utf-8"), len(data)
            except UnicodeDecodeError as exc:
                if not at_eof and exc.reason == "unexpected end of data" and exc.end == len(data):
                    data = data[: exc.start]
                    continue
                raise FsError("invalid_utf8", "file chunk is not valid UTF-8") from exc
        return "", 0

    # --- host.fs.write (chunked temp + single commit) ---

    def _sweep_sessions(self) -> None:
        now = self._clock()
        with self._sessions_lock:
            expired = [
                wid
                for wid, session in self._sessions.items()
                if now - session.last_touch > WRITE_SESSION_TTL_S
            ]
            sessions = [self._sessions.pop(wid) for wid in expired]
        for session in sessions:
            # Never discard a staging file out from under an
            # in-flight chunk/commit; a busy session stays
            # registered and is retried on the next sweep.
            if not session.lock.acquire(blocking=False):
                with self._sessions_lock:
                    self._sessions.setdefault(session.write_id, session)
                continue
            try:
                if not session.finished:
                    session.finished = True
                    self._discard_session_temp(session)
            finally:
                session.lock.release()

    def _discard_session_temp(self, session: _WriteSession) -> None:
        root = self._registry.by_id(session.root_id)
        if root is None:
            return
        try:
            fd = self._open_dirs(root, session.dir_parts)
            try:
                os.unlink(session.temp_name, dir_fd=fd)
            finally:
                os.close(fd)
        except (FsError, OSError):
            pass  # best-effort cleanup

    def _sweep_orphan_temps(self, dir_fd: int, now: float) -> None:
        try:
            names = os.listdir(dir_fd)
        except OSError:
            return
        for name in names:
            if not _TEMP_NAME_RE.match(name):
                continue
            try:
                st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                if now - st.st_mtime > WRITE_SESSION_TTL_S:
                    os.unlink(name, dir_fd=dir_fd)
            except OSError:
                pass

    def _open_staging(self, dir_fd: int, session: _WriteSession, flags: int) -> int:
        """Open the session's staging file without following symlinks.

        ``O_NOFOLLOW`` refuses a symlink swapped in under the staging
        name (``ELOOP`` → ``path_outside_root``), and the recorded
        ``st_dev``+``st_ino`` identity refuses any other replacement:
        the write can only ever touch the file this session created.
        """
        fd = os.open(
            session.temp_name, flags | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd
        )
        try:
            st = os.fstat(fd)
            if not stat_mod.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (
                session.temp_dev,
                session.temp_ino,
            ):
                raise FsError("path_outside_root", "staging file identity changed")
        except BaseException:
            os.close(fd)
            raise
        return fd

    @staticmethod
    def _verify_staging_name(dir_fd: int, session: _WriteSession) -> None:
        """Verify the staging name still denotes the session's file."""
        st = os.stat(session.temp_name, dir_fd=dir_fd, follow_symlinks=False)
        if not stat_mod.S_ISREG(st.st_mode) or (st.st_dev, st.st_ino) != (
            session.temp_dev,
            session.temp_ino,
        ):
            raise FsError("path_outside_root", "staging file identity changed")

    def _write(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(
            params,
            frozenset(
                {
                    "root_id",
                    "path",
                    "mode",
                    "content",
                    "offset",
                    "write_id",
                    "commit",
                    "abort",
                    "create_parents",
                }
            ),
        )
        target = self._resolve(
            params.get("root_id"), params.get("path"), mutation=True, allow_root=False
        )
        mode = params.get("mode", "create")
        if mode not in ("create", "overwrite"):
            raise _invalid_params("params.mode must be 'create' or 'overwrite'")
        content = params.get("content", "")
        if not isinstance(content, str):
            raise _invalid_params("params.content must be a string")
        data = content.encode("utf-8")
        if len(data) > MAX_FS_CHUNK_BYTES:
            raise FsError("content_too_large", "chunk exceeds the maximum chunk size")
        offset = _int_param(params, "offset", default=0, minimum=0, maximum=MAX_WRITE_BYTES)
        commit = _bool_param(params, "commit", default=False)
        abort = _bool_param(params, "abort", default=False)
        create_parents = _bool_param(params, "create_parents", default=True)
        write_id = _str_param(params, "write_id", required=False)
        if abort and commit:
            raise _invalid_params("params.abort and params.commit are mutually exclusive")
        self._sweep_sessions()

        if abort:
            if write_id is None:
                raise _invalid_params("params.write_id is required to abort")
            with self._sessions_lock:
                session = self._sessions.pop(write_id, None)
            if session is None:
                raise _invalid_params("unknown write_id")
            # Serialize with any in-flight chunk/commit on this
            # session: the staging file is discarded only once no
            # other request can still write or publish it.
            with session.lock:
                if session.finished:
                    raise _invalid_params("write_id already finished")
                session.finished = True
                self._discard_session_temp(session)
            return {
                "root_id": target.root.id,
                "path": target.display,
                "size": 0,
                "bytes_written": 0,
                "aborted": True,
            }

        if write_id is None:
            if offset != 0:
                raise _invalid_params("params.write_id is required when offset > 0")
            write_id = uuid.uuid4().hex
            session = _WriteSession(
                write_id=write_id,
                root_id=target.root.id,
                dir_parts=target.parts[:-1],
                target_name=target.parts[-1],
                temp_name=_staging_name(),
                mode=str(mode),
                last_touch=self._clock(),
            )
            with self._sessions_lock:
                self._sessions[write_id] = session
            try:
                fd = self._open_dirs(target.root, session.dir_parts, create=create_parents)
                try:
                    self._sweep_orphan_temps(fd, time.time())
                    tfd = os.open(
                        session.temp_name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                        0o600,
                        dir_fd=fd,
                    )
                    try:
                        tst = os.fstat(tfd)
                        session.temp_dev = tst.st_dev
                        session.temp_ino = tst.st_ino
                    finally:
                        os.close(tfd)
                finally:
                    os.close(fd)
            except BaseException:
                with self._sessions_lock:
                    self._sessions.pop(write_id, None)
                raise
        else:
            with self._sessions_lock:
                session = self._sessions.get(write_id)
            if session is None:
                raise _invalid_params("unknown write_id")
            if (
                session.root_id != target.root.id
                or session.dir_parts != target.parts[:-1]
                or session.target_name != target.parts[-1]
            ):
                raise _invalid_params("write_id does not match root_id/path")

        # Serialize chunk/commit on this session: a delayed
        # pwrite must never land after commit published the
        # inode, and two commits must not race the same staging
        # file. State is re-checked under the lock: an abort or
        # expiry that won the race makes this request fail typed
        # instead of touching a discarded staging file.
        with session.lock:
            if session.finished:
                raise _invalid_params("unknown write_id")
            fd = self._open_dirs(target.root, session.dir_parts)
            try:
                if data:
                    if offset > session.size:
                        raise _invalid_params("offset leaves a gap in the write session")
                    if offset + len(data) > MAX_WRITE_BYTES:
                        raise FsError("content_too_large", "write session exceeds size limit")
                    tfd = self._open_staging(fd, session, os.O_WRONLY)
                    try:
                        os.pwrite(tfd, data, offset)
                    finally:
                        os.close(tfd)
                    session.size = max(session.size, offset + len(data))
                    session.last_touch = self._clock()
                elif offset > session.size:
                    raise _invalid_params("offset leaves a gap in the write session")
                if not commit:
                    return {
                        "root_id": target.root.id,
                        "path": target.display,
                        "size": session.size,
                        "bytes_written": len(data),
                        "write_id": write_id,
                    }
                # Commit: fsync the temp, then the cancellation gate, then
                # exactly one publish syscall. After the gate passes, a
                # lost response means outcome-unknown (documented); the
                # publish itself is atomic either way.
                tfd = self._open_staging(fd, session, os.O_RDONLY)
                try:
                    os.fsync(tfd)
                finally:
                    os.close(tfd)
                if self._commit_hook is not None:
                    self._commit_hook()
                token.check()
                # The publish syscalls address the staging file by name;
                # re-verify its identity immediately before publishing so
                # a swapped-in symlink/file is never published.
                self._verify_staging_name(fd, session)
                if session.mode == "create":
                    # linkat fails with EEXIST if the target appeared:
                    # absence is verified atomically, no exists()+replace race.
                    try:
                        os.link(
                            session.temp_name,
                            session.target_name,
                            src_dir_fd=fd,
                            dst_dir_fd=fd,
                        )
                    except FileExistsError as exc:
                        raise FsError(
                            "already_exists", f"target already exists: {target.display}"
                        ) from exc
                    os.unlink(session.temp_name, dir_fd=fd)
                else:
                    os.rename(session.temp_name, session.target_name, src_dir_fd=fd, dst_dir_fd=fd)
                session.finished = True
                with self._sessions_lock:
                    self._sessions.pop(write_id, None)
                return {
                    "root_id": target.root.id,
                    "path": target.display,
                    "size": session.size,
                    "bytes_written": len(data),
                    "committed": True,
                }
            finally:
                os.close(fd)

    # --- host.fs.patch ---

    def _patch(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"root_id", "path", "patch", "dry_run"}))
        target = self._resolve(
            params.get("root_id"), params.get("path"), mutation=True, allow_root=False
        )
        patch_text = _str_param(params, "patch")
        assert patch_text is not None
        dry_run = _bool_param(params, "dry_run", default=False)
        violation = validate_patch_contract(patch_text)
        if violation is not None:
            raise FsError("invalid_patch", violation[:512])
        if target.lexical in self._mount_points():
            raise FsError("path_outside_root", "nested mount traversal is forbidden in v1")
        pfd, name = self._open_parent(target)
        try:
            # O_NONBLOCK: a FIFO target must fail typed below, not
            # block the worker in open() past the deadline.
            fd = os.open(
                name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=pfd
            )
            try:
                st = os.fstat(fd)
                if stat_mod.S_ISDIR(st.st_mode):
                    raise FsError("is_a_directory", f"path is a directory: {target.display}")
                if not stat_mod.S_ISREG(st.st_mode):
                    raise FsError("not_a_file", f"path is not a regular file: {target.display}")
                if st.st_size > MAX_PATCH_FILE_BYTES:
                    raise FsError("file_too_large", "file exceeds the patch size limit")
                raw = os.pread(fd, st.st_size, 0)
            finally:
                os.close(fd)
            if b"\x00" in raw:
                raise FsError("binary_file", "binary file cannot be patched as text")
            try:
                current = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FsError("invalid_utf8", "file is not valid UTF-8") from exc
            try:
                updated = apply_unified_patch(current, patch_text)
            except (ValueError, IndexError) as exc:
                raise FsError("patch_mismatch", str(exc)[:512]) from exc
            if dry_run:
                return {
                    "root_id": target.root.id,
                    "path": target.display,
                    "applied": False,
                    "dry_run": True,
                    "size": len(updated.encode("utf-8")),
                }
            temp_name = _staging_name()
            tfd = os.open(
                temp_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                0o600,
                dir_fd=pfd,
            )
            try:
                _write_all(tfd, updated.encode("utf-8"))
                # Preserve the original file's permission bits: the
                # publish replaces the inode, so without this an
                # executable 0755 file would silently become 0600.
                os.fchmod(tfd, stat_mod.S_IMODE(st.st_mode))
                os.fsync(tfd)
            finally:
                os.close(tfd)
            try:
                if self._commit_hook is not None:
                    self._commit_hook()
                token.check()
                os.rename(temp_name, name, src_dir_fd=pfd, dst_dir_fd=pfd)
            except BaseException:
                try:
                    os.unlink(temp_name, dir_fd=pfd)
                except OSError:
                    pass
                raise
        finally:
            os.close(pfd)
        return {
            "root_id": target.root.id,
            "path": target.display,
            "applied": True,
            "dry_run": False,
            "size": len(updated.encode("utf-8")),
        }

    # --- subtree checks for move / recursive delete ---

    def _check_subtree(self, target: _Target, token: CancelToken) -> None:
        """Refuse when any descendant is protected, write-excluded or on
        another mount. Symlink entries are classified by their resolved
        path (conservative) and never descended.

        The walk holds at most one directory fd at a time: each
        directory is opened via ``_open_dirs`` (which re-enforces the
        fd-relative symlink/mount rules for the whole path), scanned,
        and closed before its children are visited, so the fd count
        does not grow with the number of directories in the tree.
        """
        mount_points = self._mount_points()
        seen = 0
        root_dev: int | None = None
        stack: list[tuple[str, ...]] = [target.parts]
        while stack:
            parts = stack.pop()
            fd = self._open_dirs(target.root, parts)
            try:
                if root_dev is None:
                    root_dev = os.fstat(fd).st_dev
                # Stat entries while the directory fd is open (DirEntry
                # stats are fd-relative for fd-based scandir).
                with os.scandir(fd) as it:
                    entries = [(entry.name, entry.stat(follow_symlinks=False)) for entry in it]
            finally:
                os.close(fd)
            current_lexical = target.root.path.joinpath(*parts) if parts else target.root.path
            for name, st in entries:
                token.check()
                seen += 1
                if seen > MAX_SUBTREE_ENTRIES:
                    raise FsError("subtree_too_large", "subtree exceeds the entry limit")
                lexical = current_lexical / name
                self._require_normal(lexical, mutation=True)
                probe = _Target(
                    root=target.root,
                    parts=lexical.relative_to(target.root.path).parts,
                    lexical=lexical,
                    display=target.display,
                )
                self._check_write_excluded(probe)
                if stat_mod.S_ISDIR(st.st_mode):
                    if st.st_dev != root_dev or lexical in mount_points:
                        raise FsError(
                            "path_outside_root",
                            "subtree crosses a nested mount",
                        )
                    stack.append((*parts, name))

    # --- host.fs.move (single rename) ---

    def _move(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(
            params,
            frozenset({"root_id", "path", "dest_root_id", "dest_path", "overwrite"}),
        )
        source = self._resolve(
            params.get("root_id"), params.get("path"), mutation=True, allow_root=False
        )
        dest = self._resolve(
            params.get("dest_root_id", params.get("root_id")),
            params.get("dest_path"),
            mutation=True,
            allow_root=False,
        )
        overwrite = _bool_param(params, "overwrite", default=False)
        if source.root.id == dest.root.id and source.parts == dest.parts:
            raise FsError("invalid_move", "source and destination are the same")
        if (
            source.root.id == dest.root.id
            and len(dest.parts) > len(source.parts)
            and dest.parts[: len(source.parts)] == source.parts
        ):
            raise FsError("invalid_move", "cannot move a directory into its own subtree")
        sfd, sname = self._open_parent(source)
        try:
            sst = os.stat(sname, dir_fd=sfd, follow_symlinks=False)
            if stat_mod.S_ISDIR(sst.st_mode):
                self._check_subtree(source, token)
            dfd, dname = self._open_parent(dest, create=True)
            try:
                token.check()
                try:
                    if overwrite:
                        os.rename(sname, dname, src_dir_fd=sfd, dst_dir_fd=dfd)
                    else:
                        # Atomic no-replace: a destination created
                        # concurrently (after any stat) is never
                        # destroyed; the syscall itself refuses.
                        _rename_noreplace(sname, sfd, dname, dfd)
                except OSError as exc:
                    if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                        raise FsError(
                            "already_exists",
                            f"destination exists: {dest.display}",
                        ) from exc
                    raise
            finally:
                os.close(dfd)
        finally:
            os.close(sfd)
        return {
            "root_id": source.root.id,
            "path": source.display,
            "dest_root_id": dest.root.id,
            "dest_path": dest.display,
            "moved": True,
        }

    # --- host.fs.delete (trash only) ---

    def _topdir_of(self, path: Path) -> Path | None:
        """Longest mountpoint containing ``path`` (freedesktop topdir)."""
        try:
            bindings = self._mount_bindings_fn()
        except (OSError, ValueError):
            return None
        best: Path | None = None
        for mount, _ in bindings:
            if path.is_relative_to(mount) and (best is None or len(mount.parts) > len(best.parts)):
                best = mount
        return best

    def _trash_dirs(self, source_lexical: Path, source_dev: int) -> tuple[Path, Path]:
        """Return (files_dir, info_dir) for the source's mount.

        Freedesktop trash: home trash when it shares the source's
        device, else the topdir ``.Trash/<uid>`` (only when a sticky
        ``.Trash`` exists) or ``.Trash-<uid>``. No trash on the same
        mount -> ``trash_unavailable`` and the source stays untouched.
        """
        uid = os.getuid()
        if self._trash_root is not None:
            files_dir = self._trash_root / "files"
            info_dir = self._trash_root / "info"
            files_dir.mkdir(parents=True, exist_ok=True)
            info_dir.mkdir(parents=True, exist_ok=True)
            return files_dir, info_dir
        home_trash = self._registry.home / ".local" / "share" / "Trash"
        try:
            if home_trash.stat().st_dev == source_dev:
                files_dir = home_trash / "files"
                info_dir = home_trash / "info"
                files_dir.mkdir(parents=True, exist_ok=True)
                info_dir.mkdir(parents=True, exist_ok=True)
                return files_dir, info_dir
        except OSError:
            pass
        mount_root = self._topdir_of(source_lexical)
        if mount_root is not None:
            shared = mount_root / ".Trash"
            try:
                st = shared.stat(follow_symlinks=False)
                if stat_mod.S_ISDIR(st.st_mode) and st.st_mode & stat_mod.S_ISVTX:
                    files_dir = shared / str(uid) / "files"
                    info_dir = shared / str(uid) / "info"
                    files_dir.mkdir(parents=True, exist_ok=True)
                    info_dir.mkdir(parents=True, exist_ok=True)
                    return files_dir, info_dir
            except OSError:
                pass
            own = mount_root / f".Trash-{uid}"
            try:
                files_dir = own / "files"
                info_dir = own / "info"
                files_dir.mkdir(parents=True, exist_ok=True)
                info_dir.mkdir(parents=True, exist_ok=True)
                return files_dir, info_dir
            except OSError:
                pass
        raise FsError("trash_unavailable", "no trash available on the source mount")

    def _delete(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(params, frozenset({"root_id", "path", "recursive"}))
        target = self._resolve(
            params.get("root_id"), params.get("path"), mutation=True, allow_root=False
        )
        recursive = _bool_param(params, "recursive", default=False)
        pfd, name = self._open_parent(target)
        try:
            st = os.stat(name, dir_fd=pfd, follow_symlinks=False)
            if stat_mod.S_ISDIR(st.st_mode):
                if not recursive:
                    raise FsError(
                        "is_a_directory",
                        f"path is a directory; recursive=true required: {target.display}",
                    )
                self._check_subtree(target, token)
            files_dir, info_dir = self._trash_dirs(target.lexical, st.st_dev)
            token.check()
            # Exclusive .trashinfo first; on failure the metadata is
            # removed again and the source stays untouched.
            info_fd = os.open(str(info_dir), _DIR_OPEN_FLAGS)
            files_fd = os.open(str(files_dir), _DIR_OPEN_FLAGS)
            try:
                trash_name: str | None = None
                for attempt in range(10):
                    candidate = (
                        name
                        if attempt == 0
                        else f"{Path(name).stem}-{uuid.uuid4().hex[:8]}{Path(name).suffix}"
                    )
                    info_name = f"{candidate}.trashinfo"
                    try:
                        ifd = os.open(
                            info_name,
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
                            0o644,
                            dir_fd=info_fd,
                        )
                    except FileExistsError:
                        continue
                    try:
                        body = (
                            "[Trash Info]\n"
                            f"Path={quote(str(target.lexical), safe='/')}\n"
                            f"DeletionDate={datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                        )
                        _write_all(ifd, body.encode("utf-8"))
                    finally:
                        os.close(ifd)
                    try:
                        # Cancel gate immediately before the publish
                        # syscall: if the deadline fired while the
                        # .trashinfo was being written, the source
                        # stays in place and the cleanup below removes
                        # the metadata (cancel-before-commit).
                        token.check()
                        # Atomic no-replace publish: an existing
                        # Trash/files entry (e.g. orphaned without
                        # .trashinfo by another trash client) must
                        # never be destroyed by this delete.
                        _rename_noreplace(name, pfd, candidate, files_fd)
                    except OSError as exc:
                        try:
                            os.unlink(info_name, dir_fd=info_fd)
                        except OSError:
                            pass
                        if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                            continue  # name collision: pick another name
                        raise
                    except BaseException:
                        try:
                            os.unlink(info_name, dir_fd=info_fd)
                        except OSError:
                            pass
                        raise
                    trash_name = candidate
                    break
                if trash_name is None:
                    raise FsError("already_exists", "no free trash name available")
            finally:
                os.close(info_fd)
                os.close(files_fd)
        finally:
            os.close(pfd)
        return {
            "root_id": target.root.id,
            "path": target.display,
            "trash_name": trash_name,
            "recoverable": True,
        }

    # --- host.fs.search (name-only, v1) ---

    def _search(
        self, params: dict[str, object], request_id: str, token: CancelToken
    ) -> dict[str, object]:
        _check_params(
            params,
            frozenset({"root_id", "query", "path", "limit", "modified_after"}),
        )
        target = self._resolve(
            params.get("root_id"),
            params.get("path", "."),
            mutation=False,
            allow_root=True,
        )
        query = _str_param(params, "query")
        assert query is not None
        if not query:
            raise _invalid_params("params.query must not be empty")
        limit = _int_param(
            params, "limit", default=DEFAULT_SEARCH_LIMIT, minimum=1, maximum=MAX_SEARCH_LIMIT
        )
        modified_after: datetime | None = None
        raw_after = _str_param(params, "modified_after", required=False)
        if raw_after is not None:
            try:
                modified_after = datetime.fromisoformat(raw_after)
            except ValueError as exc:
                raise _invalid_params("params.modified_after is not ISO-8601") from exc
            if modified_after.tzinfo is None:
                modified_after = modified_after.astimezone()
        needle = query.casefold()
        collected: list[tuple[float, str, os.stat_result]] = []
        visited = 0
        mount_points = self._mount_points()
        root_dev: int | None = None
        # Bounded-fd walk: each directory is opened via _open_dirs
        # (fd-relative symlink/mount rules re-enforced per path),
        # scanned and closed before its children are visited, so the
        # open-fd count does not grow with the tree size. Nested
        # mounts (including same-device bind mounts) are never
        # descended, matching direct list behavior.
        stack: list[tuple[tuple[str, ...], str]] = [(target.parts, "")]
        while stack:
            parts, rel = stack.pop()
            fd = self._open_dirs(target.root, parts)
            try:
                if root_dev is None:
                    root_dev = os.fstat(fd).st_dev
                with os.scandir(fd) as it:
                    scanned: list[tuple[str, os.stat_result | None]] = []
                    for entry in it:
                        try:
                            scanned.append((entry.name, entry.stat(follow_symlinks=False)))
                        except FileNotFoundError:
                            scanned.append((entry.name, None))
            finally:
                os.close(fd)
            for name, st in scanned:
                token.check()
                visited += 1
                if visited > MAX_SEARCH_VISITED:
                    raise FsError("io_error", "search exceeded the visited-entry limit")
                rel_path = f"{rel}/{name}" if rel else name
                lexical = target.lexical / rel_path
                if self._classify(lexical, mutation=False) != "normal":
                    continue  # denied entries omitted, dirs not descended
                if st is None:
                    continue  # vanished mid-scan
                if needle in name.casefold():
                    if (
                        modified_after is None
                        or datetime.fromtimestamp(st.st_mtime).astimezone() >= modified_after
                    ):
                        collected.append((st.st_mtime, rel_path, st))
                        if len(collected) > MAX_SEARCH_COLLECTED:
                            raise FsError(
                                "io_error",
                                "search exceeded the collected-result limit",
                            )
                if (
                    stat_mod.S_ISDIR(st.st_mode)
                    and st.st_dev == root_dev
                    and lexical not in mount_points
                ):
                    stack.append(((*parts, name), rel_path))
        collected.sort(key=lambda item: item[0], reverse=True)
        matches = [
            {
                "path": rel_path,
                "type": _type_name(st.st_mode),
                "size": st.st_size,
                "modified_at": _iso(st.st_mtime),
            }
            for _, rel_path, st in collected[:limit]
        ]
        result: dict[str, object] = {
            "root_id": target.root.id,
            "path": target.display,
            "matches": matches,
            "truncated": len(collected) > limit,
        }
        while matches and not self._frame_fits(result, request_id):
            matches.pop()
            result["truncated"] = True
        return result
