"""Browser session lifecycle, separated from the browser tool.

``DesktopBrowserTool`` is a thin adapter: it validates the request, honours
cancellation and delegates every browser operation to a ``BrowserService``.
The service owns the durable part — which backend instance serves which
profile, when it is created and when it is torn down. The generic browser
actions themselves stay in ``PlaywrightBrowserBackend`` untouched
(action implementation reuse, E02-T01 delete-vs-wrap decision).

A profile is identity plus ownership only. Persistence (dedicated
``user_data_dir`` / persistent context, E02-T02) and the active-writer
lease (E02-T03) live here; login handoff is a later E02 ticket and
deliberately has no surface here.

Persistence (E02-T02): every profile owns one dedicated directory under
``DEFAULT_BROWSER_PROFILE_ROOT`` (``~/.local/state/slavikai/browser-profiles``,
the same XDG state home the repo already uses for durable agent state —
``shared/sandbox.py``: ``CHATGPT_PLAN_STORAGE``). The directory is derived
from profile identity only; identity values are encoded into single safe
path components (plain ids stay readable, production principals such as
``email:owner@example.com`` become ``email%3Aowner%40example.com``), so no
caller can point the browser at an arbitrary directory, the owner's main
Chrome profile can never be selected and no password database is ever
copied — see :func:`validate_browser_profile_dir`. Below the root every
path component must be a real directory, never a symlink: an alias (one
profile or owner directory symlinked to another) would let two identities
drive one ``user_data_dir`` under two different leases, so it is refused.

Lease (E02-T03): lease files live in their own root,
``<profile-root>.leases`` next to the profile root
(``~/.local/state/slavikai/browser-profiles.leases`` by default), as
``<lease-root>/<owner_id>/<profile_id>.lease``. Profile directories and
lease files never share a namespace, so no profile id can collide with a
lease file, and existing profile paths are unchanged.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from core.browser_lease import (
    BrowserProfileLease,
    BrowserProfileLeaseError,
    acquire_profile_lease,
)
from shared.models import ToolRequest, ToolResult

DEFAULT_PROFILE_ID = "default"
DEFAULT_OWNER_ID = "local"

DEFAULT_BROWSER_PROFILE_ROOT = Path.home() / ".local" / "state" / "slavikai" / "browser-profiles"

_LEASE_ROOT_SUFFIX = ".leases"
_LEASE_FILE_SUFFIX = ".lease"


class BrowserAutomationBackend(Protocol):
    def handle(self, request: ToolRequest) -> ToolResult: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class BrowserProfile:
    """Identity and ownership of one browser session.

    ``profile_id`` names the session within one owner; ``owner_id`` is the
    principal the session belongs to (the agent ``user_id`` in production).
    The pair ``(owner_id, profile_id)`` is the session key, so two owners
    never share a backend even when they use the same profile id.
    """

    profile_id: str = DEFAULT_PROFILE_ID
    owner_id: str = DEFAULT_OWNER_ID

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be blank")
        if not self.owner_id.strip():
            raise ValueError("owner_id must not be blank")

    @classmethod
    def default(cls, owner_id: str = DEFAULT_OWNER_ID) -> BrowserProfile:
        return cls(profile_id=DEFAULT_PROFILE_ID, owner_id=owner_id)


def resolve_browser_profile_dir(profile: BrowserProfile, *, root: Path | None = None) -> Path:
    """Dedicated on-disk directory for one browser profile.

    ``<root>/<owner_id>/<profile_id>``. The directory is derived from
    profile identity only and validated to stay under the dedicated root,
    so the browser can never be pointed at the owner's main Chrome profile
    (``~/.config/google-chrome`` and friends live outside the root) and no
    foreign profile — passwords included — is ever read or copied. Identity
    values are encoded, not rejected, so production principals such as
    ``email:owner@example.com`` work: see :func:`_profile_component`.
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    return validate_browser_profile_dir(
        base
        / _profile_component(profile.owner_id, "owner_id")
        / _profile_component(profile.profile_id, "profile_id"),
        root=base,
    )


def validate_browser_profile_dir(path: Path, *, root: Path | None = None) -> Path:
    """Require a browser profile directory to live under the dedicated root.

    Returns the canonical path. Anything outside the root — the owner's main
    Chrome profile, the root itself, a ``..`` traversal escaping it — is
    refused with ``ValueError`` before any browser is launched, and so is
    an alias: a symlink anywhere below the root (see
    :func:`_canonical_under_root`).
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    return _canonical_under_root(
        path,
        base,
        what="browser profile directory",
        error=ValueError,
    )


def _canonical_under_root(
    path: Path,
    root: Path,
    *,
    what: str,
    error: type[Exception],
) -> Path:
    """Canonical ``path`` strictly under ``root``, without following aliases.

    The root itself may sit behind a symlink (it is resolved once); below
    it the path is checked lexically (``..`` collapsed, never resolved) and
    every existing component is ``lstat``-ed: a symlink is refused with
    ``error`` instead of being followed, so two identities can never reach
    one directory through two different paths.
    """
    base_lexical = _lexical_root(root)
    base_resolved = _canonical_root(root)
    lexical = Path(os.path.normpath(os.path.abspath(Path(path).expanduser())))
    relative: Path | None = None
    for base in (base_lexical, base_resolved):
        if lexical != base and base in lexical.parents:
            relative = lexical.relative_to(base)
            break
    if relative is None:
        raise error(f"{what} must live under its dedicated root")
    current = base_resolved
    for part in relative.parts:
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise error(f"cannot inspect {what} path {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise error(f"{what} must not be a symlink alias: {current}")
    return base_resolved / relative


def _lexical_root(root: Path) -> Path:
    """Absolute, ``..``-collapsed spelling of ``root`` (not resolved)."""
    return Path(os.path.normpath(os.path.abspath(Path(root).expanduser())))


def _canonical_root(root: Path) -> Path:
    """The one canonical path of a dedicated root.

    Both the profile directory (:func:`_canonical_under_root`) and the
    sibling lease root (:func:`browser_profile_lease_root`) are derived from
    this single value, so two spellings of one root (``/x/profiles`` and a
    symlink ``/x/alias`` pointing at it) always meet the same lease file.
    """
    return _lexical_root(root).resolve()


def browser_profile_lease_root(root: Path | None = None) -> Path:
    """Dedicated lease root for a profile root: ``<profile-root>.leases``.

    A sibling of the *canonical* profile root, never inside it, so lease
    files and profile directories can never collide, and an alias of the
    profile root (a symlink to it) resolves to the same lease root as the
    root itself: one actual ``user_data_dir`` always has exactly one lease.
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    canonical = _canonical_root(base)
    return canonical.with_name(canonical.name + _LEASE_ROOT_SUFFIX)


def resolve_browser_profile_lease_path(
    profile: BrowserProfile, *, root: Path | None = None
) -> Path:
    """Dedicated lease file for one browser profile.

    ``<lease-root>/<owner_id>/<profile_id>.lease`` in the lease root of
    ``root`` (:func:`browser_profile_lease_root`), never inside the profile
    root: Chromium owns ``user_data_dir`` contents and must never see our
    lock file, and no profile id can collide with a lease file. Components
    are encoded exactly like the profile directory, so the lease path is
    derived from identity only; parent components are checked for symlink
    aliases, and the last component is left to the lease open
    (``O_NOFOLLOW``).
    """
    lease_root = browser_profile_lease_root(root)
    candidate = (
        lease_root
        / _profile_component(profile.owner_id, "owner_id")
        / (_profile_component(profile.profile_id, "profile_id") + _LEASE_FILE_SUFFIX)
    )
    parent = _canonical_under_root(
        candidate.parent,
        lease_root,
        what="browser profile lease directory",
        error=BrowserProfileLeaseError,
    )
    return parent / candidate.name


def acquire_browser_profile_lease(
    profile: BrowserProfile, *, root: Path | None = None
) -> BrowserProfileLease:
    """Take the active-writer lease for one profile.

    Raises :class:`BrowserProfileBusyError` (a ``RuntimeError``) when
    another live run holds the profile, and
    :class:`BrowserProfileLeaseError` when the lease itself is unusable.
    The caller must release the returned lease.
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    return acquire_profile_lease(
        resolve_browser_profile_lease_path(profile, root=base),
        profile_dir=resolve_browser_profile_dir(profile, root=base),
        holder=f"{profile.owner_id}/{profile.profile_id}",
    )


def _profile_component(value: str, field: str) -> str:
    """Encode one identity value as a single safe path component.

    ``urllib.parse.quote`` with ``safe="._-"`` is deterministic and
    injective: plain ids such as ``local`` stay readable, while production
    principals such as ``email:owner@example.com`` become
    ``email%3Aowner%40example.com`` instead of being rejected — and two
    different principals can never share one directory. ``.`` and ``..``
    are refused outright: quoting leaves them unchanged and they would
    escape the profile directory.
    """
    encoded = quote(value, safe="._-")
    if not encoded or encoded in {".", ".."}:
        raise ValueError(f"{field} must not be empty, '.' or '..'")
    return encoded


@dataclass
class _BrowserSession:
    """One live session: the backend plus its profile lease, if any."""

    backend: BrowserAutomationBackend
    lease: BrowserProfileLease | None = None


class BrowserService:
    """Owns browser backends keyed by profile.

    A backend is created lazily on the first operation for a profile and
    reused by every later operation for the same profile, until the profile
    session is closed. Closing drops the backend from the map, so the next
    operation starts a fresh session instead of reusing a closed one.

    The service never interprets browser operations beyond the ``close``
    lifecycle operation; all generic actions are delegated to the backend
    unchanged. When a ``lease_provider`` is given, starting a session also
    takes the profile's active-writer lease (E02-T03) and holds it until
    the session closes: a second run on the same profile gets
    ``profile_busy`` instead of a second Chromium on one ``user_data_dir``.
    Without a provider (disposable backends, tests) there is no lease.
    """

    def __init__(
        self,
        backend_factory: Callable[[BrowserProfile], BrowserAutomationBackend],
        *,
        lease_provider: Callable[[BrowserProfile], BrowserProfileLease] | None = None,
    ) -> None:
        self._backend_factory = backend_factory
        self._lease_provider = lease_provider
        self._sessions: dict[tuple[str, str], _BrowserSession] = {}

    def handle(self, profile: BrowserProfile, request: ToolRequest) -> ToolResult:
        session = self._sessions.get(self._key(profile))
        if session is None:
            lease = self._acquire_lease(profile)
            try:
                backend = self._backend_factory(profile)
            except BaseException:
                self._release_lease(lease)
                raise
            session = _BrowserSession(backend=backend, lease=lease)
            self._sessions[self._key(profile)] = session
        result = session.backend.handle(request)
        if result.ok and _operation(request) == "close":
            self._sessions.pop(self._key(profile), None)
            self._release_lease(session.lease)
        return result

    def close(self, profile: BrowserProfile) -> None:
        session = self._sessions.pop(self._key(profile), None)
        if session is not None:
            try:
                session.backend.close()
            finally:
                self._release_lease(session.lease)

    def close_all(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        first_error: Exception | None = None
        for session in sessions:
            try:
                session.backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                if first_error is None:
                    first_error = exc
            finally:
                self._release_lease(session.lease)
        if first_error is not None:
            raise first_error

    def _acquire_lease(self, profile: BrowserProfile) -> BrowserProfileLease | None:
        if self._lease_provider is None:
            return None
        return self._lease_provider(profile)

    @staticmethod
    def _release_lease(lease: BrowserProfileLease | None) -> None:
        if lease is not None:
            lease.release()

    def is_active(self, profile: BrowserProfile) -> bool:
        return self._key(profile) in self._sessions

    def active_profiles(self) -> tuple[BrowserProfile, ...]:
        return tuple(
            BrowserProfile(profile_id=profile_id, owner_id=owner_id)
            for owner_id, profile_id in self._sessions
        )

    @staticmethod
    def _key(profile: BrowserProfile) -> tuple[str, str]:
        return (profile.owner_id, profile.profile_id)


def _operation(request: ToolRequest) -> str:
    raw = request.args.get("operation")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else "snapshot"
