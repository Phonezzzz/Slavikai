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
copied — see :func:`validate_browser_profile_dir`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from core.browser_lease import BrowserProfileLease, acquire_profile_lease
from shared.models import ToolRequest, ToolResult

DEFAULT_PROFILE_ID = "default"
DEFAULT_OWNER_ID = "local"

DEFAULT_BROWSER_PROFILE_ROOT = Path.home() / ".local" / "state" / "slavikai" / "browser-profiles"


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

    Returns the resolved path. Anything outside the root — the owner's main
    Chrome profile, the root itself, a ``..`` traversal escaping it — is
    refused with ``ValueError`` before any browser is launched.
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    resolved = Path(path).expanduser().resolve()
    base_resolved = base.expanduser().resolve()
    if resolved == base_resolved or base_resolved not in resolved.parents:
        raise ValueError(
            "browser profile directory must live under the dedicated browser profile root"
        )
    return resolved


def _require_under_root(path: Path, root: Path) -> Path:
    """Resolve ``path`` and require it to live strictly under ``root``."""
    resolved = Path(path).expanduser().resolve()
    base_resolved = Path(root).expanduser().resolve()
    if resolved == base_resolved or base_resolved not in resolved.parents:
        raise ValueError("browser profile path must live under the dedicated browser profile root")
    return resolved


def resolve_browser_profile_lease_path(
    profile: BrowserProfile, *, root: Path | None = None
) -> Path:
    """Dedicated lease file for one browser profile.

    ``<root>/<owner_id>/.lease-<profile_id>`` — a sibling of the profile
    directory, never inside it: Chromium owns ``user_data_dir`` contents
    and must never see our lock file. The component is encoded exactly
    like the profile directory, so the lease path is derived from identity
    only and can never escape the dedicated root.
    """
    base = Path(root) if root is not None else DEFAULT_BROWSER_PROFILE_ROOT
    profile_dir = resolve_browser_profile_dir(profile, root=base)
    lease_path = profile_dir.parent / (
        ".lease-" + _profile_component(profile.profile_id, "profile_id")
    )
    return _require_under_root(lease_path, base)


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
