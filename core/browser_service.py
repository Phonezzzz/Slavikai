"""Browser session lifecycle, separated from the browser tool.

``DesktopBrowserTool`` is a thin adapter: it validates the request, honours
cancellation and delegates every browser operation to a ``BrowserService``.
The service owns the durable part — which backend instance serves which
profile, when it is created and when it is torn down. The generic browser
actions themselves stay in ``PlaywrightBrowserBackend`` untouched
(action implementation reuse, E02-T01 delete-vs-wrap decision).

A profile is identity plus ownership only. Persistence (dedicated
``user_data_dir`` / persistent context), leases and login handoff are later
E02 tickets and deliberately have no surface here.

Persistence (E02-T02): every profile owns one dedicated directory under
``DEFAULT_BROWSER_PROFILE_ROOT`` (``~/.local/state/slavikai/browser-profiles``,
the same XDG state home the repo already uses for durable agent state —
``shared/sandbox.py``: ``CHATGPT_PLAN_STORAGE``). The directory is derived
from profile identity only; no caller can point the browser at an arbitrary
directory, so the owner's main Chrome profile can never be selected and no
password database is ever copied — see :func:`validate_browser_profile_dir`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shared.models import ToolRequest, ToolResult

DEFAULT_PROFILE_ID = "default"
DEFAULT_OWNER_ID = "local"

DEFAULT_BROWSER_PROFILE_ROOT = Path.home() / ".local" / "state" / "slavikai" / "browser-profiles"

_PROFILE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


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
    foreign profile — passwords included — is ever read or copied.
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


def _profile_component(value: str, field: str) -> str:
    if not _PROFILE_COMPONENT.fullmatch(value):
        raise ValueError(f"{field} must be a single safe path component")
    return value


class BrowserService:
    """Owns browser backends keyed by profile.

    A backend is created lazily on the first operation for a profile and
    reused by every later operation for the same profile, until the profile
    session is closed. Closing drops the backend from the map, so the next
    operation starts a fresh session instead of reusing a closed one.

    The service never interprets browser operations beyond the ``close``
    lifecycle operation; all generic actions are delegated to the backend
    unchanged.
    """

    def __init__(
        self,
        backend_factory: Callable[[BrowserProfile], BrowserAutomationBackend],
    ) -> None:
        self._backend_factory = backend_factory
        self._sessions: dict[tuple[str, str], BrowserAutomationBackend] = {}

    def handle(self, profile: BrowserProfile, request: ToolRequest) -> ToolResult:
        backend = self._sessions.get(self._key(profile))
        if backend is None:
            backend = self._backend_factory(profile)
            self._sessions[self._key(profile)] = backend
        result = backend.handle(request)
        if result.ok and _operation(request) == "close":
            self._sessions.pop(self._key(profile), None)
        return result

    def close(self, profile: BrowserProfile) -> None:
        backend = self._sessions.pop(self._key(profile), None)
        if backend is not None:
            backend.close()

    def close_all(self) -> None:
        sessions = list(self._sessions.values())
        self._sessions.clear()
        first_error: Exception | None = None
        for backend in sessions:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

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
