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
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from shared.models import ToolRequest, ToolResult

DEFAULT_PROFILE_ID = "default"
DEFAULT_OWNER_ID = "local"


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
