"""BrowserService lifecycle: profiles own sessions, the tool is a thin adapter.

Covers E02-T01: the service creates one backend per (owner, profile), reuses
it across operations, drops it on close, and isolates profiles from each
other. Generic browser actions reach the backend unchanged through the
service; the real Playwright backend is exercised through the service with
fake pages (its own operations are covered in test_desktop_browser_tool.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.browser_service import BrowserProfile, BrowserService
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest, ToolResult
from tests.test_desktop_browser_tool import InteractivePage
from tools.desktop_browser_tool import DesktopBrowserTool, PlaywrightBrowserBackend


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []
        self.closed = False

    def handle(self, request: ToolRequest) -> ToolResult:
        self.requests.append(request)
        if request.args.get("operation") == "close":
            self.closed = True
        return ToolResult.success(
            {"operation": request.args.get("operation", ""), "output": "semantic result"}
        )

    def close(self) -> None:
        self.closed = True


class FailingCloseBackend(FakeBackend):
    def close(self) -> None:
        raise RuntimeError("injected close failure")


def _profile(profile_id: str = "default", owner_id: str = "local") -> BrowserProfile:
    return BrowserProfile(profile_id=profile_id, owner_id=owner_id)


def _request(operation: str = "snapshot") -> ToolRequest:
    return ToolRequest("desktop_browser", {"operation": operation})


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def test_profile_rejects_blank_identity() -> None:
    for profile_id, owner_id in (("", "local"), ("   ", "local"), ("default", "")):
        with pytest.raises(ValueError):
            BrowserProfile(profile_id=profile_id, owner_id=owner_id)
    assert BrowserProfile.default() == BrowserProfile(profile_id="default", owner_id="local")
    assert BrowserProfile.default(owner_id="alice") == BrowserProfile(
        profile_id="default", owner_id="alice"
    )


def test_service_reuses_backend_for_same_profile() -> None:
    created: list[BrowserProfile] = []
    backends: list[FakeBackend] = []

    def factory(profile: BrowserProfile) -> FakeBackend:
        created.append(profile)
        backend = FakeBackend()
        backends.append(backend)
        return backend

    service = BrowserService(factory)
    profile = _profile()

    first = service.handle(profile, _request("snapshot"))
    second = service.handle(profile, _request("tabs"))

    assert first.ok and second.ok
    assert created == [profile]
    assert len(backends) == 1
    assert len(backends[0].requests) == 2
    assert service.is_active(profile)
    assert service.active_profiles() == (profile,)


def test_service_isolates_profiles_and_owners() -> None:
    created: list[BrowserProfile] = []

    def factory(profile: BrowserProfile) -> FakeBackend:
        created.append(profile)
        return FakeBackend()

    service = BrowserService(factory)
    alice = _profile("default", "alice")
    bob = _profile("default", "bob")
    alice_second = _profile("work", "alice")

    service.handle(alice, _request())
    service.handle(bob, _request())
    service.handle(alice_second, _request())
    service.handle(alice, _request())

    assert created == [alice, bob, alice_second]
    assert set(service.active_profiles()) == {alice, bob, alice_second}


def test_service_close_drops_session_and_next_handle_starts_fresh() -> None:
    backends: list[FakeBackend] = []
    service = BrowserService(lambda _profile: backends.append(FakeBackend()) or backends[-1])
    profile = _profile()

    service.handle(profile, _request())
    service.close(profile)

    assert backends[0].closed
    assert not service.is_active(profile)

    service.handle(profile, _request())
    assert len(backends) == 2
    assert not backends[1].closed


def test_service_close_operation_drops_session() -> None:
    backends: list[FakeBackend] = []
    service = BrowserService(lambda _profile: backends.append(FakeBackend()) or backends[-1])
    profile = _profile()

    service.handle(profile, _request("snapshot"))
    closed = service.handle(profile, _request("close"))

    assert closed.ok
    assert backends[0].closed
    assert not service.is_active(profile)

    service.handle(profile, _request("snapshot"))
    assert len(backends) == 2


def test_service_close_all_closes_every_session() -> None:
    backends: dict[str, FakeBackend] = {}

    def factory(profile: BrowserProfile) -> FakeBackend:
        backend = FakeBackend()
        backends[profile.owner_id] = backend
        return backend

    service = BrowserService(factory)
    service.handle(_profile("default", "alice"), _request())
    service.handle(_profile("default", "bob"), _request())

    service.close_all()

    assert all(backend.closed for backend in backends.values())
    assert service.active_profiles() == ()


def test_service_close_all_is_best_effort() -> None:
    good = FakeBackend()
    failing = FailingCloseBackend()
    calls = {"count": 0}

    def factory(profile: BrowserProfile) -> FakeBackend:
        calls["count"] += 1
        return failing if profile.owner_id == "alice" else good

    service = BrowserService(factory)
    service.handle(_profile("default", "alice"), _request())
    service.handle(_profile("default", "bob"), _request())

    with pytest.raises(RuntimeError, match="injected close failure"):
        service.close_all()

    assert good.closed
    assert service.active_profiles() == ()


def test_service_factory_failure_leaves_no_session() -> None:
    calls = {"count": 0}

    def factory(_profile: BrowserProfile) -> FakeBackend:
        calls["count"] += 1
        raise RuntimeError("injected backend failure")

    service = BrowserService(factory)
    profile = _profile()

    with pytest.raises(RuntimeError, match="injected backend failure"):
        service.handle(profile, _request())
    with pytest.raises(RuntimeError, match="injected backend failure"):
        service.handle(profile, _request())

    assert calls["count"] == 2
    assert not service.is_active(profile)


def test_tool_routes_generic_actions_through_shared_service(tmp_path: Path) -> None:
    backends: list[FakeBackend] = []
    service = BrowserService(lambda _profile: backends.append(FakeBackend()) or backends[-1])
    profile = _profile()
    first_tool = DesktopBrowserTool(_security(tmp_path), service=service, profile=profile)
    second_tool = DesktopBrowserTool(_security(tmp_path), service=service, profile=profile)

    snapshot = first_tool.handle(_request("snapshot"))
    tabs = second_tool.handle(_request("tabs"))

    assert snapshot.ok and tabs.ok
    assert len(backends) == 1
    assert len(backends[0].requests) == 2
    assert service.is_active(profile)

    first_tool.close()
    assert backends[0].closed
    assert not service.is_active(profile)


def test_tool_cancellation_still_prevents_backend_creation(tmp_path: Path) -> None:
    created = False

    def factory(_profile: BrowserProfile) -> FakeBackend:
        nonlocal created
        created = True
        return FakeBackend()

    tool = DesktopBrowserTool(
        _security(tmp_path),
        service=BrowserService(factory),
        cancelled=lambda: True,
    )
    result = tool.handle(_request("snapshot"))

    assert not result.ok
    assert not created


def test_real_playwright_backend_works_through_service(tmp_path: Path) -> None:
    page = InteractivePage()
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._browser = object()  # type: ignore[assignment]
    backend._context = object()  # type: ignore[assignment]
    backend._pages["page-1"] = page  # type: ignore[assignment]
    backend._active_page_id = "page-1"
    service = BrowserService(lambda _profile: backend)

    snapshot = service.handle(_profile(), _request("snapshot"))
    read = service.handle(
        _profile(),
        ToolRequest(
            "desktop_browser",
            {"operation": "read", "selector_type": "css", "selector": "main"},
        ),
    )

    assert snapshot.ok and snapshot.data["title"] == "Example page"
    assert read.ok and read.data["text"] == "content"
    assert service.is_active(_profile())
