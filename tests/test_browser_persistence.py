"""Dedicated persistent Chromium profile (E02-T02).

The browser keeps one ``user_data_dir`` per (owner, profile) under the
dedicated root (``~/.local/state/slavikai/browser-profiles`` by default):
cookies, local storage and logins survive a browser restart. The directory
is derived from profile identity only — no request argument and no
constructor path outside the root can select the owner's main Chrome
profile, and no password database is ever copied.

Fast tests run in every ``make check`` (resolver/validator, backend wiring
against a recording fake Playwright). The real-browser acceptance test is
opt-in (``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``): it logs into a local
test site, stops the browser, starts it again on the same profile, and
requires the authenticated state to still be there.
"""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.browser_service import (
    BrowserProfile,
    resolve_browser_profile_dir,
    validate_browser_profile_dir,
)
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest
from tools import desktop_browser_tool
from tools.desktop_browser_tool import DesktopBrowserTool, PlaywrightBrowserBackend

LOGIN_HTML = """<!doctype html><html><body>
<button onclick="document.cookie='slavik_session=valid; path=/; max-age=3600'; \
localStorage.setItem('slavik_auth','yes'); \
document.getElementById('state').textContent='logged in';">Log in</button>
<p id="state">anonymous</p>
</body></html>
"""

STATUS_HTML = """<!doctype html><html><body><p id="state">loading</p>
<script>
const cookie = document.cookie.includes('slavik_session=valid')
  ? 'cookie=yes' : 'cookie=no';
const stored = localStorage.getItem('slavik_auth') === 'yes'
  ? 'storage=yes' : 'storage=no';
document.getElementById('state').textContent = cookie + ' ' + stored;
</script></body></html>
"""


class _AuthHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path in {"/login", "/status"}:
            body = LOGIN_HTML if self.path == "/login" else STATUS_HTML
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        del args


class _RecordingContext:
    def __init__(self) -> None:
        self.browser: _RecordingBrowser | None = None
        self.closed = False
        self.timeout_ms: int | None = None

    def set_default_timeout(self, timeout: int) -> None:
        self.timeout_ms = timeout

    def new_page(self) -> None:
        raise AssertionError("recording context has no pages")

    def close(self) -> None:
        self.closed = True


class _RecordingBrowser:
    def __init__(self, context: _RecordingContext | None = None) -> None:
        self._context = context
        self.closed = False

    def new_context(self, **kwargs: object) -> _RecordingContext:
        del kwargs
        if self._context is None:
            self._context = _RecordingContext()
            self._context.browser = self
        return self._context

    def close(self) -> None:
        self.closed = True


class _RecordingChromium:
    def __init__(self) -> None:
        self.persistent_calls: list[dict[str, object]] = []
        self.launch_calls: list[dict[str, object]] = []
        self.context = _RecordingContext()
        self.browser = _RecordingBrowser(self.context)
        self.context.browser = self.browser

    def launch_persistent_context(self, user_data_dir: str, **kwargs: object) -> object:
        self.persistent_calls.append({"user_data_dir": user_data_dir, **kwargs})
        return self.context

    def launch(self, **kwargs: object) -> _RecordingBrowser:
        self.launch_calls.append(kwargs)
        return self.browser


class _RecordingPlaywright:
    def __init__(self) -> None:
        self.chromium = _RecordingChromium()
        self.stopped = False

    def start(self) -> _RecordingPlaywright:
        return self

    def stop(self) -> None:
        self.stopped = True


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def _profile(profile_id: str = "default", owner_id: str = "local") -> BrowserProfile:
    return BrowserProfile(profile_id=profile_id, owner_id=owner_id)


def test_profile_dir_is_derived_from_identity(tmp_path: Path) -> None:
    resolved = resolve_browser_profile_dir(_profile("work", "alice"), root=tmp_path)
    assert resolved == (tmp_path / "alice" / "work").resolve()
    default = resolve_browser_profile_dir(BrowserProfile.default())
    assert default == (
        Path.home() / ".local" / "state" / "slavikai" / "browser-profiles" / "local" / "default"
    )


def test_profile_dir_encodes_production_principal(tmp_path: Path) -> None:
    resolved = resolve_browser_profile_dir(
        BrowserProfile(profile_id="default", owner_id="email:owner@example.com"),
        root=tmp_path,
    )
    assert resolved == (tmp_path / "email%3Aowner%40example.com" / "default").resolve()
    other = resolve_browser_profile_dir(
        BrowserProfile(profile_id="default", owner_id="email:other@example.com"),
        root=tmp_path,
    )
    assert other != resolved
    assert (
        resolve_browser_profile_dir(_profile("work", "alice"), root=tmp_path)
        == (tmp_path / "alice" / "work").resolve()
    )


def test_profile_dir_encodes_separators_but_rejects_dot_components(
    tmp_path: Path,
) -> None:
    assert (
        resolve_browser_profile_dir(
            BrowserProfile(profile_id="a/b", owner_id="local"), root=tmp_path
        )
        == (tmp_path / "local" / "a%2Fb").resolve()
    )
    for profile in (
        BrowserProfile(profile_id="..", owner_id="local"),
        BrowserProfile(profile_id=".", owner_id="local"),
    ):
        with pytest.raises(ValueError):
            resolve_browser_profile_dir(profile, root=tmp_path)


def test_validator_rejects_owner_chrome_profile(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    chrome_default = Path.home() / ".config" / "google-chrome" / "Default"
    with pytest.raises(ValueError):
        validate_browser_profile_dir(chrome_default, root=root)
    with pytest.raises(ValueError):
        validate_browser_profile_dir(root, root=root)
    with pytest.raises(ValueError):
        validate_browser_profile_dir(root / ".." / "google-chrome", root=root)
    assert (
        validate_browser_profile_dir(root / "alice" / "work", root=root)
        == (root / "alice" / "work").resolve()
    )


def test_backend_rejects_non_dedicated_user_data_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PlaywrightBrowserBackend(
            _security(tmp_path),
            cancelled=lambda: False,
            user_data_dir=Path.home() / ".config" / "google-chrome" / "Default",
        )


def test_backend_uses_persistent_context_for_dedicated_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _RecordingPlaywright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: recording)
    profile_dir = resolve_browser_profile_dir(_profile(), root=tmp_path)
    backend = PlaywrightBrowserBackend(
        _security(tmp_path),
        cancelled=lambda: False,
        user_data_dir=profile_dir,
        profile_root=tmp_path,
    )

    started = backend.handle(ToolRequest("desktop_browser", {"operation": "start"}))

    assert started.ok
    assert recording.chromium.persistent_calls == [
        {
            "user_data_dir": str(profile_dir),
            "headless": True,
            "accept_downloads": True,
        }
    ]
    assert recording.chromium.launch_calls == []

    backend.close()
    assert recording.chromium.context.closed
    # A persistent context owns its browser: the browser is closed with the
    # context, closing it again would be a double close.
    assert not recording.chromium.context.browser.closed
    assert recording.stopped


def test_backend_without_profile_dir_keeps_disposable_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _RecordingPlaywright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: recording)
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)

    started = backend.handle(ToolRequest("desktop_browser", {"operation": "start"}))

    assert started.ok
    assert recording.chromium.persistent_calls == []
    assert recording.chromium.launch_calls == [{"headless": True}]

    backend.close()
    assert recording.chromium.context.closed
    assert recording.chromium.context.browser.closed
    assert recording.stopped


def test_tool_starts_persistent_context_for_production_principal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _RecordingPlaywright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: recording)
    root = tmp_path / "profiles"
    profile = BrowserProfile(profile_id="default", owner_id="email:owner@example.com")
    tool = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)

    started = tool.handle(ToolRequest("desktop_browser", {"operation": "start"}))

    assert started.ok
    assert recording.chromium.persistent_calls == [
        {
            "user_data_dir": str(resolve_browser_profile_dir(profile, root=root)),
            "headless": True,
            "accept_downloads": True,
        }
    ]
    tool.close()


def test_request_cannot_select_owner_chrome_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = _RecordingPlaywright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: recording)
    root = tmp_path / "profiles"
    profile = _profile()
    tool = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)

    started = tool.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "start",
                "user_data_dir": str(Path.home() / ".config" / "google-chrome" / "Default"),
            },
        )
    )

    assert started.ok
    assert recording.chromium.persistent_calls == [
        {
            "user_data_dir": str(resolve_browser_profile_dir(profile, root=root)),
            "headless": True,
            "accept_downloads": True,
        }
    ]
    tool.close()


def _read_state(tool: DesktopBrowserTool, url: str) -> str:
    navigated = tool.handle(ToolRequest("desktop_browser", {"operation": "navigate", "url": url}))
    assert navigated.ok, navigated.data
    read = tool.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "read", "selector_type": "css", "selector": "#state"},
        )
    )
    assert read.ok, read.data
    return str(read.data["text"])


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser persistence E2E, opt-in only",
)
def test_login_survives_browser_restart(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    profile = _profile()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)
        opened = first.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{base}/login"})
        )
        assert opened.ok, opened.data
        clicked = first.handle(
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "click",
                    "selector_type": "role",
                    "selector": "button",
                    "name": "Log in",
                },
            )
        )
        assert clicked.ok, clicked.data
        assert _read_state(first, f"{base}/status") == "cookie=yes storage=yes"
        first.close()

        profile_dir = resolve_browser_profile_dir(profile, root=root)
        assert profile_dir.is_dir()
        assert any(profile_dir.iterdir())

        second = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)
        opened_again = second.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{base}/status"})
        )
        assert opened_again.ok, opened_again.data
        assert _read_state(second, f"{base}/status") == "cookie=yes storage=yes"
        second.close()
    finally:
        server.shutdown()
        server.server_close()
