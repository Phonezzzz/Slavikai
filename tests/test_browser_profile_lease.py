"""Browser profile lease (E02-T03).

One active writer lease per profile: a second run — another service in
the same process, another tool, or another process — gets
``profile_busy`` instead of a second Chromium on the same
``user_data_dir``. A crashed holder never blocks the profile forever:
the kernel releases ``flock`` when the holder dies, and the next run
re-acquires after health validation, with the previous holder pid
reported as recovered.

Fast tests run in every ``make check``. The real-browser acceptance test
is opt-in (``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``): it logs into a
local test site, requires the second concurrent run to get
``profile_busy``, and requires the login to still be there once the
first run closes.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.browser_lease import (
    BrowserProfileBusyError,
    BrowserProfileLease,
    BrowserProfileLeaseError,
)
from core.browser_service import (
    BrowserProfile,
    BrowserService,
    acquire_browser_profile_lease,
    resolve_browser_profile_dir,
    resolve_browser_profile_lease_path,
)
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest, ToolResult
from tools import desktop_browser_tool
from tools.desktop_browser_tool import DesktopBrowserTool

REPO_ROOT = Path(__file__).resolve().parents[1]

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

    def set_default_timeout(self, timeout: int) -> None:
        del timeout

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


class _FakeBackend:
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


def _profile(profile_id: str = "default", owner_id: str = "local") -> BrowserProfile:
    return BrowserProfile(profile_id=profile_id, owner_id=owner_id)


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def _request(operation: str = "snapshot") -> ToolRequest:
    return ToolRequest("desktop_browser", {"operation": operation})


def _lease_provider(root: Path) -> Callable[[BrowserProfile], BrowserProfileLease]:
    def _acquire(profile: BrowserProfile) -> BrowserProfileLease:
        return acquire_browser_profile_lease(profile, root=root)

    return _acquire


def test_lease_path_is_derived_from_identity(tmp_path: Path) -> None:
    assert (
        resolve_browser_profile_lease_path(_profile("work", "alice"), root=tmp_path)
        == (tmp_path / "alice" / ".lease-work").resolve()
    )
    encoded = resolve_browser_profile_lease_path(
        BrowserProfile(profile_id="default", owner_id="email:owner@example.com"),
        root=tmp_path,
    )
    assert encoded == (tmp_path / "email%3Aowner%40example.com" / ".lease-default").resolve()
    other = resolve_browser_profile_lease_path(
        BrowserProfile(profile_id="default", owner_id="email:other@example.com"),
        root=tmp_path,
    )
    assert other != encoded


def test_lease_acquire_release_roundtrip(tmp_path: Path) -> None:
    profile = _profile()
    lease = acquire_browser_profile_lease(profile, root=tmp_path)
    try:
        assert lease.lease_path.is_file()
        assert lease.holder_pid == os.getpid()
        assert lease.recovered_from_pid is None
        assert not lease.closed
        record = json.loads(lease.lease_path.read_text())
        assert record["pid"] == os.getpid()
        assert record["holder"] == "local/default"
    finally:
        lease.release()
    assert lease.closed
    lease.release()  # idempotent
    again = acquire_browser_profile_lease(profile, root=tmp_path)
    try:
        assert again.recovered_from_pid is None
    finally:
        again.release()


def test_second_acquire_in_process_is_busy(tmp_path: Path) -> None:
    profile = _profile()
    first = acquire_browser_profile_lease(profile, root=tmp_path)
    try:
        with pytest.raises(BrowserProfileBusyError) as excinfo:
            acquire_browser_profile_lease(profile, root=tmp_path)
        assert "profile_busy" in str(excinfo.value)
        assert str(os.getpid()) in str(excinfo.value)
    finally:
        first.release()
    second = acquire_browser_profile_lease(profile, root=tmp_path)
    try:
        assert second.holder_pid == os.getpid()
    finally:
        second.release()


def test_lease_file_with_loose_permissions_is_refused(tmp_path: Path) -> None:
    profile = _profile()
    lease_path = resolve_browser_profile_lease_path(profile, root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease_path.write_text("{}")
    lease_path.chmod(0o644)
    with pytest.raises(BrowserProfileLeaseError):
        acquire_browser_profile_lease(profile, root=tmp_path)


def test_lease_path_symlink_is_refused(tmp_path: Path) -> None:
    profile = _profile()
    lease_path = resolve_browser_profile_lease_path(profile, root=tmp_path)
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "elsewhere"
    target.write_text("{}")
    lease_path.symlink_to(target)
    with pytest.raises(BrowserProfileLeaseError):
        acquire_browser_profile_lease(profile, root=tmp_path)


def test_profile_dir_that_is_a_file_is_refused(tmp_path: Path) -> None:
    profile = _profile()
    profile_dir = resolve_browser_profile_dir(profile, root=tmp_path)
    profile_dir.parent.mkdir(parents=True, exist_ok=True)
    profile_dir.write_text("not a directory")
    with pytest.raises(BrowserProfileLeaseError):
        acquire_browser_profile_lease(profile, root=tmp_path)


def test_crash_holder_lease_is_recoverable(tmp_path: Path) -> None:
    profile = _profile()
    lease_path = resolve_browser_profile_lease_path(profile, root=tmp_path)
    profile_dir = resolve_browser_profile_dir(profile, root=tmp_path)
    holder_code = (
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, os.environ['SLAVIK_REPO_ROOT'])\n"
        "from core.browser_lease import acquire_profile_lease\n"
        "lease = acquire_profile_lease(\n"
        "    Path(os.environ['SLAVIK_LEASE_PATH']),\n"
        "    profile_dir=Path(os.environ['SLAVIK_PROFILE_DIR']),\n"
        "    holder='crash-holder',\n"
        ")\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    env = {
        **os.environ,
        "SLAVIK_REPO_ROOT": str(REPO_ROOT),
        "SLAVIK_LEASE_PATH": str(lease_path),
        "SLAVIK_PROFILE_DIR": str(profile_dir),
    }
    child = subprocess.Popen(
        [sys.executable, "-c", holder_code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    try:
        ready, _, _ = select.select([child.stdout], [], [], 10)
        assert ready, "holder process did not start"
        assert child.stdout.readline().strip() == "READY"
        with pytest.raises(BrowserProfileBusyError) as excinfo:
            acquire_browser_profile_lease(profile, root=tmp_path)
        assert "profile_busy" in str(excinfo.value)
        assert str(child.pid) in str(excinfo.value)
        child.kill()  # SIGKILL: no cleanup, the crash case
        child.wait(timeout=10)
        assert profile_dir.is_dir()
        recovered = acquire_browser_profile_lease(profile, root=tmp_path)
        try:
            assert recovered.recovered_from_pid == child.pid
            assert recovered.holder_pid == os.getpid()
        finally:
            recovered.release()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if child.stderr is not None:
            child.stderr.close()


def test_service_second_run_gets_profile_busy(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    profile = _profile()
    first = BrowserService(lambda _p: _FakeBackend(), lease_provider=_lease_provider(root))
    second = BrowserService(lambda _p: _FakeBackend(), lease_provider=_lease_provider(root))
    assert first.handle(profile, _request()).ok
    with pytest.raises(BrowserProfileBusyError) as excinfo:
        second.handle(profile, _request())
    assert "profile_busy" in str(excinfo.value)
    first.close(profile)
    assert second.handle(profile, _request()).ok
    second.close(profile)


def test_service_releases_lease_when_backend_factory_fails(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    profile = _profile()

    def _failing(_p: BrowserProfile) -> _FakeBackend:
        raise RuntimeError("injected backend failure")

    failing = BrowserService(_failing, lease_provider=_lease_provider(root))
    with pytest.raises(RuntimeError, match="injected backend failure"):
        failing.handle(profile, _request())
    working = BrowserService(lambda _p: _FakeBackend(), lease_provider=_lease_provider(root))
    assert working.handle(profile, _request()).ok
    working.close(profile)


def test_service_releases_lease_on_close_operation(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    profile = _profile()
    first = BrowserService(lambda _p: _FakeBackend(), lease_provider=_lease_provider(root))
    second = BrowserService(lambda _p: _FakeBackend(), lease_provider=_lease_provider(root))
    assert first.handle(profile, _request()).ok
    closed = first.handle(profile, _request("close"))
    assert closed.ok
    assert second.handle(profile, _request()).ok
    second.close(profile)


def test_service_without_lease_provider_is_unchanged(tmp_path: Path) -> None:
    profile = _profile()
    first = BrowserService(lambda _p: _FakeBackend())
    second = BrowserService(lambda _p: _FakeBackend())
    assert first.handle(profile, _request()).ok
    assert second.handle(profile, _request()).ok
    first.close(profile)
    second.close(profile)


def test_tool_reports_profile_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recording = _RecordingPlaywright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: recording)
    root = tmp_path / "profiles"
    profile = _profile()
    first = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)
    second = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)
    assert first.handle(_request("start")).ok
    busy = second.handle(_request("start"))
    assert not busy.ok
    assert busy.error is not None and "profile_busy" in busy.error
    first.close()
    assert second.handle(_request("start")).ok
    assert recording.chromium.persistent_calls
    second.close()


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
    reason="real-browser lease E2E, opt-in only",
)
def test_second_real_run_gets_profile_busy_and_login_survives(tmp_path: Path) -> None:
    root = tmp_path / "profiles"
    profile = _profile()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AuthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    first: DesktopBrowserTool | None = None
    second: DesktopBrowserTool | None = None
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

        second = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=root)
        busy = second.handle(_request("start"))
        assert not busy.ok
        assert busy.error is not None and "profile_busy" in busy.error
        first.close()
        first = None

        assert second.handle(_request("start")).ok
        opened_again = second.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{base}/status"})
        )
        assert opened_again.ok, opened_again.data
        assert _read_state(second, f"{base}/status") == "cookie=yes storage=yes"
    finally:
        if second is not None:
            second.close()
        if first is not None:
            first.close()
        server.shutdown()
        server.server_close()


def test_lease_is_recoverable_after_process_exit_without_release(tmp_path: Path) -> None:
    # A holder that exits normally without releasing still frees the flock:
    # the lease never outlives its holder process as a block.
    profile = _profile()
    lease_path = resolve_browser_profile_lease_path(profile, root=tmp_path)
    profile_dir = resolve_browser_profile_dir(profile, root=tmp_path)
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, os.environ['SLAVIK_REPO_ROOT'])\n"
        "from core.browser_lease import acquire_profile_lease\n"
        "acquire_profile_lease(\n"
        "    Path(os.environ['SLAVIK_LEASE_PATH']),\n"
        "    profile_dir=Path(os.environ['SLAVIK_PROFILE_DIR']),\n"
        "    holder='short-holder',\n"
        ")\n"
    )
    env = {
        **os.environ,
        "SLAVIK_REPO_ROOT": str(REPO_ROOT),
        "SLAVIK_LEASE_PATH": str(lease_path),
        "SLAVIK_PROFILE_DIR": str(profile_dir),
    }
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30
    )
    assert done.returncode == 0, done.stderr
    previous_pid = int(json.loads(lease_path.read_text())["pid"])
    recovered = acquire_browser_profile_lease(profile, root=tmp_path)
    try:
        assert recovered.recovered_from_pid == previous_pid
    finally:
        recovered.release()
