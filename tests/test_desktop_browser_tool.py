from __future__ import annotations

from pathlib import Path

import pytest

from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest, ToolResult
from tools.desktop_browser_tool import (
    DesktopBrowserTool,
    PlaywrightBrowserBackend,
    _locator,
    _validated_url,
)


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []
        self.closed = False

    def handle(self, request: ToolRequest) -> ToolResult:
        self.requests.append(request)
        return ToolResult.success(
            {"operation": request.args.get("operation", ""), "output": "semantic result"}
        )

    def close(self) -> None:
        self.closed = True


class FakeDownload:
    suggested_filename = "artifact.zip"

    def __init__(self, payload: bytes, *, fail: bool = False) -> None:
        self.payload = payload
        self.fail = fail

    def failure(self) -> str | None:
        return None

    def save_as(self, path: str) -> None:
        if self.fail:
            raise RuntimeError("injected interrupted download")
        Path(path).write_bytes(self.payload)


class DownloadContext:
    def __init__(self, download: FakeDownload) -> None:
        self.value = download

    def __enter__(self) -> DownloadContext:
        return self

    def __exit__(self, *args: object) -> None:
        del args


class FakeLocator:
    def click(self, *, timeout: int) -> None:
        assert timeout > 0


class FakePage:
    url = "https://example.test/download"

    def __init__(self, download: FakeDownload) -> None:
        self.download = download
        self.selector_calls: list[tuple[str, str]] = []

    def expect_download(self, *, timeout: int) -> DownloadContext:
        assert timeout > 0
        return DownloadContext(self.download)

    def is_closed(self) -> bool:
        return False

    def get_by_text(self, value: str, *, exact: bool) -> FakeLocator:
        self.selector_calls.append(("text", value))
        assert exact is False
        return FakeLocator()


class SelectorPage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_by_role(self, role: str, *, name: str | None = None, exact: bool = True) -> str:
        self.calls.append(("role", f"{role}:{name}:{exact}"))
        return "role-locator"

    def get_by_label(self, value: str, *, exact: bool = True) -> str:
        self.calls.append(("label", f"{value}:{exact}"))
        return "label-locator"


class InteractiveLocator:
    def __init__(self, text: str = "content") -> None:
        self.text = text
        self.value = ""
        self.clicked = False
        self.pressed: list[str] = []
        self.waited_for: list[str] = []

    def aria_snapshot(self, *, timeout: float, mode: str) -> str:
        assert timeout > 0 and mode == "ai"
        return "- document: Example"

    def inner_text(self, *, timeout: float) -> str:
        assert timeout > 0
        return self.text

    def click(self, *, timeout: float) -> None:
        assert timeout > 0
        self.clicked = True

    def fill(self, value: str, *, timeout: float) -> None:
        assert timeout > 0
        self.value = value

    def input_value(self, *, timeout: float) -> str:
        assert timeout > 0
        return self.value

    def select_option(self, *, value: str, timeout: float) -> list[str]:
        assert timeout > 0
        self.value = value
        return [value]

    def press(self, key: str, *, timeout: float) -> None:
        assert timeout > 0
        self.pressed.append(key)

    def wait_for(self, *, state: str, timeout: float) -> None:
        assert timeout > 0
        self.waited_for.append(state)

    def count(self) -> int:
        return 2

    def nth(self, index: int) -> InteractiveLocator:
        return InteractiveLocator(f"item-{index}")

    def is_visible(self) -> bool:
        return True


class FakeResponse:
    status = 204


class InteractivePage:
    def __init__(self, *, url: str = "https://example.test/start") -> None:
        self.url = url
        self.closed = False
        self.front = False
        self.locators: dict[str, InteractiveLocator] = {}

    def _locator_for(self, key: str) -> InteractiveLocator:
        return self.locators.setdefault(key, InteractiveLocator())

    def locator(self, selector: str) -> InteractiveLocator:
        return self._locator_for(f"css:{selector}")

    def get_by_role(
        self,
        role: str,
        *,
        name: str | None = None,
        exact: bool = True,
    ) -> InteractiveLocator:
        return self._locator_for(f"role:{role}:{name}:{exact}")

    def get_by_label(self, value: str, *, exact: bool = True) -> InteractiveLocator:
        return self._locator_for(f"label:{value}:{exact}")

    def get_by_text(self, value: str, *, exact: bool = True) -> InteractiveLocator:
        return self._locator_for(f"text:{value}:{exact}")

    def get_by_placeholder(self, value: str, *, exact: bool = True) -> InteractiveLocator:
        return self._locator_for(f"placeholder:{value}:{exact}")

    def get_by_test_id(self, value: str) -> InteractiveLocator:
        return self._locator_for(f"test-id:{value}")

    def title(self) -> str:
        return "Example page"

    def wait_for_timeout(self, timeout: int) -> None:
        assert timeout >= 0

    def bring_to_front(self) -> None:
        self.front = True

    def close(self) -> None:
        self.closed = True

    def is_closed(self) -> bool:
        return self.closed

    def goto(self, url: str, *, wait_until: str, timeout: float) -> FakeResponse:
        assert wait_until == "domcontentloaded" and timeout > 0
        self.url = url
        return FakeResponse()


class InteractiveContext:
    def __init__(self) -> None:
        self.closed = False
        self.created_pages: list[InteractivePage] = []

    def new_page(self) -> InteractivePage:
        page = InteractivePage(url="about:blank")
        self.created_pages.append(page)
        return page

    def close(self) -> None:
        self.closed = True


class ClosableBrowser:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class StoppablePlaywright:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def test_browser_wrapper_preserves_backend_abstraction_and_closes(tmp_path: Path) -> None:
    backend = FakeBackend()
    tool = DesktopBrowserTool(_security(tmp_path), backend_factory=lambda: backend)

    result = tool.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "find", "selector_type": "role", "role": "button"},
        )
    )
    tool.close()

    assert result.ok
    assert backend.requests[0].args["selector_type"] == "role"
    assert backend.closed


def test_browser_cancellation_prevents_backend_creation(tmp_path: Path) -> None:
    created = False

    def create() -> FakeBackend:
        nonlocal created
        created = True
        return FakeBackend()

    tool = DesktopBrowserTool(_security(tmp_path), backend_factory=create, cancelled=lambda: True)
    result = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))

    assert not result.ok
    assert not created


def test_browser_url_validation_rejects_local_and_credential_urls() -> None:
    assert _validated_url("https://example.test/path") == "https://example.test/path"
    for value in ("file:///etc/passwd", "javascript:alert(1)", "https://user:pass@example.test"):
        with pytest.raises(ValueError):
            _validated_url(value)


def test_semantic_selector_prefers_role_and_label() -> None:
    page = SelectorPage()

    role = _locator(
        page,  # type: ignore[arg-type]
        ToolRequest(
            "desktop_browser",
            {"selector_type": "role", "selector": "button", "name": "Save"},
        ),
    )
    label = _locator(
        page,  # type: ignore[arg-type]
        ToolRequest(
            "desktop_browser",
            {"selector_type": "label", "selector": "Email"},
        ),
    )

    assert role == "role-locator"
    assert label == "label-locator"
    assert page.calls == [("role", "button:Save:False"), ("label", "Email:False")]


def test_download_is_saved_as_verified_first_class_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "downloads" / "artifact.zip"
    page = FakePage(FakeDownload(b"PK\x03\x04test-data"))
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._pages["page-1"] = page  # type: ignore[assignment]
    backend._active_page_id = "page-1"

    result = backend._download(  # noqa: SLF001
        ToolRequest(
            "desktop_browser",
            {
                "operation": "download",
                "selector_type": "text",
                "selector": "Download",
                "destination": str(destination),
            },
        )
    )

    assert result.ok and result.data["verified"] is True
    assert result.data["path"] == str(destination)
    assert result.data["size_bytes"] == len(b"PK\x03\x04test-data")
    assert result.data["mime_type"] == "application/zip"
    assert destination.read_bytes() == b"PK\x03\x04test-data"
    assert page.selector_calls == [("text", "Download")]


def test_interrupted_download_is_not_reported_as_success_and_cleans_staging(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.zip"
    page = FakePage(FakeDownload(b"partial", fail=True))
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._pages["page-1"] = page  # type: ignore[assignment]
    backend._active_page_id = "page-1"

    with pytest.raises(RuntimeError, match="interrupted"):
        backend._download(  # noqa: SLF001
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "download",
                    "selector_type": "text",
                    "selector": "Download",
                    "destination": str(destination),
                },
            )
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".artifact.zip.*.download"))


def test_browser_semantic_operations_share_page_and_verify_results(tmp_path: Path) -> None:
    page = InteractivePage()
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._browser = object()  # type: ignore[assignment]
    backend._context = object()  # type: ignore[assignment]
    backend._pages["page-1"] = page  # type: ignore[assignment]
    backend._active_page_id = "page-1"

    snapshot = backend.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))
    read = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "read", "selector_type": "css", "selector": "main"},
        )
    )
    found = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "find", "selector_type": "text", "selector": "result", "limit": 1},
        )
    )
    clicked = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "click", "selector_type": "role", "selector": "button"},
        )
    )
    entered = backend.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "input",
                "selector_type": "label",
                "selector": "Email",
                "value": "user@example.test",
            },
        )
    )
    selected = backend.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "select",
                "selector_type": "test_id",
                "selector": "country",
                "value": "LT",
            },
        )
    )
    submitted = backend.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "submit",
                "selector_type": "placeholder",
                "selector": "Search",
            },
        )
    )
    waited = backend.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "wait",
                "selector_type": "text",
                "selector": "Done",
                "state": "visible",
            },
        )
    )
    invalid_wait = backend.handle(
        ToolRequest(
            "desktop_browser",
            {
                "operation": "wait",
                "selector_type": "text",
                "selector": "Done",
                "state": "unknown",
            },
        )
    )
    navigated = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "navigate", "url": "https://example.test/next"},
        )
    )
    tabs = backend.handle(ToolRequest("desktop_browser", {"operation": "tabs"}))
    switched = backend.handle(
        ToolRequest("desktop_browser", {"operation": "switch_tab", "page_id": "page-1"})
    )
    unknown = backend.handle(ToolRequest("desktop_browser", {"operation": "unknown"}))

    assert snapshot.ok and snapshot.data["title"] == "Example page"
    assert read.ok and read.data["text"] == "content"
    assert found.ok and found.data["truncated"] is True
    assert found.data["items"] == [{"index": 0, "text": "item-0", "visible": True}]
    assert clicked.ok and clicked.data["requires_followup_observation"] is True
    assert entered.ok and entered.data["verified"] is True
    assert selected.ok and selected.data["selected"] == ["LT"]
    assert submitted.ok and submitted.data["requires_followup_observation"] is True
    assert waited.ok and waited.data["verified"] is True
    assert not invalid_wait.ok
    assert navigated.ok and navigated.data["http_status"] == 204
    assert tabs.ok and tabs.data["items"][0]["active"] is True
    assert switched.ok and page.front
    assert not unknown.ok

    closed = backend.handle(
        ToolRequest("desktop_browser", {"operation": "close_tab", "page_id": "page-1"})
    )
    assert closed.ok and closed.data["verified"] is True


def test_browser_opens_blank_and_url_tabs_then_closes_runtime(tmp_path: Path) -> None:
    context = InteractiveContext()
    browser = ClosableBrowser()
    playwright = StoppablePlaywright()
    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._context = context  # type: ignore[assignment]
    backend._browser = browser  # type: ignore[assignment]
    backend._playwright = playwright  # type: ignore[assignment]

    started = backend.handle(ToolRequest("desktop_browser", {"operation": "start"}))
    blank = backend.handle(ToolRequest("desktop_browser", {"operation": "new_tab"}))
    opened = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "open", "url": "https://example.test/opened"},
        )
    )
    closed = backend.handle(ToolRequest("desktop_browser", {"operation": "close"}))

    assert started.ok and started.data["pages"] == []
    assert blank.ok and blank.data["url"] == "about:blank"
    assert opened.ok and opened.data["url"] == "https://example.test/opened"
    assert closed.ok and closed.data["verified"] is True
    assert context.closed and browser.closed and playwright.stopped
