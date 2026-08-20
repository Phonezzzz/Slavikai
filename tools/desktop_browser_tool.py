from __future__ import annotations

import hashlib
import mimetypes
import os
import uuid
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

from playwright.sync_api import (
    Browser,
    BrowserContext,
    Locator,
    Page,
    Playwright,
    sync_playwright,
)
from playwright.sync_api import (
    Error as PlaywrightError,
)

from core.desktop_security import DesktopPathSecurity
from shared.models import JSONValue, ToolRequest, ToolResult

MAX_BROWSER_TEXT_CHARS = 30_000
MAX_BROWSER_FIND_RESULTS = 50
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024


class BrowserAutomationBackend(Protocol):
    def handle(self, request: ToolRequest) -> ToolResult: ...

    def close(self) -> None: ...


class DesktopBrowserTool:
    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        backend_factory: Callable[[], BrowserAutomationBackend] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._security = security
        self._cancelled = cancelled or (lambda: False)
        self._backend_factory = backend_factory or (
            lambda: PlaywrightBrowserBackend(
                security,
                cancelled=self._cancelled,
            )
        )
        self._backend: BrowserAutomationBackend | None = None

    def handle(self, request: ToolRequest) -> ToolResult:
        if self._cancelled():
            return ToolResult.failure("Browser operation cancelled before execution.")
        try:
            backend = self._backend
            if backend is None:
                backend = self._backend_factory()
                self._backend = backend
            return backend.handle(request)
        except (OSError, RuntimeError, ValueError, PlaywrightError) as exc:
            return ToolResult.failure(f"Browser automation failed: {exc}")

    def close(self) -> None:
        backend = self._backend
        self._backend = None
        if backend is not None:
            backend.close()


class PlaywrightBrowserBackend:
    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        cancelled: Callable[[], bool],
    ) -> None:
        self._security = security
        self._cancelled = cancelled
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[str, Page] = {}
        self._active_page_id: str | None = None

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request, default="snapshot")
        if operation == "start":
            self._ensure_started(request)
            return ToolResult.success(
                {
                    "output": "Browser automation session started.",
                    "operation": operation,
                    "pages": self._tab_items(),
                }
            )
        self._ensure_started(request)
        if self._cancelled():
            return ToolResult.failure("Browser operation cancelled.")
        if operation in {"open", "new_tab"}:
            return self._open(request)
        if operation == "navigate":
            page_id, page = self._page(request)
            return self._navigate(page_id, page, _required_url(request), request=request)
        if operation == "snapshot":
            page_id, page = self._page(request)
            snapshot = page.locator("body").aria_snapshot(
                timeout=_timeout_ms(request),
                mode="ai",
            )
            visible, truncated = _truncate(snapshot)
            return ToolResult.success(
                {
                    "output": visible,
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                    "title": page.title(),
                    "snapshot": visible,
                    "truncated": truncated,
                    "total_chars": len(snapshot),
                }
            )
        if operation == "read":
            page_id, page = self._page(request)
            locator = _locator(page, request)
            text = locator.inner_text(timeout=_timeout_ms(request))
            visible, truncated = _truncate(text)
            return ToolResult.success(
                {
                    "output": visible,
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                    "text": visible,
                    "truncated": truncated,
                    "total_chars": len(text),
                }
            )
        if operation == "find":
            return self._find(request)
        if operation == "click":
            page_id, page = self._page(request)
            before_url = page.url
            _locator(page, request).click(timeout=_timeout_ms(request))
            page.wait_for_timeout(50)
            return ToolResult.success(
                {
                    "output": "Browser element clicked; inspect resulting page state.",
                    "operation": operation,
                    "page_id": page_id,
                    "before_url": before_url,
                    "url": page.url,
                    "requires_followup_observation": True,
                }
            )
        if operation == "input":
            page_id, page = self._page(request)
            value = _required_string(request, "value", allow_empty=True)
            locator = _locator(page, request)
            locator.fill(value, timeout=_timeout_ms(request))
            verified = locator.input_value(timeout=_timeout_ms(request)) == value
            if not verified:
                return ToolResult.failure("Browser input verification failed.")
            return ToolResult.success(
                {
                    "output": "Browser input filled and verified.",
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                    "verified": True,
                    "value_chars": len(value),
                    "value_sha256": hashlib.sha256(value.encode()).hexdigest(),
                }
            )
        if operation == "select":
            page_id, page = self._page(request)
            value = _required_string(request, "value")
            selected = _locator(page, request).select_option(
                value=value,
                timeout=_timeout_ms(request),
            )
            verified = value in selected
            if not verified:
                return ToolResult.failure("Browser select verification failed.")
            return ToolResult.success(
                {
                    "output": "Browser option selected and verified.",
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                    "selected": selected,
                    "verified": True,
                }
            )
        if operation == "submit":
            page_id, page = self._page(request)
            before_url = page.url
            _locator(page, request).press("Enter", timeout=_timeout_ms(request))
            page.wait_for_timeout(50)
            return ToolResult.success(
                {
                    "output": "Browser form submitted; inspect resulting page state.",
                    "operation": operation,
                    "page_id": page_id,
                    "before_url": before_url,
                    "url": page.url,
                    "requires_followup_observation": True,
                }
            )
        if operation == "wait":
            page_id, page = self._page(request)
            locator = _locator(page, request)
            state = _optional_string(request, "state") or "visible"
            if state not in {"attached", "detached", "visible", "hidden"}:
                return ToolResult.failure("state must be attached|detached|visible|hidden")
            locator.wait_for(state=state, timeout=_timeout_ms(request))  # type: ignore[arg-type]
            return ToolResult.success(
                {
                    "output": f"Browser wait condition reached: {state}.",
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                    "state": state,
                    "verified": True,
                }
            )
        if operation == "tabs":
            return ToolResult.success(
                {
                    "output": "Browser tabs listed.",
                    "operation": operation,
                    "items": self._tab_items(),
                }
            )
        if operation == "switch_tab":
            page_id, page = self._page(request, require_explicit=True)
            page.bring_to_front()
            self._active_page_id = page_id
            return ToolResult.success(
                {
                    "output": f"Switched to {page_id}.",
                    "operation": operation,
                    "page_id": page_id,
                    "url": page.url,
                }
            )
        if operation == "close_tab":
            page_id, page = self._page(request, require_explicit=True)
            page.close()
            self._pages.pop(page_id, None)
            self._active_page_id = next(iter(self._pages), None)
            return ToolResult.success(
                {
                    "output": f"Closed browser tab {page_id}.",
                    "operation": operation,
                    "page_id": page_id,
                    "verified": page_id not in self._pages,
                }
            )
        if operation == "download":
            return self._download(request)
        if operation == "close":
            self.close()
            return ToolResult.success(
                {
                    "output": "Browser automation session closed.",
                    "operation": operation,
                    "verified": True,
                }
            )
        return ToolResult.failure(
            "operation must be start|open|new_tab|navigate|snapshot|read|find|click|input|"
            "select|submit|wait|tabs|switch_tab|close_tab|download|close"
        )

    def close(self) -> None:
        context, browser, playwright = self._context, self._browser, self._playwright
        self._pages.clear()
        self._active_page_id = None
        self._context = None
        self._browser = None
        self._playwright = None
        try:
            if context is not None:
                context.close()
        finally:
            try:
                if browser is not None:
                    browser.close()
            finally:
                if playwright is not None:
                    playwright.stop()

    def _ensure_started(self, request: ToolRequest) -> None:
        if self._browser is not None and self._context is not None:
            return
        headless_raw = request.args.get("headless")
        headless = headless_raw if isinstance(headless_raw, bool) else True
        playwright = sync_playwright().start()
        try:
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context(accept_downloads=True)
            context.set_default_timeout(15_000)
        except Exception:
            playwright.stop()
            raise
        self._playwright = playwright
        self._browser = browser
        self._context = context

    def _open(self, request: ToolRequest) -> ToolResult:
        context = self._context
        if context is None:
            raise RuntimeError("browser context unavailable")
        page = context.new_page()
        page_id = f"page-{uuid.uuid4().hex[:10]}"
        self._pages[page_id] = page
        self._active_page_id = page_id
        url = _optional_string(request, "url")
        if url is None:
            return ToolResult.success(
                {
                    "output": f"Opened blank browser tab {page_id}.",
                    "operation": "new_tab",
                    "page_id": page_id,
                    "url": page.url,
                }
            )
        try:
            return self._navigate(page_id, page, _validated_url(url), request=request)
        except Exception:
            page.close()
            self._pages.pop(page_id, None)
            self._active_page_id = next(iter(self._pages), None)
            raise

    def _navigate(
        self,
        page_id: str,
        page: Page,
        url: str,
        *,
        request: ToolRequest,
    ) -> ToolResult:
        response = page.goto(url, wait_until="domcontentloaded", timeout=_timeout_ms(request))
        final_url = page.url
        status = response.status if response is not None else None
        if not final_url or final_url == "about:blank":
            return ToolResult.failure("Browser navigation produced no page.")
        return ToolResult.success(
            {
                "output": f"Browser navigated to {final_url}.",
                "operation": "navigate",
                "page_id": page_id,
                "url": final_url,
                "title": page.title(),
                "http_status": status,
                "verified": True,
            }
        )

    def _find(self, request: ToolRequest) -> ToolResult:
        page_id, page = self._page(request)
        locator = _locator(page, request)
        total = locator.count()
        limit = _bounded_int(request.args.get("limit"), 20, 1, MAX_BROWSER_FIND_RESULTS)
        items: list[JSONValue] = []
        for index in range(min(total, limit)):
            item = locator.nth(index)
            try:
                text = item.inner_text(timeout=2000)[:1000]
            except PlaywrightError:
                text = ""
            items.append({"index": index, "text": text, "visible": item.is_visible()})
        return ToolResult.success(
            {
                "output": f"Found {total} browser element(s).",
                "operation": "find",
                "page_id": page_id,
                "url": page.url,
                "items": items,
                "total": total,
                "truncated": total > limit,
            }
        )

    def _download(self, request: ToolRequest) -> ToolResult:
        page_id, page = self._page(request)
        destination_raw = _required_string(request, "destination")
        resolved = self._security.require_not_denied(destination_raw, mutation=True)
        destination = resolved.canonical
        overwrite = request.args.get("overwrite") is True
        if destination.exists() and not overwrite:
            return ToolResult.failure("Download destination exists; set overwrite=true.")
        self._security.require_not_denied(str(destination.parent))
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.download")
        locator = _locator(page, request)
        try:
            with page.expect_download(timeout=_timeout_ms(request)) as download_info:
                locator.click(timeout=_timeout_ms(request))
            download = download_info.value
            failure = download.failure()
            if failure:
                return ToolResult.failure(f"Browser download failed: {failure}")
            download.save_as(str(staging))
            size = staging.stat().st_size
            if size > MAX_DOWNLOAD_BYTES:
                return ToolResult.failure("Browser download exceeds maximum artifact size.")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                staging.unlink()
        if not destination.is_file():
            return ToolResult.failure("Downloaded artifact does not exist after save.")
        mime = _detect_mime(destination)
        size = destination.stat().st_size
        return ToolResult.success(
            {
                "output": f"Downloaded and verified {destination} ({size} bytes).",
                "operation": "download",
                "page_id": page_id,
                "url": page.url,
                "path": str(destination),
                "size_bytes": size,
                "mime_type": mime,
                "suggested_filename": download.suggested_filename,
                "verified": True,
            }
        )

    def _page(
        self,
        request: ToolRequest,
        *,
        require_explicit: bool = False,
    ) -> tuple[str, Page]:
        page_id = _optional_string(request, "page_id")
        if page_id is None and not require_explicit:
            page_id = self._active_page_id
        if page_id is None:
            raise ValueError("page_id is required; open a browser tab first")
        page = self._pages.get(page_id)
        if page is None or page.is_closed():
            raise ValueError(f"Unknown or closed browser page: {page_id}")
        return page_id, page

    def _tab_items(self) -> list[JSONValue]:
        items: list[JSONValue] = []
        for page_id, page in self._pages.items():
            if page.is_closed():
                continue
            items.append(
                {
                    "page_id": page_id,
                    "url": page.url,
                    "title": page.title(),
                    "active": page_id == self._active_page_id,
                }
            )
        return items


def _locator(page: Page, request: ToolRequest) -> Locator:
    selector_type = _optional_string(request, "selector_type") or "role"
    selector = _required_string(request, "selector")
    exact = request.args.get("exact") is True
    if selector_type == "role":
        name = _optional_string(request, "name")
        locator = page.get_by_role(selector, name=name, exact=exact)  # type: ignore[arg-type]
    elif selector_type == "label":
        locator = page.get_by_label(selector, exact=exact)
    elif selector_type == "text":
        locator = page.get_by_text(selector, exact=exact)
    elif selector_type == "placeholder":
        locator = page.get_by_placeholder(selector, exact=exact)
    elif selector_type == "test_id":
        locator = page.get_by_test_id(selector)
    elif selector_type == "css":
        locator = page.locator(selector)
    else:
        raise ValueError("selector_type must be role|label|text|placeholder|test_id|css")
    index_raw = request.args.get("index")
    if isinstance(index_raw, int) and not isinstance(index_raw, bool):
        if index_raw < 0:
            raise ValueError("index must be non-negative")
        locator = locator.nth(index_raw)
    return locator


def _operation(request: ToolRequest, *, default: str) -> str:
    raw = request.args.get("operation")
    return raw.strip().lower() if isinstance(raw, str) and raw.strip() else default


def _optional_string(request: ToolRequest, key: str) -> str | None:
    raw = request.args.get(key)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _required_string(request: ToolRequest, key: str, *, allow_empty: bool = False) -> str:
    raw = request.args.get(key)
    if not isinstance(raw, str) or (not allow_empty and not raw.strip()):
        raise ValueError(f"{key} is required")
    return raw if allow_empty else raw.strip()


def _required_url(request: ToolRequest) -> str:
    return _validated_url(_required_string(request, "url"))


def _validated_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute http/https browser URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in browser URLs are not allowed.")
    return url


def _timeout_ms(request: ToolRequest) -> float:
    raw = request.args.get("timeout_seconds")
    seconds = _bounded_int(raw, 20, 1, 120)
    return float(seconds * 1000)


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(maximum, value))


def _truncate(value: str) -> tuple[str, bool]:
    if len(value) <= MAX_BROWSER_TEXT_CHARS:
        return value, False
    return value[:MAX_BROWSER_TEXT_CHARS] + "\n...[browser content truncated]", True


def _detect_mime(path: os.PathLike[str] | str) -> str:
    destination = os.fspath(path)
    try:
        with open(destination, "rb") as handle:
            prefix = handle.read(16)
    except OSError:
        prefix = b""
    signatures = (
        (b"PK\x03\x04", "application/zip"),
        (b"\x1f\x8b", "application/gzip"),
        (b"%PDF-", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
    )
    for signature, mime in signatures:
        if prefix.startswith(signature):
            return mime
    guessed, _ = mimetypes.guess_type(destination)
    return guessed or "application/octet-stream"
