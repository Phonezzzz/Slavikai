from __future__ import annotations

import hashlib
import mimetypes
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
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

from core.browser_lease import BrowserProfileLease
from core.browser_service import (
    BrowserAutomationBackend,
    BrowserProfile,
    BrowserService,
    acquire_browser_profile_lease,
    resolve_browser_profile_dir,
    validate_browser_profile_dir,
)
from core.desktop_security import DesktopPathSecurity
from core.user_auth import UserAuthRequest, UserAuthRequired
from shared.models import JSONValue, ToolRequest, ToolResult

__all__ = ["BrowserAutomationBackend", "DesktopBrowserTool", "PlaywrightBrowserBackend"]

MAX_BROWSER_TEXT_CHARS = 30_000
MAX_BROWSER_FIND_RESULTS = 50
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
SENSITIVE_VALUE_MASK = "[redacted]"

# E02-T04: Playwright aria_snapshot печатает value текстовых полей, включая
# type=password (проверено на Chromium 2026-11-19). Перед снимком значения
# чувствительных полей временно подменяются маской и сразу восстанавливаются:
# снимок модели и tool log никогда не видят пароль или MFA-код. Для обычного
# snapshot критерий — структурные признаки поля (type=password,
# autocomplete), без угадывания по тексту страницы. Для observation после
# ручного входа (``_finish_login_handoff``) этого мало: MFA-форма может
# держать код в обычном текстовом поле без autocomplete, и структурно такое
# поле неотличимо от любого другого. Поэтому там маскируются ВСЕ непустые
# текстовые поля, а если отсутствие секретов подтвердить нельзя — snapshot
# модели не уходит (fail-closed, см. _VERIFY_ALL_MASKED_JS).
# E02-T04 rework round 3: строгая маскировка — свойство состояния сессии,
# а не одной операции _finish_login_handoff(). После «Готово» MFA-код может
# остаться в обычном поле той же страницы, и следующий обычный snapshot
# отдал бы его модели. Поэтому страница ручного входа остаётся в строгом
# режиме (``_strict_masking_pages``) до её закрытия; все model-visible
# observations этой страницы идут через строгую маскировку. Другие
# страницы сессии не затронуты: обычный просмотр не теряет содержимое
# полей, а выход из строгого режима — явное закрытие вкладки, не тихое
# ослабление по таймеру или навигации.
_MASK_SENSITIVE_JS = """(mask) => {
  const tokens = new Set(['current-password', 'new-password', 'one-time-code']);
  let masked = 0;
  const visit = (root) => {
    for (const el of root.querySelectorAll('input')) {
      const auto = (el.getAttribute('autocomplete') || '').toLowerCase().split(/\\s+/);
      const sensitive = el.type === 'password' || auto.some((token) => tokens.has(token));
      if (!sensitive || !el.value || el.__slavikaiMasked !== undefined) continue;
      el.__slavikaiMasked = el.value;
      el.value = mask;
      masked += 1;
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) visit(el.shadowRoot);
    }
  };
  visit(document);
  return masked;
}"""
_RESTORE_SENSITIVE_JS = """() => {
  const visit = (root) => {
    for (const el of root.querySelectorAll('input, textarea')) {
      if (el.__slavikaiMasked === undefined) continue;
      el.value = el.__slavikaiMasked;
      delete el.__slavikaiMasked;
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) visit(el.shadowRoot);
    }
  };
  visit(document);
}"""
# Проверка после маскировки: сколько чувствительных полей всё ещё держат
# реальное значение (включая открытые shadow roots). Ноль — маскировка
# подтверждена; иначе snapshot модели не уходит (fail-closed).
_VERIFY_MASKED_JS = """(mask) => {
  const tokens = new Set(['current-password', 'new-password', 'one-time-code']);
  let exposed = 0;
  const visit = (root) => {
    for (const el of root.querySelectorAll('input')) {
      const auto = (el.getAttribute('autocomplete') || '').toLowerCase().split(/\\s+/);
      const sensitive = el.type === 'password' || auto.some((token) => tokens.has(token));
      if (sensitive && el.value && el.value !== mask) exposed += 1;
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) visit(el.shadowRoot);
    }
  };
  visit(document);
  return exposed;
}"""
# Строгая маскировка для observation после ручного входа: ВСЕ непустые
# текстовые поля (input и textarea, кроме file — ему value не присвоить),
# включая открытые shadow roots. MFA-код в обычном поле без
# autocomplete="one-time-code" структурно неотличим от любого текста,
# поэтому после handoff модели не уходит ни одно введённое значение.
_MASK_ALL_TEXT_FIELDS_JS = """(mask) => {
  let masked = 0;
  const visit = (root) => {
    for (const el of root.querySelectorAll('input, textarea')) {
      if (el.type === 'file' || !el.value || el.__slavikaiMasked !== undefined) continue;
      el.__slavikaiMasked = el.value;
      el.value = mask;
      masked += 1;
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) visit(el.shadowRoot);
    }
  };
  visit(document);
  return masked;
}"""
# Подтверждение строгой маскировки: ни одно текстовое поле не держит
# реальное значение. Ноль — snapshot безопасен; иначе он не уходит модели.
_VERIFY_ALL_MASKED_JS = """(mask) => {
  let exposed = 0;
  const visit = (root) => {
    for (const el of root.querySelectorAll('input, textarea')) {
      if (el.type === 'file') continue;
      if (el.value && el.value !== mask) exposed += 1;
    }
    for (const el of root.querySelectorAll('*')) {
      if (el.shadowRoot) visit(el.shadowRoot);
    }
  };
  visit(document);
  return exposed;
}"""


class DesktopBrowserTool:
    """Thin adapter: request validation and cancellation, then the service.

    Session lifecycle (which backend serves which profile, creation on first
    use, teardown on close) lives in :class:`BrowserService`; this tool
    holds no backend itself. ``service``/``profile`` are injectable so one
    service can be shared; by default the tool builds its own service with
    the owner's default profile. The default backend is persistent (E02-T02):
    one dedicated ``user_data_dir`` per profile, resolved from profile
    identity only — a request can never select an arbitrary directory, and
    the owner's main Chrome profile is never used. The default persistent
    backend also takes the profile's active-writer lease (E02-T03): a
    second run on the same profile fails with ``profile_busy`` instead of
    starting a second Chromium on one ``user_data_dir``.
    """

    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        backend_factory: Callable[[], BrowserAutomationBackend] | None = None,
        cancelled: Callable[[], bool] | None = None,
        service: BrowserService | None = None,
        profile: BrowserProfile | None = None,
        profile_root: Path | None = None,
    ) -> None:
        self._security = security
        self._cancelled = cancelled or (lambda: False)
        self._profile = profile or BrowserProfile.default()
        if service is not None:
            self._service = service
        else:
            lease_provider: Callable[[BrowserProfile], BrowserProfileLease] | None = None
            if backend_factory is None:

                def _lease_for(profile: BrowserProfile) -> BrowserProfileLease:
                    return acquire_browser_profile_lease(profile, root=profile_root)

                lease_provider = _lease_for
            self._service = BrowserService(
                self._make_backend_factory(backend_factory, profile_root),
                lease_provider=lease_provider,
            )

    def _make_backend_factory(
        self,
        backend_factory: Callable[[], BrowserAutomationBackend] | None,
        profile_root: Path | None,
    ) -> Callable[[BrowserProfile], BrowserAutomationBackend]:
        if backend_factory is not None:
            supplied = backend_factory

            def _from_supplied(_profile: BrowserProfile) -> BrowserAutomationBackend:
                return supplied()

            return _from_supplied

        def _persistent(profile: BrowserProfile) -> BrowserAutomationBackend:
            return PlaywrightBrowserBackend(
                self._security,
                cancelled=self._cancelled,
                user_data_dir=resolve_browser_profile_dir(profile, root=profile_root),
                profile_root=profile_root,
            )

        return _persistent

    def current_origin(self, page_id: str | None = None) -> str | None:
        """Origin фактической целевой страницы профиля (E02-T09)."""
        return self._service.current_origin(self._profile, page_id)

    def handle(self, request: ToolRequest) -> ToolResult:
        if self._cancelled():
            return ToolResult.failure("Browser operation cancelled before execution.")
        operation = _operation(request, default="snapshot")
        if operation in {"input", "select"} and request.expected_browser_origin is not None:
            # E02-T09: повторная проверка origin непосредственно перед
            # исполнением. Redirect, ручной переход или переключение
            # вкладки между policy-решением и исполнением меняют
            # фактический origin — действие не выполняется по старому
            # разрешению, policy переоценивается на следующем вызове.
            page_id_raw = request.args.get("page_id")
            page_id_hint = (
                page_id_raw.strip()
                if isinstance(page_id_raw, str) and page_id_raw.strip()
                else None
            )
            actual_origin = self.current_origin(page_id_hint)
            if actual_origin != request.expected_browser_origin:
                return ToolResult.failure(
                    "Browser origin changed since the policy decision "
                    f"(expected {request.expected_browser_origin}, "
                    f"actual {actual_origin or 'unknown'}); "
                    "the edit was not executed and policy must be re-evaluated."
                )
        try:
            return self._service.handle(self._profile, request)
        except (OSError, RuntimeError, ValueError, PlaywrightError) as exc:
            return ToolResult.failure(f"Browser automation failed: {exc}")

    def close(self) -> None:
        self._service.close(self._profile)


class PlaywrightBrowserBackend:
    """Playwright backend with a dedicated persistent profile per session.

    With ``user_data_dir`` the backend starts a persistent context
    (``launch_persistent_context``): cookies, local storage and logins live
    in that directory and survive a browser restart. Without it the backend
    keeps the old disposable behaviour (``launch`` + ``new_context``).
    The directory is validated at construction to live under the dedicated
    browser profile root, so the owner's main Chrome profile can never be
    selected and no foreign profile data is ever read or copied.
    """

    def __init__(
        self,
        security: DesktopPathSecurity,
        *,
        cancelled: Callable[[], bool],
        user_data_dir: Path | None = None,
        profile_root: Path | None = None,
    ) -> None:
        self._security = security
        self._cancelled = cancelled
        self._user_data_dir = (
            validate_browser_profile_dir(user_data_dir, root=profile_root)
            if user_data_dir is not None
            else None
        )
        # A persistent context owns its browser: closing the context already
        # closes the browser, so the browser must not be closed twice.
        self._owns_browser = user_data_dir is None
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._pages: dict[str, Page] = {}
        self._active_page_id: str | None = None
        # E02-T04: активная попытка ручного входа (attempt_id, page_id).
        self._handoff: tuple[str, str] | None = None
        # E02-T04 rework round 3: страницы в строгом режиме маскировки.
        # Включается при открытии окна ручного входа, снимается только при
        # закрытии страницы (close_tab/close) — секреты, введённые вручную,
        # могут оставаться в полях страницы после «Готово».
        self._strict_masking_pages: set[str] = set()
        # E02-T04 rework: ошибка неподтверждённого закрытия Chromium. Пока она
        # сохранена, close() не превращается в подтверждённое закрытие, а
        # новый контекст на этом профиле не стартует.
        self._close_error: Exception | None = None

    def handle(self, request: ToolRequest) -> ToolResult:
        operation = _operation(request, default="snapshot")
        if operation == "login_handoff":
            return self._login_handoff(request)
        if request.user_auth_resume is not None:
            raise ValueError("user_auth_resume is only valid for login_handoff")
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
            return ToolResult.success(
                {"operation": operation, **self._observation(page_id, page, request)}
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
            # Страница закрыта — строгий режим маскировки для неё окончен.
            self._strict_masking_pages.discard(page_id)
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
            "select|submit|wait|tabs|switch_tab|close_tab|download|close|login_handoff"
        )

    def _observation(
        self,
        page_id: str,
        page: Page,
        request: ToolRequest,
        *,
        mask_all_text_fields: bool = False,
    ) -> dict[str, JSONValue]:
        """Accessibility snapshot страницы без значений password/MFA-полей.

        Строгая маскировка (все непустые текстовые поля) действует, когда
        явно запрошена (``mask_all_text_fields``, observation после ручного
        входа) ИЛИ страница находится в строгом режиме сессии
        (``_strict_masking_pages`` — страница ручного входа до её закрытия):
        MFA-код может лежать в обычном поле без ``autocomplete``, и
        подтвердить отсутствие секретов иначе нельзя.
        """
        strict = mask_all_text_fields or page_id in self._strict_masking_pages
        mask_js = _MASK_ALL_TEXT_FIELDS_JS if strict else _MASK_SENSITIVE_JS
        verify_js = _VERIFY_ALL_MASKED_JS if strict else _VERIFY_MASKED_JS
        frames = list(page.frames)
        masked = 0
        try:
            for frame in frames:
                masked += int(frame.evaluate(mask_js, SENSITIVE_VALUE_MASK))
            # Маскировка подтверждается до снимка: если хоть одно поле
            # (включая открытые shadow roots) осталось с реальным
            # значением, snapshot модели не уходит — явный отказ, не fallback.
            for frame in frames:
                exposed = int(frame.evaluate(verify_js, SENSITIVE_VALUE_MASK))
                if exposed:
                    raise RuntimeError(
                        "sensitive field masking could not be confirmed; snapshot withheld"
                    )
            snapshot = page.locator("body").aria_snapshot(timeout=_timeout_ms(request), mode="ai")
        finally:
            for frame in frames:
                if not frame.is_detached():
                    frame.evaluate(_RESTORE_SENSITIVE_JS)
        visible, truncated = _truncate(snapshot)
        return {
            "output": visible,
            "page_id": page_id,
            "url": page.url,
            "title": page.title(),
            "snapshot": visible,
            "truncated": truncated,
            "total_chars": len(snapshot),
            "masked_sensitive_fields": masked,
            # Честный сигнал модели: страница в строгом режиме, значения
            # всех текстовых полей скрыты до закрытия вкладки.
            "strict_masking": strict,
        }

    def _login_handoff(self, request: ToolRequest) -> ToolResult:
        """E02-T04: передать вход пользователю в видимом окне того же профиля.

        Первый вызов: headless-сессия закрывается, тот же ``user_data_dir``
        запускается headed (profile lease держит BrowserService, он не
        отпускается), страница входа открывается в окне, и когда окно готово,
        бросается :class:`UserAuthRequired`. Вызов с ``user_auth_resume``
        (только после «Готово» владельца run) отвечает свежим observation —
        это не подтверждение входа, решение остаётся за моделью.
        """
        if request.user_auth_resume is not None:
            return self._finish_login_handoff(request, request.user_auth_resume)
        if self._user_data_dir is None:
            return ToolResult.failure("login_handoff requires a persistent browser profile.")
        if self._handoff is not None:
            return ToolResult.failure("A login handoff is already waiting for the user.")
        # url обязателен: явный retry повторяет тот же вызов после закрытия окна,
        # и страница входа не должна зависеть от уже закрытой вкладки.
        explicit_url = _optional_string(request, "url")
        if explicit_url is None:
            raise ValueError("login_handoff requires url of the login page")
        url = _validated_url(explicit_url)
        page_id = f"page-{uuid.uuid4().hex[:10]}"
        # Тот же профиль, видимое окно: headless-контекст закрывается, lease нет.
        self.close()
        self._ensure_started(request, headless=False)
        context = self._context
        if context is None:
            raise RuntimeError("browser context unavailable")
        page = context.new_page()
        self._pages[page_id] = page
        self._active_page_id = page_id
        page.goto(url, wait_until="domcontentloaded", timeout=_timeout_ms(request))
        page.bring_to_front()
        # Строгий режим маскировки для этой страницы — с момента открытия
        # окна и до закрытия вкладки: пользователь вводит секреты вручную,
        # и они могут остаться в полях после «Готово».
        self._strict_masking_pages.add(page_id)
        attempt_id = uuid.uuid4().hex
        self._handoff = (attempt_id, page_id)
        raise UserAuthRequired(
            UserAuthRequest(
                attempt_id=attempt_id,
                tool="desktop_browser",
                page_id=page_id,
                url=page.url,
                title=page.title(),
                window_ready_monotonic=time.monotonic(),
            )
        )

    def _finish_login_handoff(self, request: ToolRequest, attempt_id: str) -> ToolResult:
        handoff, self._handoff = self._handoff, None
        if handoff is None or handoff[0] != attempt_id:
            self._handoff = handoff
            return ToolResult.failure(
                "Login handoff attempt is not active.",
                meta={"reason_code": "user_auth_attempt_unavailable"},
            )
        page_id = handoff[1]
        page = self._pages.get(page_id)
        if page is None or page.is_closed():
            return ToolResult.failure(
                "The login window was closed; login state is unknown.",
                meta={"reason_code": "user_auth_window_closed"},
            )
        return ToolResult.success(
            {
                "operation": "login_handoff",
                # Строгая маскировка: пользователь только что вводил секреты
                # вручную, и MFA-код мог остаться в обычном текстовом поле.
                **self._observation(page_id, page, request, mask_all_text_fields=True),
                "user_returned_control": True,
                # «Готово» — возврат управления, не доказательство входа.
                "login_verified": False,
            }
        )

    def close(self) -> None:
        if self._close_error is not None:
            # Закрытие уже падало: повторный cleanup не подтверждает его задним
            # числом — BrowserService удержит profile lease (fail-closed).
            raise self._close_error
        context, browser, playwright = self._context, self._browser, self._playwright
        self._pages.clear()
        self._active_page_id = None
        self._handoff = None
        self._strict_masking_pages.clear()
        self._context = None
        self._browser = None
        self._playwright = None
        error: Exception | None = None
        try:
            if context is not None:
                context.close()
        except Exception as exc:  # noqa: BLE001 - ошибка сохраняется ниже
            error = exc
        try:
            if self._owns_browser and browser is not None:
                browser.close()
        except Exception as exc:  # noqa: BLE001 - ошибка сохраняется ниже
            error = error or exc
        try:
            if playwright is not None:
                playwright.stop()
        except Exception as exc:  # noqa: BLE001 - ошибка сохраняется ниже
            error = error or exc
        if error is not None:
            # E02-T04 rework: неподтверждённое закрытие (в т.ч. при перезапуске
            # headless -> headed внутри login_handoff) сохраняется на backend,
            # чтобы последующий cleanup не отпустил profile lease.
            self._close_error = error
            raise error

    def _ensure_started(self, request: ToolRequest, *, headless: bool | None = None) -> None:
        if self._close_error is not None:
            raise RuntimeError(
                "browser close was not confirmed; refusing to start another "
                "Chromium on this profile"
            )
        if self._browser is not None and self._context is not None:
            return
        if headless is None:
            headless_raw = request.args.get("headless")
            headless = headless_raw if isinstance(headless_raw, bool) else True
        playwright = sync_playwright().start()
        try:
            if self._user_data_dir is not None:
                context = playwright.chromium.launch_persistent_context(
                    str(self._user_data_dir),
                    headless=headless,
                    accept_downloads=True,
                )
                context.set_default_timeout(15_000)
                browser = context.browser
                if browser is None:
                    raise RuntimeError("persistent browser context has no browser")
            else:
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

    def page_url(self, page_id: str | None = None) -> str | None:
        """URL целевой страницы для провайдера origin (E02-T09).

        Явный ``page_id`` резолвится в backend-состоянии страниц, без
        него — активная страница. Нет страницы/неизвестный id — None,
        без исключений: policy-провайдер обязан быть fail-closed.
        """
        resolved = page_id if page_id is not None else self._active_page_id
        if resolved is None:
            return None
        page = self._pages.get(resolved)
        if page is None:
            return None
        try:
            return page.url
        except Exception:  # noqa: BLE001 - закрытая страница не должна ронять policy
            return None

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
