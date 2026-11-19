"""Ручной вход пользователя в браузере (E02-T04, manual login handoff).

Решение владельца (вариант A): пауза run — существующий continuation, фаза
``WAITING_USER_INPUT``, причина ``NEEDS_USER_AUTH``. Тот же persistent profile
перезапускается в видимом Chromium с удержанием lease; «Готово» — только
возврат управления, после него агент снимает свежий observation. Таймаут
5 минут от готовности окна: окно закрывается, lease профиля отпускается
только после подтверждённого закрытия, «Готово» больше не принимается,
доступен явный retry. Пароль/MFA не попадают ни в модель, ни в tool log,
ни в effect log, ни в snapshot.

Быстрые тесты используют fake Playwright (headed-окно на мониторе владельца
в автотестах не открывается). Реальная проверка редакции aria_snapshot —
opt-in, только headless: ``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``.
"""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.approval_policy import ApprovalContext, ApprovalRequired
from core.browser_lease import BrowserProfileBusyError
from core.browser_service import BrowserProfile, acquire_browser_profile_lease
from core.computer_activity_log import ComputerActivityLog
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
)
from core.desktop_runtime import (
    DesktopContinuationUnavailable,
    DesktopExecutionControl,
    DesktopRunCoordinator,
    DesktopRuntime,
)
from core.desktop_security import DesktopPathSecurity
from core.response_preferences import ResponsePreferences
from core.run_contract import RunPolicy
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from core.user_auth import NEEDS_USER_AUTH, UserAuthRequest, UserAuthRequired
from llm.brain_base import Brain
from llm.stream_model import Done, TextDelta, ToolCallCompleted
from llm.types import LLMResult, ToolCall
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools import desktop_browser_tool
from tools.desktop_browser_tool import (
    SENSITIVE_VALUE_MASK,
    DesktopBrowserTool,
    PlaywrightBrowserBackend,
)
from tools.tool_logger import ToolCallLogger
from tools.tool_registry import ToolRegistry

PASSWORD = "S3cr3t-PW-xyz"
OTP = "424242"
LOGIN_URL = "https://login.example.test/signin"


# --- fake Playwright: страница входа с password/OTP-полями ------------------


class _Field:
    def __init__(
        self, name: str, *, sensitive: bool, value: str = "", maskable: bool = True
    ) -> None:
        self.name = name
        self.sensitive = sensitive
        self.value = value
        self.maskable = maskable
        self.stash: str | None = None


class _Frame:
    """Эмулирует DOM-семантику скриптов маскировки по их идентичности."""

    def __init__(self, page: _LoginPage) -> None:
        self.page = page
        self.detached = False

    def _visible_fields(self, script: str) -> list[_Field]:
        # Скрипт без обхода shadowRoot полей в shadow DOM не видит — как в
        # реальном Chromium с document.querySelectorAll.
        fields = list(self.page.fields)
        if "shadowRoot" in script:
            fields += self.page.shadow_fields
        return fields

    def evaluate(self, script: str, arg: object = None) -> int | None:
        if script == desktop_browser_tool._MASK_SENSITIVE_JS:
            masked = 0
            for field in self._visible_fields(script):
                if field.sensitive and field.maskable and field.value and field.stash is None:
                    field.stash, field.value = field.value, str(arg)
                    masked += 1
            return masked
        if script == desktop_browser_tool._VERIFY_MASKED_JS:
            return sum(
                1
                for field in self._visible_fields(script)
                if field.sensitive and field.value and field.value != str(arg)
            )
        if script == desktop_browser_tool._RESTORE_SENSITIVE_JS:
            for field in self._visible_fields(script):
                if field.stash is not None:
                    field.value, field.stash = field.stash, None
            return None
        raise AssertionError(f"unexpected script: {script[:40]}")

    def is_detached(self) -> bool:
        return self.detached


class _BodyLocator:
    def __init__(self, page: _LoginPage) -> None:
        self.page = page

    def aria_snapshot(self, *, timeout: float, mode: str) -> str:
        # Как реальный Chromium: значение textbox печатается открытым текстом.
        assert timeout > 0 and mode == "ai"
        lines = [f'- heading "{self.page.heading}"']
        lines.extend(f'- textbox "{field.name}": {field.value}' for field in self.page.fields)
        # Реальный aria_snapshot обходит Shadow DOM и печатает значения оттуда.
        lines.extend(
            f'- textbox "{field.name}": {field.value}' for field in self.page.shadow_fields
        )
        return "\n".join(lines)


class _LoginPage:
    def __init__(self) -> None:
        self.url = "about:blank"
        self.heading = "Sign in"
        self.closed = False
        self.front = False
        self.fields = [
            _Field("Email", sensitive=False),
            _Field("Password", sensitive=True),
            _Field("Code", sensitive=True),
        ]
        self.shadow_fields = [
            _Field("ShadowPassword", sensitive=True),
            _Field("ShadowCode", sensitive=True),
        ]
        self.frames = [_Frame(self)]

    def locator(self, selector: str) -> _BodyLocator:
        assert selector == "body"
        return _BodyLocator(self)

    def goto(self, url: str, *, wait_until: str, timeout: float) -> None:
        assert wait_until == "domcontentloaded" and timeout > 0
        self.url = url

    def title(self) -> str:
        return "Sign in"

    def bring_to_front(self) -> None:
        self.front = True

    def is_closed(self) -> bool:
        return self.closed

    def close(self) -> None:
        self.closed = True

    def type_credentials(self) -> None:
        """Пользователь сам вводит логин, пароль и MFA в видимом окне."""
        self.fields[0].value = "owner@example.test"
        self.fields[1].value = PASSWORD
        self.fields[2].value = OTP
        self.heading = "Welcome back"

    def type_shadow_credentials(self) -> None:
        """Пользователь ввёл пароль и MFA в форму внутри открытого shadow root."""
        self.shadow_fields[0].value = PASSWORD
        self.shadow_fields[1].value = OTP
        self.heading = "Welcome back"


class _Context:
    def __init__(self, chromium: _Chromium, *, headless: bool) -> None:
        self.chromium = chromium
        self.headless = headless
        self.closed = False
        self.pages: list[_LoginPage] = []
        self.browser = object()

    def set_default_timeout(self, timeout: int) -> None:
        assert timeout > 0

    def new_page(self) -> _LoginPage:
        page = _LoginPage()
        self.pages.append(page)
        return page

    def close(self) -> None:
        if self.chromium.fail_close:
            raise RuntimeError("chromium did not exit")
        self.closed = True
        for page in self.pages:
            page.closed = True


class _Chromium:
    def __init__(self) -> None:
        self.contexts: list[_Context] = []
        self.persistent_calls: list[dict[str, object]] = []
        self.fail_close = False

    def launch_persistent_context(self, user_data_dir: str, **kwargs: object) -> _Context:
        self.persistent_calls.append({"user_data_dir": user_data_dir, **kwargs})
        context = _Context(self, headless=bool(kwargs["headless"]))
        self.contexts.append(context)
        return context


class _Playwright:
    def __init__(self) -> None:
        self.chromium = _Chromium()

    def start(self) -> _Playwright:
        return self

    def stop(self) -> None:
        return None


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path, policy_store_path=tmp_path / ".run" / "desktop-approvals.json"
    )


def _profile() -> BrowserProfile:
    return BrowserProfile(profile_id="default", owner_id="local")


@pytest.fixture
def playwright(monkeypatch: pytest.MonkeyPatch) -> _Playwright:
    fake = _Playwright()
    monkeypatch.setattr(desktop_browser_tool, "sync_playwright", lambda: fake)
    return fake


def _tool(tmp_path: Path) -> DesktopBrowserTool:
    return DesktopBrowserTool(
        _security(tmp_path), profile=_profile(), profile_root=tmp_path / "profiles"
    )


def _handoff(url: str = LOGIN_URL) -> ToolRequest:
    return ToolRequest("desktop_browser", {"operation": "login_handoff", "url": url})


def _lease_is_held(tmp_path: Path) -> bool:
    try:
        lease = acquire_browser_profile_lease(_profile(), root=tmp_path / "profiles")
    except BrowserProfileBusyError:
        return True
    lease.release()
    return False


# --- backend / service ------------------------------------------------------


def test_handoff_relaunches_same_profile_headed_and_keeps_lease(
    tmp_path: Path, playwright: _Playwright
) -> None:
    tool = _tool(tmp_path)
    opened = tool.handle(
        ToolRequest("desktop_browser", {"operation": "open", "url": "https://shop.example.test/"})
    )
    assert opened.ok
    headless_context = playwright.chromium.contexts[0]
    assert _lease_is_held(tmp_path)

    with pytest.raises(UserAuthRequired) as paused:
        tool.handle(_handoff())

    calls = playwright.chromium.persistent_calls
    assert [call["headless"] for call in calls] == [True, False]
    assert calls[0]["user_data_dir"] == calls[1]["user_data_dir"]
    assert headless_context.closed
    headed_page = playwright.chromium.contexts[1].pages[0]
    assert headed_page.url == LOGIN_URL and headed_page.front
    request = paused.value.request
    assert request.reason == NEEDS_USER_AUTH
    assert request.url == LOGIN_URL and request.window_ready_monotonic > 0
    # Lease не отпускался ни на перезапуске, ни во время ожидания входа.
    assert _lease_is_held(tmp_path)
    tool.close()
    assert not _lease_is_held(tmp_path)


def test_handoff_resume_returns_fresh_masked_observation_once(
    tmp_path: Path, playwright: _Playwright
) -> None:
    tool = _tool(tmp_path)
    with pytest.raises(UserAuthRequired) as paused:
        tool.handle(_handoff())
    page = playwright.chromium.contexts[0].pages[0]
    page.type_credentials()
    attempt = paused.value.request.attempt_id
    resumed = ToolRequest(
        "desktop_browser",
        {"operation": "login_handoff", "url": LOGIN_URL},
        user_auth_resume=attempt,
    )

    result = tool.handle(resumed)

    assert result.ok
    assert result.data["user_returned_control"] is True
    # «Готово» — не доказательство входа: решение за моделью по observation.
    assert result.data["login_verified"] is False
    snapshot = str(result.data["snapshot"])
    assert "Welcome back" in snapshot
    assert PASSWORD not in snapshot and OTP not in snapshot
    assert snapshot.count(SENSITIVE_VALUE_MASK) == 2
    assert result.data["masked_sensitive_fields"] == 2
    # Значения в окне пользователя восстановлены после снимка.
    assert page.fields[1].value == PASSWORD and page.fields[2].value == OTP

    again = tool.handle(resumed)
    assert not again.ok
    assert again.meta == {"reason_code": "user_auth_attempt_unavailable"}
    tool.close()


def test_snapshot_never_contains_password_or_otp(tmp_path: Path, playwright: _Playwright) -> None:
    tool = _tool(tmp_path)
    opened = tool.handle(ToolRequest("desktop_browser", {"operation": "open", "url": LOGIN_URL}))
    assert opened.ok
    playwright.chromium.contexts[0].pages[0].type_credentials()

    snapshot = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))

    assert snapshot.ok
    text = repr(snapshot.data)
    assert PASSWORD not in text and OTP not in text
    assert "owner@example.test" in text
    tool.close()


def test_user_auth_resume_is_only_valid_for_handoff(
    tmp_path: Path, playwright: _Playwright
) -> None:
    backend = PlaywrightBrowserBackend(
        _security(tmp_path),
        cancelled=lambda: False,
        user_data_dir=tmp_path / "profiles" / "local" / "default",
        profile_root=tmp_path / "profiles",
    )
    with pytest.raises(ValueError, match="user_auth_resume"):
        backend.handle(
            ToolRequest("desktop_browser", {"operation": "snapshot"}, user_auth_resume="x")
        )
    unknown = backend.handle(
        ToolRequest(
            "desktop_browser",
            {"operation": "login_handoff", "url": LOGIN_URL},
            user_auth_resume="never-issued",
        )
    )
    assert not unknown.ok
    assert unknown.meta == {"reason_code": "user_auth_attempt_unavailable"}
    assert playwright.chromium.persistent_calls == []


def test_handoff_requires_url_and_persistent_profile(tmp_path: Path) -> None:
    disposable = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    refused = disposable.handle(_handoff())
    assert not refused.ok and "persistent" in str(refused.error)
    tool = _tool(tmp_path)
    missing = tool.handle(ToolRequest("desktop_browser", {"operation": "login_handoff"}))
    assert not missing.ok and "url" in str(missing.error)


def test_failed_close_keeps_profile_lease(tmp_path: Path, playwright: _Playwright) -> None:
    tool = _tool(tmp_path)
    with pytest.raises(UserAuthRequired):
        tool.handle(_handoff())
    playwright.chromium.fail_close = True

    with pytest.raises(RuntimeError, match="chromium did not exit"):
        tool.close()

    # Chromium может быть жив на этом user_data_dir: lease не отпускается.
    assert _lease_is_held(tmp_path)
    playwright.chromium.fail_close = False


# --- DesktopRuntime: пауза, «Готово», таймаут, retry -------------------------


class _ScriptedBrain(Brain):
    supports_native_tools = True

    def __init__(self, results: list[LLMResult]) -> None:
        self.results = list(results)
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # type: ignore[override]
        del config, tools
        self.messages_seen.append(list(messages))
        return self.results.pop(0)


class _Parent:
    response_preferences = ResponsePreferences()

    def __init__(self, brain: Brain, registry: ToolRegistry, tool: DesktopBrowserTool) -> None:
        self.brain = brain
        self.tool_registry = registry
        self.tool = tool
        self.main_config = None
        self.desktop_execution_control = DesktopExecutionControl()
        self.activity = ComputerActivityLog()
        self.closed_resources = 0

    def _get_main_brain(self) -> Brain:
        return self.brain

    def _build_tool_gateway(self, *, budget=None) -> ToolGateway:  # noqa: ANN001
        del budget
        return ToolGateway(
            self.tool_registry,
            pre_call=self.activity.pre_call,
            post_call=self.activity.post_call,
            run_policy=RunPolicy(mode="desktop", execution_target="desktop"),
        )

    def close_desktop_resources(self) -> None:
        self.closed_resources += 1
        self.tool.close()


def _handoff_call(call_id: str = "login-1") -> LLMResult:
    return LLMResult(
        text="need login",
        tool_calls=[
            ToolCall(
                id=call_id,
                name="desktop_browser",
                arguments={"operation": "login_handoff", "url": LOGIN_URL},
            )
        ],
    )


def _runtime_setup(
    tmp_path: Path, results: list[LLMResult], *, timeout: float = 300.0
) -> tuple[DesktopRuntime, _Parent, _ScriptedBrain, Path]:
    log_path = tmp_path / "tool_calls.log"
    registry = ToolRegistry(call_logger=ToolCallLogger(log_path))
    tool = _tool(tmp_path)
    registry.register(
        "desktop_browser",
        tool,
        capability="exec",
        risk_classes=["network", "external_side_effect"],
        execution_targets={"desktop"},
    )
    brain = _ScriptedBrain(results)
    parent = _Parent(brain, registry, tool)
    runtime = DesktopRuntime(
        parent,  # type: ignore[arg-type]
        run_coordinator=DesktopRunCoordinator(),
        user_auth_timeout_seconds=timeout,
    )
    return runtime, parent, brain, log_path


def _assert_no_secret_leak(parent: _Parent, brain: _ScriptedBrain, log_path: Path) -> None:
    model_text = repr(brain.messages_seen)
    assert PASSWORD not in model_text and OTP not in model_text
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    assert PASSWORD not in log_text and OTP not in log_text
    activity = repr(parent.activity.drain())
    assert PASSWORD not in activity and OTP not in activity


def test_runtime_pauses_then_done_resumes_with_fresh_observation(
    tmp_path: Path, playwright: _Playwright
) -> None:
    runtime, parent, brain, log_path = _runtime_setup(
        tmp_path, [_handoff_call(), LLMResult(text="Logged in, continuing.")]
    )
    with pytest.raises(UserAuthRequired) as paused:
        runtime.run("buy the thing")
    continuation = paused.value.continuation
    assert continuation is not None
    pause = continuation.approval
    assert isinstance(pause, UserAuthRequest)
    assert pause.timeout_seconds == 300.0 and pause.expires_at_epoch is not None
    assert runtime.has_continuation(continuation)
    assert _lease_is_held(tmp_path)
    assert len(brain.messages_seen) == 1
    playwright.chromium.contexts[0].pages[0].type_credentials()

    outcome = runtime.resume("buy the thing", continuation, user_auth="done")

    assert outcome.text == "Logged in, continuing."
    # Модель получила свежий observation как результат того же tool call.
    tool_messages = [m for m in brain.messages_seen[-1] if m.role == "tool"]
    assert len(tool_messages) == 1
    assert "Welcome back" in tool_messages[0].content
    assert '"login_verified": false' in tool_messages[0].content
    _assert_no_secret_leak(parent, brain, log_path)
    # Run завершён: окно закрыто, lease профиля и хоста отпущены.
    assert not runtime.has_continuation(continuation)
    assert not _lease_is_held(tmp_path)
    with pytest.raises(DesktopContinuationUnavailable):
        runtime.resume("buy the thing", continuation, user_auth="done")
    runtime.close()


def test_runtime_refuses_wrong_resume_kind_and_early_retry(
    tmp_path: Path, playwright: _Playwright
) -> None:
    runtime, _parent, _brain, _log = _runtime_setup(tmp_path, [_handoff_call()])
    with pytest.raises(UserAuthRequired) as paused:
        runtime.run("login")
    continuation = paused.value.continuation
    assert continuation is not None
    with pytest.raises(DesktopContinuationUnavailable, match="approval_continuation"):
        runtime.resume("login", continuation)
    with pytest.raises(DesktopContinuationUnavailable, match="user_auth_retry_unavailable"):
        runtime.resume("login", continuation, user_auth="retry")
    # Отказ не снимает паузу: run всё ещё ждёт входа, окно открыто.
    assert runtime.has_continuation(continuation)
    assert not playwright.chromium.contexts[0].closed
    runtime.cancel_pending()
    assert not _lease_is_held(tmp_path)
    runtime.close()


def test_timeout_closes_window_keeps_needs_user_auth_and_allows_retry(
    tmp_path: Path, playwright: _Playwright
) -> None:
    runtime, parent, brain, log_path = _runtime_setup(
        tmp_path,
        [_handoff_call(), LLMResult(text="Logged in after retry.")],
        timeout=0.05,
    )
    with pytest.raises(UserAuthRequired) as paused:
        runtime.run("login")
    continuation = paused.value.continuation
    assert continuation is not None
    deadline = time.monotonic() + 5
    while not runtime.user_auth_expired(continuation) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runtime.user_auth_expired(continuation)
    # Попытка закончилась: окно закрыто, lease профиля отпущен после закрытия.
    assert playwright.chromium.contexts[0].closed
    assert parent.closed_resources == 1
    assert not _lease_is_held(tmp_path)
    # Run остаётся в NEEDS_USER_AUTH: пауза жива, просроченное «Готово» — отказ.
    assert runtime.has_continuation(continuation)
    with pytest.raises(DesktopContinuationUnavailable, match="user_auth_attempt_expired"):
        runtime.resume("login", continuation, user_auth="done")
    assert runtime.has_continuation(continuation)

    with pytest.raises(UserAuthRequired) as retried:
        runtime.resume("login", continuation, user_auth="retry")

    # Явный retry — новая попытка в новом видимом окне того же профиля.
    assert [call["headless"] for call in playwright.chromium.persistent_calls] == [False, False]
    retry_continuation = retried.value.continuation
    assert retry_continuation is not None
    assert retried.value.request.attempt_id != paused.value.request.attempt_id
    assert _lease_is_held(tmp_path)
    playwright.chromium.contexts[1].pages[0].type_credentials()
    outcome = runtime.resume("login", retry_continuation, user_auth="done")
    assert outcome.text == "Logged in after retry."
    _assert_no_secret_leak(parent, brain, log_path)
    runtime.close()


def test_timeout_with_failed_close_keeps_profile_lease(
    tmp_path: Path, playwright: _Playwright
) -> None:
    runtime, parent, _brain, _log = _runtime_setup(tmp_path, [_handoff_call()], timeout=0.05)
    with pytest.raises(UserAuthRequired) as paused:
        runtime.run("login")
    continuation = paused.value.continuation
    assert continuation is not None
    playwright.chromium.fail_close = True
    deadline = time.monotonic() + 5
    while not runtime.user_auth_expired(continuation) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert runtime.user_auth_expired(continuation)
    assert parent.closed_resources == 1
    # Закрытие Chromium не подтверждено — lease профиля не освобождается.
    assert _lease_is_held(tmp_path)
    playwright.chromium.fail_close = False
    runtime.cancel_pending()
    runtime.close()


def test_handoff_tool_log_records_pause_without_window_contents(
    tmp_path: Path, playwright: _Playwright
) -> None:
    runtime, _parent, _brain, log_path = _runtime_setup(tmp_path, [_handoff_call()])
    with pytest.raises(UserAuthRequired):
        runtime.run("login")
    records = ToolCallLogger(log_path).read_recent()
    assert [(item.tool, item.ok, item.error) for item in records] == [
        ("desktop_browser", False, NEEDS_USER_AUTH)
    ]
    assert records[0].args == {"operation": "login_handoff", "url": LOGIN_URL}
    runtime.cancel_pending()
    runtime.close()


def test_gateway_resume_refuses_foreign_subject(tmp_path: Path) -> None:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "log"))
    calls: list[ToolRequest] = []

    def handler(request: ToolRequest) -> ToolResult:
        calls.append(request)
        return ToolResult.success({"ok": True})

    registry.register("desktop_browser", handler, capability="exec", execution_targets={"desktop"})
    gateway = ToolGateway(
        registry, run_policy=RunPolicy(mode="desktop", execution_target="desktop")
    )
    pause = UserAuthRequest(
        attempt_id="a1",
        tool="desktop_browser",
        page_id="p",
        url=LOGIN_URL,
        title="Sign in",
        window_ready_monotonic=1.0,
        control_returned=True,
    )
    request = ToolRequest("desktop_browser", {"operation": "login_handoff", "url": LOGIN_URL})
    with pytest.raises(ValueError, match="user_auth_subject_mismatch"):
        gateway.call_user_auth_resumed(ToolRequest("other", {}), pause)
    with pytest.raises(ValueError, match="user_auth_subject_mismatch"):
        gateway.call_user_auth_resumed(
            request, UserAuthRequest(**{**pause.__dict__, "session_id": "s2"})
        )
    with pytest.raises(ValueError, match="user_auth_subject_mismatch"):
        gateway.call_user_auth_resumed(
            request, UserAuthRequest(**{**pause.__dict__, "control_returned": False})
        )
    assert calls == []
    assert gateway.call_user_auth_resumed(request, pause).ok
    assert calls[0].user_auth_resume == "a1"


# --- E02-T04 rework (PR #181 round 1): регрессии на findings Codex ------------


def test_snapshot_masks_shadow_dom_password_and_otp(
    tmp_path: Path, playwright: _Playwright
) -> None:
    tool = _tool(tmp_path)
    opened = tool.handle(ToolRequest("desktop_browser", {"operation": "open", "url": LOGIN_URL}))
    assert opened.ok
    page = playwright.chromium.contexts[0].pages[0]
    page.type_shadow_credentials()

    snapshot = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))

    assert snapshot.ok
    text = str(snapshot.data["snapshot"])
    assert PASSWORD not in text and OTP not in text
    assert snapshot.data["masked_sensitive_fields"] == 2
    # Значения в shadow DOM восстановлены после снимка.
    assert page.shadow_fields[0].value == PASSWORD
    assert page.shadow_fields[1].value == OTP
    tool.close()


def test_snapshot_withheld_when_masking_unconfirmed(
    tmp_path: Path, playwright: _Playwright
) -> None:
    tool = _tool(tmp_path)
    opened = tool.handle(ToolRequest("desktop_browser", {"operation": "open", "url": LOGIN_URL}))
    assert opened.ok
    page = playwright.chromium.contexts[0].pages[0]
    page.fields[1].maskable = False  # маскировка не может подтвердить поле
    page.type_credentials()

    snapshot = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))

    assert not snapshot.ok
    assert "masking could not be confirmed" in str(snapshot.error)
    assert "snapshot" not in snapshot.data
    assert PASSWORD not in repr(snapshot.data)
    tool.close()


def test_failed_close_during_handoff_restart_keeps_lease(
    tmp_path: Path, playwright: _Playwright
) -> None:
    tool = _tool(tmp_path)
    opened = tool.handle(
        ToolRequest("desktop_browser", {"operation": "open", "url": "https://shop.example.test/"})
    )
    assert opened.ok
    playwright.chromium.fail_close = True

    # Перезапуск headless -> headed падает на закрытии: headed-окно не
    # открывается, handoff не начинается.
    refused = tool.handle(_handoff())
    assert not refused.ok
    assert [call["headless"] for call in playwright.chromium.persistent_calls] == [True]

    # Повторный cleanup не превращает упавшее закрытие в подтверждённое:
    # lease профиля остаётся удержанным.
    with pytest.raises(RuntimeError, match="chromium did not exit"):
        tool.close()
    assert _lease_is_held(tmp_path)
    assert tool._service.unconfirmed_close_count() == 1
    # Повторный cleanup — no-op на уровне сервиса: lease не освобождается.
    tool.close()
    assert _lease_is_held(tmp_path)
    assert tool._service.unconfirmed_close_count() == 1
    playwright.chromium.fail_close = False


def test_gateway_approval_on_resume_keeps_attempt_id(tmp_path: Path) -> None:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "log"))
    calls: list[ToolRequest] = []

    def handler(request: ToolRequest) -> ToolResult:
        calls.append(request)
        return ToolResult.success({"ok": True})

    registry.register("desktop_browser", handler, capability="exec", execution_targets={"desktop"})
    scope = DesktopApprovalScope(
        tool="desktop_browser",
        action="login_handoff",
        target_pattern=LOGIN_URL,
        risk_class="user_handoff",
    )
    ask_rule = DesktopApprovalRule.create(effect="ask", source="persistent", scope=scope)
    security = DesktopPathSecurity(
        home=tmp_path, policy_store_path=tmp_path / ".run" / "desktop_approvals.json"
    )

    def gateway_with(rules: list[DesktopApprovalRule]) -> ToolGateway:
        return ToolGateway(
            registry,
            approval_context=ApprovalContext(
                safe_mode=True,
                session_id="desktop-session",
                approved_categories=set(),
                execution_target="desktop",
                desktop_policy=DesktopPolicyRuntime(rules),
                desktop_security=security,
            ),
            run_policy=RunPolicy(mode="desktop", execution_target="desktop"),
        )

    pause = UserAuthRequest(
        attempt_id="a1",
        tool="desktop_browser",
        page_id="p",
        url=LOGIN_URL,
        title="Sign in",
        window_ready_monotonic=1.0,
        session_id="desktop-session",
        control_returned=True,
    )
    request = ToolRequest("desktop_browser", {"operation": "login_handoff", "url": LOGIN_URL})

    with pytest.raises(ApprovalRequired) as required:
        gateway_with([ask_rule]).call_user_auth_resumed(request, pause)
    # Approval на продолжение несёт тот же attempt_id.
    assert required.value.request.user_auth_attempt_id == "a1"
    assert calls == []

    once_allow = DesktopApprovalRule.create(effect="allow", source="once", scope=scope)
    result = gateway_with([ask_rule, once_allow]).call_approved_once(
        request, required.value.request
    )

    assert result.ok
    # Подтверждение завершило ту же попытку, а не запустило новый handoff.
    assert calls[0].user_auth_resume == "a1"


class _StreamingBrain:
    supports_native_tools = True

    def __init__(self) -> None:
        self.calls = 0
        self.messages_seen: list[list[LLMMessage]] = []

    def generate(self, messages, config=None, tools=None):  # type: ignore[no-untyped-def]
        del config, tools
        self.calls += 1
        self.messages_seen.append(list(messages))
        if self.calls == 1:
            return _handoff_call()
        return LLMResult(text="streaming cannot login")

    def generate_stream_events(self, messages, config=None, tools=None, cancellation_token=None):  # type: ignore[no-untyped-def]
        result = self.generate(messages, config, tools)
        for call in result.tool_calls:
            yield ToolCallCompleted(call=call)
        if result.text:
            yield TextDelta(text=result.text)
        yield Done()


def test_streaming_path_refuses_login_handoff_before_window(
    tmp_path: Path, playwright: _Playwright
) -> None:
    registry = ToolRegistry(call_logger=ToolCallLogger(tmp_path / "log"))
    tool = _tool(tmp_path)
    registry.register(
        "desktop_browser",
        tool,
        capability="exec",
        risk_classes=["network", "external_side_effect"],
        execution_targets={"desktop"},
    )
    policy = RunPolicy(mode="desktop", execution_target="desktop")
    brain = _StreamingBrain()

    events = list(
        AgentToolLoop(max_iterations=4).run_stream_events(
            brain=brain,  # type: ignore[arg-type]
            gateway=ToolGateway(registry, run_policy=policy),
            messages=[LLMMessage(role="user", content="login please")],
            tools=registry.list_tool_specs(policy=policy),
        )
    )

    completed = [event for event in events if isinstance(event, ToolCallCompleted)]
    assert len(completed) == 1
    assert completed[0].result is not None
    assert not completed[0].result.ok
    assert completed[0].result.meta == {
        "reason_code": "user_auth_streaming_unsupported",
        "execution_started": False,
    }
    # Отказ до открытия окна: Chromium не стартовал вовсе.
    assert playwright.chromium.persistent_calls == []
    assert brain.calls == 2
    tool.close()


# --- opt-in: реальный headless Chromium, редакция aria_snapshot --------------

LOGIN_FORM_HTML = """<!doctype html><html><body><h1>Sign in</h1>
<label>Email <input name="email" autocomplete="username"></label>
<label>Password <input type="password" name="pw"></label>
<label>Code <input name="otp" autocomplete="one-time-code"></label>
</body></html>
"""


class _FormHandler(BaseHTTPRequestHandler):
    html = LOGIN_FORM_HTML

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        payload = self.html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        del args


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real headless Chromium redaction check is opt-in",
)
def test_real_chromium_snapshot_redacts_password_and_otp(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FormHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/login"
    tool = _tool(tmp_path)
    try:
        opened = tool.handle(ToolRequest("desktop_browser", {"operation": "open", "url": url}))
        assert opened.ok, opened.error
        for label, value in (
            ("Email", "owner@example.test"),
            ("Password", PASSWORD),
            ("Code", OTP),
        ):
            typed = tool.handle(
                ToolRequest(
                    "desktop_browser",
                    {
                        "operation": "input",
                        "selector_type": "label",
                        "selector": label,
                        "value": value,
                    },
                )
            )
            assert typed.ok, typed.error
        snapshot = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))
        assert snapshot.ok, snapshot.error
        text = str(snapshot.data["snapshot"])
        assert PASSWORD not in text and OTP not in text
        assert "owner@example.test" in text
        assert snapshot.data["masked_sensitive_fields"] == 2
        read_back = tool.handle(
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "input",
                    "selector_type": "label",
                    "selector": "Password",
                    "value": PASSWORD,
                },
            )
        )
        assert read_back.ok, read_back.error
    finally:
        tool.close()
        server.shutdown()


SHADOW_PASSWORD = "shadow-pw-synthetic-123"
SHADOW_OTP = "987654"

SHADOW_FORM_HTML = """<!doctype html><html><body><h1>Sign in</h1>
<label>Email <input name="email" autocomplete="username"></label>
<label>Password <input type="password" name="pw"></label>
<div id="shadow-host"></div>
<script>
  const root = document.getElementById('shadow-host').attachShadow({mode: 'open'});
  root.innerHTML =
    '<label>Vault key <input type="password" name="spw"></label>' +
    '<label>Vault code <input name="sotp" autocomplete="one-time-code"></label>';
  root.querySelector('[name=spw]').value = 'SHADOW_PASSWORD_VALUE';
  root.querySelector('[name=sotp]').value = 'SHADOW_OTP_VALUE';
</script>
</body></html>
""".replace("SHADOW_PASSWORD_VALUE", SHADOW_PASSWORD).replace("SHADOW_OTP_VALUE", SHADOW_OTP)


class _ShadowFormHandler(_FormHandler):
    html = SHADOW_FORM_HTML


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real headless Chromium redaction check is opt-in",
)
def test_real_chromium_snapshot_redacts_shadow_dom_secrets(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ShadowFormHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/login"
    tool = _tool(tmp_path)
    try:
        opened = tool.handle(ToolRequest("desktop_browser", {"operation": "open", "url": url}))
        assert opened.ok, opened.error
        for label, value in (("Email", "owner@example.test"), ("Password", PASSWORD)):
            typed = tool.handle(
                ToolRequest(
                    "desktop_browser",
                    {
                        "operation": "input",
                        "selector_type": "label",
                        "selector": label,
                        "value": value,
                    },
                )
            )
            assert typed.ok, typed.error
        snapshot = tool.handle(ToolRequest("desktop_browser", {"operation": "snapshot"}))
        assert snapshot.ok, snapshot.error
        text = str(snapshot.data["snapshot"])
        assert PASSWORD not in text
        assert SHADOW_PASSWORD not in text and SHADOW_OTP not in text
        assert "owner@example.test" in text
        # Light password + shadow password + shadow OTP замаскированы.
        assert snapshot.data["masked_sensitive_fields"] == 3
    finally:
        tool.close()
        server.shutdown()
