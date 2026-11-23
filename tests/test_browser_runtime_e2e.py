"""E02-T08 (вариант C): единый real-browser E2E через полный стек.

Scripted brain → DesktopRuntime → ToolGateway → настоящий
DesktopBrowserTool → Chromium с persistent profile. Сценарий
автоматического среза T08:

1. Первый runtime логинится в тестовом authenticated app (серверная
   сессия; cookie живёт в persistent profile).
2. Runtime завершён: браузер остановлен, profile lease освобождён.
3. НОВЫЙ DesktopRuntime на том же profile: агент читает private
   page, меняет значение записи X (click + browser_state), и
   канонический outcome — DONE + final_success.
4. Серверная инвалидация сессии: попытка изменения значения успех
   не публикует, значение на сервере не меняется, реальная ошибка
   действия сохраняется.

Шаги 2 и 8 буквального T08 (видимый headed handoff; агент сам
выбирает login_handoff после разлогина) — ручная приёмка на ПК,
в этот тест не входят: T08 остаётся PARTIAL (решение владельца
2026-11-23, вариант C). Автодетект разлогина в production-код не
добавлялся.

Opt-in, только headless: ``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``.
"""

from __future__ import annotations

import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.approval_policy import ApprovalContext
from core.browser_service import BrowserProfile, resolve_browser_profile_dir
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
)
from core.desktop_runtime import DesktopExecutionControl, DesktopRuntime
from core.desktop_security import DesktopPathSecurity
from core.response_preferences import ResponsePreferences
from core.run_contract import RunPolicy
from core.run_outcome import TerminalReason
from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ToolCall
from shared.models import JSONValue
from tools.desktop_browser_tool import DesktopBrowserTool
from tools.tool_registry import ToolRegistry

_LOGIN_HTML = """<!doctype html><html><body>
<button id="login-btn">Log in</button>
<p id="state">anonymous</p>
<script>
  document.getElementById('login-btn').onclick = async () => {
    await fetch('/api/login', {method: 'POST'});
    localStorage.setItem('slavik_auth', 'yes');
    const marker = document.createElement('div');
    marker.id = 'logged-in';
    marker.textContent = 'logged in';
    document.body.appendChild(marker);
    document.getElementById('state').textContent = 'logged in';
  };
</script>
</body></html>"""

_STATUS_HTML = """<!doctype html><html><body><p id="state">loading</p>
<script>
  const cookie = document.cookie.includes('slavik_session=')
    ? 'cookie=yes' : 'cookie=no';
  const stored = localStorage.getItem('slavik_auth') === 'yes'
    ? 'storage=yes' : 'storage=no';
  document.getElementById('state').textContent = cookie + ' ' + stored;
</script></body></html>"""

_PRIVATE_TEMPLATE = """<!doctype html><html><body>
<p id="private-marker">private content</p>
<div id="order-X"><span class="status">__STATUS__</span><button id="pay-x">Оплатить</button></div>
<script>
  document.getElementById('pay-x').onclick = async () => {
    const response = await fetch('/api/pay', {method: 'POST'});
    if (response.ok) {
      document.querySelector('#order-X .status').textContent = 'Оплачено';
    }
  };
</script>
</body></html>"""


class _AppState:
    """Серверное состояние тестового приложения: сессии и запись X."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions: set[str] = set()
        self.order_x = "Черновик"

    def new_session(self) -> str:
        session_id = uuid.uuid4().hex
        with self.lock:
            self.sessions.add(session_id)
        return session_id

    def session_valid(self, cookie_header: str | None) -> bool:
        if not cookie_header:
            return False
        for part in cookie_header.split(";"):
            name, _, value = part.strip().partition("=")
            if name == "slavik_session":
                with self.lock:
                    return value in self.sessions
        return False

    def pay(self) -> None:
        with self.lock:
            self.order_x = "Оплачено"

    def reset_order(self) -> None:
        with self.lock:
            self.order_x = "Черновик"

    def invalidate_sessions(self) -> None:
        with self.lock:
            self.sessions.clear()

    def order_status(self) -> str:
        with self.lock:
            return self.order_x


class _AppHandler(BaseHTTPRequestHandler):
    state: _AppState

    def _send(self, body: str, *, cookie: str | None = None) -> None:
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        if cookie is not None:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/login":
            self._send(_LOGIN_HTML)
        elif self.path == "/status":
            self._send(_STATUS_HTML)
        elif self.path == "/private":
            if self.state.session_valid(self.headers.get("Cookie")):
                self._send(_PRIVATE_TEMPLATE.replace("__STATUS__", self.state.order_status()))
            else:
                self._send(_LOGIN_HTML)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/api/login":
            session_id = self.state.new_session()
            self._send("{}", cookie=f"slavik_session={session_id}; Path=/; Max-Age=3600")
        elif self.path == "/api/pay":
            if self.state.session_valid(self.headers.get("Cookie")):
                self.state.pay()
                self._send("{}")
            else:
                self.send_response(401)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args: object) -> None:
        del args


class _ScriptedBrain(Brain):
    """Fixture модели: заданная последовательность вызовов, затем ответ."""

    supports_native_tools = True

    def __init__(self, calls: list[ToolCall]) -> None:
        self._calls = calls
        self._index = 0

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del config, tools
        if self._index < len(self._calls):
            call = self._calls[self._index]
            self._index += 1
            return LLMResult(usage=LLMUsage(0, 0, 0), text="", tool_calls=[call])
        return LLMResult(usage=LLMUsage(0, 0, 0), text="Готово")


class _RuntimeParent:
    """Родитель DesktopRuntime с production-формой gateway (desktop target).

    Регистрация desktop_browser повторяет production (core/agent.py):
    настоящий DesktopBrowserTool, capability exec, risk network +
    external_side_effect, только desktop target. Закрытие ресурсов
    останавливает браузер и освобождает profile lease — как
    Agent.close_desktop_resources.
    """

    response_preferences = ResponsePreferences()

    def __init__(self, brain, registry, workspace_root, rules, tool):  # noqa: ANN001
        self.brain = brain
        self.tool_registry = registry
        self.main_config = None
        self.desktop_execution_control = DesktopExecutionControl()
        self._workspace_root = workspace_root
        self._rules = rules
        self._tool = tool

    def _get_main_brain(self):  # noqa: ANN202
        return self.brain

    def _build_tool_gateway(self, *, budget=None):  # noqa: ANN001, ANN202
        del budget
        security = DesktopPathSecurity(
            home=self._workspace_root,
            policy_store_path=self._workspace_root / ".run" / "desktop_approvals.json",
        )
        return ToolGateway(
            self.tool_registry,
            approval_context=ApprovalContext(
                safe_mode=True,
                session_id="session-test",
                approved_categories=set(),
                execution_target="desktop",
                desktop_policy=DesktopPolicyRuntime(self._rules),
                desktop_security=security,
            ),
            run_policy=RunPolicy(mode="desktop", execution_target="desktop"),
        )

    def close_desktop_resources(self) -> None:
        self._tool.close()


def _allow_rule(action: str, target: str, risk_class: str) -> DesktopApprovalRule:
    return DesktopApprovalRule.create(
        effect="allow",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action=action,
            target_pattern=target,
            risk_class=risk_class,
        ),
    )


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def _run_phase(tmp_path: Path, profile_root: Path, profile: BrowserProfile, calls, rules):  # noqa: ANN001, ANN202
    """Один DesktopRuntime на настоящем браузере; runtime закрыт до возврата."""
    tool = DesktopBrowserTool(_security(tmp_path), profile=profile, profile_root=profile_root)
    registry = ToolRegistry()
    registry.register(
        "desktop_browser",
        tool,
        description="Real browser tool for E02-T08 acceptance",
        parameters_schema={"type": "object", "properties": {"operation": {"type": "string"}}},
        capability="exec",
        risk_classes=["network", "external_side_effect"],
        execution_targets={"desktop"},
    )
    parent = _RuntimeParent(_ScriptedBrain(calls), registry, tmp_path, rules, tool)
    runtime = DesktopRuntime(parent)
    try:
        return runtime.run("browser task")
    finally:
        runtime.close()


def _call(call_id: str, arguments: dict[str, JSONValue]) -> ToolCall:
    return ToolCall(id=call_id, name="desktop_browser", arguments=arguments)


def _read_text(outcome, call_id: str) -> str:  # noqa: ANN001
    for executed in outcome.loop_result.tool_calls:
        if executed.call.id == call_id:
            assert executed.result.ok, executed.result.data
            return str(executed.result.data["text"])
    raise AssertionError(f"call {call_id} not executed")


def _pay_declaration() -> dict[str, JSONValue]:
    return {
        "entity_id": "order-X",
        "selector_type": "css",
        "selector": ".status",
        "expect": {"kind": "text", "before": "Черновик", "after": "Оплачено"},
    }


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_persistent_browser_full_runtime_acceptance(tmp_path: Path) -> None:
    state = _AppState()
    handler = type("BoundHandler", (_AppHandler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    profile = BrowserProfile(profile_id="default", owner_id="local")
    profile_root = tmp_path / "profiles"
    try:
        # Фаза 1: первый runtime логинится; сессия остаётся в profile.
        login_outcome = _run_phase(
            tmp_path,
            profile_root,
            profile,
            [
                _call("open-1", {"operation": "open", "url": f"{origin}/login"}),
                _call(
                    "login-1",
                    {"operation": "click", "selector_type": "css", "selector": "#login-btn"},
                ),
                _call(
                    "wait-1",
                    {
                        "operation": "wait",
                        "selector_type": "css",
                        "selector": "#logged-in",
                        "state": "visible",
                        "timeout_seconds": 5,
                    },
                ),
                _call("nav-1", {"operation": "navigate", "url": f"{origin}/status"}),
                _call(
                    "read-1",
                    {"operation": "read", "selector_type": "css", "selector": "#state"},
                ),
            ],
            [_allow_rule("click", "#login-btn", "external_side_effect")],
        )
        assert login_outcome.loop_result.error is None
        assert _read_text(login_outcome, "read-1") == "cookie=yes storage=yes"

        # Runtime завершён: profile на диске, браузер остановлен.
        profile_dir = resolve_browser_profile_dir(profile, root=profile_root)
        assert profile_dir.is_dir()
        assert any(profile_dir.iterdir())

        # Фаза 2: НОВЫЙ DesktopRuntime на том же profile читает private
        # page и меняет значение записи X → DONE + final_success.
        change_outcome = _run_phase(
            tmp_path,
            profile_root,
            profile,
            [
                _call("open-2", {"operation": "open", "url": f"{origin}/private"}),
                _call(
                    "read-2",
                    {"operation": "read", "selector_type": "css", "selector": "#order-X .status"},
                ),
                _call(
                    "pay-2",
                    {
                        "operation": "click",
                        "selector_type": "css",
                        "selector": "#pay-x",
                        "timeout_seconds": 5,
                        "browser_state": _pay_declaration(),
                    },
                ),
                _call("snap-2", {"operation": "snapshot"}),
            ],
            [_allow_rule("click", "#pay-x", "external_side_effect")],
        )
        assert _read_text(change_outcome, "read-2") == "Черновик"
        assert change_outcome.verification.ok
        assert change_outcome.canonical_outcome is not None
        assert change_outcome.canonical_outcome.ui_label == "DONE"
        assert change_outcome.canonical_outcome.is_full_done is True
        assert change_outcome.canonical_outcome.communication_class == "final_success"
        assert state.order_status() == "Оплачено"

        # Фаза 3: серверная инвалидация сессии. Значение сброшено на
        # сервере; попытка изменения не публикует успех, значение не
        # меняется, реальная ошибка действия сохраняется.
        state.reset_order()
        state.invalidate_sessions()
        invalidated_outcome = _run_phase(
            tmp_path,
            profile_root,
            profile,
            [
                _call("open-3", {"operation": "open", "url": f"{origin}/private"}),
                _call(
                    "pay-3",
                    {
                        "operation": "click",
                        "selector_type": "css",
                        "selector": "#pay-x",
                        "timeout_seconds": 3,
                        "browser_state": _pay_declaration(),
                    },
                ),
            ],
            [_allow_rule("click", "#pay-x", "external_side_effect")],
        )
        assert state.order_status() == "Черновик"
        assert invalidated_outcome.canonical_outcome is not None
        assert invalidated_outcome.canonical_outcome.is_full_done is False
        assert invalidated_outcome.canonical_outcome.communication_class != "final_success"
        assert invalidated_outcome.canonical_outcome.ui_label != "DONE"
        # Реальная ошибка действия не подменена неизвестностью.
        assert invalidated_outcome.loop_result.error is not None
        assert invalidated_outcome.loop_result.error in invalidated_outcome.text
        assert invalidated_outcome.canonical_outcome.terminal_reason is TerminalReason.FAILED
    finally:
        server.shutdown()
        server.server_close()
