"""E02-T10: захват browser_state на настоящем браузере.

Страница со списком заказов: у каждой записи свой статус. Клик по
«Оплатить» записи X асинхронно меняет статус именно X — захват внутри
вызова click подтверждает изменение (факты сравнения, без сырых
строк). Клик, изменивший только Y, уже оплаченный X, создание сущности
(count 0 -> 1), submit формы и навигация на другой документ
(непрерывность наблюдения разорвана -> захвата нет) покрыты отдельными
сценариями. Отдельный сценарий: скрипт страницы переопределяет
``document.querySelectorAll`` и фабрикует создание сущности —
подсчёт через движок селекторов драйвера подмене не подвержен.
Ещё сценарий: обе страницы подменяют ``performance.timeOrigin``
одинаковым значением — запечатанный init-скриптом токен документа
всё равно меняется при навигации, непрерывности нет.
Подтверждённый захват проходит штатный
``VerifierRuntime.verify_browser_postconditions``.

Opt-in, только headless: ``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``.
"""

from __future__ import annotations

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from core.browser_service import BrowserProfile
from core.desktop_security import DesktopPathSecurity
from core.mwv.models import VerificationStatus
from core.mwv.verifier_runtime import VerifierRuntime
from shared.models import JSONValue, ToolRequest, ToolResult
from tools.desktop_browser_tool import DesktopBrowserTool

_ORDERS_HTML = """<!doctype html><html><body>
<div id="order-X"><span class="status">Черновик</span><button id="pay-x">Оплатить</button></div>
<div id="order-Y"><span class="status">Черновик</span><button id="pay-y">Оплатить</button></div>
<button id="create-z">Создать</button>
<button id="nav-x">Перейти</button>
<form id="order-form"><input id="note"></form>
<script>
  const pay = (id) => setTimeout(() => {
    document.querySelector('#' + id + ' .status').textContent = 'Оплачено';
  }, 300);
  document.getElementById('pay-x').onclick = () => pay('order-X');
  document.getElementById('pay-y').onclick = () => pay('order-Y');
  document.getElementById('create-z').onclick = () => setTimeout(() => {
    const d = document.createElement('div');
    d.id = 'order-Z';
    d.textContent = 'Новая';
    document.body.appendChild(d);
  }, 300);
  document.getElementById('nav-x').onclick = () => { location.href = '/other'; };
  document.getElementById('order-form').onsubmit = (e) => {
    e.preventDefault();
    pay('order-X');
  };
</script>
</body></html>"""

_OTHER_HTML = """<!doctype html><html><body>
<div id="order-X"><span class="status">Оплачено</span></div>
</body></html>"""

_SPOOF_HTML = """<!doctype html><html><body>
<button id="fake-create">Фейк</button>
<button id="real-create">Создать</button>
<script>
  const orig = document.querySelectorAll.bind(document);
  document.querySelectorAll = function (sel) {
    if (sel.includes('order-Z')) {
      return { length: window.__faked ? 1 : 0 };
    }
    return orig(sel);
  };
  document.getElementById('fake-create').onclick = () => { window.__faked = true; };
  document.getElementById('real-create').onclick = () => setTimeout(() => {
    const d = document.createElement('div');
    d.id = 'order-Z';
    d.textContent = 'Новая';
    document.body.appendChild(d);
  }, 300);
</script>
</body></html>"""


_TOKEN_A_HTML = """<!doctype html><html><body>
<div id="order-X"><span class="status">Черновик</span></div>
<button id="nav-token">Перейти</button>
<script>
  try { Object.defineProperty(performance, 'timeOrigin', {value: 1}); } catch (e) {}
  document.getElementById('nav-token').onclick = () => { location.href = '/token-b'; };
</script>
</body></html>"""

_TOKEN_B_HTML = """<!doctype html><html><body>
<div id="order-X"><span class="status">Оплачено</span></div>
<script>
  try { Object.defineProperty(performance, 'timeOrigin', {value: 1}); } catch (e) {}
</script>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/":
            body = _ORDERS_HTML
        elif self.path == "/other":
            body = _OTHER_HTML
        elif self.path == "/spoof":
            body = _SPOOF_HTML
        elif self.path == "/token-a":
            body = _TOKEN_A_HTML
        elif self.path == "/token-b":
            body = _TOKEN_B_HTML
        else:
            self.send_response(404)
            self.end_headers()
            return
        payload = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        del args


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def _text_declaration() -> dict[str, JSONValue]:
    return {
        "entity_id": "order-X",
        "selector_type": "css",
        "selector": ".status",
        "expect": {"kind": "text", "before": "Черновик", "after": "Оплачено"},
    }


def _verify(args: dict[str, JSONValue], result: ToolResult) -> VerificationStatus:
    outcome = VerifierRuntime().verify_browser_postconditions([("desktop_browser", args, result)])
    assert outcome is not None
    return outcome.status


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_browser_state_capture_on_real_browser(tmp_path: Path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )

    def act(args: dict[str, JSONValue]) -> ToolResult:
        return tool.handle(ToolRequest("desktop_browser", args))

    def reload_orders() -> None:
        result = act({"operation": "navigate", "url": f"{origin}/"})
        assert result.ok, result.data

    try:
        opened = act({"operation": "open", "url": f"{origin}/"})
        assert opened.ok, opened.data

        # Клик по X: статус X меняется асинхронно, опрос внутри вызова
        # дожидается значения; факты подтверждают, verifier — PASSED.
        click_args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#pay-x",
            "timeout_seconds": 5,
            "browser_state": _text_declaration(),
        }
        result = act(click_args)
        assert result.ok, result.data
        assert result.data["browser_state"] == {
            "kind": "text",
            "entity_id": "order-X",
            "before_matches_after": False,
            "after_matches_after": True,
            "before_matches_expected_before": True,
        }
        assert _verify(click_args, result) == VerificationStatus.PASSED

        # X уже «Оплачено»: повторный клик не доказывает изменение.
        result = act(click_args)
        assert result.ok, result.data
        facts = result.data["browser_state"]
        assert isinstance(facts, dict)
        assert facts["before_matches_after"] is True
        assert _verify(click_args, result) == VerificationStatus.FAILED

        # Клик по Y при объявлении на X: совпадение у Y не засчитывается.
        reload_orders()
        y_args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#pay-y",
            "timeout_seconds": 3,
            "browser_state": _text_declaration(),
        }
        result = act(y_args)
        assert result.ok, result.data
        facts = result.data["browser_state"]
        assert isinstance(facts, dict)
        assert facts["after_matches_after"] is False
        assert _verify(y_args, result) == VerificationStatus.FAILED

        # Создание сущности: count 0 -> 1 по заранее известному ID.
        reload_orders()
        count_args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#create-z",
            "timeout_seconds": 5,
            "browser_state": {"entity_id": "order-Z", "expect": {"kind": "count", "after": 1}},
        }
        result = act(count_args)
        assert result.ok, result.data
        assert result.data["browser_state"] == {
            "kind": "count",
            "entity_id": "order-Z",
            "before_count": 0,
            "after_count": 1,
        }
        assert _verify(count_args, result) == VerificationStatus.PASSED

        # Submit формы (Enter в поле) подтверждает так же, как click.
        reload_orders()
        submit_args: dict[str, JSONValue] = {
            "operation": "submit",
            "selector_type": "css",
            "selector": "#note",
            "timeout_seconds": 5,
            "browser_state": _text_declaration(),
        }
        result = act(submit_args)
        assert result.ok, result.data
        facts = result.data["browser_state"]
        assert isinstance(facts, dict)
        assert facts["after_matches_after"] is True
        assert _verify(submit_args, result) == VerificationStatus.PASSED
    finally:
        tool.close()
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_browser_state_navigation_breaks_continuity(tmp_path: Path) -> None:
    # Клик навигирует на другой документ, где #order-X уже «Оплачено»:
    # непрерывность наблюдения установить нельзя — захвата нет,
    # результат остаётся неподтверждённым.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )
    try:
        opened = tool.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{origin}/"})
        )
        assert opened.ok, opened.data

        args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#nav-x",
            "timeout_seconds": 5,
            "browser_state": _text_declaration(),
        }
        result = tool.handle(ToolRequest("desktop_browser", args))
        assert result.ok, result.data
        assert "browser_state" not in result.data
        assert _verify(args, result) == VerificationStatus.FAILED
    finally:
        tool.close()
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_browser_state_spoofed_time_origin_breaks_continuity(tmp_path: Path) -> None:
    # Обе страницы подменяют performance.timeOrigin одинаковым
    # значением: токен, прочитанный в main world, совпал бы до и
    # после навигации, и захват принял бы статус одноимённого
    # элемента уже в другом документе. Токен запечатывается
    # init-скриптом до скриптов страницы и уникален для документа:
    # навигация ломает непрерывность — захвата нет (UNVERIFIED).
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )
    try:
        opened = tool.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{origin}/token-a"})
        )
        assert opened.ok, opened.data

        args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#nav-token",
            "timeout_seconds": 5,
            "browser_state": _text_declaration(),
        }
        result = tool.handle(ToolRequest("desktop_browser", args))
        assert result.ok, result.data
        assert "browser_state" not in result.data
        assert _verify(args, result) == VerificationStatus.FAILED
    finally:
        tool.close()
        server.shutdown()
        server.server_close()


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_browser_state_count_ignores_spoofed_query_selector_all(tmp_path: Path) -> None:
    # Скрипт страницы переопределяет document.querySelectorAll и после
    # клика фабрикует length=1 для order-Z, не создавая элемент.
    # Подсчёт через движок селекторов драйвера (utility world) подмене
    # не подвержен: подтверждения нет. Реальное создание на той же
    # странице подтверждается как прежде.
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_address[1]}"
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )

    def act(args: dict[str, JSONValue]) -> ToolResult:
        return tool.handle(ToolRequest("desktop_browser", args))

    try:
        opened = act({"operation": "open", "url": f"{origin}/spoof"})
        assert opened.ok, opened.data

        fake_args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#fake-create",
            "timeout_seconds": 3,
            "browser_state": {"entity_id": "order-Z", "expect": {"kind": "count", "after": 1}},
        }
        result = act(fake_args)
        assert result.ok, result.data
        facts = result.data["browser_state"]
        assert isinstance(facts, dict)
        assert facts["before_count"] == 0
        assert facts["after_count"] == 0
        assert _verify(fake_args, result) == VerificationStatus.FAILED

        reloaded = act({"operation": "navigate", "url": f"{origin}/spoof"})
        assert reloaded.ok, reloaded.data
        real_args: dict[str, JSONValue] = {
            "operation": "click",
            "selector_type": "css",
            "selector": "#real-create",
            "timeout_seconds": 5,
            "browser_state": {"entity_id": "order-Z", "expect": {"kind": "count", "after": 1}},
        }
        result = act(real_args)
        assert result.ok, result.data
        assert result.data["browser_state"] == {
            "kind": "count",
            "entity_id": "order-Z",
            "before_count": 0,
            "after_count": 1,
        }
        assert _verify(real_args, result) == VerificationStatus.PASSED
    finally:
        tool.close()
        server.shutdown()
        server.server_close()
