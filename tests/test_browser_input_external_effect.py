"""E02-T05 rework, проверка п.3: являются ли input/select только local_mutation?

Классификация ``local_mutation`` (core/browser_effects.py) исходит из того,
что ввод и выбор значения ничего не отправляют наружу до явного
submit/click. Этот тест проверяет это утверждение на настоящем браузере:
страница вешает обработчики ``input``/``change``, которые сразу шлют
``fetch`` локальному тестовому HTTP-серверу (автосейв/реактивный UI), без
отдельной кнопки Submit. Операции выполняются штатным DesktopBrowserTool
(Playwright ``fill``/``select_option``), и фиксируется, дошёл ли запрос
до сервера.

Результат (зафиксирован при запуске на хосте, см. ROADMAP/STATE):
внешний эффект ПРОИСХОДИТ — ``fill`` диспатчит ``input``, ``select_option``
диспатчит ``change``/``input``, и обработчики страницы успевают отправить
запрос до любого submit/click. Следовательно, ``local_mutation`` не
является гарантией отсутствия внешнего эффекта; класс означает только
«агент не выполняет явный submit». Обнаружение запроса после отправки
не считается предотвращением.

Opt-in, только headless: ``SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1``.
"""

from __future__ import annotations

import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from core.browser_service import BrowserProfile
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest
from tools.desktop_browser_tool import DesktopBrowserTool

FORM_HTML = """<!doctype html><html><body>
<input id="name" type="text">
<select id="kind">
  <option value="a">A</option>
  <option value="b">B</option>
</select>
<script>
const beacon = (kind) => {
  fetch('/beacon?kind=' + encodeURIComponent(kind), {method: 'POST'});
};
document.getElementById('name').addEventListener('input', () => beacon('input'));
document.getElementById('name').addEventListener('change', () => beacon('input-change'));
document.getElementById('kind').addEventListener('change', () => beacon('select'));
document.getElementById('kind').addEventListener('input', () => beacon('select-input'));
</script>
</body></html>
"""

BEACONS: list[str] = []
BEACONS_LOCK = threading.Lock()


class _AutosaveHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path == "/form":
            payload = FORM_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        parsed = urlparse(self.path)
        if parsed.path == "/beacon":
            kinds = parse_qs(parsed.query).get("kind", [])
            with BEACONS_LOCK:
                BEACONS.extend(kinds)
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        del args


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop-approvals.json",
    )


def _wait_for_beacons(expected: set[str], timeout_s: float = 5.0) -> set[str]:
    deadline = time.monotonic() + timeout_s
    seen: set[str] = set()
    while time.monotonic() < deadline:
        with BEACONS_LOCK:
            seen = set(BEACONS)
        if expected <= seen:
            return seen
        time.sleep(0.05)
    return seen


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_input_and_select_trigger_external_requests_without_submit(
    tmp_path: Path,
) -> None:
    with BEACONS_LOCK:
        BEACONS.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _AutosaveHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )
    try:
        opened = tool.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{base}/form"})
        )
        assert opened.ok, opened.data

        filled = tool.handle(
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "input",
                    "selector_type": "css",
                    "selector": "#name",
                    "value": "draft",
                },
            )
        )
        assert filled.ok, filled.data

        selected = tool.handle(
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "select",
                    "selector_type": "css",
                    "selector": "#kind",
                    "value": "b",
                },
            )
        )
        assert selected.ok, selected.data

        # Никакого submit/click не было. Если обработчики страницы
        # отправили fetch, внешний эффект произошёл уже на input/select.
        seen = _wait_for_beacons({"input", "select"})
        assert "input" in seen, f"input beacon missing, saw: {sorted(seen)}"
        assert "select" in seen, f"select beacon missing, saw: {sorted(seen)}"
    finally:
        tool.close()
        server.shutdown()
        server.server_close()
