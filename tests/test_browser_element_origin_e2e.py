"""E02-T09 rework round 5: origin документа элемента на настоящем браузере.

Cross-origin iframe подменяет ``ownerDocument`` и ``window.origin``
через ``Object.defineProperty`` — проверка origin документа элемента
не должна принимать подставленное значение: ввод в чужой iframe по
гранту верхней страницы не выполняется. Same-origin iframe и srcdoc
(унаследованный origin) по гранту работают; opaque origin (sandbox
без ``allow-same-origin``) — fail-closed.

Rework round 6: мутация привязана к проверенному элементу и
документу — навигация same-origin iframe на другой origin между
проверкой и вводом не приводит к заполнению чужого документа.

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
from shared.models import ToolRequest
from tools.desktop_browser_tool import _DOCUMENT_ORIGIN_PROPERTY, DesktopBrowserTool


def _sub_html(foreign_origin: str) -> str:
    # Same-origin iframe: скрипт пытается перезаписать запечатанный
    # init-скриптом origin чужим значением. Истинный origin документа
    # захвачен до скриптов страницы и подмене не поддаётся.
    return f"""<!doctype html><html><body>
<input id="sub-input">
<script>
  try {{ Object.defineProperty(
    window, '{_DOCUMENT_ORIGIN_PROPERTY}', {{value: '{foreign_origin}'}}); }} catch (e) {{}}
  try {{ window.{_DOCUMENT_ORIGIN_PROPERTY} = '{foreign_origin}'; }} catch (e) {{}}
  try {{ delete window.{_DOCUMENT_ORIGIN_PROPERTY}; }} catch (e) {{}}
</script>
</body></html>"""


def _main_html(origin_a: str, origin_b: str) -> str:
    return f"""<!doctype html><html><body>
<input id="main-input">
<iframe id="xo" src="{origin_b}/page"></iframe>
<iframe id="srcdoc" srcdoc="&lt;input id='sd-input'&gt;"></iframe>
<iframe id="srcdoc-sandbox" sandbox="allow-scripts" srcdoc="&lt;input id='sds-input'&gt;"></iframe>
<iframe id="same" src="{origin_a}/sub"></iframe>
</body></html>"""


def _xo_html(origin_a: str) -> str:
    return f"""<!doctype html><html><body>
<input id="xo-input">
<script>
  const el = document.getElementById('xo-input');
  Object.defineProperty(el, 'ownerDocument',
    {{value: {{defaultView: {{location: {{origin: '{origin_a}'}}}}}}}});
  try {{ Object.defineProperty(window, 'origin', {{value: '{origin_a}'}}); }} catch (e) {{}}
  try {{ Object.defineProperty(
    window, '{_DOCUMENT_ORIGIN_PROPERTY}', {{value: '{origin_a}'}}); }} catch (e) {{}}
  try {{ window.{_DOCUMENT_ORIGIN_PROPERTY} = '{origin_a}'; }} catch (e) {{}}
  try {{ delete window.{_DOCUMENT_ORIGIN_PROPERTY}; }} catch (e) {{}}
</script>
</body></html>"""


def _navsub_html(origin_b: str) -> str:
    # Same-origin iframe с недоступным для редактирования полем:
    # fill ждёт готовности, а документ за это время навигируется на
    # другой origin. Мутация на проверенном ElementHandle обязана
    # упасть (detached), а не заполнить поле чужого документа.
    return f"""<!doctype html><html><body>
<input id="nav-input" disabled>
<script>
  setTimeout(() => {{ location.href = '{origin_b}/navtarget'; }}, 800);
</script>
</body></html>"""


def _navtarget_html() -> str:
    # Документ другого origin с полем того же id: locator.fill с
    # повторным разрешением селектора заполнил бы его по старому
    # гранту. oninput сообщает серверу о любом вводе.
    return """<!doctype html><html><body>
<input id="nav-input" oninput="fetch('/report?v=' + encodeURIComponent(this.value))">
</body></html>"""


def _navmain_html(origin_a: str) -> str:
    return f"""<!doctype html><html><body>
<iframe id="nav" src="{origin_a}/navsub"></iframe>
</body></html>"""


class _Handler(BaseHTTPRequestHandler):
    origin_a = ""
    origin_b = ""
    reports: list[str] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server naming
        if self.path.startswith("/report"):
            type(self).reports.append(self.path)
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/":
            body = _main_html(self.origin_a, self.origin_b)
        elif self.path == "/sub":
            body = _sub_html(self.origin_b)
        elif self.path == "/page":
            body = _xo_html(self.origin_a)
        elif self.path == "/navmain":
            body = _navmain_html(self.origin_a)
        elif self.path == "/navsub":
            body = _navsub_html(self.origin_b)
        elif self.path == "/navtarget":
            body = _navtarget_html()
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


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_element_origin_check_resists_spoofing(tmp_path: Path) -> None:
    server_a = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server_b = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.origin_a = f"http://127.0.0.1:{server_a.server_address[1]}"
    _Handler.origin_b = f"http://127.0.0.1:{server_b.server_address[1]}"
    for server in (server_a, server_b):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    origin_a = _Handler.origin_a
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )
    try:
        opened = tool.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{origin_a}/"})
        )
        assert opened.ok, opened.data

        def edit(selector: str) -> bool:
            result = tool.handle(
                ToolRequest(
                    "desktop_browser",
                    {
                        "operation": "input",
                        "selector_type": "css",
                        "selector": selector,
                        "value": "x",
                    },
                    expected_browser_origin=origin_a,
                )
            )
            return result.ok

        # Документ верхней страницы и same-origin iframe — по гранту
        # (скрипт iframe пытается перезаписать запечатанный origin
        # чужим значением — истинный origin не подменяется).
        assert edit("#main-input")
        assert edit("#same >> internal:control=enter-frame >> #sub-input")
        # srcdoc наследует origin родителя — по гранту.
        assert edit("#srcdoc >> internal:control=enter-frame >> #sd-input")
        # Cross-origin iframe с подменой ownerDocument/window.origin —
        # ввод не выполняется.
        assert not edit("#xo >> internal:control=enter-frame >> #xo-input")
        # Opaque origin (sandbox без allow-same-origin) — fail-closed.
        assert not edit("#srcdoc-sandbox >> internal:control=enter-frame >> #sds-input")
    finally:
        tool.close()
        for server in (server_a, server_b):
            server.shutdown()
            server.server_close()


@pytest.mark.skipif(
    os.environ.get("SLAVIK_RUN_BROWSER_PERSISTENCE_E2E") != "1",
    reason="real-browser E2E is opt-in (SLAVIK_RUN_BROWSER_PERSISTENCE_E2E=1)",
)
def test_edit_bound_to_verified_element_document(tmp_path: Path) -> None:
    # Rework round 6 (P1): поле в same-origin iframe недоступно для
    # редактирования, fill ждёт готовности; iframe за это время
    # навигируется на другой origin, где есть поле с тем же id.
    # Мутация на проверенном ElementHandle падает (detached) и не
    # заполняет чужой документ по старому гранту.
    import time

    server_a = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server_b = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.origin_a = f"http://127.0.0.1:{server_a.server_address[1]}"
    _Handler.origin_b = f"http://127.0.0.1:{server_b.server_address[1]}"
    _Handler.reports = []
    for server in (server_a, server_b):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    origin_a = _Handler.origin_a
    tool = DesktopBrowserTool(
        _security(tmp_path),
        profile=BrowserProfile(profile_id="default", owner_id="local"),
        profile_root=tmp_path / "profiles",
    )
    try:
        opened = tool.handle(
            ToolRequest("desktop_browser", {"operation": "open", "url": f"{origin_a}/navmain"})
        )
        assert opened.ok, opened.data

        result = tool.handle(
            ToolRequest(
                "desktop_browser",
                {
                    "operation": "input",
                    "selector_type": "css",
                    "selector": "#nav >> internal:control=enter-frame >> #nav-input",
                    "value": "x",
                    "timeout_seconds": 5,
                },
                expected_browser_origin=origin_a,
            )
        )
        assert not result.ok
        # Чужой документ ввода не получил: oninput не срабатывал.
        time.sleep(0.5)
        assert _Handler.reports == []
    finally:
        tool.close()
        for server in (server_a, server_b):
            server.shutdown()
            server.server_close()
