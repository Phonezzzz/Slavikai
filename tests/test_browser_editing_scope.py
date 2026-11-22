"""E02-T09: scoped browser editing authorization.

Editing-грант на input/select по точному origin в пределах одного run:
первый ввод спрашивает, грант покрывает разные поля того же origin,
смена origin/run — переоценка, конкретные deny/ask поверх гранта,
origin только из фактической страницы, перепроверка перед исполнением.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.approval_policy import ApprovalContext, build_approval_request, decide_request
from core.browser_service import BrowserProfile, BrowserService, browser_origin_of_url
from core.desktop_policy import (
    DesktopAction,
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
    DesktopPolicyStore,
)
from core.desktop_security import DesktopPathSecurity
from core.tool_gateway import ToolGateway
from shared.models import ToolRequest, ToolResult
from tools.desktop_browser_tool import DesktopBrowserTool
from tools.tool_registry import ToolRegistry

ORIGIN = "https://example.test"
RUN_ID = "run-1"


def _security(tmp_path: Path) -> DesktopPathSecurity:
    return DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop_approvals.json",
    )


def _context(
    tmp_path: Path,
    *,
    rules: list[DesktopApprovalRule] | None = None,
    origin: str | None = ORIGIN,
    run_id: str | None = RUN_ID,
    provider: bool = True,
) -> ApprovalContext:
    return ApprovalContext(
        safe_mode=True,
        session_id="desktop-session",
        approved_categories=set(),
        execution_target="desktop",
        desktop_policy=DesktopPolicyRuntime(rules),
        desktop_security=_security(tmp_path),
        run_id=run_id,
        browser_origin=(lambda _page_id: origin) if provider else None,
    )


def _grant_rule(
    *,
    origin: str = ORIGIN,
    run_id: str | None = RUN_ID,
    source: str = "session",
) -> DesktopApprovalRule:
    return DesktopApprovalRule.create(
        effect="allow",
        source=source,  # type: ignore[arg-type]
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action="input",
            target_pattern=None,
            risk_class="local_mutation",
            origin_pattern=origin,
            actions=("input", "select"),
            run_id=run_id,
        ),
    )


def _input_request(selector: str = "#name", **extra: object) -> ToolRequest:
    return ToolRequest(
        "desktop_browser",
        {"operation": "input", "selector": selector, "value": "x", **extra},
    )


def test_first_input_without_grant_asks_and_scope_is_grant(tmp_path: Path) -> None:
    decision, scope = decide_request(context=_context(tmp_path), request=_input_request())

    assert decision.status == "require_approval"
    assert decision.reason == "browser_edit_requires_scope"
    assert scope is not None
    assert scope.origin_pattern == ORIGIN
    assert scope.actions == ("input", "select")
    assert scope.target_pattern is None
    assert scope.run_id == RUN_ID


def test_grant_allows_input_and_select_on_different_fields(tmp_path: Path) -> None:
    context = _context(tmp_path, rules=[_grant_rule()])

    first, _ = decide_request(context=context, request=_input_request("#name"))
    second, _ = decide_request(context=context, request=_input_request("#email"))
    select, _ = decide_request(
        context=context,
        request=ToolRequest(
            "desktop_browser",
            {"operation": "select", "selector": "#country", "value": "LT"},
        ),
    )

    assert first.status == "allow"
    assert second.status == "allow"
    assert select.status == "allow"
    assert first.expected_browser_origin == ORIGIN


def test_grant_does_not_cover_other_origin_subdomain_or_port(tmp_path: Path) -> None:
    rules = [_grant_rule()]
    for other_origin in (
        "https://other.test",
        "https://sub.example.test",
        "https://example.test:8443",
        "http://example.test",
    ):
        decision, _ = decide_request(
            context=_context(tmp_path, rules=rules, origin=other_origin),
            request=_input_request(),
        )
        assert decision.status == "require_approval", other_origin


def test_grant_does_not_cover_operation_outside_explicit_list() -> None:
    # Операция класса local_mutation вне явного списка {input, select}
    # грантом не покрывается, даже на том же origin и в том же run.
    action = DesktopAction(
        tool="desktop_browser",
        action="type_text",
        target="#name",
        risk_class="local_mutation",
        origin=ORIGIN,
        run_id=RUN_ID,
    )
    resolution = DesktopPolicyRuntime([_grant_rule()]).resolve(
        action,
        default_effect="ask",
        default_reason="browser_edit_requires_scope",
    )
    assert resolution.effect == "ask"
    assert resolution.rule_id is None


def test_origin_from_model_args_is_ignored(tmp_path: Path) -> None:
    # Модель подставляет чужой url в args: origin берётся только из
    # провайдера фактической страницы.
    rules = [_grant_rule()]
    decision, _ = decide_request(
        context=_context(tmp_path, rules=rules),
        request=_input_request(url="https://evil.test/phish"),
    )
    assert decision.status == "allow"

    evil_grant = [_grant_rule(origin="https://evil.test")]
    decision, _ = decide_request(
        context=_context(tmp_path, rules=evil_grant),
        request=_input_request(url="https://evil.test/phish"),
    )
    assert decision.status == "require_approval"


def test_missing_provider_or_page_asks(tmp_path: Path) -> None:
    rules = [_grant_rule()]
    no_provider, scope = decide_request(
        context=_context(tmp_path, rules=rules, provider=False),
        request=_input_request(),
    )
    assert no_provider.status == "require_approval"
    # Без известного origin грант не создаётся: legacy exact scope.
    assert scope is not None
    assert scope.origin_pattern is None
    assert scope.target_pattern == "#name"

    no_page, _ = decide_request(
        context=_context(tmp_path, rules=rules, origin=None),
        request=_input_request(),
    )
    assert no_page.status == "require_approval"


def test_grant_does_not_cover_submit_click_or_close_tab(tmp_path: Path) -> None:
    context = _context(tmp_path, rules=[_grant_rule()])
    for operation, extra in (
        ("submit", {"selector": "#send"}),
        ("click", {"selector": "#publish"}),
        ("close_tab", {"page_id": "page-1"}),
        ("upload", {"selector": "#file"}),
    ):
        decision, _ = decide_request(
            context=context,
            request=ToolRequest("desktop_browser", {"operation": operation, **extra}),
        )
        assert decision.status == "require_approval", operation


def test_explicit_deny_beats_grant(tmp_path: Path) -> None:
    deny = DesktopApprovalRule.create(
        effect="deny",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action="input",
            target_pattern="#password",
            risk_class="local_mutation",
        ),
    )
    decision, _ = decide_request(
        context=_context(tmp_path, rules=[_grant_rule(), deny]),
        request=_input_request("#password"),
    )
    assert decision.status == "block"
    assert decision.reason == "explicit_deny"


def test_migrated_old_deny_still_blocks_inside_grant(tmp_path: Path) -> None:
    # Старый persistent deny со superseded risk class (external_side_effect)
    # мигрирует на local_mutation при загрузке и продолжает блокировать.
    store_path = tmp_path / "desktop-approvals.json"
    store_path.write_text(
        json.dumps(
            [
                {
                    "rule_id": "legacy-deny",
                    "effect": "deny",
                    "source": "persistent",
                    "created_at": "2026-11-01T00:00:00+00:00",
                    "description": "",
                    "subject_principal_id": "legacy",
                    "scope": {
                        "tool": "desktop_browser",
                        "action": "input",
                        "target_pattern": "#secret",
                        "risk_class": "external_side_effect",
                        "execution_target": "desktop",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    migrated = DesktopPolicyStore(store_path).list_rules()
    decision, _ = decide_request(
        context=_context(tmp_path, rules=[_grant_rule(), *migrated]),
        request=_input_request("#secret"),
    )
    assert decision.status == "block"


def test_concrete_ask_inside_grant_still_asks(tmp_path: Path) -> None:
    ask = DesktopApprovalRule.create(
        effect="ask",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action="input",
            target_pattern="#card",
            risk_class="local_mutation",
        ),
    )
    context = _context(tmp_path, rules=[_grant_rule(), ask])
    decision, _ = decide_request(context=context, request=_input_request("#card"))
    assert decision.status == "require_approval"
    assert decision.reason == "explicit_ask"
    assert decision.policy_rule_id == ask.rule_id

    other, _ = decide_request(context=context, request=_input_request("#name"))
    assert other.status == "allow"


def test_new_run_does_not_inherit_grant(tmp_path: Path) -> None:
    rules = [_grant_rule()]
    same_run, _ = decide_request(
        context=_context(tmp_path, rules=rules, run_id=RUN_ID),
        request=_input_request(),
    )
    assert same_run.status == "allow"

    new_run, _ = decide_request(
        context=_context(tmp_path, rules=rules, run_id="run-2"),
        request=_input_request(),
    )
    assert new_run.status == "require_approval"

    no_run, _ = decide_request(
        context=_context(tmp_path, rules=rules, run_id=None),
        request=_input_request(),
    )
    assert no_run.status == "require_approval"


def test_persistent_grant_is_rejected(tmp_path: Path) -> None:
    scope = _grant_rule().scope
    assert scope.supports_persistent_allow is False
    store = DesktopPolicyStore(tmp_path / "desktop-approvals.json")
    with pytest.raises(ValueError, match="не может быть persistent"):
        store.add_rule(DesktopApprovalRule.create(effect="allow", source="persistent", scope=scope))


def test_scope_serialization_round_trip_and_back_compat() -> None:
    scope = _grant_rule().scope
    restored = DesktopApprovalScope.from_dict(scope.to_dict())
    assert restored == scope

    legacy = DesktopApprovalScope.from_dict(
        {
            "tool": "desktop_browser",
            "action": "input",
            "target_pattern": "#name",
            "risk_class": "local_mutation",
            "execution_target": "desktop",
        }
    )
    assert legacy.origin_pattern is None
    assert legacy.actions is None
    assert legacy.run_id is None
    # Старое правило без origin_pattern не расширяется: target None не матчится.
    action = DesktopAction(
        tool="desktop_browser",
        action="input",
        target="#other",
        risk_class="local_mutation",
        origin=ORIGIN,
        run_id=RUN_ID,
    )
    assert not legacy.matches(action)


def test_browser_origin_of_url_exactness() -> None:
    assert browser_origin_of_url("https://example.test/path?q=1") == "https://example.test"
    assert browser_origin_of_url("https://example.test:443/x") == "https://example.test"
    assert browser_origin_of_url("http://example.test:80/x") == "http://example.test"
    assert browser_origin_of_url("https://example.test:8443/x") == "https://example.test:8443"
    assert browser_origin_of_url("https://sub.example.test/") == "https://sub.example.test"
    assert browser_origin_of_url("about:blank") is None
    assert browser_origin_of_url("data:text/html,hi") is None
    assert browser_origin_of_url("") is None
    assert browser_origin_of_url(None) is None


def test_approval_prompt_describes_grant_scope(tmp_path: Path) -> None:
    context = _context(tmp_path)
    decision, scope = decide_request(context=context, request=_input_request())
    request = build_approval_request(context=context, decision=decision, scope=scope)
    assert request is not None
    assert ORIGIN in request.prompt.what
    assert "текущего запуска" in request.prompt.what
    assert "autosave" in request.prompt.risk


class _FakeBackend:
    """Backend с живой страницей: page_url можно менять между вызовами."""

    def __init__(self, url: str = "https://example.test/form") -> None:
        self.url = url
        self.calls: list[ToolRequest] = []

    def handle(self, request: ToolRequest) -> ToolResult:
        self.calls.append(request)
        return ToolResult.success({"operation": "input", "verified": True})

    def close(self) -> None:
        return None

    def page_url(self, page_id: str | None = None) -> str | None:
        return self.url


def test_service_current_origin_does_not_create_session(tmp_path: Path) -> None:
    created: list[BrowserProfile] = []

    def factory(profile: BrowserProfile) -> _FakeBackend:
        created.append(profile)
        return _FakeBackend()

    service = BrowserService(factory)
    profile = BrowserProfile.default(owner_id="local")
    assert service.current_origin(profile) is None
    assert created == []

    service.handle(profile, ToolRequest("desktop_browser", {"operation": "input"}))
    assert service.current_origin(profile) == ORIGIN


def test_tool_rechecks_origin_before_execution(tmp_path: Path) -> None:
    backend = _FakeBackend()
    tool = DesktopBrowserTool(_security(tmp_path), backend_factory=lambda: backend)
    request = ToolRequest(
        "desktop_browser",
        {"operation": "input", "selector": "#name", "value": "x"},
        expected_browser_origin=ORIGIN,
    )
    # Сессия поднимается первым вызовом без expected origin (в реальном
    # флоу expected origin появляется только когда провайдер уже видел
    # живую страницу на момент решения).
    warmup = ToolRequest(
        "desktop_browser", {"operation": "input", "selector": "#name", "value": "x"}
    )
    assert tool.handle(warmup).ok
    backend.calls.clear()

    # Origin совпадает с decision-time — исполняется.
    assert tool.handle(request).ok
    assert len(backend.calls) == 1

    # Redirect на другой origin между решением и исполнением: не исполняется.
    backend.url = "https://other.test/form"
    result = tool.handle(request)
    assert not result.ok
    assert "origin changed" in (result.error or "")
    assert len(backend.calls) == 1


def test_gateway_attaches_expected_origin(tmp_path: Path) -> None:
    seen: list[ToolRequest] = []

    def handler(request: ToolRequest) -> ToolResult:
        seen.append(request)
        return ToolResult.success({"operation": "input", "verified": True})

    registry = ToolRegistry()
    registry.register(
        "desktop_browser",
        handler,
        description="Fake browser",
        parameters_schema={"type": "object", "properties": {}},
        capability="exec",
        risk_classes=["external_side_effect"],
    )
    gateway = ToolGateway(
        registry,
        approval_context=_context(tmp_path, rules=[_grant_rule()]),
    )
    result = gateway.call(_input_request())
    assert result.ok
    assert seen[0].expected_browser_origin == ORIGIN


# --- Rework round 1 (findings Codex по PR #187) ---


def test_ipv6_origin_keeps_brackets_and_grant_does_not_leak(tmp_path: Path) -> None:
    # P2: без скобок http://[2001:db8::1]:8080 и http://[2001:db8::1:8080]
    # нормализовались в один origin — грант протекал между сайтами.
    first = browser_origin_of_url("http://[2001:db8::1]:8080/form")
    second = browser_origin_of_url("http://[2001:db8::1:8080]/form")
    assert first == "http://[2001:db8::1]:8080"
    assert second == "http://[2001:db8::1:8080]"
    assert first != second

    grant = _grant_rule(origin=first or "")
    allowed, _ = decide_request(
        context=_context(tmp_path, rules=[grant], origin=first),
        request=_input_request(),
    )
    assert allowed.status == "allow"

    leaked, _ = decide_request(
        context=_context(tmp_path, rules=[grant], origin=second),
        request=_input_request(),
    )
    assert leaked.status == "require_approval"


def _card_ask_rule() -> DesktopApprovalRule:
    return DesktopApprovalRule.create(
        effect="ask",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action="input",
            target_pattern="#card",
            risk_class="local_mutation",
        ),
    )


@pytest.mark.parametrize("source", ["once", "session"])
@pytest.mark.parametrize("with_grant", [True, False])
def test_explicit_ask_approval_passes_confirmed_action(
    tmp_path: Path, source: str, with_grant: bool
) -> None:
    # P2: подтверждение explicit ask создавало широкий грант, и при resume
    # ask выбирался снова — бесконечный approval. Scope подтверждения
    # должен быть точным на selector.
    rules: list[DesktopApprovalRule] = [_card_ask_rule()]
    if with_grant:
        rules.insert(0, _grant_rule())
    context = _context(tmp_path, rules=rules)

    decision, scope = decide_request(context=context, request=_input_request("#card"))
    assert decision.status == "require_approval"
    assert decision.reason == "explicit_ask"
    assert scope is not None
    assert scope.target_pattern == "#card"
    assert scope.origin_pattern is None

    # Пользователь подтвердил: правило создаётся из возвращённого scope.
    approved = DesktopApprovalRule.create(effect="allow", source=source, scope=scope)  # type: ignore[arg-type]
    resumed = _context(tmp_path, rules=[*rules, approved])
    passed, _ = decide_request(context=resumed, request=_input_request("#card"))
    assert passed.status == "allow"
    assert passed.expected_browser_origin == ORIGIN

    again, _ = decide_request(context=resumed, request=_input_request("#card"))
    if source == "once":
        # Once-разрешение израсходовано: ask спрашивает снова.
        assert again.status == "require_approval"
        assert again.reason == "explicit_ask"
    else:
        assert again.status == "allow"

    # Точное подтверждение не создаёт грант на другие поля.
    other, _ = decide_request(context=resumed, request=_input_request("#name"))
    assert other.status == ("allow" if with_grant else "require_approval")


class _FakeLocator:
    def __init__(self) -> None:
        self.filled: list[str] = []
        self.selected: list[str] = []
        # Security origin документа элемента для evaluate(location.origin);
        # None — документ страницы (page._fresh_url). "null" — opaque
        # origin (sandbox без allow-same-origin). raise_on_evaluate — сбой
        # протокольного чтения origin.
        self.document_origin: str | None = None
        self.raise_on_evaluate = False
        self.page: _FakePage | None = None

    def __call__(self, selector: str) -> _FakeLocator:
        # page.locator(selector) при selector_type="css".
        return self

    def evaluate(self, expression: str, timeout: float | None = None) -> str | None:
        if self.raise_on_evaluate:
            raise RuntimeError("evaluate failed")
        if self.document_origin is not None:
            return self.document_origin
        return self.page._fresh_url if self.page is not None else None

    def fill(self, value: str, timeout: int | None = None) -> None:
        self.filled.append(value)

    def select_option(self, value: str | None = None, timeout: int | None = None) -> list[str]:
        if value is not None:
            self.selected.append(value)
        return self.selected

    def input_value(self, timeout: int | None = None) -> str:
        return self.filled[-1] if self.filled else ""


class _FakePage:
    """Страница с устаревшим кешем Page.url и свежим href документа."""

    def __init__(self, cached_url: str, fresh_url: str) -> None:
        self.url = cached_url
        self._fresh_url = fresh_url
        self.locator = _FakeLocator()
        self.locator.page = self

    def evaluate(self, expression: str) -> str:
        assert expression == "window.location.href"
        return self._fresh_url

    def is_closed(self) -> bool:
        return False

    def get_by_role(self, *args: object, **kwargs: object) -> _FakeLocator:
        return self.locator


def _backend_with_page(tmp_path: Path, page: _FakePage):  # noqa: ANN202
    from tools.desktop_browser_tool import PlaywrightBrowserBackend

    backend = PlaywrightBrowserBackend(_security(tmp_path), cancelled=lambda: False)
    backend._browser = object()  # type: ignore[assignment]
    backend._context = object()  # type: ignore[assignment]
    backend._pages["page-1"] = page  # type: ignore[assignment]
    backend._active_page_id = "page-1"
    return backend


def test_page_url_reads_fresh_document_not_stale_cache(tmp_path: Path) -> None:
    # P1: Page.url — кеш Frame._url. После навигации во время approval
    # провайдер origin должен видеть фактический документ.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://other.test/form",
    )
    backend = _backend_with_page(tmp_path, page)
    assert backend.page_url() == "https://other.test/form"


def test_backend_rechecks_fresh_origin_before_fill(tmp_path: Path) -> None:
    # P1: кеш говорит example.test (разрешённый origin), но документ уже
    # на other.test — fill не выполняется.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://other.test/form",
    )
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {"operation": "input", "selector": "#name", "value": "x"},
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert not result.ok
    assert "origin changed" in (result.error or "")
    assert page.locator.filled == []

    # Документ на разрешённом origin — ввод выполняется.
    page._fresh_url = "https://example.test/form"
    result = backend.handle(request)
    assert result.ok
    assert page.locator.filled == ["x"]


def test_exact_allow_with_origin_beats_concrete_ask(tmp_path: Path) -> None:
    # Точное allow на selector (даже с origin_pattern) — не широкий грант:
    # конкретный ask его не перехватывает.
    exact_allow = DesktopApprovalRule.create(
        effect="allow",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action="input",
            target_pattern="#card",
            risk_class="local_mutation",
            origin_pattern=ORIGIN,
            actions=("input", "select"),
            run_id=RUN_ID,
        ),
    )
    context = _context(tmp_path, rules=[exact_allow, _card_ask_rule()])
    decision, _ = decide_request(context=context, request=_input_request("#card"))
    assert decision.status == "allow"


# --- Rework round 2 (finding Codex P1 по PR #187) ---


def test_backend_rejects_edit_in_cross_origin_iframe(tmp_path: Path) -> None:
    # P1: грант на верхний документ example.test не покрывает поле в
    # iframe other.test — origin документа элемента проверяется перед fill.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://example.test/form",
    )
    page.locator.document_origin = "https://other.test"
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {
            "operation": "input",
            "selector_type": "css",
            "selector": "iframe >> internal:control=enter-frame >> input",
            "value": "x",
        },
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert not result.ok
    assert "different origin" in (result.error or "")
    assert page.locator.filled == []

    # Поле в документе того же origin по гранту работает как прежде.
    page.locator.document_origin = "https://example.test"
    result = backend.handle(request)
    assert result.ok
    assert page.locator.filled == ["x"]


def test_backend_rejects_select_in_cross_origin_iframe(tmp_path: Path) -> None:
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://example.test/form",
    )
    page.locator.document_origin = "https://other.test"
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {"operation": "select", "selector": "#country", "value": "LT"},
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert not result.ok
    assert "different origin" in (result.error or "")
    assert page.locator.selected == []


def test_backend_element_origin_check_is_fail_closed(tmp_path: Path) -> None:
    # Origin документа элемента прочитать нельзя — ввод не выполняется.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://example.test/form",
    )
    page.locator.raise_on_evaluate = True
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {"operation": "input", "selector": "#name", "value": "x"},
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert not result.ok
    assert page.locator.filled == []


# --- Rework round 3 (finding Codex P2 по PR #187) ---


def test_backend_allows_edit_in_srcdoc_iframe_with_inherited_origin(tmp_path: Path) -> None:
    # P2: iframe без src/с srcdoc наследует origin родителя, хотя URL его
    # документа — about:blank/about:srcdoc. location.origin документа
    # элемента равен origin верхней страницы — ввод по гранту проходит.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://example.test/form",
    )
    page.locator.document_origin = ORIGIN
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {
            "operation": "input",
            "selector_type": "css",
            "selector": "iframe >> internal:control=enter-frame >> input",
            "value": "x",
        },
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert result.ok
    assert page.locator.filled == ["x"]


def test_backend_rejects_edit_in_opaque_origin_iframe(tmp_path: Path) -> None:
    # Sandbox без allow-same-origin: location.origin документа элемента —
    # строка "null" (opaque origin). Грант верхней страницы такой ввод не
    # покрывает — fail-closed, как и для cross-origin iframe.
    page = _FakePage(
        cached_url="https://example.test/form",
        fresh_url="https://example.test/form",
    )
    page.locator.document_origin = "null"
    backend = _backend_with_page(tmp_path, page)
    request = ToolRequest(
        "desktop_browser",
        {
            "operation": "input",
            "selector_type": "css",
            "selector": "iframe >> internal:control=enter-frame >> input",
            "value": "x",
        },
        expected_browser_origin=ORIGIN,
    )
    result = backend.handle(request)
    assert not result.ok
    assert "different origin" in (result.error or "")
    assert page.locator.filled == []
