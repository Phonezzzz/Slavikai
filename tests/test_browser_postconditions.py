"""E02-T06: browser postconditions (семантика B, решение владельца 2026-11-22).

Browser observation verification подключается к канонической деривации
``outcome_from_loop_result`` без перехвата FAILED_VERIFIER: неподтверждённый
browser postcondition даёт UNVERIFIED, а успешный процессный verifier его
не перекрывает. Факт re-observation (snapshot/read/find) сам по себе не
подтверждает целевое состояние click/submit — DONE только при
подтверждённом результате. ``find`` с total=0 не подтверждает ничего.

Позитивный DONE для click/submit требует контракта ожидаемого
postcondition, которого пока нет: здесь click/submit остаются
UNVERIFIED, а self-verified операции (navigate/input/select/download)
могут дать DONE.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from core.agent import Agent
from core.agent_response import ResponseProduced
from core.approval_policy import ApprovalContext, ApprovalRequired
from core.auto_preset import (
    AutoPresetRunner,
    _browser_postcondition_result,
    _canonical_verification,
)
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
    DesktopPolicyStore,
)
from core.desktop_runtime import DesktopExecutionControl, DesktopRuntime
from core.desktop_security import DesktopPathSecurity
from core.managed_workspace import ManagedWorkspace, WorkspaceAccess
from core.mwv.models import MWV_REPORT_PREFIX, VerificationResult, VerificationStatus
from core.mwv.verifier_runtime import VerifierRuntime, desktop_last_state_change_succeeded
from core.response_preferences import ResponsePreferences
from core.run_contract import RunPolicy
from core.run_outcome import TerminalReason, VerificationOutcome, outcome_from_loop_result
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoopResult, ExecutedToolCall
from llm.brain_base import Brain
from llm.stream_model import TextDelta
from llm.types import LLMResult, LLMUsage, ToolCall
from server.http.handlers.ui_chat import _iterate_stream_in_thread
from shared.auto_models import AutoRunStatus
from shared.models import JSONValue, LLMMessage, ToolRequest, ToolResult
from tests.test_canonical_loop_e2e import _E2EAgent
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import workspace_root_context

PAGE = "page-1"


def _call(
    operation: str,
    data: dict[str, JSONValue],
    *,
    ok: bool = True,
    error: str | None = None,
    args: dict[str, JSONValue] | None = None,
) -> tuple[str, dict[str, JSONValue], ToolResult]:
    call_args: dict[str, JSONValue] = {"operation": operation}
    if args:
        call_args.update(args)
    result = ToolResult.success(data) if ok else ToolResult.failure(error or "failed")
    return ("desktop_browser", call_args, result)


def _click(*, verified: bool = False) -> tuple[str, dict[str, JSONValue], ToolResult]:
    data: dict[str, JSONValue] = {
        "page_id": PAGE,
        "requires_followup_observation": True,
        "before_url": "https://example.test/form",
        "url": "https://example.test/form",
    }
    if verified:
        data["verified"] = True
    return _call("click", data, args={"selector": "#publish"})


def _snapshot() -> tuple[str, dict[str, JSONValue], ToolResult]:
    return _call(
        "snapshot",
        {"page_id": PAGE, "snapshot": "published post text"},
        args={"page_id": PAGE},
    )


def _find(total: int) -> tuple[str, dict[str, JSONValue], ToolResult]:
    return _call(
        "find",
        {"page_id": PAGE, "total": total, "items": []},
        args={"selector": ".post"},
    )


def _navigate() -> tuple[str, dict[str, JSONValue], ToolResult]:
    return _call(
        "navigate",
        {"page_id": PAGE, "url": "https://example.test/", "verified": True},
        args={"url": "https://example.test/"},
    )


def _input() -> tuple[str, dict[str, JSONValue], ToolResult]:
    return _call(
        "input",
        {"page_id": PAGE, "url": "https://example.test/", "verified": True},
        args={"selector": "#name", "value": "x"},
    )


def _process_result(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(
        status=status,
        command=["make", "-B", "-r", "-f", "Makefile", "check"],
        exit_code=0 if status is VerificationStatus.PASSED else 1,
        stdout="",
        stderr="",
        duration_seconds=0.0,
        error=None if status is VerificationStatus.PASSED else "canonical check failed",
        excerpt="",
        verifier_profile="canonical",
    )


def _loop_result(
    calls: list[tuple[str, dict[str, JSONValue], ToolResult]],
) -> AgentToolLoopResult:
    return AgentToolLoopResult(
        text="Готово",
        messages=[],
        tool_calls=[
            ExecutedToolCall(
                call=ToolCall(id=f"call-{index}", name=name, arguments=dict(args)),
                result=result,
            )
            for index, (name, args, result) in enumerate(calls)
        ],
        iterations=1,
    )


# --- критерий отказа выполнения (rework round 2) --------------------------


def test_last_state_change_succeeded_requires_successful_last_change() -> None:
    failed_click = _call(
        "click", {}, ok=False, error="locator timeout", args={"selector": "#publish"}
    )
    assert desktop_last_state_change_succeeded([failed_click]) is False
    assert desktop_last_state_change_succeeded([_click()]) is True
    # Ошибка, исправленная последующим успешным действием, не отказ.
    assert desktop_last_state_change_succeeded([failed_click, _click()]) is True
    # Успешное действие, за которым следует отказ, — выполнение завершилось отказом.
    assert desktop_last_state_change_succeeded([_navigate(), failed_click]) is False
    # Нет state-changing действий — нормализовать нечего.
    assert desktop_last_state_change_succeeded([_snapshot()]) is False
    assert desktop_last_state_change_succeeded([]) is False


# --- find(total=0): desktop gate semantics -------------------------------


def test_find_with_zero_total_does_not_confirm_desktop_change() -> None:
    rejected = VerifierRuntime().verify_desktop_observations([_click(), _find(0)])
    accepted = VerifierRuntime().verify_desktop_observations([_click(), _find(2)])

    assert rejected.status == VerificationStatus.FAILED
    assert accepted.status == VerificationStatus.PASSED


def test_snapshot_still_satisfies_desktop_reobservation_gate() -> None:
    # Desktop gate проверяет факт re-observation перед финишем; его
    # семантика не меняется (канонический DONE — отдельный вопрос ниже).
    result = VerifierRuntime().verify_desktop_observations([_click(), _snapshot()])

    assert result.status == VerificationStatus.PASSED


# --- verify_browser_postconditions: canonical semantics -------------------


def test_browser_postconditions_not_applicable_without_changes() -> None:
    verifier = VerifierRuntime()

    assert verifier.verify_browser_postconditions([]) is None
    assert verifier.verify_browser_postconditions([_snapshot()]) is None


def test_click_without_observation_is_unconfirmed() -> None:
    result = VerifierRuntime().verify_browser_postconditions([_click()])

    assert result is not None
    assert result.status == VerificationStatus.FAILED
    assert result.error == "browser_postcondition_unconfirmed:click"


def test_click_with_reobservation_is_still_unconfirmed() -> None:
    # Факт повторной observation не доказывает целевое состояние:
    # snapshot и find(total>=1) не превращаются в PASSED ради DONE.
    verifier = VerifierRuntime()

    for later in (_snapshot(), _find(3)):
        result = verifier.verify_browser_postconditions([_click(), later])
        assert result is not None
        assert result.status == VerificationStatus.FAILED
        assert result.error == "browser_postcondition_unconfirmed:click"


def test_submit_with_reobservation_is_unconfirmed() -> None:
    submit = _call(
        "submit",
        {"page_id": PAGE, "requires_followup_observation": True},
        args={"selector": "#form"},
    )

    result = VerifierRuntime().verify_browser_postconditions([submit, _snapshot()])

    assert result is not None
    assert result.status == VerificationStatus.FAILED
    assert result.error == "browser_postcondition_unconfirmed:submit"


def test_self_verified_browser_changes_are_confirmed() -> None:
    result = VerifierRuntime().verify_browser_postconditions([_navigate(), _input()])

    assert result is not None
    assert result.status == VerificationStatus.PASSED


def test_self_verified_click_is_confirmed() -> None:
    # Шов для будущего контракта ожидаемого postcondition: если tool
    # сам подтвердил результат (verified=True), каноническая проверка
    # его пропускает без изменений.
    result = VerifierRuntime().verify_browser_postconditions([_click(verified=True)])

    assert result is not None
    assert result.status == VerificationStatus.PASSED


def test_blank_new_tab_confirmed_by_later_observation() -> None:
    new_tab = _call("new_tab", {"page_id": PAGE, "url": "about:blank"})

    result = VerifierRuntime().verify_browser_postconditions([new_tab, _snapshot()])

    assert result is not None
    assert result.status == VerificationStatus.PASSED


def test_failed_browser_change_is_not_confirmed() -> None:
    failed_click = _call(
        "click",
        {},
        ok=False,
        error="locator timeout",
        args={"selector": "#publish"},
    )

    result = VerifierRuntime().verify_browser_postconditions([failed_click])

    assert result is not None
    assert result.status == VerificationStatus.FAILED
    assert result.error == "locator timeout"


# --- canonical combination: semantics B ----------------------------------


def test_process_passed_does_not_override_unconfirmed_browser() -> None:
    loop_result = _loop_result([_click(), _snapshot()])
    browser = _browser_postcondition_result(loop_result)
    assert browser is not None and browser.status == VerificationStatus.FAILED

    verification = _canonical_verification(_process_result(VerificationStatus.PASSED), browser)
    assert verification is not None
    assert verification.outcome is VerificationOutcome.FAILED
    assert {check.name for check in verification.checks} == {
        "process_verifier",
        "browser_postconditions",
    }

    outcome = outcome_from_loop_result(loop_result, state_changing=True, verification=verification)
    assert outcome.ui_label == "UNVERIFIED"
    assert outcome.is_full_done is False
    assert outcome.communication_class is None


def test_confirmed_browser_postconditions_can_be_done() -> None:
    loop_result = _loop_result([_navigate(), _input()])
    browser = _browser_postcondition_result(loop_result)
    assert browser is not None and browser.status == VerificationStatus.PASSED

    verification = _canonical_verification(None, browser)
    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED

    outcome = outcome_from_loop_result(loop_result, state_changing=True, verification=verification)
    assert outcome.ui_label == "DONE"
    assert outcome.is_full_done is True


def test_click_without_observation_is_unverified_not_failed_verifier() -> None:
    # Негатив тикета: click() success без observation -> UNVERIFIED.
    # Каноническая деривация: completed + full + verification failed,
    # терминальная причина не переписывается в FAILED.
    loop_result = _loop_result([_click()])
    browser = _browser_postcondition_result(loop_result)
    verification = _canonical_verification(None, browser)

    outcome = outcome_from_loop_result(loop_result, state_changing=True, verification=verification)
    assert outcome.ui_label == "UNVERIFIED"
    assert outcome.is_full_done is False


def test_no_browser_calls_keeps_process_verification_unchanged() -> None:
    verification = _canonical_verification(_process_result(VerificationStatus.PASSED), None)

    assert verification is not None
    assert verification.outcome is VerificationOutcome.PASSED
    assert [check.name for check in verification.checks] == ["verifier"]


def test_no_verification_at_all_stays_none() -> None:
    assert _canonical_verification(None, None) is None


# --- AutoPresetRunner wiring (scripted brain, fake browser tool) ----------


class _BrowserBrain(Brain):
    """Fixture модели: заданная последовательность browser-вызовов, затем ответ."""

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


def _fake_browser_handler(request: ToolRequest) -> ToolResult:
    operation = str(request.args.get("operation") or "snapshot")
    if operation == "navigate":
        return ToolResult.success(
            {
                "operation": operation,
                "page_id": PAGE,
                "url": "https://example.test/",
                "verified": True,
            }
        )
    if operation == "input":
        return ToolResult.success(
            {
                "operation": operation,
                "page_id": PAGE,
                "url": "https://example.test/",
                "verified": True,
            }
        )
    if operation == "click":
        return ToolResult.success(
            {
                "operation": operation,
                "page_id": PAGE,
                "before_url": "https://example.test/form",
                "url": "https://example.test/form",
                "requires_followup_observation": True,
            }
        )
    if operation == "snapshot":
        return ToolResult.success(
            {"operation": operation, "page_id": PAGE, "snapshot": "published post text"}
        )
    return ToolResult.failure(f"unexpected operation {operation}")


def _failing_click_handler(request: ToolRequest) -> ToolResult:
    operation = str(request.args.get("operation") or "snapshot")
    if operation == "click":
        return ToolResult.failure("locator timeout")
    return _fake_browser_handler(request)


def _flaky_click_handler():  # noqa: ANN202
    """Первый click падает с locator timeout, последующие — как обычно."""
    state = {"clicks": 0}

    def handler(request: ToolRequest) -> ToolResult:
        operation = str(request.args.get("operation") or "snapshot")
        if operation == "click":
            state["clicks"] += 1
            if state["clicks"] == 1:
                return ToolResult.failure("locator timeout")
        return _fake_browser_handler(request)

    return handler


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


class _BrowserE2EAgent(_E2EAgent):
    """_E2EAgent с desktop policy, разрешающей click/input тестовых селекторов."""

    def _build_tool_gateway(self, *, budget=None):  # noqa: ANN001, ANN202
        del budget
        security = DesktopPathSecurity(
            home=self._workspace.root,
            policy_store_path=self._workspace.root / ".run" / "desktop_approvals.json",
        )
        rules = [
            _allow_rule("click", "#publish", "external_side_effect"),
            _allow_rule("input", "#name", "local_mutation"),
        ]
        return ToolGateway(
            self.tool_registry,
            approval_context=ApprovalContext(
                safe_mode=True,
                session_id="session-test",
                approved_categories=set(),
                execution_target="desktop",
                desktop_policy=DesktopPolicyRuntime(rules),
                desktop_security=security,
            ),
            workspace_access=WorkspaceAccess(self._workspace, "test-principal", "session-test"),
        )


def _build_browser_e2e(tmp_path: Path) -> tuple[ManagedWorkspace, ToolRegistry]:
    workspace = ManagedWorkspace.create(tmp_path, principal_id="test-principal")
    registry = ToolRegistry()
    registry.register(
        "desktop_browser",
        _fake_browser_handler,
        description="Fake browser tool for postcondition tests",
        parameters_schema={"type": "object", "properties": {"operation": {"type": "string"}}},
        capability="exec",
        risk_classes=["network", "external_side_effect"],
    )
    return workspace, registry


def _run_preset(tmp_path: Path, monkeypatch, calls: list[ToolCall]):  # noqa: ANN001, ANN202
    workspace, registry = _build_browser_e2e(tmp_path)
    monkeypatch.chdir(workspace.root)
    agent = _BrowserE2EAgent(_BrowserBrain(calls), registry, workspace)
    with workspace_root_context(workspace.root):
        return AutoPresetRunner(agent, workspace_root=workspace.root).run("browser task")


def test_preset_click_with_snapshot_is_unverified_not_done(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    outcome = _run_preset(
        tmp_path,
        monkeypatch,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
            ToolCall(
                id="snap-1",
                name="desktop_browser",
                arguments={"operation": "snapshot", "page_id": PAGE},
            ),
        ],
    )

    # Семантика B: run завершается COMPLETED (не FAILED_VERIFIER), но
    # канонический outcome — UNVERIFIED: re-observation не доказывает
    # результат клика.
    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.verifier is None
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.canonical_outcome.is_full_done is False


def test_preset_self_verified_browser_run_is_done(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    outcome = _run_preset(
        tmp_path,
        monkeypatch,
        [
            ToolCall(
                id="nav-1",
                name="desktop_browser",
                arguments={"operation": "navigate", "url": "https://example.test/"},
            ),
            ToolCall(
                id="input-1",
                name="desktop_browser",
                arguments={"operation": "input", "selector": "#name", "value": "x"},
            ),
        ],
    )

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "DONE"
    assert outcome.canonical_outcome.is_full_done is True


# --- Пользовательский ответ Auto из канонического outcome ----------------
# Отдельный PR от main (решение владельца 2026-11-22): в
# AutoPresetRunner._execute_frame пользовательский текст и статус
# верификатора выводятся из RunOutcome, как в Desktop-проекции (#183).
# COMPLETED — факт остановки прогона; DONE и публикацию успеха
# определяет RunOutcome. Тот же путь обслуживает resume.


class _CapturingBrowserE2EAgent(_BrowserE2EAgent):
    """_BrowserE2EAgent, записывающий kwargs report/stop блоков."""

    def __init__(self, brain, registry, workspace):  # noqa: ANN001
        super().__init__(brain, registry, workspace)
        self.report_kwargs_history: list[dict] = []
        self.stop_kwargs_history: list[dict] = []

    def _append_report_block(self, text: str, **kwargs):  # noqa: ANN003, ANN202
        self.report_kwargs_history.append(kwargs)
        return text

    def _format_stop_response(self, **kwargs):  # noqa: ANN003, ANN202
        self.stop_kwargs_history.append(kwargs)
        return f"stop:{kwargs.get('what', '')}"


class _MutableRulesBrowserAgent(_CapturingBrowserE2EAgent):
    """Gateway строится из изменяемого списка правил (для resume)."""

    def __init__(self, brain, registry, workspace):  # noqa: ANN001
        super().__init__(brain, registry, workspace)
        self.rules: list[DesktopApprovalRule] = []

    def _build_tool_gateway(self, *, budget=None):  # noqa: ANN001, ANN202
        del budget
        security = DesktopPathSecurity(
            home=self._workspace.root,
            policy_store_path=self._workspace.root / ".run" / "desktop_approvals.json",
        )
        return ToolGateway(
            self.tool_registry,
            approval_context=ApprovalContext(
                safe_mode=True,
                session_id="session-test",
                approved_categories=set(),
                execution_target="desktop",
                desktop_policy=DesktopPolicyRuntime(self.rules),
                desktop_security=security,
            ),
            workspace_access=WorkspaceAccess(self._workspace, "test-principal", "session-test"),
        )


def _run_preset_capturing(tmp_path: Path, monkeypatch, calls: list[ToolCall], handler=None):  # noqa: ANN001, ANN202
    workspace, _registry = _build_browser_e2e(tmp_path)
    registry = ToolRegistry()
    registry.register(
        "desktop_browser",
        handler or _fake_browser_handler,
        description="Fake browser tool for postcondition tests",
        parameters_schema={"type": "object", "properties": {"operation": {"type": "string"}}},
        capability="exec",
        risk_classes=["network", "external_side_effect"],
    )
    monkeypatch.chdir(workspace.root)
    agent = _CapturingBrowserE2EAgent(_BrowserBrain(calls), registry, workspace)
    with workspace_root_context(workspace.root):
        outcome = AutoPresetRunner(agent, workspace_root=workspace.root).run("browser task")
    return outcome, agent


def test_auto_user_response_unverified_click_is_not_success(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Действие выполнено, результат не подтверждён: пользователь видит
    # UNVERIFIED-текст и не-ok статус верификатора, а не финальный
    # текст модели «Готово».
    outcome, agent = _run_preset_capturing(tmp_path, monkeypatch, [_click_call(), _snapshot_call()])

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.canonical_outcome.is_full_done is False
    assert outcome.text == "Действие выполнено, результат не подтверждён"
    assert "Готово" not in outcome.text
    report_kwargs = agent.report_kwargs_history[-1]
    assert report_kwargs["verifier"].ok is False
    assert "canonical=UNVERIFIED" in report_kwargs["execution_summary"]
    assert "verification=passed" not in report_kwargs["execution_summary"]


def test_auto_user_response_confirmed_run_is_done(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Ожидаемый результат подтверждён (self-verified navigate + input):
    # DONE, текст модели сохраняется, статус верификатора — ok.
    outcome, agent = _run_preset_capturing(
        tmp_path,
        monkeypatch,
        [
            ToolCall(
                id="nav-1",
                name="desktop_browser",
                arguments={"operation": "navigate", "url": "https://example.test/"},
            ),
            ToolCall(
                id="input-1",
                name="desktop_browser",
                arguments={"operation": "input", "selector": "#name", "value": "x"},
            ),
        ],
    )

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "DONE"
    assert outcome.canonical_outcome.is_full_done is True
    assert outcome.text == "Готово"
    report_kwargs = agent.report_kwargs_history[-1]
    assert report_kwargs["verifier"].ok is True
    assert "canonical=DONE" in report_kwargs["execution_summary"]


def test_auto_user_response_failed_click_keeps_error(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Действие завершилось реальной ошибкой: FAILED_WORKER с фактической
    # причиной в stop-ответе, успех не публикуется.
    outcome, agent = _run_preset_capturing(
        tmp_path, monkeypatch, [_click_call()], handler=_failing_click_handler
    )

    assert outcome.status == AutoRunStatus.FAILED_WORKER
    assert outcome.canonical_outcome is None
    assert "Готово" not in outcome.text
    assert "неустранённый сбой tool call" in outcome.text
    stop_kwargs = agent.stop_kwargs_history[-1]
    assert "verifier evidence" in stop_kwargs["why"]


def test_auto_user_response_unverified_after_resume(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    # Resume после approval идёт через тот же _execute_frame: проекция
    # по RunOutcome применяется и к продолженному прогону.
    workspace, registry = _build_browser_e2e(tmp_path)
    monkeypatch.chdir(workspace.root)
    agent = _MutableRulesBrowserAgent(
        _BrowserBrain([_click_call(), _snapshot_call()]), registry, workspace
    )
    preset = AutoPresetRunner(agent, workspace_root=workspace.root)
    with workspace_root_context(workspace.root):
        with pytest.raises(ApprovalRequired):
            preset.run("browser task")
        assert agent.last_auto_state["status"] == AutoRunStatus.WAITING_APPROVAL.value
        agent.rules = [_allow_rule("click", "#publish", "external_side_effect")]
        outcome = preset.resume(agent.last_auto_state["run_id"])

    assert outcome.status == AutoRunStatus.COMPLETED
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.text == "Действие выполнено, результат не подтверждён"
    assert "Готово" not in outcome.text
    report_kwargs = agent.report_kwargs_history[-1]
    assert report_kwargs["verifier"].ok is False
    assert "canonical=UNVERIFIED" in report_kwargs["execution_summary"]


# --- DesktopRuntime integration (real registry/gateway/runtime) -----------
#
# Rework по P2 Codex (PR #183): каноническая browser verification должна
# работать на реальном production-пути ToolRegistry -> ToolGateway ->
# DesktopRuntime, а не только в AutoPresetRunner с подменённым
# desktop-контекстом. Gateway здесь строится как в production Agent:
# ApprovalContext(execution_target="desktop", desktop_policy=...) +
# RunPolicy(mode="desktop", execution_target="desktop"); desktop_browser
# зарегистрирован только для desktop target. Никакого monkeypatch
# execution target или policy.


class _DesktopBrowserParent:
    """Родитель DesktopRuntime с production-формой gateway (desktop target)."""

    response_preferences = ResponsePreferences()

    def __init__(self, brain, registry, workspace_root, rules):  # noqa: ANN001
        self.brain = brain
        self.tool_registry = registry
        self.main_config = None
        self.desktop_execution_control = DesktopExecutionControl()
        self._workspace_root = workspace_root
        self._rules = rules

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


def _desktop_browser_registry_with(handler) -> ToolRegistry:  # noqa: ANN001
    registry = ToolRegistry()
    registry.register(
        "desktop_browser",
        handler,
        description="Fake browser tool for postcondition tests",
        parameters_schema={"type": "object", "properties": {"operation": {"type": "string"}}},
        capability="exec",
        risk_classes=["network", "external_side_effect"],
        execution_targets={"desktop"},
    )
    return registry


def _desktop_browser_registry() -> ToolRegistry:
    return _desktop_browser_registry_with(_fake_browser_handler)


def _run_desktop_browser(tmp_path: Path, calls: list[ToolCall], rules, handler=None):  # noqa: ANN001, ANN202
    registry = (
        _desktop_browser_registry() if handler is None else _desktop_browser_registry_with(handler)
    )
    parent = _DesktopBrowserParent(_BrowserBrain(calls), registry, tmp_path, rules)
    runtime = DesktopRuntime(parent)
    try:
        return runtime.run("browser task")
    finally:
        runtime.close()


def test_desktop_runtime_click_with_snapshot_is_unverified(tmp_path: Path) -> None:
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
            ToolCall(
                id="snap-1",
                name="desktop_browser",
                arguments={"operation": "snapshot", "page_id": PAGE},
            ),
        ],
        [_allow_rule("click", "#publish", "external_side_effect")],
    )

    # Desktop gate (re-observation) проходит, но канонический outcome —
    # UNVERIFIED: snapshot доказывает факт наблюдения, а не результат
    # клика. Не DONE и не FAILED_VERIFIER.
    assert outcome.verification.ok
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.canonical_outcome.is_full_done is False


def test_desktop_runtime_self_verified_browser_run_is_done(tmp_path: Path) -> None:
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="nav-1",
                name="desktop_browser",
                arguments={"operation": "navigate", "url": "https://example.test/"},
            ),
            ToolCall(
                id="input-1",
                name="desktop_browser",
                arguments={"operation": "input", "selector": "#name", "value": "x"},
            ),
        ],
        [_allow_rule("input", "#name", "local_mutation")],
    )

    assert outcome.verification.ok
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.ui_label == "DONE"
    assert outcome.canonical_outcome.is_full_done is True


def test_desktop_runtime_click_without_observation_gate_exhaustion_is_unverified(
    tmp_path: Path,
) -> None:
    # Модель кликает и пытается завершить без re-observation: final_gate
    # отклоняет финальный ответ до исчерпания итераций. Отсутствие
    # подтверждения не должно подменяться ошибкой выполнения:
    # канонический outcome — UNVERIFIED (COMPLETED), а не FAILED.
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
        ],
        [_allow_rule("click", "#publish", "external_side_effect")],
    )

    assert outcome.loop_result.error is not None
    assert not outcome.verification.ok
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.canonical_outcome.is_full_done is False


def test_desktop_runtime_click_without_allow_rule_requires_approval(tmp_path: Path) -> None:
    parent = _DesktopBrowserParent(
        _BrowserBrain(
            [
                ToolCall(
                    id="click-1",
                    name="desktop_browser",
                    arguments={"operation": "click", "selector": "#publish"},
                )
            ]
        ),
        _desktop_browser_registry(),
        tmp_path,
        [],
    )
    runtime = DesktopRuntime(parent)
    try:
        with pytest.raises(ApprovalRequired):
            runtime.run("browser task")
    finally:
        runtime.close()


def test_sandbox_gateway_cannot_call_desktop_browser() -> None:
    registry = _desktop_browser_registry()
    sandbox_policy = RunPolicy(mode="auto", execution_target="sandbox")
    desktop_policy = RunPolicy(mode="desktop", execution_target="desktop")

    sandbox_specs = {spec.name for spec in registry.list_tool_specs(policy=sandbox_policy)}
    desktop_specs = {spec.name for spec in registry.list_tool_specs(policy=desktop_policy)}
    assert "desktop_browser" not in sandbox_specs
    assert "desktop_browser" in desktop_specs

    gateway = ToolGateway(registry, run_policy=sandbox_policy)
    result = gateway.call(ToolRequest("desktop_browser", {"operation": "snapshot"}))
    assert not result.ok
    assert "EXECUTION_TARGET_BLOCK" in (result.error or "")


# --- rework round 2: отказ выполнения vs неподтверждённый успех -----------


def test_desktop_runtime_locator_timeout_is_failed_with_original_error(
    tmp_path: Path,
) -> None:
    # Единственный browser-вызов возвращает ToolResult.failure, модель
    # пытается завершить до исчерпания correction loop. Реальная ошибка
    # не должна стираться нормализацией gate-exhaustion: канонический
    # outcome — FAILED с исходной ошибкой, а не completed + UNVERIFIED.
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
        ],
        [_allow_rule("click", "#publish", "external_side_effect")],
        handler=_failing_click_handler,
    )

    assert outcome.loop_result.error == "locator timeout"
    assert not outcome.verification.ok
    assert "locator timeout" in outcome.text
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.terminal_reason is TerminalReason.FAILED
    assert outcome.canonical_outcome.ui_label == "FAILED"
    assert outcome.canonical_outcome.is_full_done is False
    # Исходная ошибка сохранена в каноническом outcome.
    assert outcome.canonical_outcome.verification.checks[0].detail == "locator timeout"


def test_desktop_runtime_failed_click_then_successful_correction_is_done(
    tmp_path: Path,
) -> None:
    # Ошибочное действие, после которого агент успешно исправил ситуацию
    # (self-verified navigate + input), не превращает весь run в FAILED.
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
            ToolCall(
                id="nav-1",
                name="desktop_browser",
                arguments={"operation": "navigate", "url": "https://example.test/"},
            ),
            ToolCall(
                id="input-1",
                name="desktop_browser",
                arguments={"operation": "input", "selector": "#name", "value": "x"},
            ),
        ],
        [
            _allow_rule("click", "#publish", "external_side_effect"),
            _allow_rule("input", "#name", "local_mutation"),
        ],
        handler=_failing_click_handler,
    )

    assert outcome.verification.ok
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.canonical_outcome.ui_label == "DONE"
    assert outcome.canonical_outcome.is_full_done is True


def test_desktop_runtime_failed_click_then_unconfirmed_click_is_unverified(
    tmp_path: Path,
) -> None:
    # Первый click падает, второй проходит, но без re-observation:
    # final_gate исчерпывается. Последнее действие успешно, результат
    # не подтверждён — UNVERIFIED (COMPLETED), а не FAILED по первой
    # ошибке и не DONE.
    outcome = _run_desktop_browser(
        tmp_path,
        [
            ToolCall(
                id="click-1",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
            ToolCall(
                id="click-2",
                name="desktop_browser",
                arguments={"operation": "click", "selector": "#publish"},
            ),
        ],
        [_allow_rule("click", "#publish", "external_side_effect")],
        handler=_flaky_click_handler(),
    )

    assert outcome.loop_result.error is not None
    assert not outcome.verification.ok
    assert outcome.canonical_outcome is not None
    assert outcome.canonical_outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.canonical_outcome.ui_label == "UNVERIFIED"
    assert outcome.canonical_outcome.is_full_done is False


# --- Пользовательский ответ Desktop из канонического outcome ---------------
# Регрессия аудита на d7d5982: click -> snapshot давал canonical
# UNVERIFIED, но пользователю уходили «Готово» и verifier status ok из
# desktop gate результата. Проекция _project_desktop_outcome обязана
# выводить текст, статус верификатора и сводку проверки из
# canonical_outcome во всех путях: обычный ответ, streaming (первый
# TextDelta) и resume после approval сходятся в _finalize_desktop_response.


class _DesktopBrowserAgent(Agent):
    """Полный Agent с fake desktop_browser для проекционных регрессий."""

    def __init__(self, home: Path, calls: list[ToolCall], handler=None) -> None:  # noqa: ANN001
        super().__init__(
            brain=_BrowserBrain(calls),
            desktop_home=home,
            desktop_policy_store=DesktopPolicyStore(home / ".config/slavik/policy.json"),
        )
        self.tool_registry.register(
            "desktop_browser",
            handler or _fake_browser_handler,
            enabled=True,
            capability="exec",
            risk_classes=["network", "external_side_effect"],
            execution_targets={"desktop"},
            description="Fake browser tool for projection tests",
            parameters_schema={
                "type": "object",
                "properties": {"operation": {"type": "string"}},
            },
        )


def _split_user_response(text: str) -> tuple[str, dict]:
    marker = text.rfind(MWV_REPORT_PREFIX)
    assert marker >= 0, "MWV report block missing"
    report = json.loads(text[marker + len(MWV_REPORT_PREFIX) :])
    return text[:marker].rstrip(), report


def _run_desktop_agent_response(tmp_path, monkeypatch, calls, entry, rules, handler=None):  # noqa: ANN001, ANN202
    monkeypatch.chdir(tmp_path)
    agent = _DesktopBrowserAgent(tmp_path, calls, handler=handler)
    agent.set_session_context("session", set())
    agent.set_runtime_state(
        mode="desktop", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    first_delta = None
    try:
        if entry == "resume":
            paused = agent.respond([LLMMessage(role="user", content="browser task")])
            assert paused.runtime_result is None
            identity = agent.last_approval_resume_payload["continuation_id"]
            agent.set_desktop_policy_context(rules, "local")
            response = agent.resume_chat_approval(identity)
        elif entry == "stream":
            agent.set_desktop_policy_context(rules, "local")

            async def consume():  # noqa: ANN202
                return [
                    event
                    async for event in _iterate_stream_in_thread(
                        agent.respond_stream([LLMMessage(role="user", content="browser task")]),
                        asyncio.Event(),
                    )
                ]

            events = asyncio.run(consume())
            deltas = [event for event in events if isinstance(event, TextDelta)]
            assert deltas, "streaming must emit a TextDelta"
            first_delta = deltas[0].text
            produced = [event.response for event in events if isinstance(event, ResponseProduced)]
            assert len(produced) == 1
            response = produced[0]
        else:
            agent.set_desktop_policy_context(rules, "local")
            response = agent.respond([LLMMessage(role="user", content="browser task")])
        return response, first_delta
    finally:
        agent.close()


def _click_call() -> ToolCall:
    return ToolCall(
        id="click-1",
        name="desktop_browser",
        arguments={"operation": "click", "selector": "#publish"},
    )


def _snapshot_call() -> ToolCall:
    return ToolCall(
        id="snap-1",
        name="desktop_browser",
        arguments={"operation": "snapshot", "page_id": PAGE},
    )


@pytest.mark.parametrize("entry", ["sync", "stream", "resume"])
def test_desktop_user_response_unverified_click_is_not_success(
    tmp_path,
    monkeypatch,
    entry,  # noqa: ANN001
) -> None:
    # Действие выполнено, результат не подтверждён: пользователь видит
    # UNVERIFIED-текст и не-ok статус верификатора, а не «Готово» +
    # Verifier OK из desktop gate. Desktop gate при этом PASSED.
    response, first_delta = _run_desktop_agent_response(
        tmp_path,
        monkeypatch,
        [_click_call(), _snapshot_call()],
        entry,
        [_allow_rule("click", "#publish", "external_side_effect")],
    )

    outcome = response.runtime_result
    assert outcome.verification.ok  # desktop gate (re-observation) проходит
    canonical = outcome.canonical_outcome
    assert canonical is not None
    assert canonical.ui_label == "UNVERIFIED"
    assert canonical.is_full_done is False

    user_text, report = _split_user_response(response.text)
    assert user_text == "Действие выполнено, результат не подтверждён"
    assert report["verifier"]["status"] != "ok"
    assert "canonical=UNVERIFIED" in report["execution_summary"]
    assert "verification=passed" not in report["execution_summary"]
    if first_delta is not None:
        delta_text, delta_report = _split_user_response(first_delta)
        assert delta_text == "Действие выполнено, результат не подтверждён"
        assert delta_report["verifier"]["status"] != "ok"


@pytest.mark.parametrize("entry", ["sync", "resume"])
def test_desktop_user_response_failed_click_keeps_original_error(
    tmp_path,
    monkeypatch,
    entry,  # noqa: ANN001
) -> None:
    # Действие завершилось реальной ошибкой: FAILED с исходной причиной
    # в пользовательском тексте, статус верификатора — fail.
    response, _ = _run_desktop_agent_response(
        tmp_path,
        monkeypatch,
        [_click_call()],
        entry,
        [_allow_rule("click", "#publish", "external_side_effect")],
        handler=_failing_click_handler,
    )

    outcome = response.runtime_result
    canonical = outcome.canonical_outcome
    assert canonical is not None
    assert canonical.ui_label == "FAILED"

    user_text, report = _split_user_response(response.text)
    assert "locator timeout" in user_text
    assert report["verifier"]["status"] == "fail"
    assert "canonical=FAILED" in report["execution_summary"]


@pytest.mark.parametrize("entry", ["sync", "stream"])
def test_desktop_user_response_confirmed_run_is_done(
    tmp_path,
    monkeypatch,
    entry,  # noqa: ANN001
) -> None:
    # Ожидаемый результат подтверждён (self-verified navigate + input):
    # DONE, текст модели сохраняется, статус верификатора — ok.
    response, first_delta = _run_desktop_agent_response(
        tmp_path,
        monkeypatch,
        [
            ToolCall(
                id="nav-1",
                name="desktop_browser",
                arguments={"operation": "navigate", "url": "https://example.test/"},
            ),
            ToolCall(
                id="input-1",
                name="desktop_browser",
                arguments={"operation": "input", "selector": "#name", "value": "x"},
            ),
        ],
        entry,
        [_allow_rule("input", "#name", "local_mutation")],
    )

    outcome = response.runtime_result
    canonical = outcome.canonical_outcome
    assert canonical is not None
    assert canonical.ui_label == "DONE"
    assert canonical.is_full_done is True

    user_text, report = _split_user_response(response.text)
    assert user_text == "Готово"
    assert report["verifier"]["status"] == "ok"
    assert "canonical=DONE" in report["execution_summary"]
    if first_delta is not None:
        delta_text, delta_report = _split_user_response(first_delta)
        assert delta_text == "Готово"
        assert delta_report["verifier"]["status"] == "ok"


# --- rework round 3: предупреждение об очистке в проекции UNVERIFIED -----
# P2 Codex (PR #183, id 4658068590): desktop_launch успешен, модель
# исчерпывает final gate без подтверждения -> canonical UNVERIFIED.
# DesktopRuntime при неудачной очистке добавляет desktop_cleanup_failed
# в outcome.text, но проекция заменяла текст целиком и теряла
# предупреждение. Известный отказ rollback не должен скрываться
# неизвестностью основного результата.

CLEANUP_FAILURE_REASON = "process 43210 still running"


def _launch_allow_rule() -> DesktopApprovalRule:
    return DesktopApprovalRule.create(
        effect="allow",
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_launch",
            action="launch",
            target_pattern=None,
            risk_class="unknown",
        ),
    )


def _launch_call() -> ToolCall:
    return ToolCall(
        id="launch-1",
        name="desktop_launch",
        arguments={"argv": ["app"]},
    )


class _DesktopLaunchAgent(Agent):
    """Полный Agent с fake desktop_launch и fake cleanup для проекции."""

    def __init__(self, home: Path, calls: list[ToolCall], *, cleanup_failure: str | None) -> None:
        super().__init__(
            brain=_BrowserBrain(calls),
            desktop_home=home,
            desktop_policy_store=DesktopPolicyStore(home / ".config/slavik/policy.json"),
        )

        def launch(_request: ToolRequest) -> ToolResult:
            return ToolResult.success({"pid": 43210})

        def cleanup(_request: ToolRequest) -> ToolResult:
            if cleanup_failure is None:
                return ToolResult.success({"terminated_pids": [43210], "verified": True})
            return ToolResult.failure(cleanup_failure)

        self.tool_registry.register(
            "desktop_launch",
            launch,
            enabled=True,
            capability="exec",
            risk_classes=["execute"],
            execution_targets={"desktop"},
            description="Fake launch tool for projection tests",
            parameters_schema={
                "type": "object",
                "properties": {"argv": {"type": "array", "items": {"type": "string"}}},
            },
        )
        self.tool_registry.register(
            "desktop_cleanup_unverified_launches",
            cleanup,
            enabled=True,
            capability="exec",
            risk_classes=["execute"],
            model_exposed=False,
            execution_targets={"desktop"},
            description="Fake cleanup tool for projection tests",
            parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
        )


def _run_desktop_launch_agent_response(tmp_path, monkeypatch, entry, *, cleanup_failure):  # noqa: ANN001, ANN202
    monkeypatch.chdir(tmp_path)
    agent = _DesktopLaunchAgent(tmp_path, [_launch_call()], cleanup_failure=cleanup_failure)
    agent.set_session_context("session", set())
    agent.set_runtime_state(
        mode="desktop", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    rules = [_launch_allow_rule()]
    first_delta = None
    try:
        if entry == "resume":
            paused = agent.respond([LLMMessage(role="user", content="launch task")])
            assert paused.runtime_result is None
            identity = agent.last_approval_resume_payload["continuation_id"]
            agent.set_desktop_policy_context(rules, "local")
            response = agent.resume_chat_approval(identity)
        elif entry == "stream":
            agent.set_desktop_policy_context(rules, "local")

            async def consume():  # noqa: ANN202
                return [
                    event
                    async for event in _iterate_stream_in_thread(
                        agent.respond_stream([LLMMessage(role="user", content="launch task")]),
                        asyncio.Event(),
                    )
                ]

            events = asyncio.run(consume())
            deltas = [event for event in events if isinstance(event, TextDelta)]
            assert deltas, "streaming must emit a TextDelta"
            first_delta = deltas[0].text
            produced = [event.response for event in events if isinstance(event, ResponseProduced)]
            assert len(produced) == 1
            response = produced[0]
        else:
            agent.set_desktop_policy_context(rules, "local")
            response = agent.respond([LLMMessage(role="user", content="launch task")])
        return response, first_delta
    finally:
        agent.close()


@pytest.mark.parametrize("entry", ["sync", "stream", "resume"])
def test_desktop_user_response_unverified_launch_keeps_cleanup_warning(
    tmp_path,
    monkeypatch,
    entry,  # noqa: ANN001
) -> None:
    # Launch выполнен, результат не подтверждён (gate исчерпан), а
    # очистка запущенного процесса отказала: пользователь видит
    # UNVERIFIED-текст И предупреждение о неудачной очистке с причиной.
    response, first_delta = _run_desktop_launch_agent_response(
        tmp_path,
        monkeypatch,
        entry,
        cleanup_failure=CLEANUP_FAILURE_REASON,
    )

    outcome = response.runtime_result
    assert outcome.cleanup_failure == CLEANUP_FAILURE_REASON
    canonical = outcome.canonical_outcome
    assert canonical is not None
    assert canonical.ui_label == "UNVERIFIED"
    assert canonical.is_full_done is False

    expected_text = (
        "Действие выполнено, результат не подтверждён"
        f"; desktop_cleanup_failed: {CLEANUP_FAILURE_REASON}"
    )
    user_text, report = _split_user_response(response.text)
    assert user_text == expected_text
    assert report["verifier"]["status"] != "ok"
    assert "canonical=UNVERIFIED" in report["execution_summary"]
    if first_delta is not None:
        delta_text, delta_report = _split_user_response(first_delta)
        assert delta_text == expected_text
        assert delta_report["verifier"]["status"] != "ok"


def test_desktop_user_response_unverified_launch_successful_cleanup_no_warning(
    tmp_path,
    monkeypatch,  # noqa: ANN001
) -> None:
    # Очистка успешна: предупреждения нет, текст — ровно UNVERIFIED-сообщение.
    response, _ = _run_desktop_launch_agent_response(
        tmp_path,
        monkeypatch,
        "sync",
        cleanup_failure=None,
    )

    outcome = response.runtime_result
    assert outcome.cleanup_failure is None
    canonical = outcome.canonical_outcome
    assert canonical is not None
    assert canonical.ui_label == "UNVERIFIED"

    user_text, report = _split_user_response(response.text)
    assert user_text == "Действие выполнено, результат не подтверждён"
    assert "desktop_cleanup_failed" not in response.text
    assert report["verifier"]["status"] != "ok"
