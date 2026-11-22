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

from pathlib import Path

from core.approval_policy import ApprovalContext
from core.auto_preset import (
    AutoPresetRunner,
    _browser_postcondition_result,
    _canonical_verification,
)
from core.desktop_policy import DesktopApprovalRule, DesktopApprovalScope, DesktopPolicyRuntime
from core.desktop_security import DesktopPathSecurity
from core.managed_workspace import ManagedWorkspace, WorkspaceAccess
from core.mwv.models import VerificationResult, VerificationStatus
from core.mwv.verifier_runtime import VerifierRuntime
from core.run_outcome import VerificationOutcome, outcome_from_loop_result
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoopResult, ExecutedToolCall
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ToolCall
from shared.auto_models import AutoRunStatus
from shared.models import JSONValue, ToolRequest, ToolResult
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
