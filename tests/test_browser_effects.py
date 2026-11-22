"""E02-T05: effect classification for generic browser operations.

The classification table (core/browser_effects.py) drives the Desktop
policy defaults: reading a page must not require send approval, typing is
a local draft mutation, submit/click are external side effects, download
is a local write, and unknown operations fail closed instead of inheriting
``read``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.approval_policy import ApprovalContext, decide_request
from core.browser_effects import classify_browser_operation, known_browser_operations
from core.desktop_policy import (
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
)
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest

# Every operation DesktopBrowserTool accepts (its own error message lists
# them); the classification table must cover each one explicitly.
TOOL_OPERATIONS = {
    "start",
    "open",
    "new_tab",
    "navigate",
    "snapshot",
    "read",
    "find",
    "click",
    "input",
    "select",
    "submit",
    "wait",
    "tabs",
    "switch_tab",
    "close_tab",
    "download",
    "close",
    "login_handoff",
}


def _context(
    tmp_path: Path,
    *,
    rules: list[DesktopApprovalRule] | None = None,
) -> ApprovalContext:
    security = DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop_approvals.json",
    )
    return ApprovalContext(
        safe_mode=True,
        session_id="desktop-session",
        approved_categories=set(),
        execution_target="desktop",
        desktop_policy=DesktopPolicyRuntime(rules),
        desktop_security=security,
    )


def _decide(
    tmp_path: Path,
    args: dict[str, object],
    *,
    rules: list[DesktopApprovalRule] | None = None,
) -> str:
    decision, _ = decide_request(
        context=_context(tmp_path, rules=rules),
        request=ToolRequest("desktop_browser", args),  # type: ignore[arg-type]
    )
    return decision.status


def _rule(
    *,
    effect: str,
    action: str,
    target: str,
    risk_class: str,
) -> DesktopApprovalRule:
    return DesktopApprovalRule.create(
        effect=effect,  # type: ignore[arg-type]
        source="session",
        scope=DesktopApprovalScope(
            tool="desktop_browser",
            action=action,
            target_pattern=target,
            risk_class=risk_class,
        ),
    )


def test_classification_covers_every_tool_operation() -> None:
    assert TOOL_OPERATIONS <= known_browser_operations()


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("snapshot", "read"),
        ("read", "read"),
        ("find", "read"),
        ("wait", "read"),
        ("tabs", "read"),
        ("switch_tab", "read"),
        ("start", "read"),
        ("close", "read"),
        ("navigate", "network"),
        ("open", "network"),
        ("new_tab", "network"),
        ("input", "local_mutation"),
        ("select", "local_mutation"),
        ("submit", "external_side_effect"),
        ("click", "external_side_effect"),
        ("upload", "external_side_effect"),
        ("download", "write"),
        ("close_tab", "external_side_effect"),
        ("login_handoff", "user_handoff"),
    ],
)
def test_operation_effect_classes(operation: str, expected: str) -> None:
    assert classify_browser_operation(operation) == expected


def test_unknown_operation_fails_closed_not_read() -> None:
    assert classify_browser_operation("teleport") == "external_side_effect"
    assert classify_browser_operation("") == "external_side_effect"


def test_reading_profile_page_needs_no_approval(tmp_path: Path) -> None:
    assert (
        _decide(tmp_path, {"operation": "navigate", "url": "https://example.test/profile"})
        == "allow"
    )
    assert _decide(tmp_path, {"operation": "snapshot"}) == "allow"
    assert _decide(tmp_path, {"operation": "read", "selector": "main"}) == "allow"


def test_typing_is_local_mutation_allowed_by_default(tmp_path: Path) -> None:
    assert (
        _decide(tmp_path, {"operation": "input", "selector": "#name", "value": "draft"}) == "allow"
    )
    assert _decide(tmp_path, {"operation": "select", "selector": "#kind", "value": "a"}) == "allow"


def test_explicit_rules_still_govern_typing(tmp_path: Path) -> None:
    deny = _rule(effect="deny", action="input", target="#name", risk_class="local_mutation")
    ask = _rule(effect="ask", action="input", target="#name", risk_class="local_mutation")
    args = {"operation": "input", "selector": "#name", "value": "draft"}

    assert _decide(tmp_path, args, rules=[deny]) == "block"
    assert _decide(tmp_path, args, rules=[ask]) == "require_approval"


def test_submit_and_click_require_approval(tmp_path: Path) -> None:
    assert _decide(tmp_path, {"operation": "submit", "selector": "#send"}) == "require_approval"
    assert _decide(tmp_path, {"operation": "click", "selector": "#delete"}) == "require_approval"


def test_click_does_not_inherit_read_allowance(tmp_path: Path) -> None:
    # Negative criterion: an allow rule for observing the page must not
    # make a click on the same page allowed.
    read_allow = _rule(effect="allow", action="snapshot", target="page-1", risk_class="read")

    assert (
        _decide(tmp_path, {"operation": "snapshot", "page_id": "page-1"}, rules=[read_allow])
        == "allow"
    )
    assert (
        _decide(tmp_path, {"operation": "click", "page_id": "page-1"}, rules=[read_allow])
        == "require_approval"
    )


def test_download_is_write_but_overwrite_still_asks(tmp_path: Path) -> None:
    destination = str(tmp_path / "report.pdf")
    base = {"operation": "download", "selector": "#dl", "destination": destination}

    assert _decide(tmp_path, base) == "allow"
    assert _decide(tmp_path, {**base, "overwrite": True}) == "require_approval"


def test_unknown_operation_requires_approval_not_allow(tmp_path: Path) -> None:
    assert _decide(tmp_path, {"operation": "teleport"}) == "require_approval"


def test_login_handoff_still_needs_no_approval(tmp_path: Path) -> None:
    assert _decide(tmp_path, {"operation": "login_handoff"}) == "allow"
