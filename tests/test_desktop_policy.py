from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.approval_policy import ApprovalContext, decide_request
from core.desktop_policy import (
    DesktopAction,
    DesktopApprovalRule,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
    DesktopPolicyStore,
)
from core.desktop_security import DesktopPathSecurity
from shared.models import ToolRequest


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


def test_desktop_safe_create_is_allowed_but_delete_and_overwrite_ask(tmp_path: Path) -> None:
    create, _ = decide_request(
        context=_context(tmp_path),
        request=ToolRequest(
            "desktop_file_write",
            {"path": str(tmp_path / "note.txt"), "content": "hello"},
        ),
    )
    delete, delete_scope = decide_request(
        context=_context(tmp_path),
        request=ToolRequest("desktop_file_delete", {"path": str(tmp_path / "note.txt")}),
    )
    overwrite, _ = decide_request(
        context=_context(tmp_path),
        request=ToolRequest(
            "desktop_file_write",
            {"path": str(tmp_path / "note.txt"), "content": "x", "overwrite": True},
        ),
    )

    assert create.status == "allow"
    assert delete.status == "require_approval"
    assert delete_scope is not None
    assert delete_scope.target_pattern == str(tmp_path / "note.txt")
    assert overwrite.status == "require_approval"


def test_runtime_cleanup_is_a_deterministic_allowed_rollback(tmp_path: Path) -> None:
    decision, scope = decide_request(
        context=_context(tmp_path),
        request=ToolRequest("desktop_cleanup_unverified_launches", {}),
    )

    assert decision.status == "allow"
    assert decision.reason == "desktop_safe_default"
    assert scope is not None
    assert scope.action == "rollback"
    assert scope.target_pattern == "current-desktop-run"

    store = DesktopPolicyStore(tmp_path / "desktop-approvals.json")
    with pytest.raises(ValueError, match="Runtime cleanup policy недоступна"):
        store.add_rule(
            DesktopApprovalRule.create(
                effect="deny",
                source="persistent",
                scope=scope,
            )
        )


def test_deny_precedence_over_broad_allow() -> None:
    action = DesktopAction(
        tool="desktop_file_delete",
        action="delete",
        target="/home/user/Downloads/blocked.iso",
        risk_class="destructive",
    )
    broad_allow = DesktopApprovalRule.create(
        effect="allow",
        source="persistent",
        scope=DesktopApprovalScope(
            tool=action.tool,
            action=action.action,
            target_pattern="/home/user/Downloads/**",
            risk_class=action.risk_class,
        ),
    )
    exact_deny = DesktopApprovalRule.create(
        effect="deny",
        source="persistent",
        scope=DesktopApprovalScope(
            tool=action.tool,
            action=action.action,
            target_pattern=action.target,
            risk_class=action.risk_class,
        ),
    )

    resolution = DesktopPolicyRuntime([broad_allow, exact_deny]).resolve(
        action,
        default_effect="ask",
        default_reason="destructive_action",
    )

    assert resolution.effect == "deny"
    assert resolution.rule_id == exact_deny.rule_id


def test_once_rule_is_consumed_only_when_entire_request_is_allowed(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("data", encoding="utf-8")
    destination = tmp_path / "destination.txt"
    once = DesktopApprovalRule.create(
        effect="allow",
        source="once",
        scope=DesktopApprovalScope(
            tool="desktop_file_transfer",
            action="read_source",
            target_pattern=str(source),
            risk_class="read",
        ),
    )
    overwrite_ask = DesktopApprovalRule.create(
        effect="ask",
        source="persistent",
        scope=DesktopApprovalScope(
            tool="desktop_file_transfer",
            action="copy",
            target_pattern=str(destination),
            risk_class="overwrite",
        ),
    )
    runtime = DesktopPolicyRuntime([once, overwrite_ask])
    context = _context(tmp_path)
    context = ApprovalContext(
        safe_mode=True,
        session_id=context.session_id,
        approved_categories=set(),
        execution_target="desktop",
        desktop_policy=runtime,
        desktop_security=context.desktop_security,
    )
    request = ToolRequest(
        "desktop_file_transfer",
        {
            "operation": "copy",
            "source": str(source),
            "destination": str(destination),
            "overwrite": True,
        },
    )

    first, _ = decide_request(context=context, request=request)

    assert first.status == "require_approval"
    assert runtime.drain_consumed_rule_ids() == []


def test_shell_approval_is_bound_to_exact_command(tmp_path: Path) -> None:
    request = ToolRequest(
        "desktop_shell",
        {"argv": ["pytest", "tests/test_one.py"], "cwd": str(tmp_path)},
    )
    decision, scope = decide_request(context=_context(tmp_path), request=request)

    assert decision.status == "require_approval"
    assert scope is not None
    assert scope.command_exact == "pytest tests/test_one.py"

    rule = DesktopApprovalRule.create(effect="allow", source="session", scope=scope)
    allowed, _ = decide_request(context=_context(tmp_path, rules=[rule]), request=request)
    changed, _ = decide_request(
        context=_context(tmp_path, rules=[rule]),
        request=ToolRequest(
            "desktop_shell",
            {"argv": ["pytest", "tests/test_two.py"], "cwd": str(tmp_path)},
        ),
    )

    assert allowed.status == "allow"
    assert changed.status == "require_approval"


def test_exact_approval_treats_wildcard_characters_literally() -> None:
    path_scope = DesktopApprovalScope(
        tool="desktop_file_delete",
        action="delete",
        target_pattern="/home/user/Downloads/file*.iso",
        risk_class="destructive",
    )
    command_scope = DesktopApprovalScope(
        tool="desktop_shell",
        action="execute",
        target_pattern="/home/user",
        command_class="unknown",
        command_exact="custom '*'",
        risk_class="unknown",
    )

    assert path_scope.matches(
        DesktopAction(
            tool="desktop_file_delete",
            action="delete",
            target="/home/user/Downloads/file*.iso",
            risk_class="destructive",
        )
    )
    assert not path_scope.matches(
        DesktopAction(
            tool="desktop_file_delete",
            action="delete",
            target="/home/user/Downloads/file-secret.iso",
            risk_class="destructive",
        )
    )
    assert not command_scope.matches(
        DesktopAction(
            tool="desktop_shell",
            action="execute",
            target="/home/user",
            command_class="unknown",
            command="custom secret",
            risk_class="unknown",
        )
    )


def test_legacy_command_pattern_is_rejected_instead_of_broadening_scope() -> None:
    with pytest.raises(ValueError, match="command_pattern больше не поддерживается"):
        DesktopApprovalScope.from_dict(
            {
                "tool": "desktop_shell",
                "action": "execute",
                "command_pattern": "git *",
            }
        )


def test_policy_store_rejects_all_rules_when_one_persistent_rule_is_invalid(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "desktop-approvals.json"
    store_path.write_text(
        json.dumps(
            [
                {
                    "rule_id": "legacy-pattern",
                    "effect": "deny",
                    "source": "persistent",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "scope": {
                        "tool": "desktop_shell",
                        "action": "execute",
                        "command_pattern": "git *",
                    },
                },
                DesktopApprovalRule.create(
                    effect="allow",
                    source="persistent",
                    scope=DesktopApprovalScope(
                        tool="desktop_file_delete",
                        action="delete",
                        target_pattern="/tmp/protected",
                    ),
                ).to_dict(),
            ]
        ),
        encoding="utf-8",
    )
    store = DesktopPolicyStore(store_path)

    with pytest.raises(ValueError, match="contains invalid rules"):
        store.list_rules()
    assert "command_pattern больше не поддерживается" in store.list_load_errors()[0]


def test_transfer_checks_protected_source_and_destination(tmp_path: Path) -> None:
    request = ToolRequest(
        "desktop_file_transfer",
        {
            "operation": "copy",
            "source": "/root/private.txt",
            "destination": str(tmp_path / "copy.txt"),
        },
    )

    decision, scope = decide_request(context=_context(tmp_path), request=request)

    assert decision.status == "block"
    assert scope is not None
    assert scope.action == "read_source"


def test_symlink_and_traversal_resolve_to_protected_resource(tmp_path: Path) -> None:
    security = _context(tmp_path).desktop_security
    assert security is not None
    link = tmp_path / "root-link"
    link.symlink_to("/root", target_is_directory=True)

    assert security.resolve(str(link / "secret")).protection == "deny"
    assert security.resolve(str(tmp_path / "folder" / ".." / ".run")).protection == "deny"


def test_enforcement_config_and_its_mutating_ancestor_are_protected(tmp_path: Path) -> None:
    protected = tmp_path / "application" / "config"
    security = DesktopPathSecurity(
        home=tmp_path,
        policy_store_path=tmp_path / ".run" / "desktop_approvals.json",
        protected_paths=(protected,),
    )

    assert security.resolve(str(protected / "tools.json")).protection == "deny"
    assert security.resolve(str(tmp_path / "application")).protection == "normal"
    assert security.resolve(str(tmp_path / "application"), mutation=True).protection == "deny"
    assert security.resolve(str(tmp_path), mutation=True).protection == "deny"


def test_policy_store_round_trip_and_rejects_nonpersistent_rule(tmp_path: Path) -> None:
    store = DesktopPolicyStore(tmp_path / "approvals.json")
    scope = DesktopApprovalScope(
        tool="desktop_file_delete",
        action="delete",
        target_pattern=str(tmp_path / "Downloads" / "**"),
        risk_class="destructive",
    )
    persistent = DesktopApprovalRule.create(
        effect="allow",
        source="persistent",
        scope=scope,
        description="downloads only",
    )

    store.add_rule(persistent)

    assert store.list_rules() == [persistent]
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.update_rule(persistent.rule_id, effect="deny") is not None
    assert store.list_rules()[0].effect == "deny"
    assert store.remove_rule(persistent.rule_id)
    assert store.list_rules() == []
    with pytest.raises(ValueError):
        store.add_rule(DesktopApprovalRule.create(effect="allow", source="session", scope=scope))
    with pytest.raises(ValueError):
        store.add_rule(
            DesktopApprovalRule.create(
                effect="allow",
                source="persistent",
                scope=DesktopApprovalScope(
                    tool="desktop_shell",
                    action="execute",
                    target_pattern=str(tmp_path),
                    command_class="project_execution",
                    command_exact="pytest",
                    risk_class="arbitrary_code",
                ),
            )
        )


def test_persistent_allow_reuses_only_its_exact_file_scope(tmp_path: Path) -> None:
    approved = tmp_path / "Downloads" / "approved.iso"
    different = tmp_path / "Project" / "different.iso"
    rule = DesktopApprovalRule.create(
        effect="allow",
        source="persistent",
        scope=DesktopApprovalScope(
            tool="desktop_file_delete",
            action="delete",
            target_pattern=str(approved),
            risk_class="destructive",
        ),
    )

    allowed, _ = decide_request(
        context=_context(tmp_path, rules=[rule]),
        request=ToolRequest("desktop_file_delete", {"path": str(approved)}),
    )
    still_asks, _ = decide_request(
        context=_context(tmp_path, rules=[rule]),
        request=ToolRequest("desktop_file_delete", {"path": str(different)}),
    )

    assert allowed.status == "allow"
    assert allowed.policy_rule_id == rule.rule_id
    assert still_asks.status == "require_approval"


def test_privilege_and_interpreter_indirection_are_hard_denied(tmp_path: Path) -> None:
    commands = (
        ["sudo", "id"],
        ["python3", "-c", "print('x')"],
        ["find", ".", "-exec", "rm", "{}", ";"],
        ["timeout", "10", "systemctl", "restart", "ssh"],
        ["busybox", "rm", "-f", "protected"],
    )
    for argv in commands:
        decision, _ = decide_request(
            context=_context(tmp_path),
            request=ToolRequest("desktop_shell", {"argv": argv, "cwd": str(tmp_path)}),
        )
        assert decision.status == "block"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("desktop_system_info", {"operation": "summary"}),
        ("desktop_process", {"operation": "list"}),
        ("desktop_systemd", {"operation": "status", "unit": "example.service"}),
        ("desktop_package", {"operation": "query", "package": "python3"}),
        ("desktop_session", {"operation": "capabilities"}),
    ],
)
def test_typed_read_only_system_operations_do_not_require_approval(
    tmp_path: Path,
    tool: str,
    args: dict[str, object],
) -> None:
    decision, _ = decide_request(
        context=_context(tmp_path),
        request=ToolRequest(tool, args),  # type: ignore[arg-type]
    )

    assert decision.status == "allow"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("desktop_clipboard", {"operation": "read"}),
        ("desktop_process", {"operation": "terminate", "pid": 123, "expected_create_time": 1.0}),
        ("desktop_systemd", {"operation": "restart", "unit": "example.service"}),
        ("desktop_package", {"operation": "install", "package": "example"}),
        ("desktop_gui", {"operation": "click", "x": 10, "y": 10}),
    ],
)
def test_sensitive_typed_operations_require_approval(
    tmp_path: Path,
    tool: str,
    args: dict[str, object],
) -> None:
    decision, scope = decide_request(
        context=_context(tmp_path),
        request=ToolRequest(tool, args),  # type: ignore[arg-type]
    )

    assert decision.status == "require_approval"
    assert scope is not None and scope.tool == tool


def test_browser_download_cannot_target_protected_policy_store(tmp_path: Path) -> None:
    protected = tmp_path / ".run" / "desktop_approvals.json"

    decision, scope = decide_request(
        context=_context(tmp_path),
        request=ToolRequest(
            "desktop_browser",
            {
                "operation": "download",
                "selector_type": "text",
                "selector": "Download",
                "destination": str(protected),
            },
        ),
    )

    assert decision.status == "block"
    assert scope is not None and scope.action == "download_destination"


def test_browser_gui_and_clipboard_persistent_allow_are_rejected(tmp_path: Path) -> None:
    store = DesktopPolicyStore(tmp_path / "approvals.json")
    for scope in (
        DesktopApprovalScope(
            tool="desktop_browser",
            action="click",
            target_pattern="page-1",
            risk_class="external_side_effect",
        ),
        DesktopApprovalScope(
            tool="desktop_gui",
            action="click",
            target_pattern="screen:1:1",
            risk_class="ui_interaction",
        ),
        DesktopApprovalScope(
            tool="desktop_clipboard",
            action="read",
            target_pattern="clipboard",
            risk_class="sensitive_read",
        ),
    ):
        with pytest.raises(ValueError):
            store.add_rule(
                DesktopApprovalRule.create(effect="allow", source="persistent", scope=scope)
            )


def test_exact_systemd_persistent_rule_does_not_allow_another_unit(tmp_path: Path) -> None:
    request = ToolRequest(
        "desktop_systemd",
        {"operation": "restart", "scope": "user", "unit": "demo.service"},
    )
    initial, scope = decide_request(context=_context(tmp_path), request=request)
    assert initial.status == "require_approval" and scope is not None
    rule = DesktopApprovalRule.create(effect="allow", source="persistent", scope=scope)

    allowed, _ = decide_request(context=_context(tmp_path, rules=[rule]), request=request)
    other, _ = decide_request(
        context=_context(tmp_path, rules=[rule]),
        request=ToolRequest(
            "desktop_systemd",
            {"operation": "restart", "scope": "user", "unit": "other.service"},
        ),
    )

    assert allowed.status == "allow"
    assert other.status == "require_approval"
