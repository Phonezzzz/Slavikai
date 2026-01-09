from __future__ import annotations

from core.approval_policy import (
    ApprovalContext,
    ApprovalDecision,
    build_approval_request,
    detect_action_intents,
    summarize_intents,
)
from shared.models import ToolRequest


def test_build_approval_request_returns_none_without_intents() -> None:
    context = ApprovalContext(
        safe_mode=True,
        session_id="s1",
        approved_categories=set(),
    )
    decision = ApprovalDecision(
        status="allow",
        reason="no_intent",
        intents=[],
        required_categories=[],
    )
    assert build_approval_request(context=context, decision=decision) is None


def test_detect_action_intents_workspace_write_outside_and_secrets() -> None:
    intents = detect_action_intents(ToolRequest(name="workspace_write", args={"path": "../.env"}))
    categories = {intent.category for intent in intents}
    assert categories == {
        "FS_OUTSIDE_WORKSPACE",
        "FS_CONFIG_SECRETS",
        "FS_DELETE_OVERWRITE",
    }


def test_detect_action_intents_workspace_patch_dry_run() -> None:
    intents = detect_action_intents(
        ToolRequest(
            name="workspace_patch",
            args={"path": "project/readme.md", "dry_run": True},
        )
    )
    assert intents == []


def test_detect_action_intents_fs_write_inside_project() -> None:
    intents = detect_action_intents(
        ToolRequest(name="fs", args={"op": "write", "path": "project/config.yaml"})
    )
    categories = {intent.category for intent in intents}
    assert "FS_OUTSIDE_WORKSPACE" not in categories
    assert {"FS_CONFIG_SECRETS", "FS_DELETE_OVERWRITE"}.issubset(categories)


def test_detect_action_intents_web_and_summary() -> None:
    intents = detect_action_intents(ToolRequest(name="web", args={"query": "test"}))
    assert intents[0].category == "NETWORK_RISK"
    summary = summarize_intents(intents)
    assert "web:NETWORK_RISK" in summary


def test_detect_action_intents_shell_risks() -> None:
    intents = detect_action_intents(
        ToolRequest(name="shell", args={"command": "sudo pip install requests"})
    )
    categories = {intent.category for intent in intents}
    assert "SUDO" in categories
    assert "DEPS_INSTALL_UPDATE" in categories
    assert "EXEC_ARBITRARY" in categories
