from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from core.desktop_policy import (
    DesktopAction,
    DesktopApprovalScope,
    DesktopPolicyRuntime,
    PolicyEffect,
    desktop_actions_from_request,
    exact_scope_for_action,
)
from core.desktop_security import DesktopPathSecurity
from shared.command_safety import is_hard_unsafe_command as _is_unsafe
from shared.models import JSONValue, ToolRequest

ApprovalCategory = Literal[
    "FS_DELETE_OVERWRITE",
    "FS_OUTSIDE_WORKSPACE",
    "FS_CONFIG_SECRETS",
    "DEPS_INSTALL_UPDATE",
    "GIT_PUBLISH",
    "SYSTEM_IMPACT",
    "SUDO",
    "NETWORK_RISK",
    "EXEC_ARBITRARY",
    "HARD_DENY",
]
ApprovalDecisionStatus = Literal["allow", "require_approval", "block"]


@dataclass(frozen=True)
class ApprovalPrompt:
    what: str
    why: str
    risk: str
    changes: list[str]


@dataclass(frozen=True)
class ActionIntent:
    category: ApprovalCategory
    tool: str
    summary: str
    details: dict[str, JSONValue]
    prompt: ApprovalPrompt


@dataclass(frozen=True)
class ApprovalContext:
    safe_mode: bool
    session_id: str | None
    approved_categories: set[ApprovalCategory]
    execution_target: Literal["sandbox", "desktop"] = "sandbox"
    desktop_policy: DesktopPolicyRuntime | None = None
    desktop_security: DesktopPathSecurity | None = None


@dataclass(frozen=True)
class ApprovalDecision:
    status: ApprovalDecisionStatus
    reason: str
    intents: list[ActionIntent]
    required_categories: list[ApprovalCategory]
    policy_rule_id: str | None = None


@dataclass(frozen=True)
class ApprovalRequest:
    category: ApprovalCategory
    required_categories: list[ApprovalCategory]
    prompt: ApprovalPrompt
    tool: str
    details: dict[str, JSONValue]
    session_id: str | None
    scope: DesktopApprovalScope | None = None
    reason: str = "category_not_approved"
    policy_rule_id: str | None = None


class ApprovalRequired(RuntimeError):
    def __init__(self, request: ApprovalRequest) -> None:
        super().__init__("approval_required")
        self.request = request


ALL_CATEGORIES: Final[set[ApprovalCategory]] = {
    "FS_DELETE_OVERWRITE",
    "FS_OUTSIDE_WORKSPACE",
    "FS_CONFIG_SECRETS",
    "DEPS_INSTALL_UPDATE",
    "GIT_PUBLISH",
    "SYSTEM_IMPACT",
    "SUDO",
    "NETWORK_RISK",
    "EXEC_ARBITRARY",
    "HARD_DENY",
}

_RISK_TEXT: Final[dict[ApprovalCategory, str]] = {
    "FS_DELETE_OVERWRITE": "Можно потерять данные/историю файлов.",
    "FS_OUTSIDE_WORKSPACE": "Действие вне рабочей папки проекта.",
    "FS_CONFIG_SECRETS": "Риск утечки или поломки конфигурации.",
    "DEPS_INSTALL_UPDATE": "Изменит зависимости и окружение проекта.",
    "GIT_PUBLISH": "Публикация изменений наружу.",
    "SYSTEM_IMPACT": "Может повлиять на систему.",
    "SUDO": "Повышенные права, риск для системы.",
    "NETWORK_RISK": "Сетевой доступ во внешние сервисы.",
    "EXEC_ARBITRARY": "Выполнение произвольной команды.",
    "HARD_DENY": "Команда запрещена политикой безопасности.",
}

_CONFIG_HINTS: Final[tuple[str, ...]] = (
    ".env",
    "config",
    "конфиг",
    "secret",
    "секрет",
    "token",
    "apikey",
    "api key",
    "ключ",
    "password",
    "пароль",
    "credential",
    "ssh",
    ".pem",
    ".key",
)
_DEPS_HINTS: Final[tuple[str, ...]] = (
    "pip install",
    "pip3 install",
    "poetry add",
    "poetry update",
    "pipenv install",
    "npm install",
    "npm update",
    "yarn add",
    "yarn upgrade",
    "requirements.txt",
    "package.json",
)
_GIT_HINTS: Final[tuple[str, ...]] = (
    "git commit",
    "git push",
    "git tag",
    "publish",
    "release",
)
_SYSTEM_HINTS: Final[tuple[str, ...]] = (
    "systemctl",
    "service",
    "iptables",
    "ufw",
    "mount",
    "umount",
    "mkfs",
    "reboot",
    "shutdown",
)
_NETWORK_HINTS: Final[tuple[str, ...]] = (
    "curl ",
    "wget ",
    "http://",
    "https://",
)
_WINDOWS_ABS_RE: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z]:[/\\\\]")


def decide_action(
    *,
    context: ApprovalContext,
    intents: list[ActionIntent],
) -> ApprovalDecision:
    if not intents:
        return ApprovalDecision("allow", "no_intent", [], [])
    hard_denied = [intent for intent in intents if intent.category == "HARD_DENY"]
    if hard_denied:
        return ApprovalDecision(
            "block",
            "command_denied:hard_safety",
            hard_denied,
            ["HARD_DENY"],
        )
    if not context.safe_mode:
        return ApprovalDecision("allow", "safe_mode_disabled", intents, [])
    required = [intent for intent in intents if intent.category not in context.approved_categories]
    if required:
        categories = _unique_categories(required)
        return ApprovalDecision("require_approval", "category_not_approved", required, categories)
    return ApprovalDecision("allow", "approved", intents, [])


def decide_request(
    *,
    context: ApprovalContext,
    request: ToolRequest,
    risk_classes: Sequence[str] | None = None,
) -> tuple[ApprovalDecision, DesktopApprovalScope | None]:
    if context.execution_target != "desktop":
        intents = detect_action_intents(request, risk_classes=risk_classes)
        return decide_action(context=context, intents=intents), None
    actions = desktop_actions_from_request(request)
    if not actions:
        return ApprovalDecision("block", "desktop_tool_unknown", [], []), None
    canonical_actions = [
        _canonical_desktop_action(action, context.desktop_security) for action in actions
    ]
    defaults = [_desktop_default(action, context.desktop_security) for action in canonical_actions]
    for action, (hard_effect, hard_reason) in zip(canonical_actions, defaults, strict=True):
        if hard_effect == "deny":
            intent = _desktop_intent(action, request)
            return (
                ApprovalDecision("block", hard_reason, [intent], [intent.category]),
                exact_scope_for_action(action),
            )
    policy = context.desktop_policy or DesktopPolicyRuntime()
    resolutions = [
        policy.resolve(
            action,
            default_effect=default[0],
            default_reason=default[1],
            consume_once=False,
        )
        for action, default in zip(canonical_actions, defaults, strict=True)
    ]
    for wanted_effect in ("deny", "ask"):
        for action, resolution in zip(canonical_actions, resolutions, strict=True):
            if resolution.effect != wanted_effect:
                continue
            intent = _desktop_intent(action, request)
            status: ApprovalDecisionStatus = (
                "block" if wanted_effect == "deny" else "require_approval"
            )
            return (
                ApprovalDecision(
                    status,
                    resolution.reason,
                    [intent],
                    [intent.category],
                    policy_rule_id=resolution.rule_id,
                ),
                exact_scope_for_action(action),
            )
    policy.consume_once_rule_ids(
        {resolution.rule_id for resolution in resolutions if resolution.rule_id is not None}
    )
    intents = [_desktop_intent(action, request) for action in canonical_actions]
    reason = next(
        (resolution.reason for resolution in resolutions if resolution.rule_id is not None),
        "desktop_safe_default",
    )
    policy_rule_id = next(
        (resolution.rule_id for resolution in resolutions if resolution.rule_id is not None),
        None,
    )
    return ApprovalDecision(
        "allow",
        reason,
        intents,
        [],
        policy_rule_id=policy_rule_id,
    ), exact_scope_for_action(canonical_actions[-1])


def build_approval_request(
    *,
    context: ApprovalContext,
    decision: ApprovalDecision,
    scope: DesktopApprovalScope | None = None,
) -> ApprovalRequest | None:
    if decision.status != "require_approval" or not decision.intents:
        return None
    primary = decision.intents[0]
    return ApprovalRequest(
        category=primary.category,
        required_categories=decision.required_categories,
        prompt=primary.prompt,
        tool=primary.tool,
        details=primary.details,
        session_id=context.session_id,
        scope=scope,
        reason=decision.reason,
        policy_rule_id=decision.policy_rule_id,
    )


def _canonical_desktop_action(
    action: DesktopAction,
    security: DesktopPathSecurity | None,
) -> DesktopAction:
    if security is None or action.target is None:
        return action
    if not _desktop_target_is_path(action):
        return action
    resolved = security.resolve(
        action.target,
        mutation=_desktop_action_mutates_target(action),
    )
    return DesktopAction(
        tool=action.tool,
        action=action.action,
        target=str(resolved.canonical),
        command_class=action.command_class,
        command=action.command,
        risk_class=action.risk_class,
        execution_target=action.execution_target,
    )


def _desktop_default(
    action: DesktopAction,
    security: DesktopPathSecurity | None,
) -> tuple[PolicyEffect, str]:
    if action.target is not None and security is not None and _desktop_target_is_path(action):
        resolved = security.resolve(
            action.target,
            mutation=_desktop_action_mutates_target(action),
        )
        if resolved.protection == "deny":
            return "deny", f"protected_resource:{resolved.reason}"
        if resolved.protection == "ask":
            return "ask", f"sensitive_resource:{resolved.reason}"
    if action.action == "delete":
        return "ask", "destructive_action"
    if action.risk_class == "overwrite":
        return "ask", "overwrite_action"
    if action.risk_class in {
        "destructive",
        "external_side_effect",
        "install",
        "sensitive_read",
        "system_impact",
        "ui_interaction",
    }:
        return "ask", f"desktop_risk_requires_approval:{action.risk_class}"
    if action.command_class in {"privilege_escalation", "disk_boot", "shell_indirection"}:
        return "deny", f"command_denied:{action.command_class}"
    if action.command_class in {
        "package_management",
        "service_management",
        "network",
        "project_execution",
        "filesystem_mutation",
        "unknown",
        "malformed",
    }:
        return "ask", f"command_requires_approval:{action.command_class}"
    return "allow", "desktop_safe_default"


def _desktop_action_mutates_target(action: DesktopAction) -> bool:
    return action.action in {
        "archive_extract",
        "copy",
        "delete",
        "move",
        "move_source",
        "rename",
        "write",
        "download_destination",
    }


def _desktop_target_is_path(action: DesktopAction) -> bool:
    if action.tool in {
        "desktop_file_search",
        "desktop_file_read",
        "desktop_file_write",
        "desktop_file_transfer",
        "desktop_file_delete",
        "desktop_archive_extract",
        "desktop_shell",
        "desktop_launch",
        "desktop_verify",
    }:
        return True
    if action.tool == "desktop_process" and action.action == "launch":
        return True
    return action.tool == "desktop_browser" and action.action == "download_destination"


def _desktop_intent(action: DesktopAction, request: ToolRequest) -> ActionIntent:
    category: ApprovalCategory
    if action.command_class == "package_management" or (
        action.tool == "desktop_package"
        and action.action in {"install", "remove", "update_metadata"}
    ):
        category = "DEPS_INSTALL_UPDATE"
    elif action.action == "delete" or action.risk_class in {"destructive", "overwrite"}:
        category = "FS_DELETE_OVERWRITE"
    elif action.command_class == "service_management":
        category = "SYSTEM_IMPACT"
    elif action.tool == "desktop_systemd" or action.risk_class == "system_impact":
        category = "SYSTEM_IMPACT"
    elif action.command_class == "privilege_escalation":
        category = "SUDO"
    elif action.risk_class == "network":
        category = "NETWORK_RISK"
    elif action.risk_class == "sensitive_read":
        category = "FS_CONFIG_SECRETS"
    else:
        category = "EXEC_ARBITRARY"
    details: dict[str, JSONValue] = {
        "action": action.action,
        "target": action.target,
        "command_class": action.command_class,
        "risk_class": action.risk_class,
        "execution_target": action.execution_target,
        "arguments": request.args,
    }
    return _intent(action.tool, category, details)


def detect_action_intents(
    request: ToolRequest,
    *,
    risk_classes: Sequence[str] | None = None,
) -> list[ActionIntent]:
    tool = request.name
    args = request.args
    risk_intents = _risk_class_intents(tool, risk_classes or [])
    intents: list[ActionIntent] = []

    if tool in {"shell", "workspace_run", "workspace_terminal_run"}:
        command = str(args.get("command") or args.get("path") or "")
        if command.strip():
            if _is_unsafe(command):
                intents.append(_intent(tool, "HARD_DENY", {"command": command}))
            else:
                intents.extend(_shell_intents(tool, command))
        config_path = str(args.get("config_path") or "")
        if config_path:
            intents.append(
                _intent(
                    tool,
                    "FS_CONFIG_SECRETS",
                    {"path": config_path, "op": "shell_config_write"},
                )
            )
        return _extend_missing_intents(intents, risk_intents)

    if tool == "web":
        intents.append(_intent(tool, "NETWORK_RISK", {"tool": tool}))
        return _extend_missing_intents(intents, risk_intents)

    if tool in {"workspace_write", "workspace_create", "workspace_patch"}:
        path = str(args.get("path") or "")
        dry_run = bool(args.get("dry_run", False))
        if dry_run:
            return []
        if path:
            if _is_outside_workspace(path, workspace_relative=True):
                intents.append(
                    _intent(
                        tool,
                        "FS_OUTSIDE_WORKSPACE",
                        {"path": path, "op": "write"},
                    ),
                )
            intents.extend(_write_intents(tool, path))
        return _extend_missing_intents(intents, risk_intents)

    if tool == "workspace_rename":
        old_path = str(args.get("old_path") or "")
        new_path = str(args.get("new_path") or "")
        if old_path and _is_outside_workspace(old_path, workspace_relative=True):
            intents.append(
                _intent(tool, "FS_OUTSIDE_WORKSPACE", {"path": old_path, "op": "rename"})
            )
        if new_path and _is_outside_workspace(new_path, workspace_relative=True):
            intents.append(
                _intent(tool, "FS_OUTSIDE_WORKSPACE", {"path": new_path, "op": "rename"})
            )
        if old_path:
            intents.extend(_write_intents(tool, old_path))
        if new_path:
            intents.extend(_write_intents(tool, new_path))
        return _extend_missing_intents(intents, risk_intents)

    if tool == "workspace_move":
        from_path = str(args.get("from_path") or "")
        to_path = str(args.get("to_path") or "")
        if from_path and _is_outside_workspace(from_path, workspace_relative=True):
            intents.append(_intent(tool, "FS_OUTSIDE_WORKSPACE", {"path": from_path, "op": "move"}))
        if to_path and _is_outside_workspace(to_path, workspace_relative=True):
            intents.append(_intent(tool, "FS_OUTSIDE_WORKSPACE", {"path": to_path, "op": "move"}))
        if from_path:
            intents.extend(_write_intents(tool, from_path))
        if to_path:
            intents.extend(_write_intents(tool, to_path))
        return _extend_missing_intents(intents, risk_intents)

    if tool == "workspace_delete":
        path = str(args.get("path") or "")
        if path and _is_outside_workspace(path, workspace_relative=True):
            intents.append(_intent(tool, "FS_OUTSIDE_WORKSPACE", {"path": path, "op": "delete"}))
        if path:
            intents.extend(_write_intents(tool, path))
        return _extend_missing_intents(intents, risk_intents)

    if tool == "fs":
        op = str(args.get("op") or "").lower().strip()
        path = str(args.get("path") or "")
        intents.extend(_fs_intents(tool, op, path))
        return _extend_missing_intents(intents, risk_intents)

    return _extend_missing_intents(intents, risk_intents)


def _risk_class_intents(tool: str, risk_classes: Sequence[str]) -> list[ActionIntent]:
    intents: list[ActionIntent] = []
    mapping: dict[str, ApprovalCategory] = {
        "execute": "EXEC_ARBITRARY",
        "install": "DEPS_INSTALL_UPDATE",
        "network": "NETWORK_RISK",
        "external_side_effect": "NETWORK_RISK",
        "privileged": "SUDO",
        "destructive": "FS_DELETE_OVERWRITE",
    }
    seen: set[ApprovalCategory] = set()
    for item in risk_classes:
        category = mapping.get(item)
        if category is None or category in seen:
            continue
        seen.add(category)
        intents.append(_intent(tool, category, {"tool": tool, "risk_class": item}))
    return intents


def _extend_missing_intents(
    intents: list[ActionIntent],
    extra: list[ActionIntent],
) -> list[ActionIntent]:
    seen = {(intent.category, intent.tool) for intent in intents}
    for intent in extra:
        key = (intent.category, intent.tool)
        if key in seen:
            continue
        intents.append(intent)
        seen.add(key)
    return intents


def summarize_intents(intents: list[ActionIntent]) -> str:
    if not intents:
        return ""
    parts = [f"{intent.tool}:{intent.category}" for intent in intents]
    return ", ".join(parts)


def _fs_intents(tool: str, op: str, path: str) -> list[ActionIntent]:
    intents: list[ActionIntent] = []
    if path and _is_outside_workspace(path, workspace_relative=False):
        intents.append(
            _intent(
                tool,
                "FS_OUTSIDE_WORKSPACE",
                {"path": path, "op": op or "list"},
            ),
        )
    if op == "write" and path:
        intents.extend(_write_intents(tool, path))
    return intents


def _write_intents(tool: str, path: str) -> list[ActionIntent]:
    intents: list[ActionIntent] = []
    if not path:
        return intents
    if _is_config_or_secret(path):
        intents.append(_intent(tool, "FS_CONFIG_SECRETS", {"path": path}))
    intents.append(_intent(tool, "FS_DELETE_OVERWRITE", {"path": path}))
    return intents


def _shell_intents(tool: str, command: str) -> list[ActionIntent]:
    normalized = command.lower()
    intents: list[ActionIntent] = []
    if "sudo" in normalized:
        intents.append(_intent(tool, "SUDO", {"command": command}))
    if _contains_any(normalized, _DEPS_HINTS):
        intents.append(_intent(tool, "DEPS_INSTALL_UPDATE", {"command": command}))
    if _contains_any(normalized, _GIT_HINTS):
        intents.append(_intent(tool, "GIT_PUBLISH", {"command": command}))
    if _contains_any(normalized, _SYSTEM_HINTS):
        intents.append(_intent(tool, "SYSTEM_IMPACT", {"command": command}))
    if _contains_any(normalized, _NETWORK_HINTS):
        intents.append(_intent(tool, "NETWORK_RISK", {"command": command}))
    intents.append(_intent(tool, "EXEC_ARBITRARY", {"command": command}))
    return intents


def _intent(
    tool: str,
    category: ApprovalCategory,
    details: dict[str, JSONValue],
) -> ActionIntent:
    prompt = _build_prompt(category, tool, details)
    summary = _build_summary(tool, details)
    return ActionIntent(
        category=category,
        tool=tool,
        summary=summary,
        details=details,
        prompt=prompt,
    )


def _build_prompt(
    category: ApprovalCategory,
    tool: str,
    details: dict[str, JSONValue],
) -> ApprovalPrompt:
    what = _build_what(tool, details)
    why = "Для выполнения запроса пользователя."
    risk = _RISK_TEXT.get(category, "Есть риск.")
    changes = _build_changes(tool, details)
    return ApprovalPrompt(what=what, why=why, risk=risk, changes=changes)


def _build_what(tool: str, details: dict[str, JSONValue]) -> str:
    if tool in {"shell", "workspace_run", "workspace_terminal_run"}:
        command = str(details.get("command") or "")
        return f"Выполнить команду: {command}".strip()
    path = str(details.get("path") or "")
    if path:
        return f"Изменить файл: {path}"
    return f"Выполнить действие через инструмент {tool}"


def _build_changes(tool: str, details: dict[str, JSONValue]) -> list[str]:
    changes: list[str] = []
    if tool in {"shell", "workspace_run", "workspace_terminal_run"}:
        command = str(details.get("command") or "")
        if command:
            changes.append(f"Команда: {command}")
    path = str(details.get("path") or "")
    if path:
        changes.append(f"Путь: {path}")
    op = str(details.get("op") or "")
    if op:
        changes.append(f"Операция: {op}")
    if not changes:
        changes.append(f"Инструмент: {tool}")
    return changes[:3]


def _build_summary(tool: str, details: dict[str, JSONValue]) -> str:
    if tool in {"shell", "workspace_run", "workspace_terminal_run"}:
        command = str(details.get("command") or "")
        return f"{tool}:{command}".strip()
    path = str(details.get("path") or "")
    if path:
        return f"{tool}:{path}"
    return tool


def _is_config_or_secret(path: str) -> bool:
    normalized = path.lower()
    return any(hint in normalized for hint in _CONFIG_HINTS)


def _is_outside_workspace(path: str, *, workspace_relative: bool) -> bool:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return True
    if normalized.startswith(("~", "/", "../")):
        return True
    if _WINDOWS_ABS_RE.search(normalized):
        return True
    parts = [part for part in normalized.split("/") if part]
    if any(part == ".." for part in parts):
        return True
    if workspace_relative:
        return False
    if not parts:
        return True
    return parts[0] != "project"


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _unique_categories(intents: list[ActionIntent]) -> list[ApprovalCategory]:
    categories: list[ApprovalCategory] = []
    for intent in intents:
        if intent.category not in categories:
            categories.append(intent.category)
    return categories
