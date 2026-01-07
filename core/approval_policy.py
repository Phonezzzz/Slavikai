from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

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


@dataclass(frozen=True)
class ApprovalDecision:
    status: ApprovalDecisionStatus
    reason: str
    intents: list[ActionIntent]
    required_categories: list[ApprovalCategory]


@dataclass(frozen=True)
class ApprovalRequest:
    category: ApprovalCategory
    required_categories: list[ApprovalCategory]
    prompt: ApprovalPrompt
    tool: str
    details: dict[str, JSONValue]
    session_id: str | None


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
    if not context.safe_mode:
        return ApprovalDecision("allow", "safe_mode_disabled", intents, [])
    required = [
        intent
        for intent in intents
        if intent.category not in context.approved_categories
    ]
    if required:
        categories = _unique_categories(required)
        return ApprovalDecision(
            "require_approval", "category_not_approved", required, categories
        )
    return ApprovalDecision("allow", "approved", intents, [])


def build_approval_request(
    *,
    context: ApprovalContext,
    decision: ApprovalDecision,
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
    )


def detect_action_intents(request: ToolRequest) -> list[ActionIntent]:
    tool = request.name
    args = request.args
    intents: list[ActionIntent] = []

    if tool in {"shell", "workspace_run"}:
        command = str(args.get("command") or args.get("path") or "")
        if command.strip():
            intents.extend(_shell_intents(tool, command))
        return intents

    if tool == "web":
        intents.append(_intent(tool, "NETWORK_RISK", {"tool": tool}))
        return intents

    if tool in {"workspace_write", "workspace_patch"}:
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
        return intents

    if tool == "fs":
        op = str(args.get("op") or "").lower().strip()
        path = str(args.get("path") or "")
        intents.extend(_fs_intents(tool, op, path))
        return intents

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
    if tool in {"shell", "workspace_run"}:
        command = str(details.get("command") or "")
        return f"Выполнить команду: {command}".strip()
    path = str(details.get("path") or "")
    if path:
        return f"Изменить файл: {path}"
    return f"Выполнить действие через инструмент {tool}"


def _build_changes(tool: str, details: dict[str, JSONValue]) -> list[str]:
    changes: list[str] = []
    if tool in {"shell", "workspace_run"}:
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
    if tool in {"shell", "workspace_run"}:
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
