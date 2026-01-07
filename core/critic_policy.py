from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Literal

from shared.models import LLMMessage, TaskPlan

CriticMode = Literal["single", "dual", "critic-only"]


class CriticFailure(RuntimeError):
    """Ошибка критика, которую нельзя игнорировать в A+D режиме."""


@dataclass(frozen=True)
class CriticDecision:
    should_run_critic: bool
    reasons: list[str]


_TOOL_INTENT_MARKERS: Final[tuple[str, ...]] = (
    "прочитай",
    "открой",
    "покажи",
    "перечисли",
    "найди",
    "поиск",
    "search",
    "shell",
    "команд",
    "запусти",
    "run",
    "execute",
    "workspace",
    "project",
    "web",
    "tool",
    "инструмент",
    "ls",
    "cat",
    "grep",
    "rg",
)
_CODE_CHANGE_MARKERS: Final[tuple[str, ...]] = (
    "добав",
    "измен",
    "исправ",
    "перепис",
    "рефактор",
    "почини",
    "обнов",
    "rewrite",
    "refactor",
    "fix",
    "patch",
    "implement",
    "rename",
    "update",
)

_RISK_DELETE_MARKERS: Final[tuple[str, ...]] = (
    "удал",
    "delete",
    "remove",
    "wipe",
    "truncate",
    "перезапис",
    "overwrite",
)
_RISK_CONFIG_MARKERS: Final[tuple[str, ...]] = (
    "config",
    "конфиг",
    "secret",
    "секрет",
    "token",
    "api key",
    "apikey",
    "ключ",
    "пароль",
    "credentials",
    ".env",
    "dotenv",
    "ssh key",
    "private key",
)
_RISK_DEPS_MARKERS: Final[tuple[str, ...]] = (
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
    "dependencies",
    "зависим",
    "install deps",
    "install dependency",
)
_RISK_SYSTEM_MARKERS: Final[tuple[str, ...]] = (
    "systemctl",
    "service",
    "iptables",
    "ufw",
    "firewall",
    "mount",
    "umount",
    "mkfs",
    "fdisk",
    "reboot",
    "shutdown",
    "disk",
    "диск",
    "сервис",
    "служб",
    "сеть",
    "network",
    "netsh",
    "ifconfig",
    "route",
    "docker",
    "kubernetes",
    "kube",
)
_RISK_OUTSIDE_MARKERS: Final[tuple[str, ...]] = (
    "/etc/",
    "/var/",
    "/usr/",
    "/bin/",
    "/sbin/",
    "/opt/",
    "~/",
    "c:\\",
    "d:\\",
    "e:\\",
    "outside workspace",
    "вне проекта",
    "вне рабочей папки",
)
_RISK_GIT_MARKERS: Final[tuple[str, ...]] = (
    "git commit",
    "git push",
    "git tag",
    "git merge",
    "commit",
    "push",
    "publish",
    "release",
    "deploy",
)

_RISK_REGEX: Final[dict[str, tuple[str, ...]]] = {
    "delete": (r"\brm\b",),
    "sudo": (r"\bsudo\b",),
    "outside_workspace": (r"\.\./", r"[a-zA-Z]:\\\\"),
}

_BAD_PLAN_MARKERS: Final[tuple[str, ...]] = (
    "bad_plan",
    "bad plan",
    "отклон",
    "не выполнять",
    "reject",
)
_RISKY_MARKERS: Final[tuple[str, ...]] = (
    "risky",
    "risk",
    "опасн",
    "сомнител",
    "не уверен",
    "нужно подтверж",
)


def decide_critic(
    *,
    mode: CriticMode,
    messages: list[LLMMessage],
    plan: TaskPlan | None = None,
) -> CriticDecision:
    if mode == "critic-only":
        return CriticDecision(True, ["mode=critic-only"])
    if mode == "single":
        return CriticDecision(False, ["mode=single"])

    text = _collect_text(messages, plan)
    tool_intent = _detect_tool_intent(text, plan)
    code_change = _contains_any(text, _CODE_CHANGE_MARKERS)
    risk_flags = _detect_risk_flags(text)

    reasons: list[str] = ["mode=dual"]
    if tool_intent:
        reasons.append("task:tools")
    if code_change:
        reasons.append("task:code_change")
    for flag in risk_flags:
        reasons.append(f"risk:{flag}")

    should_run = tool_intent or code_change or bool(risk_flags)
    if not should_run:
        reasons.append("no_triggers")
    return CriticDecision(should_run, reasons)


def classify_critic_status(
    *,
    decision: CriticDecision,
    critic_text: str | None,
) -> str:
    if not decision.should_run_critic:
        return "disabled"
    if critic_text is None or not critic_text.strip():
        return "uncertain"
    lowered = critic_text.lower()
    if _contains_any(lowered, _BAD_PLAN_MARKERS):
        return "bad_plan"
    if any(reason.startswith("risk:") for reason in decision.reasons):
        return "risky"
    if _contains_any(lowered, _RISKY_MARKERS):
        return "risky"
    return "ok"


def _collect_text(messages: list[LLMMessage], plan: TaskPlan | None) -> str:
    parts = [msg.content for msg in messages if msg.role == "user"]
    if plan is not None:
        parts.append(plan.goal)
        parts.extend(step.description for step in plan.steps)
    return "\n".join(parts).lower()


def _detect_tool_intent(text: str, plan: TaskPlan | None) -> bool:
    if plan is not None and any(step.operation for step in plan.steps):
        return True
    return _contains_any(text, _TOOL_INTENT_MARKERS)


def _detect_risk_flags(text: str) -> list[str]:
    flags: list[str] = []
    if _contains_any(text, _RISK_DELETE_MARKERS) or _matches_any(
        text, _RISK_REGEX["delete"]
    ):
        flags.append("delete")
    if _contains_any(text, _RISK_CONFIG_MARKERS):
        flags.append("config")
    if _contains_any(text, _RISK_DEPS_MARKERS):
        flags.append("deps")
    if _contains_any(text, _RISK_SYSTEM_MARKERS):
        flags.append("system")
    if _contains_any(text, _RISK_OUTSIDE_MARKERS) or _matches_any(
        text, _RISK_REGEX["outside_workspace"]
    ):
        flags.append("outside_workspace")
    if _matches_any(text, _RISK_REGEX["sudo"]):
        flags.append("sudo")
    if _contains_any(text, _RISK_GIT_MARKERS):
        flags.append("git")
    return flags


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)
