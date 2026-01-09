from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from core.mwv.models import MWVMessage
from core.skills.index import SkillIndex, SkillMatchDecision
from shared.models import JSONValue, LLMMessage

type MessageLike = MWVMessage | LLMMessage


@dataclass(frozen=True)
class RouteDecision:
    route: Literal["chat", "mwv"]
    reason: str
    risk_flags: list[str]
    skill_decision: SkillMatchDecision | None = None


_CODE_CHANGE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\b(исправ\w*|почин\w*|поправ\w*|fix)\b.*\b(тест\w*|tests)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(исправ\w*|почин\w*|поправ\w*|fix)\b.*\b(баг\w*|bug|код\w*|code)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(рефактор\w*|refactor)\b", re.IGNORECASE),
    re.compile(r"\b(добав\w*|add)\b.*\b(фич\w*|feature)\b", re.IGNORECASE),
    re.compile(r"\b(напис\w*|write)\b.*\b(код\w*|code)\b", re.IGNORECASE),
    re.compile(r"\b(измен\w*|modify|change|update)\b.*\b(код\w*|code)\b", re.IGNORECASE),
)

_FILE_ACTION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\b(удал\w*|delete|remove)\b.*\b(файл\w*|file\w*|директ\w*|папк\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(созд\w*|create)\b.*\b(файл\w*|file\w*)\b", re.IGNORECASE),
    re.compile(
        r"\b(перезапиш\w*|overwrite|replace)\b.*\b(файл\w*|file\w*)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(запиш\w*|write)\b.*\b(файл\w*|file\w*)\b", re.IGNORECASE),
    re.compile(r"\b(примен\w*|apply)\b.*\b(патч\w*|patch\w*)\b", re.IGNORECASE),
)

_INSTALL_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(pip|pip3)\s+install\b", re.IGNORECASE),
    re.compile(r"\bnpm\s+install\b", re.IGNORECASE),
    re.compile(r"\b(apt(-get)?|brew)\s+install\b", re.IGNORECASE),
    re.compile(
        r"\b(установ\w*|обнов\w*)\b.*\b(зависимост\w*|deps|dependencies)\b",
        re.IGNORECASE,
    ),
)

_GIT_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(
        r"\bgit\s+(commit|push|pull|merge|rebase|checkout|reset|apply|diff)\b", re.IGNORECASE
    ),
    re.compile(r"\b(коммит|commit|пуш|push)\b", re.IGNORECASE),
    re.compile(r"\b(сделай|создай|make)\b.*\bpr\b", re.IGNORECASE),
)

_SUDO_PATTERNS: Sequence[re.Pattern[str]] = (re.compile(r"\bsudo\s+\S+", re.IGNORECASE),)

_SYSTEM_TOOL_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(systemctl|docker|docker-compose|kubectl)\s+\S+", re.IGNORECASE),
)

_SHELL_ACTION_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"\b(выполни|запусти|run|execute)\b.*\b(shell|терминал|команд)\b", re.IGNORECASE),
)


def classify_request(
    messages: Sequence[MessageLike],
    user_input: str,
    context: dict[str, JSONValue] | None = None,
    *,
    skill_index: SkillIndex | None = None,
) -> RouteDecision:
    _ = context
    text, used_fallback = _collect_text(messages, user_input)
    flags: list[str] = []
    skill_decision: SkillMatchDecision | None = None

    if skill_index is not None:
        skill_decision = skill_index.match_decision(text)
        if skill_decision.status == "matched":
            _add_flag(flags, "skill")
        elif skill_decision.status in {"deprecated", "ambiguous"}:
            _add_flag(flags, "skill")
            reason = _skill_reason(skill_decision)
            if used_fallback:
                reason = f"fallback_messages:{reason}"
            return RouteDecision(
                route="mwv",
                reason=reason,
                risk_flags=flags,
                skill_decision=skill_decision,
            )

    if _matches_any(text, _CODE_CHANGE_PATTERNS):
        _add_flag(flags, "code_change")

    if _matches_any(text, _FILE_ACTION_PATTERNS):
        _add_flag(flags, "filesystem")
        _add_flag(flags, "tools")

    if _matches_any(text, _INSTALL_PATTERNS):
        _add_flag(flags, "install")
        _add_flag(flags, "tools")

    if _matches_any(text, _GIT_PATTERNS):
        _add_flag(flags, "git")
        _add_flag(flags, "tools")

    if _matches_any(text, _SUDO_PATTERNS):
        _add_flag(flags, "sudo")
        _add_flag(flags, "tools")

    if _matches_any(text, _SYSTEM_TOOL_PATTERNS) or _matches_any(text, _SHELL_ACTION_PATTERNS):
        _add_flag(flags, "tools")

    if any(message.role == "tool" for message in messages):
        _add_flag(flags, "tools")

    if flags:
        reason = f"trigger:{','.join(flags)}"
        if used_fallback:
            reason = f"fallback_messages:{reason}"
        if skill_decision is not None:
            reason = _merge_reason(reason, skill_decision)
        return RouteDecision(
            route="mwv",
            reason=reason,
            risk_flags=flags,
            skill_decision=skill_decision,
        )

    reason = "no_triggers"
    if used_fallback:
        reason = "fallback_messages:no_triggers"
    if skill_decision is not None:
        reason = _merge_reason(reason, skill_decision)
    return RouteDecision(
        route="chat",
        reason=reason,
        risk_flags=flags,
        skill_decision=skill_decision,
    )


def _collect_text(messages: Sequence[MessageLike], user_input: str) -> tuple[str, bool]:
    stripped = user_input.strip()
    if stripped:
        return stripped.lower(), False
    parts = [message.content for message in messages if message.role == "user" and message.content]
    return " ".join(parts).lower(), True


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in patterns)


def _add_flag(flags: list[str], flag: str) -> None:
    if flag not in flags:
        flags.append(flag)


def _skill_reason(decision: SkillMatchDecision) -> str:
    if decision.status == "matched" and decision.match is not None:
        return f"skill_match:{decision.match.entry.id}"
    if decision.status == "deprecated" and decision.match is not None:
        base = f"skill_deprecated:{decision.match.entry.id}"
        if decision.replaced_by:
            return f"{base}->{decision.replaced_by}"
        return base
    if decision.status == "ambiguous":
        ids = ",".join(match.entry.id for match in decision.alternatives)
        return f"skill_ambiguous:{ids or 'unknown'}"
    return "skill_no_match"


def _merge_reason(base: str, decision: SkillMatchDecision) -> str:
    skill_reason = _skill_reason(decision)
    if base.startswith("trigger:"):
        return f"{skill_reason};{base}"
    return f"{skill_reason};{base}"
