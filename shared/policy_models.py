from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from shared.models import JSONValue


class ResponseVerbosity(str, Enum):
    CONCISE = "concise"
    NORMAL = "normal"
    DETAILED = "detailed"


class PolicyScope(str, Enum):
    GLOBAL = "global"
    USER = "user"


@dataclass(frozen=True)
class TriggerAlways:
    kind: Literal["always"] = "always"


@dataclass(frozen=True)
class TriggerUserMessageContains:
    kind: Literal["user_message_contains"] = "user_message_contains"
    substrings: list[str] = field(default_factory=list)
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        cleaned = [s.strip() for s in self.substrings if s.strip()]
        if not cleaned:
            raise ValueError(
                "TriggerUserMessageContains.substrings должен содержать хотя бы одну строку."
            )


@dataclass(frozen=True)
class TriggerUserMessageRegex:
    kind: Literal["user_message_regex"] = "user_message_regex"
    pattern: str = ""
    case_insensitive: bool = True

    def __post_init__(self) -> None:
        pattern = self.pattern.strip()
        if not pattern:
            raise ValueError("TriggerUserMessageRegex.pattern не должен быть пустым.")
        flags = re.IGNORECASE if self.case_insensitive else 0
        re.compile(pattern, flags=flags)


PolicyTrigger = TriggerAlways | TriggerUserMessageContains | TriggerUserMessageRegex


@dataclass(frozen=True)
class ActionAddInstruction:
    kind: Literal["add_instruction"] = "add_instruction"
    text: str = ""

    def __post_init__(self) -> None:
        if len(self.text.strip()) < 3:
            raise ValueError("ActionAddInstruction.text слишком короткий.")


@dataclass(frozen=True)
class ActionSetResponseStyle:
    kind: Literal["set_response_style"] = "set_response_style"
    verbosity: ResponseVerbosity = ResponseVerbosity.NORMAL


PolicyAction = ActionAddInstruction | ActionSetResponseStyle


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    user_id: str
    scope: PolicyScope
    trigger: PolicyTrigger
    action: PolicyAction
    priority: int
    confidence: float
    decay_half_life_days: int
    provenance: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("PolicyRule.rule_id не должен быть пустым.")
        if not self.user_id.strip():
            raise ValueError("PolicyRule.user_id не должен быть пустым.")
        if not self.provenance.strip():
            raise ValueError("PolicyRule.provenance обязателен.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("PolicyRule.confidence должен быть в диапазоне 0..1.")
        if self.decay_half_life_days <= 0:
            raise ValueError("PolicyRule.decay_half_life_days должен быть > 0.")
        if not self.created_at.strip():
            raise ValueError("PolicyRule.created_at обязателен.")
        if not self.updated_at.strip():
            raise ValueError("PolicyRule.updated_at обязателен.")


def policy_trigger_to_dict(trigger: PolicyTrigger) -> dict[str, JSONValue]:
    if isinstance(trigger, TriggerAlways):
        return {"kind": trigger.kind}
    if isinstance(trigger, TriggerUserMessageContains):
        return {
            "kind": trigger.kind,
            "substrings": [s.strip() for s in trigger.substrings if s.strip()],
            "case_sensitive": trigger.case_sensitive,
        }
    if isinstance(trigger, TriggerUserMessageRegex):
        return {
            "kind": trigger.kind,
            "pattern": trigger.pattern,
            "case_insensitive": trigger.case_insensitive,
        }
    raise TypeError(f"Unsupported PolicyTrigger: {type(trigger)}")


def policy_action_to_dict(action: PolicyAction) -> dict[str, JSONValue]:
    if isinstance(action, ActionAddInstruction):
        return {"kind": action.kind, "text": action.text}
    if isinstance(action, ActionSetResponseStyle):
        return {"kind": action.kind, "verbosity": action.verbosity.value}
    raise TypeError(f"Unsupported PolicyAction: {type(action)}")


def policy_trigger_from_dict(data: dict[str, object]) -> PolicyTrigger:
    kind = data.get("kind")
    if kind == "always":
        return TriggerAlways()
    if kind == "user_message_contains":
        substrings_raw = data.get("substrings")
        substrings = [str(s) for s in substrings_raw] if isinstance(substrings_raw, list) else []
        case_sensitive_raw = data.get("case_sensitive", False)
        if not isinstance(case_sensitive_raw, bool):
            raise ValueError("case_sensitive должен быть bool.")
        return TriggerUserMessageContains(substrings=substrings, case_sensitive=case_sensitive_raw)
    if kind == "user_message_regex":
        pattern_raw = data.get("pattern")
        if not isinstance(pattern_raw, str):
            raise ValueError("pattern должен быть строкой.")
        case_insensitive_raw = data.get("case_insensitive", True)
        if not isinstance(case_insensitive_raw, bool):
            raise ValueError("case_insensitive должен быть bool.")
        return TriggerUserMessageRegex(pattern=pattern_raw, case_insensitive=case_insensitive_raw)
    raise ValueError(f"Неизвестный trigger.kind: {kind!r}")


def policy_action_from_dict(data: dict[str, object]) -> PolicyAction:
    kind = data.get("kind")
    if kind == "add_instruction":
        text_raw = data.get("text")
        if not isinstance(text_raw, str):
            raise ValueError("text должен быть строкой.")
        return ActionAddInstruction(text=text_raw)
    if kind == "set_response_style":
        verbosity_raw = data.get("verbosity", ResponseVerbosity.NORMAL.value)
        if not isinstance(verbosity_raw, str):
            raise ValueError("verbosity должен быть строкой.")
        return ActionSetResponseStyle(verbosity=ResponseVerbosity(verbosity_raw))
    raise ValueError(f"Неизвестный action.kind: {kind!r}")


def policy_trigger_to_json(trigger: PolicyTrigger) -> str:
    return json.dumps(policy_trigger_to_dict(trigger), ensure_ascii=False, sort_keys=True)


def policy_action_to_json(action: PolicyAction) -> str:
    return json.dumps(policy_action_to_dict(action), ensure_ascii=False, sort_keys=True)


def policy_trigger_from_json(text: str) -> PolicyTrigger:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Policy trigger JSON должен быть объектом.")
    return policy_trigger_from_dict(parsed)


def policy_action_from_json(text: str) -> PolicyAction:
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Policy action JSON должен быть объектом.")
    return policy_action_from_dict(parsed)
