from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from shared.models import JSONValue

DecisionRisk = Literal["low", "medium", "high"]


class DecisionReason(str, Enum):
    BLOCKED = "blocked"
    RISK = "risk"
    NEED_USER_INPUT = "need_user_input"
    TOOL_FAIL = "tool_fail"
    AMBIGUOUS_SKILL = "ambiguous_skill"
    VERIFIER_FAIL = "verifier_fail"


class DecisionAction(str, Enum):
    ASK_USER = "ask_user"
    PROCEED_SAFE = "proceed_safe"
    RETRY = "retry"
    ADJUST_THRESHOLD = "adjust_threshold"
    CREATE_CANDIDATE = "create_candidate"
    SELECT_SKILL = "select_skill"
    ABORT = "abort"


@dataclass(frozen=True)
class DecisionOption:
    id: str
    title: str
    action: DecisionAction
    payload: dict[str, JSONValue] = field(default_factory=dict)
    risk: DecisionRisk = "low"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DecisionOption.id должен быть непустым")
        if not self.title:
            raise ValueError("DecisionOption.title должен быть непустым")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "title": self.title,
            "action": self.action.value,
            "payload": self.payload,
            "risk": self.risk,
        }


@dataclass(frozen=True)
class DecisionPacket:
    id: str
    created_at: datetime
    reason: DecisionReason
    summary: str
    context: dict[str, JSONValue] = field(default_factory=dict)
    options: list[DecisionOption] = field(default_factory=list)
    default_option_id: str | None = None
    ttl_seconds: int = 600
    policy: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("DecisionPacket.id должен быть непустым")
        if not self.summary.strip():
            raise ValueError("DecisionPacket.summary должен быть непустым")
        if not 3 <= len(self.options) <= 5:
            raise ValueError("DecisionPacket.options должен содержать 3-5 вариантов")
        if self.ttl_seconds <= 0:
            raise ValueError("DecisionPacket.ttl_seconds должен быть положительным")
        option_ids = [option.id for option in self.options]
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("DecisionPacket.options содержит повторяющиеся id")
        if self.default_option_id and self.default_option_id not in option_ids:
            raise ValueError("DecisionPacket.default_option_id отсутствует в options")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "reason": self.reason.value,
            "summary": self.summary,
            "context": self.context,
            "options": [option.to_dict() for option in self.options],
            "default_option_id": self.default_option_id,
            "ttl_seconds": self.ttl_seconds,
            "policy": self.policy,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
