from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from shared.models import JSONValue


class InteractionKind(StrEnum):
    CHAT = "chat"
    TOOL = "tool"


class InteractionMode(StrEnum):
    STANDARD = "standard"
    MEMORY_COMPANION = "memory_companion"


class ToolStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    BLOCKED = "blocked"


class BlockedReason(StrEnum):
    APPROVAL_REQUIRED = "approval_required"
    SAFE_MODE_BLOCKED = "safe_mode_blocked"
    SANDBOX_VIOLATION = "sandbox_violation"
    TOOL_DISABLED = "tool_disabled"
    TOOL_NOT_REGISTERED = "tool_not_registered"
    VALIDATION_ERROR = "validation_error"


class FeedbackRating(StrEnum):
    GOOD = "good"
    OK = "ok"
    BAD = "bad"


class FeedbackLabel(StrEnum):
    TOO_LONG = "too_long"
    OFF_TOPIC = "off_topic"
    NO_SOURCES = "no_sources"
    HALLUCINATION = "hallucination"
    TOO_COMPLEX = "too_complex"
    INCORRECT = "incorrect"
    OTHER = "other"


@dataclass(frozen=True)
class ChatInteractionLog:
    interaction_id: str
    user_id: str
    interaction_kind: InteractionKind
    raw_input: str
    mode: InteractionMode
    created_at: str
    response_text: str
    retrieved_memory_ids: list[str] = field(default_factory=list)
    applied_policy_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolInteractionLog:
    interaction_id: str
    user_id: str
    interaction_kind: InteractionKind
    raw_input: str
    mode: InteractionMode
    created_at: str
    tool_name: str
    tool_args: dict[str, JSONValue]
    tool_status: ToolStatus
    blocked_reason: BlockedReason | None = None
    tool_output_preview: str | None = None
    tool_error: str | None = None
    tool_meta: dict[str, JSONValue] | None = None


InteractionLog = ChatInteractionLog | ToolInteractionLog


@dataclass(frozen=True)
class FeedbackEvent:
    feedback_id: str
    interaction_id: str
    user_id: str
    rating: FeedbackRating
    created_at: str
    labels: list[FeedbackLabel] = field(default_factory=list)
    free_text: str | None = None
