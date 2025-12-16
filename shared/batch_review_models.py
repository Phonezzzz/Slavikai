from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from shared.policy_models import PolicyAction, PolicyTrigger


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class Signal(str, Enum):
    DIRECT_NEGATIVE_FEEDBACK = "direct_negative_feedback"
    EXPLICIT_POSITIVE = "explicit_positive"
    RATING_OK = "rating_ok"

    LABEL_TOO_LONG = "label_too_long"
    LABEL_OFF_TOPIC = "label_off_topic"
    LABEL_NO_SOURCES = "label_no_sources"
    LABEL_HALLUCINATION = "label_hallucination"
    LABEL_TOO_COMPLEX = "label_too_complex"
    LABEL_INCORRECT = "label_incorrect"
    LABEL_OTHER = "label_other"

    STYLE_PROFANITY_ONLY = "style_profanity_only"


class IntentKind(str, Enum):
    QUESTION = "question"
    REQUEST_ACTION = "request_action"
    REPORT_PROBLEM = "report_problem"
    PRAISE = "praise"
    CLARIFICATION = "clarification"
    OTHER = "other"


class ParadoxFlag(str, Enum):
    STYLE_CONFLICT = "style_conflict"
    ALWAYS_NEVER_CONFLICT = "always_never_conflict"
    DO_DONT_CONFLICT = "do_dont_conflict"


@dataclass(frozen=True)
class IntentHypothesis:
    intent: IntentKind
    score: float


@dataclass(frozen=True)
class EvidenceItem:
    interaction_id: str
    excerpt: str
    feedback_id: str | None = None


@dataclass(frozen=True)
class PolicyRuleCandidate:
    candidate_id: str
    batch_review_run_id: str
    user_id: str
    proposed_trigger: PolicyTrigger
    proposed_action: PolicyAction
    priority_suggestion: int
    confidence_suggestion: float
    created_at: str
    updated_at: str
    evidence: list[EvidenceItem] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    intent_hypotheses: list[IntentHypothesis] = field(default_factory=list)
    paradox_flags: list[ParadoxFlag] = field(default_factory=list)
    status: CandidateStatus = CandidateStatus.PROPOSED


@dataclass(frozen=True)
class BatchReviewRun:
    batch_review_run_id: str
    user_id: str
    period_start: str
    period_end: str
    created_at: str
    interaction_count: int
    feedback_count: int
    candidate_count: int
    report_text: str
