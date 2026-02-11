from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from shared.models import JSONValue


class ClaimType(StrEnum):
    PREFERENCE = "preference"
    POLICY = "policy"
    FACT = "fact"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    ENVIRONMENT = "environment"


class AtomStatus(StrEnum):
    ACTIVE = "active"
    CONFLICT = "conflict"
    DEPRECATED = "deprecated"


@dataclass(frozen=True)
class Claim:
    claim_type: ClaimType
    stable_key: str
    value_json: JSONValue
    confidence: float
    summary_text: str
    is_explicit: bool
    source_kind: str
    source_id: str
    created_at: str

    def __post_init__(self) -> None:
        if not self.stable_key.strip():
            raise ValueError("Claim.stable_key должен быть непустым")
        if not self.summary_text.strip():
            raise ValueError("Claim.summary_text должен быть непустым")
        if not self.source_kind.strip():
            raise ValueError("Claim.source_kind должен быть непустым")
        if not self.source_id.strip():
            raise ValueError("Claim.source_id должен быть непустым")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Claim.confidence должен быть в диапазоне 0..1")
        _validate_utc_iso(self.created_at)


@dataclass(frozen=True)
class CanonicalAtom:
    atom_id: str
    stable_key: str
    claim_type: ClaimType
    value_json: JSONValue
    confidence: float
    support_count: int
    contradict_count: int
    last_seen_at: str
    status: AtomStatus
    summary_text: str

    def __post_init__(self) -> None:
        if not self.atom_id.strip():
            raise ValueError("CanonicalAtom.atom_id должен быть непустым")
        if not self.stable_key.strip():
            raise ValueError("CanonicalAtom.stable_key должен быть непустым")
        if not self.summary_text.strip():
            raise ValueError("CanonicalAtom.summary_text должен быть непустым")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("CanonicalAtom.confidence должен быть в диапазоне 0..1")
        if self.support_count < 0:
            raise ValueError("CanonicalAtom.support_count должен быть >= 0")
        if self.contradict_count < 0:
            raise ValueError("CanonicalAtom.contradict_count должен быть >= 0")
        _validate_utc_iso(self.last_seen_at)


@dataclass(frozen=True)
class ClaimExtractionInput:
    text: str
    source_kind: str
    source_id: str
    lang_hint: str | None
    created_at: str

    def __post_init__(self) -> None:
        if not self.source_kind.strip():
            raise ValueError("ClaimExtractionInput.source_kind должен быть непустым")
        if not self.source_id.strip():
            raise ValueError("ClaimExtractionInput.source_id должен быть непустым")
        _validate_utc_iso(self.created_at)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _validate_utc_iso(value: str) -> None:
    if not value.strip():
        raise ValueError("timestamp должен быть непустой ISO-строкой")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("timestamp должен быть timezone-aware (UTC ISO)")
