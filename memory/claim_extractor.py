from __future__ import annotations

import re
from dataclasses import dataclass

from shared.canonical_atom_models import Claim, ClaimExtractionInput, ClaimType
from shared.models import JSONValue

_EXPLICIT_PREFIX = re.compile(r"^\s*(запомни|remember|/remember)\b[:\-\s]*", re.IGNORECASE)
_PREFERENCE_RU = re.compile(r"\b(?:я\s+предпочитаю|предпочитаю)\s+(?P<value>.+)", re.IGNORECASE)
_PREFERENCE_EN = re.compile(r"\b(?:i\s+prefer)\s+(?P<value>.+)", re.IGNORECASE)
_POLICY_RU = re.compile(
    r"\b(?:не\s+делай|избегай|всегда\s+делай|правило[:\s])\s*(?P<rule>.+)",
    re.IGNORECASE,
)
_POLICY_EN = re.compile(r"\b(?:do\s+not|don't|always|rule[:\s])\s*(?P<rule>.+)", re.IGNORECASE)

_NON_WORD = re.compile(r"[^a-z0-9]+")
_MULTI_SPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ExtractorConfig:
    enable_llm_enrichment: bool = False


class ClaimExtractor:
    def __init__(self, config: ExtractorConfig | None = None) -> None:
        self._config = config or ExtractorConfig()

    def extract(self, payload: ClaimExtractionInput) -> list[Claim]:
        _ = self._config  # deterministic-only by default; LLM enrichment stays behind feature flag.
        text = payload.text.strip()
        if not text:
            return []

        is_explicit = False
        explicit_match = _EXPLICIT_PREFIX.match(text)
        if explicit_match is not None:
            is_explicit = True
            text = text[explicit_match.end() :].strip()
            if not text:
                return []

        claims: list[Claim] = []

        preference = _extract_preference(text)
        if preference is not None:
            stable_key = _normalize_stable_key(ClaimType.PREFERENCE, preference.key_hint)
            claims.append(
                _build_claim(
                    claim_type=ClaimType.PREFERENCE,
                    stable_key=stable_key,
                    value_json={"value": preference.value, "raw": text},
                    confidence=0.92 if is_explicit else 0.72,
                    summary_text=f"preference:{stable_key}={preference.value}",
                    is_explicit=is_explicit,
                    payload=payload,
                )
            )

        policy = _extract_policy(text)
        if policy is not None:
            stable_key = _normalize_stable_key(ClaimType.POLICY, policy.key_hint)
            claims.append(
                _build_claim(
                    claim_type=ClaimType.POLICY,
                    stable_key=stable_key,
                    value_json={"rule": policy.value, "raw": text},
                    confidence=0.95 if is_explicit else 0.7,
                    summary_text=f"policy:{stable_key}={policy.value}",
                    is_explicit=is_explicit,
                    payload=payload,
                )
            )

        # fallback explicit capture: deterministic fact claim
        if is_explicit and not claims:
            normalized = _normalize_text(text)
            stable_key = _normalize_stable_key(ClaimType.FACT, normalized[:80])
            claims.append(
                _build_claim(
                    claim_type=ClaimType.FACT,
                    stable_key=stable_key,
                    value_json={"text": text},
                    confidence=0.65,
                    summary_text=f"fact:{stable_key}",
                    is_explicit=True,
                    payload=payload,
                )
            )

        return _dedupe_claims(claims)


@dataclass(frozen=True)
class _ExtractedValue:
    key_hint: str
    value: str


def _extract_preference(text: str) -> _ExtractedValue | None:
    match = _PREFERENCE_RU.search(text)
    if match is None:
        match = _PREFERENCE_EN.search(text)
    if match is None:
        return None
    value = _normalize_text(match.group("value"))
    if not value:
        return None
    key_hint = _preference_key_hint(value)
    return _ExtractedValue(key_hint=key_hint, value=value)


def _extract_policy(text: str) -> _ExtractedValue | None:
    match = _POLICY_RU.search(text)
    if match is None:
        match = _POLICY_EN.search(text)
    if match is None:
        return None
    value = _normalize_text(match.group("rule"))
    if not value:
        return None
    key_hint = _policy_key_hint(value)
    return _ExtractedValue(key_hint=key_hint, value=value)


def _preference_key_hint(value: str) -> str:
    lower = value.lower()
    if "корот" in lower or "concise" in lower or "short" in lower:
        return "response_length"
    if "рус" in lower or "russian" in lower:
        return "response_language"
    if "англ" in lower or "english" in lower:
        return "response_language"
    if "markdown" in lower:
        return "response_format"
    return value


def _policy_key_hint(value: str) -> str:
    lower = value.lower()
    if "emoji" in lower or "эмод" in lower:
        return "avoid_emoji"
    if "источник" in lower or "sources" in lower:
        return "include_sources"
    if "safe" in lower or "безопас" in lower:
        return "safety_guardrails"
    return value


def _normalize_stable_key(claim_type: ClaimType, raw_key: str) -> str:
    key = _normalize_text(raw_key).lower()
    ascii_key = key.encode("ascii", errors="ignore").decode("ascii")
    if not ascii_key:
        ascii_key = "general"
    slug = _NON_WORD.sub("_", ascii_key).strip("_")
    slug = _MULTI_SPACE.sub("_", slug)
    if not slug:
        slug = "general"
    slug = slug[:80]
    return f"{claim_type.value}:{slug}"


def _build_claim(
    *,
    claim_type: ClaimType,
    stable_key: str,
    value_json: JSONValue,
    confidence: float,
    summary_text: str,
    is_explicit: bool,
    payload: ClaimExtractionInput,
) -> Claim:
    return Claim(
        claim_type=claim_type,
        stable_key=stable_key,
        value_json=value_json,
        confidence=confidence,
        summary_text=summary_text,
        is_explicit=is_explicit,
        source_kind=payload.source_kind,
        source_id=payload.source_id,
        created_at=payload.created_at,
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _dedupe_claims(claims: list[Claim]) -> list[Claim]:
    deduped: dict[str, Claim] = {}
    for claim in claims:
        deduped[claim.stable_key] = claim
    return list(deduped.values())
