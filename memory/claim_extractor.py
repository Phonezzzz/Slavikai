from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from llm.brain_base import Brain
from shared.canonical_atom_models import Claim, ClaimExtractionInput, ClaimType
from shared.models import JSONValue, LLMMessage

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
_LOGGER = logging.getLogger("SlavikAI.ClaimExtractor")


@dataclass(frozen=True)
class ExtractorConfig:
    enable_llm_enrichment: bool = False
    llm_max_claims_per_call: int = 6


class ClaimExtractor:
    def __init__(
        self,
        config: ExtractorConfig | None = None,
        brain: Brain | None = None,
    ) -> None:
        self._config = config or ExtractorConfig()
        self._brain = brain

    def extract(self, payload: ClaimExtractionInput) -> list[Claim]:
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

        if self._should_run_llm_enrichment(is_explicit):
            claims = _merge_claims(
                claims,
                self._llm_extract_claims(text=text, payload=payload),
            )

        return _dedupe_claims(claims)

    def _should_run_llm_enrichment(self, is_explicit: bool) -> bool:
        return bool(is_explicit and self._config.enable_llm_enrichment and self._brain is not None)

    def _llm_extract_claims(
        self,
        *,
        text: str,
        payload: ClaimExtractionInput,
    ) -> list[Claim]:
        if self._brain is None:
            return []
        prompt = (
            "Извлеки только явно запоминаемые факты из текста пользователя.\n"
            "Верни JSON-массив объектов. Схема объекта:\n"
            "{"
            '"claim_type":"PREFERENCE|POLICY|FACT|GOAL|CONSTRAINT|DECISION|ENVIRONMENT",'
            '"stable_key":"type:short_ascii_key",'
            '"value_json":{"text":"..."},'
            '"summary_text":"краткое описание",'
            '"confidence":0.0'
            "}.\n"
            "Если фактов нет, верни []. Не добавляй пояснений.\n\n"
            f"Текст:\n{text}"
        )
        try:
            result = self._brain.generate([LLMMessage(role="user", content=prompt)])
        except Exception as exc:  # noqa: BLE001
            _LOGGER.warning("llm_claim_extract_failed: %s", exc)
            return []
        return self._parse_llm_claims(result.text, payload=payload)[
            : self._config.llm_max_claims_per_call
        ]

    def _parse_llm_claims(
        self,
        raw: str,
        *,
        payload: ClaimExtractionInput,
    ) -> list[Claim]:
        try:
            parsed = json.loads(_strip_json_code_fence(raw))
        except json.JSONDecodeError as exc:
            _LOGGER.warning("llm_claim_json_invalid: %s", exc)
            return []
        if not isinstance(parsed, list):
            return []

        claims: list[Claim] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            claim = _claim_from_llm_item(item, payload=payload)
            if claim is not None:
                claims.append(claim)
        return claims


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


def _merge_claims(primary: list[Claim], enrichment: list[Claim]) -> list[Claim]:
    by_key = {claim.stable_key: claim for claim in primary}
    for claim in enrichment:
        if claim.stable_key in by_key:
            continue
        by_key[claim.stable_key] = claim
    return list(by_key.values())


def _strip_json_code_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _claim_from_llm_item(
    item: dict[object, object],
    *,
    payload: ClaimExtractionInput,
) -> Claim | None:
    raw_type = item.get("claim_type")
    if not isinstance(raw_type, str):
        return None
    try:
        claim_type = ClaimType(raw_type.strip().lower())
    except ValueError:
        return None

    raw_value_json = item.get("value_json")
    parsed_value_json = _json_value_or_none(raw_value_json)
    if parsed_value_json is not None:
        value_json = parsed_value_json
    elif isinstance(raw_value_json, str) and raw_value_json.strip():
        value_json = {"text": raw_value_json.strip()}
    else:
        return None

    raw_summary = item.get("summary_text")
    summary_text = raw_summary.strip() if isinstance(raw_summary, str) else ""
    if not summary_text:
        text_value = value_json.get("text") if isinstance(value_json, dict) else None
        summary_text = text_value if isinstance(text_value, str) and text_value.strip() else ""
    if not summary_text:
        return None

    raw_stable_key = item.get("stable_key")
    stable_key = raw_stable_key.strip() if isinstance(raw_stable_key, str) else ""
    if not stable_key:
        stable_key = _normalize_stable_key(claim_type, summary_text)
    elif not stable_key.startswith(f"{claim_type.value}:"):
        stable_key = _normalize_stable_key(claim_type, stable_key)

    confidence = _parse_llm_confidence(item.get("confidence"))
    return _build_claim(
        claim_type=claim_type,
        stable_key=stable_key,
        value_json=value_json,
        confidence=confidence,
        summary_text=summary_text,
        is_explicit=True,
        payload=payload,
    )


def _parse_llm_confidence(raw: object) -> float:
    if isinstance(raw, bool):
        return 0.75
    if isinstance(raw, int | float):
        return min(0.9, max(0.0, float(raw)))
    return 0.75


def _json_value_or_none(value: object) -> JSONValue | None:
    if value is None or isinstance(value, str | bytes | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        items: list[JSONValue] = []
        for item in value:
            parsed = _json_value_or_none(item)
            if parsed is None and item is not None:
                return None
            items.append(parsed)
        return items
    if isinstance(value, dict):
        obj: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return None
            parsed = _json_value_or_none(item)
            if parsed is None and item is not None:
                return None
            obj[key] = parsed
        return obj
    return None
