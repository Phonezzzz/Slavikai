from __future__ import annotations

from memory.claim_extractor import ClaimExtractor
from shared.canonical_atom_models import ClaimExtractionInput, ClaimType, utc_now_iso


def _payload(text: str) -> ClaimExtractionInput:
    return ClaimExtractionInput(
        text=text,
        source_kind="chat.user_input",
        source_id="session-1",
        lang_hint="ru",
        created_at=utc_now_iso(),
    )


def test_claim_extractor_explicit_ru_preference() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract(_payload("запомни: я предпочитаю короткие ответы"))

    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_type is ClaimType.PREFERENCE
    assert claim.is_explicit is True
    assert claim.stable_key == "preference:response_length"


def test_claim_extractor_en_policy() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract(_payload("Rule: do not use emoji in answers"))

    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_type is ClaimType.POLICY
    assert claim.stable_key == "policy:avoid_emoji"


def test_claim_extractor_ignores_noise() -> None:
    extractor = ClaimExtractor()
    claims = extractor.extract(_payload("hello there, just chatting"))
    assert claims == []


def test_claim_extractor_stable_key_is_deterministic() -> None:
    extractor = ClaimExtractor()
    first = extractor.extract(_payload("remember i prefer markdown output"))
    second = extractor.extract(_payload("remember i prefer markdown output"))

    assert first and second
    assert first[0].stable_key == second[0].stable_key
    assert first[0].stable_key == "preference:response_format"
