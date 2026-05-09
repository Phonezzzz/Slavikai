from __future__ import annotations

from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.claim_extractor import ClaimExtractor, ExtractorConfig
from shared.canonical_atom_models import ClaimExtractionInput, ClaimType, utc_now_iso
from shared.models import LLMMessage


class JsonBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        del messages, config
        self.calls += 1
        return LLMResult(text=self.text)


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


def test_claim_extractor_llm_enriches_explicit_remember_only() -> None:
    brain = JsonBrain(
        """
        [
          {
            "claim_type": "ENVIRONMENT",
            "stable_key": "environment:editor",
            "value_json": {"value": "neovim"},
            "summary_text": "user editor is neovim",
            "confidence": 0.88
          }
        ]
        """
    )
    extractor = ClaimExtractor(
        ExtractorConfig(enable_llm_enrichment=True),
        brain=brain,
    )

    claims = extractor.extract(_payload("remember i prefer markdown output and use neovim"))

    stable_keys = {claim.stable_key for claim in claims}
    assert "preference:response_format" in stable_keys
    assert "environment:editor" in stable_keys
    assert brain.calls == 1


def test_claim_extractor_does_not_llm_enrich_non_explicit_text() -> None:
    brain = JsonBrain(
        '[{"claim_type":"ENVIRONMENT","stable_key":"environment:editor",'
        '"value_json":{"value":"neovim"},"summary_text":"editor","confidence":0.8}]'
    )
    extractor = ClaimExtractor(
        ExtractorConfig(enable_llm_enrichment=True),
        brain=brain,
    )

    claims = extractor.extract(_payload("i use neovim"))

    assert claims == []
    assert brain.calls == 0


def test_claim_extractor_invalid_llm_json_keeps_deterministic_fallback() -> None:
    brain = JsonBrain("not json")
    extractor = ClaimExtractor(
        ExtractorConfig(enable_llm_enrichment=True),
        brain=brain,
    )

    claims = extractor.extract(_payload("remember my laptop hostname is alpha"))

    assert [claim.stable_key for claim in claims] == ["fact:my_laptop_hostname_is_alpha"]
    assert brain.calls == 1
