from __future__ import annotations

from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.session_summarizer import SessionSummarizer
from shared.canonical_atom_models import ClaimType
from shared.models import LLMMessage


class CapturingBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[LLMMessage] = []

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        del config
        self.messages = messages
        return LLMResult(text=self.text)


def test_session_summarizer_creates_explicit_claim() -> None:
    brain = CapturingBrain("- обсудили память")
    summarizer = SessionSummarizer(brain)

    result = summarizer.summarize(
        [
            LLMMessage(role="system", content="ignored"),
            LLMMessage(role="user", content="надо сохранить итоги"),
            LLMMessage(role="assistant", content="сделаем /end-session"),
        ]
    )

    assert result is not None
    assert result.stable_key.startswith("session:")
    assert result.text == "- обсудили память"
    assert result.claim.claim_type is ClaimType.FACT
    assert result.claim.stable_key == result.stable_key
    assert result.claim.value_json == {"text": "- обсудили память"}
    assert result.claim.is_explicit is True
    assert result.claim.source_kind == "session.summary"
    assert brain.messages[0].role == "system"
    assert "надо сохранить итоги" in brain.messages[1].content


def test_session_summarizer_skips_empty_session() -> None:
    brain = CapturingBrain("- empty")
    summarizer = SessionSummarizer(brain)

    assert summarizer.summarize([LLMMessage(role="system", content="ignored")]) is None
    assert brain.messages == []


def test_session_summarizer_skips_empty_model_response() -> None:
    brain = CapturingBrain("   ")
    summarizer = SessionSummarizer(brain)

    result = summarizer.summarize([LLMMessage(role="user", content="hello")])

    assert result is None
