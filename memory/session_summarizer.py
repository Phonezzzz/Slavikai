from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from llm.brain_base import Brain
from shared.canonical_atom_models import Claim, ClaimType, utc_now_iso
from shared.models import LLMMessage

SESSION_SUMMARY_PROMPT = """Ты кратко резюмируешь рабочую сессию SlavikAI.

На входе последние реплики пользователя и ассистента.
Сделай 5-8 коротких пунктов на русском:
- что обсуждали,
- какие решения приняли,
- какие открытые вопросы остались,
- упомянутые файлы, команды и инструменты.

Не добавляй новых фактов. Верни только резюме."""

MAX_SUMMARY_MESSAGES = 40
MAX_TRANSCRIPT_MESSAGE_CHARS = 500


@dataclass(frozen=True)
class SessionSummaryResult:
    stable_key: str
    text: str
    claim: Claim


class SessionSummarizer:
    def __init__(self, brain: Brain) -> None:
        self.brain = brain

    def summarize(self, messages: list[LLMMessage]) -> SessionSummaryResult | None:
        transcript_messages = [
            message
            for message in messages
            if message.role in {"user", "assistant"} and message.content.strip()
        ]
        if not transcript_messages:
            return None

        transcript = "\n".join(
            f"{message.role}: {message.content[:MAX_TRANSCRIPT_MESSAGE_CHARS]}"
            for message in transcript_messages[-MAX_SUMMARY_MESSAGES:]
        )
        result = self.brain.generate(
            [
                LLMMessage(role="system", content=SESSION_SUMMARY_PROMPT),
                LLMMessage(role="user", content=transcript),
            ]
        )
        summary_text = result.text.strip()
        if not summary_text:
            return None

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
        stable_key = f"session:{timestamp}"
        claim = Claim(
            claim_type=ClaimType.FACT,
            stable_key=stable_key,
            value_json={"text": summary_text},
            confidence=0.9,
            summary_text=f"session summary {timestamp}",
            is_explicit=True,
            source_kind="session.summary",
            source_id=stable_key,
            created_at=utc_now_iso(),
        )
        return SessionSummaryResult(stable_key=stable_key, text=summary_text, claim=claim)
