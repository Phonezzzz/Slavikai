from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from llm.brain_base import Brain
from llm.stream_model import Done, Error, TextDelta
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

    def summarize(
        self,
        messages: list[LLMMessage],
        *,
        cancellation_token: asyncio.Event | None = None,
    ) -> SessionSummaryResult | None:
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
        summary_messages = [
            LLMMessage(role="system", content=SESSION_SUMMARY_PROMPT),
            LLMMessage(role="user", content=transcript),
        ]
        if cancellation_token is None:
            summary_text = self.brain.generate(summary_messages).text.strip()
        else:
            chunks: list[str] = []
            finished = False
            for event in self.brain.generate_stream(
                summary_messages, cancellation_token=cancellation_token
            ):
                if isinstance(event, TextDelta):
                    if event.mode == "replace":
                        chunks = [event.text]
                    else:
                        chunks.append(event.text)
                elif isinstance(event, Error):
                    raise RuntimeError(event.message)
                elif isinstance(event, Done):
                    if event.finish_reason == "cancelled":
                        return None
                    finished = True
            if not finished:
                raise RuntimeError("Session summary stream ended without Done")
            summary_text = "".join(chunks).strip()
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
