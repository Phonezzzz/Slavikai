from __future__ import annotations

from pathlib import Path
from typing import Any

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.memory_companion_models import (
    ChatInteractionLog,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionMode,
)
from shared.models import LLMMessage


class CounterBrain(Brain):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text="ok")


class FakeVectors:
    def search(self, query: str, namespace: str = "default", top_k: int = 5) -> list[Any]:
        return []


class FakeMemory:
    def get_recent(self, limit: int, kind=None):
        return []

    def get_user_prefs(self):
        return []


def test_context_uses_hints_meta(tmp_path: Path) -> None:
    main = CounterBrain()
    agent = Agent(brain=main, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.vectors = FakeVectors()
    agent.memory = FakeMemory()
    agent._interaction_store.log_interaction(  # noqa: SLF001
        ChatInteractionLog(
            interaction_id="1",
            user_id=agent.user_id,
            interaction_kind=InteractionKind.CHAT,
            raw_input="hi",
            mode=InteractionMode.STANDARD,
            created_at="2024-01-01 00:00:00",
            response_text="ok",
        )
    )
    agent.record_feedback_event(
        interaction_id="1",
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.HALLUCINATION],
    )
    ctx = agent._build_context_messages([LLMMessage(role="user", content="hi")], "hi")  # noqa: SLF001
    assert any(msg.role == "system" for msg in ctx)
    assert agent.last_hints_used
    assert agent.last_hints_meta and agent.last_hints_meta[0]["severity"] == "major"
