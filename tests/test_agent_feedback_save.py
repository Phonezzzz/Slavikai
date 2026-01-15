from __future__ import annotations

from shared.memory_companion_models import (
    ChatInteractionLog,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionMode,
)


def test_save_feedback_variants(tmp_path) -> None:
    from core.agent import Agent
    from llm.brain_base import Brain
    from llm.types import LLMResult, ModelConfig

    class BrainStub(Brain):
        def generate(self, messages, config: ModelConfig | None = None):  # type: ignore[override]
            return LLMResult(text="ok")

    agent = Agent(
        brain=BrainStub(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )

    for interaction_id in ("1", "2", "3"):
        agent._interaction_store.log_interaction(  # noqa: SLF001
            ChatInteractionLog(
                interaction_id=interaction_id,
                user_id=agent.user_id,
                interaction_kind=InteractionKind.CHAT,
                raw_input="prompt",
                mode=InteractionMode.STANDARD,
                created_at="2024-01-01 00:00:00",
                response_text="answer",
            )
        )

    agent.record_feedback_event(
        interaction_id="1",
        rating=FeedbackRating.GOOD,
        labels=[],
        free_text=None,
    )
    agent.record_feedback_event(
        interaction_id="2",
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.INCORRECT],
        free_text="fix",
    )
    agent.record_feedback_event(
        interaction_id="3",
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.OFF_TOPIC],
        free_text=None,
    )

    hints = agent._collect_feedback_hints(5)  # noqa: SLF001
    ratings = {hint["rating"] for hint in hints}
    assert "bad" in ratings
    assert any(hint["severity"] in {"major", "fatal"} for hint in hints)
    assert any("fix" in hint["hint"] for hint in hints)
