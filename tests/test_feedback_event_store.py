from __future__ import annotations

from pathlib import Path

import pytest

from memory.memory_companion_store import MemoryCompanionStore
from shared.memory_companion_models import (
    ChatInteractionLog,
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionMode,
)


def test_store_adds_and_reads_feedback_event(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    store = MemoryCompanionStore(db_path)

    store.log_interaction(
        ChatInteractionLog(
            interaction_id="chat-1",
            user_id="local",
            interaction_kind=InteractionKind.CHAT,
            raw_input="hello",
            mode=InteractionMode.STANDARD,
            created_at="2025-01-01 00:00:01",
            response_text="ok",
            retrieved_memory_ids=[],
            applied_policy_ids=[],
        )
    )

    store.add_feedback_event(
        FeedbackEvent(
            feedback_id="fb-1",
            interaction_id="chat-1",
            user_id="local",
            rating=FeedbackRating.GOOD,
            created_at="2025-01-01 00:00:02",
            labels=[FeedbackLabel.NO_SOURCES, FeedbackLabel.OTHER],
            free_text="nice",
        )
    )

    events = store.get_recent_feedback(user_id="local", limit=10)
    assert len(events) == 1
    ev = events[0]
    assert ev.feedback_id == "fb-1"
    assert ev.interaction_id == "chat-1"
    assert ev.rating == FeedbackRating.GOOD
    assert ev.labels == [FeedbackLabel.NO_SOURCES, FeedbackLabel.OTHER]
    assert ev.free_text == "nice"


def test_store_rejects_feedback_without_interaction(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    store = MemoryCompanionStore(db_path)

    with pytest.raises(ValueError, match="interaction_id"):
        store.add_feedback_event(
            FeedbackEvent(
                feedback_id="fb-1",
                interaction_id="missing",
                user_id="local",
                rating=FeedbackRating.BAD,
                created_at="2025-01-01 00:00:02",
                labels=[FeedbackLabel.INCORRECT],
                free_text=None,
            )
        )


def test_store_feedback_empty_when_silent(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    store = MemoryCompanionStore(db_path)

    store.log_interaction(
        ChatInteractionLog(
            interaction_id="chat-1",
            user_id="local",
            interaction_kind=InteractionKind.CHAT,
            raw_input="hello",
            mode=InteractionMode.STANDARD,
            created_at="2025-01-01 00:00:01",
            response_text="ok",
            retrieved_memory_ids=[],
            applied_policy_ids=[],
        )
    )

    assert store.get_recent_feedback(user_id="local", limit=10) == []
