from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from shared.batch_review_models import CandidateStatus
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import LLMMessage
from shared.policy_models import ActionSetResponseStyle, ResponseVerbosity


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def test_batch_review_is_manual_and_generates_candidates(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    inbox_db_path = db_path.with_name("memory_inbox.db")
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(db_path),
        memory_inbox_db_path=str(inbox_db_path),
    )
    agent._build_context_messages = (  # type: ignore[method-assign]
        lambda messages, query: messages  # noqa: ARG005
    )
    agent.save_to_memory = lambda prompt, answer: None  # type: ignore[method-assign]  # noqa: ARG005

    _ = agent.respond([LLMMessage(role="user", content="hello")])
    interaction_id = agent.last_chat_interaction_id
    assert interaction_id

    agent.record_feedback_event(
        interaction_id=interaction_id,
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.TOO_LONG],
        free_text="слишком длинно",
    )

    store = MemoryCompanionStore(db_path)
    assert store.get_recent_batch_review_runs(user_id="local", limit=10) == []
    assert store.list_policy_rule_candidates(user_id="local", limit=10) == []

    run = agent.run_batch_review(period_days=1)

    store2 = MemoryCompanionStore(db_path)
    runs = store2.get_recent_batch_review_runs(user_id="local", limit=10)
    assert runs and runs[0].batch_review_run_id == run.batch_review_run_id

    candidates = store2.list_policy_rule_candidates(user_id="local", run_id=run.batch_review_run_id)
    assert candidates
    assert all(c.status == CandidateStatus.PROPOSED for c in candidates)

    # BatchReview не должен создавать PolicyRule.
    assert store2.list_policy_rules("local") == []

    # В этом сценарии должен появиться кандидат на "консистентную краткость".
    concise_candidates = [
        c
        for c in candidates
        if isinstance(c.proposed_action, ActionSetResponseStyle)
        and c.proposed_action.verbosity == ResponseVerbosity.CONCISE
    ]
    assert concise_candidates
