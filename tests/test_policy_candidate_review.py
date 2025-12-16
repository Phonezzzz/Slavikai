from __future__ import annotations

from pathlib import Path

import pytest

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from shared.batch_review_models import CandidateStatus
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import LLMMessage
from shared.policy_models import ActionAddInstruction, TriggerUserMessageContains


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(db_path))
    agent._build_context_messages = (  # type: ignore[method-assign]
        lambda messages, query: messages  # noqa: ARG005
    )
    agent.save_to_memory = lambda prompt, answer: None  # type: ignore[method-assign]  # noqa: ARG005
    return agent


def _seed_candidate(agent: Agent) -> str:
    _ = agent.respond([LLMMessage(role="user", content="hello")])
    interaction_id = agent.last_chat_interaction_id
    assert interaction_id
    agent.record_feedback_event(
        interaction_id=interaction_id,
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.TOO_LONG],
        free_text="слишком длинно",
    )
    run = agent.run_batch_review(period_days=1)
    candidates = agent.list_policy_rule_candidates(run_id=run.batch_review_run_id, limit=10)
    assert candidates
    assert candidates[0].status is CandidateStatus.PROPOSED
    return candidates[0].candidate_id


def test_approve_candidate_creates_policy_rule_and_marks_candidate_approved(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    candidate_id = _seed_candidate(agent)

    rule = agent.approve_policy_rule_candidate(candidate_id=candidate_id)
    assert rule.rule_id
    assert "batch_review_run_id:" in rule.provenance
    assert f"candidate_id:{candidate_id}" in rule.provenance

    store = MemoryCompanionStore(tmp_path / "memory_companion.db")
    rules = store.list_policy_rules("local")
    assert [r.rule_id for r in rules] == [rule.rule_id]

    cand = store.get_policy_rule_candidate(candidate_id=candidate_id)
    assert cand is not None
    assert cand.status is CandidateStatus.APPROVED


def test_edit_candidate_then_approve_uses_updated_values(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    candidate_id = _seed_candidate(agent)

    updated = agent.update_policy_rule_candidate_suggestion(
        candidate_id=candidate_id,
        proposed_trigger_json='{"kind":"user_message_contains","substrings":["hello"],"case_sensitive":false}',
        proposed_action_json='{"kind":"add_instruction","text":"Всегда отвечай кратко и по делу."}',
        priority_suggestion=42,
        confidence_suggestion=0.55,
    )
    assert updated.candidate_id == candidate_id
    assert isinstance(updated.proposed_trigger, TriggerUserMessageContains)
    assert isinstance(updated.proposed_action, ActionAddInstruction)
    assert updated.priority_suggestion == 42
    assert updated.confidence_suggestion == 0.55

    rule = agent.approve_policy_rule_candidate(candidate_id=candidate_id)
    assert isinstance(rule.trigger, TriggerUserMessageContains)
    assert isinstance(rule.action, ActionAddInstruction)
    assert rule.priority == 42
    assert rule.confidence == 0.55


def test_reject_candidate_blocks_approval(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    candidate_id = _seed_candidate(agent)

    agent.reject_policy_rule_candidate(candidate_id=candidate_id)

    with pytest.raises(ValueError, match="status must be proposed"):
        _ = agent.approve_policy_rule_candidate(candidate_id=candidate_id)

    store = MemoryCompanionStore(tmp_path / "memory_companion.db")
    assert store.list_policy_rules("local") == []
