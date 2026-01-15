from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from shared.memory_companion_models import ChatInteractionLog, InteractionKind
from shared.models import LLMMessage
from shared.policy_models import (
    ActionAddInstruction,
    ActionSetResponseStyle,
    PolicyRule,
    PolicyScope,
    ResponseVerbosity,
    TriggerAlways,
    TriggerUserMessageContains,
)


class CapturingBrain(Brain):
    def __init__(self) -> None:
        self.last_messages: list[LLMMessage] | None = None

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.last_messages = messages
        return LLMResult(text="ok")


def _get_last_chat_log(store: MemoryCompanionStore) -> ChatInteractionLog:
    for item in store.get_recent(100):
        if item.interaction_kind == InteractionKind.CHAT and isinstance(item, ChatInteractionLog):
            return item
    raise AssertionError("ChatInteractionLog not found")


def test_agent_applies_policies_and_logs_applied_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    inbox_db_path = db_path.with_name("memory_inbox.db")
    store = MemoryCompanionStore(db_path)

    r1 = PolicyRule(
        rule_id="r1",
        user_id="local",
        scope=PolicyScope.USER,
        trigger=TriggerAlways(),
        action=ActionAddInstruction(text="Instruction A"),
        priority=10,
        confidence=0.9,
        decay_half_life_days=30,
        provenance="batch_review_run_id:1",
        created_at="2025-01-01 00:00:01",
        updated_at="2025-01-01 00:00:01",
    )
    r2 = PolicyRule(
        rule_id="r2",
        user_id="local",
        scope=PolicyScope.GLOBAL,
        trigger=TriggerAlways(),
        action=ActionAddInstruction(text="Instruction B"),
        priority=5,
        confidence=0.8,
        decay_half_life_days=30,
        provenance="batch_review_run_id:1",
        created_at="2025-01-01 00:00:02",
        updated_at="2025-01-01 00:00:02",
    )
    r3 = PolicyRule(
        rule_id="r3",
        user_id="local",
        scope=PolicyScope.USER,
        trigger=TriggerAlways(),
        action=ActionSetResponseStyle(verbosity=ResponseVerbosity.CONCISE),
        priority=7,
        confidence=0.7,
        decay_half_life_days=30,
        provenance="batch_review_run_id:1",
        created_at="2025-01-01 00:00:03",
        updated_at="2025-01-01 00:00:03",
    )
    r4_ignored_style = PolicyRule(
        rule_id="r4",
        user_id="local",
        scope=PolicyScope.GLOBAL,
        trigger=TriggerAlways(),
        action=ActionSetResponseStyle(verbosity=ResponseVerbosity.DETAILED),
        priority=3,
        confidence=0.7,
        decay_half_life_days=30,
        provenance="batch_review_run_id:1",
        created_at="2025-01-01 00:00:04",
        updated_at="2025-01-01 00:00:04",
    )

    store.add_policy_rule(r1)
    store.add_policy_rule(r2)
    store.add_policy_rule(r3)
    store.add_policy_rule(r4_ignored_style)

    brain = CapturingBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(db_path),
        memory_inbox_db_path=str(inbox_db_path),
    )
    agent._build_context_messages = (  # type: ignore[method-assign]
        lambda messages, query: messages  # noqa: ARG005
    )
    agent.save_to_memory = lambda prompt, answer: None  # type: ignore[method-assign]  # noqa: ARG005

    reply = agent.respond([LLMMessage(role="user", content="hello")])
    assert reply.startswith("ok")

    assert brain.last_messages is not None
    policy_msgs = [m for m in brain.last_messages if m.role == "system" and "Политики" in m.content]
    assert policy_msgs, "Expected policy system message"
    content = policy_msgs[-1].content
    assert "Instruction A" in content
    assert "Instruction B" in content
    assert content.index("Instruction A") < content.index("Instruction B")
    assert "Отвечай кратко" in content

    store2 = MemoryCompanionStore(db_path)
    chat = _get_last_chat_log(store2)
    assert chat.applied_policy_ids == ["r1", "r3", "r2"]


def test_policies_not_applied_when_trigger_not_matched(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    inbox_db_path = db_path.with_name("memory_inbox.db")
    store = MemoryCompanionStore(db_path)
    store.add_policy_rule(
        PolicyRule(
            rule_id="r1",
            user_id="local",
            scope=PolicyScope.USER,
            trigger=TriggerUserMessageContains(substrings=["hello"]),
            action=ActionAddInstruction(text="Instruction X"),
            priority=1,
            confidence=1.0,
            decay_half_life_days=30,
            provenance="batch_review_run_id:1",
            created_at="2025-01-01 00:00:01",
            updated_at="2025-01-01 00:00:01",
        )
    )

    brain = CapturingBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(db_path),
        memory_inbox_db_path=str(inbox_db_path),
    )
    agent._build_context_messages = (  # type: ignore[method-assign]
        lambda messages, query: messages  # noqa: ARG005
    )
    agent.save_to_memory = lambda prompt, answer: None  # type: ignore[method-assign]  # noqa: ARG005

    _ = agent.respond([LLMMessage(role="user", content="bye")])

    assert brain.last_messages is not None
    assert not any(m.role == "system" and "Политики" in m.content for m in brain.last_messages)

    store2 = MemoryCompanionStore(db_path)
    chat = _get_last_chat_log(store2)
    assert chat.applied_policy_ids == []
