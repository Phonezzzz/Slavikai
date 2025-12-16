from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from shared.memory_companion_models import InteractionKind
from shared.models import LLMMessage


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def test_agent_logs_chat_and_tool(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(db_path))

    # Изолируем тест от тяжёлых зависимостей (VectorIndex / память).
    agent._build_context_messages = (  # type: ignore[method-assign]
        lambda messages, query: messages  # noqa: ARG005
    )

    reply = agent.respond([LLMMessage(role="user", content="hello")])
    assert reply == "ok"

    _ = agent.handle_tool_command("/fs list")

    store = MemoryCompanionStore(db_path)
    recent = store.get_recent(20)
    assert any(item.interaction_kind == InteractionKind.CHAT for item in recent)
    assert any(item.interaction_kind == InteractionKind.TOOL for item in recent)
