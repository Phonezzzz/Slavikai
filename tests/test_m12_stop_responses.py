from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class FailingBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        raise RuntimeError("boom")


def test_chat_internal_error_returns_error_response_without_mwv(tmp_path: Path) -> None:
    # E01-T08: MWV-пути нет; внутренняя ошибка chat loop — видимая ошибка,
    # а не stop-ответ MWV.
    agent = Agent(
        brain=FailingBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])

    assert not hasattr(agent, "_run_mwv_flow")
    assert "mwv" not in response.text.lower()
    assert response.text.strip()
