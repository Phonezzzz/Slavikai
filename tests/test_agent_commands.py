from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class SimpleBrain(Brain):
    def generate(
        self, messages: list[LLMMessage], config: ModelConfig | None = None
    ) -> LLMResult:
        return LLMResult(text="ok")


def test_agent_unknown_tool_command(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    resp = agent.handle_tool_command("/unknown")
    assert "неизвестен" in resp.lower() or "неактивен" in resp.lower()


def test_agent_shell_disabled_in_safe_mode(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
    )
    resp = agent.handle_tool_command("/sh ls")
    assert "требуется подтверждение" in resp.lower()
