from __future__ import annotations

from pathlib import Path
from typing import Any

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
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


class FakeFeedback:
    def get_recent_hints_meta(self, limit: int = 3, severity_filter: list[str] | None = None):
        return [{"hint": "test major", "severity": "major", "timestamp": "now"}]

    def get_recent_hints(self, limit: int = 3, severity_filter: list[str] | None = None):
        return ["test major"]


class FakeMemory:
    def get_recent(self, limit: int, kind=None):
        return []

    def get_user_prefs(self):
        return []


def test_critic_only_does_not_execute_plan(tmp_path: Path) -> None:
    main = CounterBrain()
    critic = CounterBrain()
    agent = Agent(brain=main, critic=critic, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.set_mode("critic-only")
    response = agent.respond([LLMMessage(role="user", content="План построй это")])
    assert "Критик" in response or response
    assert main.calls == 0  # main модель не вызывалась
    assert critic.calls >= 1


def test_context_uses_hints_meta(tmp_path: Path) -> None:
    main = CounterBrain()
    agent = Agent(brain=main, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.vectors = FakeVectors()
    agent.feedback = FakeFeedback()
    agent.memory = FakeMemory()
    ctx = agent._build_context_messages([LLMMessage(role="user", content="hi")], "hi")  # noqa: SLF001
    assert any(msg.role == "system" for msg in ctx)
    assert agent.last_hints_used == ["test major"]
    assert agent.last_hints_meta and agent.last_hints_meta[0]["severity"] == "major"
