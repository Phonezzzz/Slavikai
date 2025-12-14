from __future__ import annotations

from core.auto_agent import AutoAgent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class FailingBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        raise RuntimeError("fail")


class Parent:
    def __init__(self) -> None:
        self.brain = FailingBrain()


def test_auto_agent_handles_errors() -> None:
    auto = AutoAgent(Parent())
    subtasks = ["a"]
    results = auto.run_parallel(subtasks)
    assert results  # at least one result
    assert "Ошибка" in results[0][1]
