from __future__ import annotations

from core.auto_agent import AutoAgent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text=f"echo:{messages[-1].content}")


class FakeAgent:
    def __init__(self) -> None:
        self.brain = SimpleBrain()


def test_auto_agent_parallel() -> None:
    auto = AutoAgent(FakeAgent())
    tasks = auto.generate_subtasks("проверить и запустить")
    assert tasks
    results = auto.run_parallel(tasks[:2])
    assert len(results) == 2
    assert all("echo" in res for _, res in results)
