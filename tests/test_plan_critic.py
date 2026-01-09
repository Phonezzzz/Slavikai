from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.types import LLMResult
from shared.models import PlanStep, TaskPlan


class DummyBrain:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages, config=None):  # noqa: ANN001
        return LLMResult(text=self._text)


def test_critic_plan_is_noop_without_dualbrain(tmp_path: Path) -> None:
    main = DummyBrain("unused")
    agent = Agent(brain=main, memory_companion_db_path=str(tmp_path / "mc.db"))
    plan = TaskPlan(goal="goal", steps=[PlanStep(description="old")])

    improved = agent._critic_plan(plan)  # noqa: SLF001
    assert improved is plan
