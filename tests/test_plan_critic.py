from __future__ import annotations

from core.agent import Agent
from llm.dual_brain import DualBrain
from llm.types import LLMResult
from shared.models import PlanStep, TaskPlan


class DummyBrain:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages, config=None):  # noqa: ANN001
        return LLMResult(text=self._text)


def test_plan_critic_rewrites_plan() -> None:
    main = DummyBrain("unused")
    critic = DummyBrain("1. new step\n2. finish")
    dual = DualBrain(main, critic)
    agent = Agent(brain=dual, critic=None)
    plan = TaskPlan(goal="goal", steps=[PlanStep(description="old")])

    improved = agent._critic_plan(plan)  # noqa: SLF001
    assert len(improved.steps) == 2
    assert "new step" in improved.steps[0].description
