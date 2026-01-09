from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, PlanStep, PlanStepStatus, TaskPlan


class SimpleBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


class FakePlanner:
    def classify_complexity(self, _: str):
        from shared.models import TaskComplexity

        return TaskComplexity.COMPLEX

    def build_plan(self, goal: str, brain=None, model_config=None) -> TaskPlan:
        return TaskPlan(
            goal=goal, steps=[PlanStep(description="step1"), PlanStep(description="step2")]
        )

    def _parse_plan_text(self, text: str):
        return [line.strip() for line in text.splitlines() if line.strip()]


class FakeExecutor:
    def __init__(self) -> None:
        self.run_called = False

    def run(self, plan: TaskPlan, tool_gateway=None) -> TaskPlan:
        self.run_called = True
        for step in plan.steps:
            step.status = PlanStepStatus.DONE
            step.result = "ok"
        return plan


def test_agent_simple_response(tmp_path: Path) -> None:
    brain = SimpleBrain("hello")
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    response = agent.respond([LLMMessage(role="user", content="привет")])
    assert "hello" in response
    assert brain.calls >= 1


def test_agent_plan_execution_path(tmp_path: Path) -> None:
    brain = SimpleBrain("ok")
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.planner = FakePlanner()  # type: ignore[assignment]
    agent.executor = FakeExecutor()  # type: ignore[assignment]
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    result = agent.respond([LLMMessage(role="user", content="планируй задачу")])
    assert "ok" in result
    assert not agent.executor.run_called  # type: ignore[attr-defined]
