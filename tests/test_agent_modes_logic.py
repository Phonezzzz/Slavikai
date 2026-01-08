from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.dual_brain import DualBrain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, PlanStep, PlanStepStatus, TaskPlan


class CountingBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


class StubPlanner:
    def __init__(self) -> None:
        self._parse = None

    def classify_complexity(self, goal: str):  # noqa: ANN001
        from shared.models import TaskComplexity

        return TaskComplexity.COMPLEX

    def build_plan(self, goal: str, brain=None, model_config=None) -> TaskPlan:  # noqa: ANN001
        return TaskPlan(
            goal=goal, steps=[PlanStep(description="orig1"), PlanStep(description="orig2")]
        )

    def _parse_plan_text(self, text: str):
        from core.planner import Planner

        return Planner()._parse_plan_text(text)  # noqa: SLF001

    def parse_plan_text(self, text: str):
        return self._parse_plan_text(text)

    def assign_operations(self, plan: TaskPlan) -> TaskPlan:
        return plan


class FakeExecutor:
    def __init__(self) -> None:
        self.run_called = False
        self.received_plan: TaskPlan | None = None

    def run(self, plan: TaskPlan, tool_gateway=None, critic_callback=None) -> TaskPlan:  # noqa: ANN001
        self.run_called = True
        self.received_plan = plan
        for step in plan.steps:
            step.status = PlanStepStatus.DONE
            step.result = "ok"
        return plan


def _prepare_agent(
    mode: str, *, tmp_path: Path
) -> tuple[Agent, FakeExecutor, CountingBrain, CountingBrain | None]:
    main = CountingBrain("main")
    critic = CountingBrain("1. rewritten\n2. done") if mode != "single" else None
    brain = DualBrain(main, critic) if critic else main
    if isinstance(brain, DualBrain):
        brain.set_mode(mode)
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.planner = StubPlanner()  # type: ignore[assignment]
    executor = FakeExecutor()
    agent.executor = executor  # type: ignore[assignment]
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent, executor, main, critic


def test_agent_single_executes_plan(tmp_path: Path) -> None:
    agent, executor, main, critic = _prepare_agent("single", tmp_path=tmp_path)
    resp = agent.respond([LLMMessage(role="user", content="план задачи")])
    assert "orig1" in resp
    assert agent.last_plan and all(
        step.status == PlanStepStatus.DONE for step in agent.last_plan.steps
    )
    assert critic is None
    assert critic is None


def test_agent_dual_uses_critic_plan(tmp_path: Path) -> None:
    agent, executor, main, critic = _prepare_agent("dual", tmp_path=tmp_path)
    agent.respond([LLMMessage(role="user", content="план задачи")])
    assert agent.last_plan
    assert agent.last_plan and agent.last_plan.steps[0].description.startswith("rewritten")
    assert critic and critic.calls >= 1


def test_agent_critic_only_not_execute_tools(tmp_path: Path) -> None:
    agent, executor, main, critic = _prepare_agent("critic-only", tmp_path=tmp_path)
    resp = agent.respond([LLMMessage(role="user", content="план задачи")])
    assert not executor.run_called
    assert "rewritten" in resp or "done" in resp
    assert critic and critic.calls >= 1
    assert main.calls == 0  # main не вызывается в critic-only при плане
