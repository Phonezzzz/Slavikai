from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_manager import MemoryManager
from shared.memory_companion_models import (
    ChatInteractionLog,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionMode,
)
from shared.models import LLMMessage, MemoryKind, MemoryRecord, PlanStep, PlanStepStatus, TaskPlan


class ErrorBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        raise RuntimeError("boom")


class SimpleBrain(Brain):
    def __init__(self, text: str = "ok") -> None:
        self.text = text

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text=self.text)


class StubPlanner:
    def __init__(self) -> None:
        self.executed = False

    def build_plan(self, goal: str, brain=None, model_config=None) -> TaskPlan:
        return TaskPlan(goal=goal, steps=[PlanStep(description="step-one")])

    def execute_plan(self, plan: TaskPlan) -> TaskPlan:
        self.executed = True
        for step in plan.steps:
            step.status = PlanStepStatus.DONE
            step.result = "done"
        return plan


class StubExecutor:
    def __init__(self) -> None:
        self.run_called = False

    def run(self, plan: TaskPlan, tool_gateway=None) -> TaskPlan:  # noqa: ANN001
        self.run_called = True
        for step in plan.steps:
            step.status = PlanStepStatus.DONE
            step.result = "done"
        return plan


def _log_interaction(agent: Agent, interaction_id: str) -> None:
    agent._interaction_store.log_interaction(  # noqa: SLF001
        ChatInteractionLog(
            interaction_id=interaction_id,
            user_id=agent.user_id,
            interaction_kind=InteractionKind.CHAT,
            raw_input="prompt",
            mode=InteractionMode.STANDARD,
            created_at="2024-01-01 00:00:00",
            response_text="answer",
        )
    )


def test_agent_llm_error_path(tmp_path: Path) -> None:
    agent = Agent(
        brain=ErrorBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory = MemoryManager(str(tmp_path / "mem.db"))
    agent.memory.save(
        MemoryRecord(id="1", content="c", tags=[], timestamp="t", kind=MemoryKind.NOTE)
    )
    resp = agent.respond([LLMMessage(role="user", content="fail me")])
    assert "Ошибка модели" in resp


def test_agent_plan_command_with_stub_planner(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    stub = StubPlanner()
    executor = StubExecutor()
    agent.planner = stub  # type: ignore[assignment]
    agent.executor = executor  # type: ignore[assignment]
    result = agent.handle_tool_command("/plan goal")
    assert "step-one" in result
    assert executor.run_called


def test_save_feedback_major_hint(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    _log_interaction(agent, "1")
    agent.record_feedback_event(
        interaction_id="1",
        rating=FeedbackRating.BAD,
        labels=[FeedbackLabel.HALLUCINATION],
    )
    hints = agent._collect_feedback_hints(1, severity_filter=["major", "fatal"])  # noqa: SLF001
    assert hints and hints[0]["severity"] in {"major", "fatal"}
    assert hints[0]["hint"]
