from __future__ import annotations

import json
from pathlib import Path

from config.mode_config import load_mode, save_mode
from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.feedback_manager import FeedbackManager
from memory.memory_manager import MemoryManager
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


def test_agent_llm_error_path(tmp_path: Path) -> None:
    agent = Agent(brain=ErrorBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.memory = MemoryManager(str(tmp_path / "mem.db"))
    agent.feedback = FeedbackManager(str(tmp_path / "fb.db"))
    agent.memory.save(
        MemoryRecord(id="1", content="c", tags=[], timestamp="t", kind=MemoryKind.NOTE)
    )
    resp = agent.respond([LLMMessage(role="user", content="fail me")])
    assert "Ошибка модели" in resp


def test_agent_plan_command_with_stub_planner(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    stub = StubPlanner()
    agent.planner = stub  # type: ignore[assignment]
    result = agent.handle_tool_command("/plan goal")
    assert "step-one" in result
    assert stub.executed


def test_agent_set_mode_invalid(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    try:
        agent.set_mode("dual")
    except ValueError as exc:
        assert "Режимы" in str(exc)
    else:  # pragma: no cover - safety
        raise AssertionError("Должно было выбросить ValueError")


def test_save_feedback_major_hint(tmp_path: Path) -> None:
    agent = Agent(brain=SimpleBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.feedback = FeedbackManager(str(tmp_path / "fb.db"))
    agent.save_feedback("p", "a", "bad", hint=None)
    records = agent.feedback.get_recent_records(1)
    assert records and records[0]["severity"] == "major"
    assert records[0]["hint"]


def test_mode_config_invalid_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "mode.json"
    bad_path.write_text("{notjson", encoding="utf-8")
    try:
        load_mode(bad_path)
    except RuntimeError:
        pass
    save_mode("single", bad_path)
    assert json.loads(bad_path.read_text())["mode"] == "single"
