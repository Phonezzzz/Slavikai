from __future__ import annotations

from collections.abc import Callable
from typing import Final

from config.system_prompts import PLANNER_PROMPT
from core.executor import Executor
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.types import ModelConfig
from shared.models import LLMMessage, PlanStep, TaskComplexity, TaskPlan
from shared.plan_models import PlanStepSchema, TaskPlanSchema

MIN_STEPS: Final[int] = 2
MAX_STEPS: Final[int] = 8

DEFAULT_PLAN_STEPS: Final[tuple[str, ...]] = (
    "Clarify the requested outcome",
    "Run the necessary native tool calls through the gateway",
    "Summarize the result and any remaining blockers",
)


class Planner:
    """Builds plan metadata without mapping prose to tools."""

    def __init__(self) -> None:
        self.tracer = Tracer()

    def build_plan(
        self,
        goal: str,
        brain: Brain | None = None,
        model_config: ModelConfig | None = None,
    ) -> TaskPlan:
        cleaned_goal = goal.strip()
        self.tracer.log("planning_start", f"Создание плана для: {cleaned_goal}")
        steps = self._llm_plan(cleaned_goal, brain, model_config) if brain else None
        plan = self._build_task_plan(cleaned_goal, steps or list(DEFAULT_PLAN_STEPS))
        if not self._has_explicit_tool_steps(plan) and not self._is_plan_valid(plan):
            self.tracer.log("planning_invalid_plan", "План не прошёл валидацию")
            plan = self._build_task_plan(cleaned_goal, list(DEFAULT_PLAN_STEPS))
        self.tracer.log("planning_done", f"Сформирован план из {len(plan.steps)} шагов")
        return plan

    def execute_plan(
        self,
        plan: TaskPlan,
        agent_callback: Callable[[PlanStep], str] | None = None,
        tool_gateway: ToolGateway | None = None,
    ) -> TaskPlan:
        return Executor(self.tracer).run(
            plan,
            agent_callback=agent_callback,
            tool_gateway=tool_gateway,
        )

    def classify_complexity(self, goal: str) -> TaskComplexity:
        tokens = goal.strip().split()
        if len(tokens) > 16:
            return TaskComplexity.COMPLEX
        return TaskComplexity.SIMPLE

    def _llm_plan(
        self, goal: str, brain: Brain | None, model_config: ModelConfig | None
    ) -> list[PlanStep] | None:
        if brain is None:
            return None
        try:
            messages = [
                LLMMessage(role="system", content=PLANNER_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "Create a short execution plan for this task. "
                        "Use native tool calls when a tool is required.\n"
                        f"Task: {goal}"
                    ),
                ),
            ]
            result = brain.generate(messages, model_config)
            if result.tool_calls:
                return [
                    PlanStep(
                        description=f"Call tool {tool_call.name}",
                        operation=tool_call.name,
                        tool_args=dict(tool_call.arguments),
                    )
                    for tool_call in result.tool_calls
                ]
            steps = self._parse_plan_text(result.text)
            if steps:
                return [PlanStep(description=step) for step in steps]
            self.tracer.log("planning_validation_failed", "LLM план не прошёл валидацию")
            return None
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("planning_llm_error", str(exc))
            return None

    def _parse_plan_text(self, text: str) -> list[str] | None:
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned: list[str] = []
        for line in raw_lines:
            stripped = line.lstrip("-•0123456789. ").strip()
            if stripped:
                cleaned.append(stripped)
        unique_steps = []
        for step in cleaned:
            if step not in unique_steps:
                unique_steps.append(step)
        if len(unique_steps) < MIN_STEPS or len(unique_steps) > MAX_STEPS:
            return None
        return unique_steps

    def _is_plan_valid(self, plan: TaskPlan) -> bool:
        if len(plan.steps) < MIN_STEPS or len(plan.steps) > MAX_STEPS:
            return False
        schema = TaskPlanSchema(
            goal=plan.goal,
            steps=[
                PlanStepSchema(
                    description=step.description,
                    status=step.status.value,
                    operation=step.operation,
                    result=step.result,
                )
                for step in plan.steps
            ],
        )
        return schema.is_valid(MIN_STEPS, MAX_STEPS)

    def _has_explicit_tool_steps(self, plan: TaskPlan) -> bool:
        return any(step.operation for step in plan.steps)

    def _build_task_plan(self, goal: str, steps: list[str] | list[PlanStep]) -> TaskPlan:
        plan_steps = [
            step if isinstance(step, PlanStep) else PlanStep(description=step) for step in steps
        ]
        return TaskPlan(goal=goal, steps=plan_steps)
