from __future__ import annotations

from collections.abc import Callable

from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.types import ModelConfig
from shared.models import PlanStep, TaskComplexity, TaskPlan

class Planner:
    tracer: object

    def build_plan(
        self,
        goal: str,
        brain: Brain | None = ...,
        model_config: ModelConfig | None = ...,
    ) -> TaskPlan: ...
    def execute_plan(
        self,
        plan: TaskPlan,
        agent_callback: Callable[[PlanStep], str] | None = ...,
        tool_gateway: ToolGateway | None = ...,
    ) -> TaskPlan: ...
    def classify_complexity(self, goal: str) -> TaskComplexity: ...
    def parse_plan_text(self, text: str) -> list[str] | None: ...
    def assign_operations(self, plan: TaskPlan) -> TaskPlan: ...
