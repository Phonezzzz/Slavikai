from __future__ import annotations

from core.executor import Executor
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult


class DummyGateway(ToolGateway):
    def __init__(self) -> None:
        super().__init__(None)  # type: ignore[arg-type]

    def call(self, request: ToolRequest) -> ToolResult:  # type: ignore[override]
        if request.name == "web":
            return ToolResult.success({"output": "ok"})
        return ToolResult.failure("blocked")


def test_executor_critic_rejects() -> None:
    tracer = Tracer()
    executor = Executor(tracer)
    plan = TaskPlan(goal="test", steps=[PlanStep(description="web step")])

    def critic(step: PlanStep):
        return False, "reject"

    finished = executor.run(plan, tool_gateway=DummyGateway(), critic_callback=critic)
    assert finished.steps[0].status == PlanStepStatus.ERROR
    assert "reject" in (finished.steps[0].result or "")
