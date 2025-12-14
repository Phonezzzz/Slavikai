from __future__ import annotations

from core.executor import Executor
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult


class FlakyGateway(ToolGateway):
    def __init__(self) -> None:
        super().__init__(None)  # type: ignore[arg-type]
        self.calls = 0

    def call(self, request: ToolRequest) -> ToolResult:  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            return ToolResult.success({"output": "ok"})
        return ToolResult.failure("fail")


def test_executor_stops_after_error() -> None:
    tracer = Tracer()
    executor = Executor(tracer)
    plan = TaskPlan(
        goal="test",
        steps=[
            PlanStep(description="web search"),
            PlanStep(description="shell command"),
            PlanStep(description="other"),
        ],
    )
    finished = executor.run(plan, tool_gateway=FlakyGateway())
    assert finished.steps[0].status == PlanStepStatus.DONE
    assert finished.steps[1].status == PlanStepStatus.ERROR
    assert finished.steps[2].status == PlanStepStatus.PENDING


def test_executor_critic_rejects_step() -> None:
    tracer = Tracer()
    executor = Executor(tracer)
    plan = TaskPlan(goal="test", steps=[PlanStep(description="web search")])

    def critic(step: PlanStep):
        return False, "nope"

    finished = executor.run(plan, tool_gateway=FlakyGateway(), critic_callback=critic)
    assert finished.steps[0].status == PlanStepStatus.ERROR
    assert "nope" in (finished.steps[0].result or "")
