from __future__ import annotations

from core.executor import Executor
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult


class DummyGateway(ToolGateway):
    def __init__(self, ok: bool = True) -> None:
        super().__init__(None)  # type: ignore[arg-type]
        self.ok = ok

    def call(self, request: ToolRequest) -> ToolResult:  # type: ignore[override]
        if self.ok:
            return ToolResult.success({"output": f"done {request.name}"})
        return ToolResult.failure("err")


def test_executor_success_and_error() -> None:
    tracer = Tracer()
    executor = Executor(tracer)
    plan = TaskPlan(
        goal="test",
        steps=[
            PlanStep(description="web", operation="web"),
            PlanStep(description="fail", operation="web"),
        ],
    )
    finished = executor.run(plan, tool_gateway=DummyGateway(ok=True), critic_callback=None)
    assert all(step.status == PlanStepStatus.DONE for step in finished.steps[:1])

    finished_error = executor.run(plan, tool_gateway=DummyGateway(ok=False), critic_callback=None)
    assert any(step.status == PlanStepStatus.ERROR for step in finished_error.steps)
