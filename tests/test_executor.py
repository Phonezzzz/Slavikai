from __future__ import annotations

from core.executor import Executor
from core.tool_gateway import ToolGateway
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


def test_executor_with_tool_and_critic_rejects() -> None:
    registry = ToolRegistry()

    def echo_handler(_: ToolRequest) -> ToolResult:
        return ToolResult.success({"output": "ok"})

    registry.register("web", echo_handler, enabled=True)
    gateway = ToolGateway(registry)
    plan = TaskPlan(
        goal="test",
        steps=[
            PlanStep(description="web step", operation="web"),
            PlanStep(description="next"),
        ],
    )

    def critic(step: PlanStep) -> tuple[bool, str | None]:
        if "web" in step.description:
            return False, "no web"
        return True, None

    executor = Executor()
    executed = executor.run(plan, tool_gateway=gateway, critic_callback=critic)
    assert executed.steps[0].status == PlanStepStatus.ERROR
    assert "no web" in (executed.steps[0].result or "")
