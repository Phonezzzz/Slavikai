from __future__ import annotations

from core.executor import Executor
from core.tool_gateway import ToolGateway
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult
from tools.tool_registry import ToolRegistry


def test_executor_with_tool_failure_marks_error() -> None:
    registry = ToolRegistry()

    def failing_handler(_: ToolRequest) -> ToolResult:
        return ToolResult.failure("fail")

    registry.register("web", failing_handler, enabled=True)
    gateway = ToolGateway(registry)
    plan = TaskPlan(
        goal="test",
        steps=[
            PlanStep(description="call web", operation="web", tool_args={"query": "test"}),
            PlanStep(description="next"),
        ],
    )

    executed = Executor().run(plan, tool_gateway=gateway)
    assert executed.steps[0].status == PlanStepStatus.ERROR
    assert "fail" in (executed.steps[0].result or "")


class DummyGateway:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []

    def call(self, request: ToolRequest) -> ToolResult:  # type: ignore[override]
        self.requests.append(request)
        return ToolResult.success({"output": f"ok:{request.name}"})


def test_executor_executes_explicit_tool_requests() -> None:
    gateway = DummyGateway()
    plan = TaskPlan(
        goal="explicit tools",
        steps=[
            PlanStep(description="read file", operation="fs", tool_args={"op": "read"}),
            PlanStep(description="run shell", operation="shell", tool_args={"command": "pwd"}),
            PlanStep(description="project search", operation="project", tool_args={"cmd": "find"}),
        ],
    )

    result = Executor().run(plan, tool_gateway=gateway)

    assert all(step.status == PlanStepStatus.DONE for step in result.steps)
    assert [req.name for req in gateway.requests] == ["fs", "shell", "project"]
    assert gateway.requests[0].args == {"op": "read"}
    assert gateway.requests[1].args == {"command": "pwd"}


def test_executor_does_not_infer_tool_arguments_from_description() -> None:
    gateway = DummyGateway()
    plan = TaskPlan(
        goal="workspace/docs/readme.txt",
        steps=[
            PlanStep(
                description="Прочитать workspace/docs/readme.txt",
                operation="workspace_read",
            )
        ],
    )

    result = Executor().run(plan, tool_gateway=gateway)

    assert result.steps[0].status == PlanStepStatus.DONE
    assert gateway.requests == [ToolRequest(name="workspace_read", args={})]
