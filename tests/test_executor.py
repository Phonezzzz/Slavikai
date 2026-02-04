from __future__ import annotations

import pytest

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
            PlanStep(description="web step", operation="web"),
            PlanStep(description="next"),
        ],
    )

    executor = Executor()
    executed = executor.run(plan, tool_gateway=gateway)
    assert executed.steps[0].status == PlanStepStatus.ERROR
    assert "fail" in (executed.steps[0].result or "")


class DummyGateway:
    def __init__(self) -> None:
        self.requests: list[ToolRequest] = []

    def call(self, request: ToolRequest) -> ToolResult:  # type: ignore[override]
        self.requests.append(request)
        return ToolResult.success({"output": f"ok:{request.name}"})


def test_executor_executes_tool_operations() -> None:
    executor = Executor()
    gateway = DummyGateway()
    plan = TaskPlan(
        goal="docs/readme.md",
        steps=[
            PlanStep(description="прочитать docs/readme.md", operation="fs"),
            PlanStep(description="shell cmd", operation="shell"),
            PlanStep(description="project search", operation="project"),
            PlanStep(description="tts", operation="tts"),
            PlanStep(description="image gen", operation="image_generate"),
        ],
    )

    result = executor.run(plan, tool_gateway=gateway)
    assert all(step.status == PlanStepStatus.DONE for step in result.steps)
    assert [req.name for req in gateway.requests] == [
        "fs",
        "shell",
        "project",
        "tts",
        "image_generate",
    ]
    assert gateway.requests[0].args.get("op") == "read"


def test_executor_maps_workspace_operations_to_requests() -> None:
    executor = Executor()
    gateway = DummyGateway()
    patch_text = "@@ -1,1 +1,1 @@\\n-old\\n+new\\n"
    plan = TaskPlan(
        goal="workspace/docs/readme.txt",
        steps=[
            PlanStep(description="Прочитать workspace/docs/readme.txt", operation="workspace_read"),
            PlanStep(
                description="Записать workspace/docs/new.txt content=hello",
                operation="workspace_write",
            ),
            PlanStep(
                description=f"Применить patch workspace/docs/new.txt patch={patch_text}",
                operation="workspace_patch",
            ),
        ],
    )

    result = executor.run(plan, tool_gateway=gateway)
    assert all(step.status == PlanStepStatus.DONE for step in result.steps)
    assert [req.name for req in gateway.requests] == [
        "workspace_read",
        "workspace_write",
        "workspace_patch",
    ]
    assert gateway.requests[0].args["path"] == "workspace/docs/readme.txt"
    assert gateway.requests[1].args["content"] == "hello"
    assert gateway.requests[2].args["patch"] == "@@ -1,1 +1,1 @@\n-old\n+new\n"


def test_executor_requires_media_paths() -> None:
    executor = Executor()
    gateway = DummyGateway()
    plan = TaskPlan(goal="test", steps=[])

    with pytest.raises(RuntimeError):
        executor._execute_with_tools(
            PlanStep(description="workspace read", operation="workspace_read"),
            plan,
            gateway,
        )
    with pytest.raises(RuntimeError):
        executor._execute_with_tools(
            PlanStep(description="workspace write", operation="workspace_write"),
            plan,
            gateway,
        )
    with pytest.raises(RuntimeError):
        executor._execute_with_tools(
            PlanStep(description="workspace patch file.txt", operation="workspace_patch"),
            plan,
            gateway,
        )
    with pytest.raises(RuntimeError):
        executor._execute_with_tools(PlanStep(description="stt", operation="stt"), plan, gateway)
    with pytest.raises(RuntimeError):
        executor._execute_with_tools(
            PlanStep(description="image", operation="image_analyze"), plan, gateway
        )
    with pytest.raises(RuntimeError):
        executor._execute_with_tools(
            PlanStep(description="run", operation="workspace_run"), plan, gateway
        )
