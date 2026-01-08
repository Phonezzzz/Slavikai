from __future__ import annotations

import pytest

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


def test_executor_requires_media_paths() -> None:
    executor = Executor()
    gateway = DummyGateway()
    plan = TaskPlan(goal="test", steps=[])

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
