from __future__ import annotations

from pathlib import Path

import pytest

import tools.workspace_tools as workspace_tools
from core.executor import Executor
from core.tool_gateway import ToolGateway
from llm.types import LLMResult, ToolCall
from shared.models import LLMMessage, PlanStepStatus
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import ApplyPatchTool, ReadFileTool, WriteFileTool


class ToolCallBrain:
    def __init__(self, calls: list[ToolCall]) -> None:
        self.calls = calls

    def generate(self, messages, config=None, tools=None):  # type: ignore[override]
        del messages, config, tools
        return LLMResult(text="", tool_calls=self.calls)


def _workspace_gateway() -> ToolGateway:
    registry = ToolRegistry()
    registry.register("workspace_read", ReadFileTool(), enabled=True)
    registry.register("workspace_write", WriteFileTool(), enabled=True)
    registry.register("workspace_patch", ApplyPatchTool(), enabled=True)
    return ToolGateway(registry)


def test_tool_call_plan_executor_workspace_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)
    target = tmp_path / "docs" / "readme.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("line-1\nline-2\n", encoding="utf-8")

    from core.planner import Planner

    plan = Planner().build_plan(
        "read",
        brain=ToolCallBrain(
            [ToolCall(id="call-1", name="workspace_read", arguments={"path": "docs/readme.txt"})]
        ),  # type: ignore[arg-type]
    )

    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    assert "line-1" in (executed.steps[0].result or "")


def test_tool_call_plan_executor_workspace_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)

    from core.planner import Planner

    plan = Planner().build_plan(
        "write",
        brain=ToolCallBrain(
            [
                ToolCall(
                    id="call-1",
                    name="workspace_write",
                    arguments={"path": "docs/new.txt", "content": "hello"},
                )
            ]
        ),  # type: ignore[arg-type]
    )

    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    assert (tmp_path / "docs" / "new.txt").read_text(encoding="utf-8") == "hello"


def test_tool_call_plan_executor_workspace_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)
    target = tmp_path / "docs" / "patch.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")

    from core.planner import Planner

    plan = Planner().build_plan(
        "patch",
        brain=ToolCallBrain(
            [
                ToolCall(
                    id="call-1",
                    name="workspace_patch",
                    arguments={
                        "path": "docs/patch.txt",
                        "patch": "@@ -1,1 +1,1 @@\n-old\n+new\n",
                    },
                )
            ]
        ),  # type: ignore[arg-type]
    )

    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    assert target.read_text(encoding="utf-8") == "new\n"


def test_tool_call_brain_receives_regular_messages() -> None:
    brain = ToolCallBrain([])
    result = brain.generate([LLMMessage(role="user", content="hello")])
    assert result.tool_calls == []
