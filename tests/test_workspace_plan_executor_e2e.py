from __future__ import annotations

from pathlib import Path

import pytest

import tools.workspace_tools as workspace_tools
from core.executor import Executor
from core.planner import Planner
from core.tool_gateway import ToolGateway
from shared.models import PlanStepStatus
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import ApplyPatchTool, ReadFileTool, WriteFileTool


def _workspace_gateway() -> ToolGateway:
    registry = ToolRegistry()
    registry.register("workspace_read", ReadFileTool(), enabled=True)
    registry.register("workspace_write", WriteFileTool(), enabled=True)
    registry.register("workspace_patch", ApplyPatchTool(), enabled=True)
    return ToolGateway(registry)


def test_plan_executor_workspace_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)
    target = tmp_path / "docs" / "readme.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("line-1\nline-2\n", encoding="utf-8")

    planner = Planner()
    plan = planner.build_plan("Прочитай файл docs/readme.txt")

    assert any(step.operation == "workspace_read" for step in plan.steps)
    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    read_steps = [step for step in executed.steps if step.operation == "workspace_read"]
    assert read_steps and "line-1" in (read_steps[0].result or "")


def test_plan_executor_workspace_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)
    planner = Planner()
    plan = planner.build_plan("Создай файл docs/new.txt content=hello")

    assert any(step.operation == "workspace_write" for step in plan.steps)
    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    assert (tmp_path / "docs" / "new.txt").read_text(encoding="utf-8") == "hello"


def test_plan_executor_workspace_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(workspace_tools, "WORKSPACE_ROOT", tmp_path)
    target = tmp_path / "docs" / "patch.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("old\n", encoding="utf-8")

    planner = Planner()
    plan = planner.build_plan("Измени файл docs/patch.txt patch=@@ -1,1 +1,1 @@\\n-old\\n+new\\n")

    assert any(step.operation == "workspace_patch" for step in plan.steps)
    executed = Executor().run(plan, tool_gateway=_workspace_gateway())
    assert all(step.status == PlanStepStatus.DONE for step in executed.steps)
    assert target.read_text(encoding="utf-8") == "new\n"
