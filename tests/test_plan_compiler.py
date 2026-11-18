from __future__ import annotations

import pytest

from core.plan_compiler import PlanCompilationError, compile_structured_plan_steps
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall, ToolSpec


class _StructuredPlanBrain(Brain):
    supports_native_tools = True

    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del messages, config
        assert tools is not None
        assert [tool.name for tool in tools] == ["submit_plan"]
        operation_schema = tools[0].parameters_schema["properties"]
        assert isinstance(operation_schema, dict)
        return LLMResult(
            text="",
            tool_calls=[
                ToolCall(
                    id="plan-1",
                    name="submit_plan",
                    arguments={
                        "steps": [
                            {
                                "title": "Write file",
                                "description": "Create the requested file.",
                                "operation": "workspace_create",
                                "tool_args": {"path": "hello.txt", "content": "hello"},
                                "expected_outputs": ["hello.txt exists"],
                                "acceptance_checks": ["write tool succeeds"],
                            }
                        ]
                    },
                )
            ],
        )


class _UnsupportedBrain(Brain):
    def generate(self, messages, config=None, tools=None):  # noqa: ANN001
        del messages, config, tools
        return LLMResult(text="unused")


def test_plan_compiler_produces_explicit_executable_step() -> None:
    steps = compile_structured_plan_steps(
        brain=_StructuredPlanBrain(),
        config=None,
        goal="Create hello.txt",
        audit_log=[],
        available_tools=[
            ToolSpec(
                name="workspace_create",
                description="Write a workspace file",
                parameters_schema={"type": "object"},
            )
        ],
    )

    assert steps[0]["allowed_tool_kinds"] == ["workspace_create"]
    assert steps[0]["inputs"] == {
        "operation": "workspace_create",
        "tool_args": {"path": "hello.txt", "content": "hello"},
    }


def test_plan_compiler_rejects_provider_without_native_tools() -> None:
    with pytest.raises(PlanCompilationError) as exc_info:
        compile_structured_plan_steps(
            brain=_UnsupportedBrain(),
            config=None,
            goal="Create hello.txt",
            audit_log=[],
            available_tools=[],
        )

    assert exc_info.value.code == "native_tools_required"
