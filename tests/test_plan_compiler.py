from __future__ import annotations

import pytest

from core.agent import Agent
from core.mwv.models import RunContext, WorkStatus
from core.plan_compiler import PlanCompilationError, compile_structured_plan_steps
from llm.brain_base import Brain
from llm.types import LLMResult, ToolCall, ToolSpec
from server.http.common import workflow_runtime


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


def test_real_agent_executes_compiled_plan_step_through_tool_gateway(tmp_path) -> None:
    agent = Agent(
        brain=_StructuredPlanBrain(),
        memory_companion_db_path=str(tmp_path / "companion.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.set_session_context("session-1", {"FS_DELETE_OVERWRITE"})
    steps = agent.compile_plan_steps("Create hello.txt", [])
    plan = workflow_runtime.build_plan_draft(
        goal="Create hello.txt",
        audit_log=[],
        steps=steps,
        verifier={},
        utc_now_iso_fn=lambda: "2026-01-01T00:00:00+00:00",
        plan_hash_payload_fn=lambda payload: "plan-hash",
    )
    packet = workflow_runtime.compile_plan_to_task_packet(
        plan=plan,
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
    )

    result = agent._mwv_worker_runner(
        packet,
        RunContext(
            session_id="session-1",
            trace_id="trace-1",
            workspace_root=str(tmp_path),
            safe_mode=True,
        ),
    )

    assert result.status == WorkStatus.SUCCESS
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello"
    assert result.tool_calls_used == 1
