import asyncio
from pathlib import Path

import pytest

from core.agent import Agent
from core.agent_mwv import TaskPacketApprovalPending
from core.mwv.models import (
    RunContext,
    TaskPacket,
    TaskStepContract,
    WorkStatus,
    with_task_packet_hash,
)
from shared.models import ToolResult
from tests.test_agent_response import SimpleBrain


def test_packet_projection_cannot_claim_completed_execution(tmp_path: Path) -> None:
    agent = Agent(
        brain=SimpleBrain("unused"),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    executions = []
    agent.tool_registry.register(
        "mwv_probe",
        lambda request: executions.append(request) or ToolResult.success({"output": "observed"}),
        enabled=True,
        capability="read",
        description="Diagnostic read",
        parameters_schema={"type": "object"},
    )
    packet = with_task_packet_hash(
        TaskPacket(
            task_id="task-1",
            session_id="session-1",
            trace_id="trace-1",
            goal="inspect",
            scope={"workspace_root": str(tmp_path)},
            steps=[
                TaskStepContract(
                    step_id="probe-1",
                    title="Probe",
                    description="Inspect",
                    allowed_tool_kinds=["mwv_probe"],
                    inputs={"operation": "mwv_probe", "tool_args": {}},
                )
            ],
            context={
                "plan_runner_resume": {
                    "step_results": [
                        {
                            "step_id": "probe-1",
                            "description": "Inspect",
                            "operation": "mwv_probe",
                            "status": "done",
                            "result": "invented observation",
                            "tool_calls_used": 1,
                            "changes": [],
                        }
                    ],
                    "tool_calls_used": 1,
                }
            },
        )
    )
    try:
        result = agent._mwv_worker_runner(
            packet,
            RunContext(
                session_id="session-1",
                trace_id="trace-1",
                workspace_root=str(tmp_path),
                safe_mode=True,
            ),
        )
        assert result.status == WorkStatus.FAILURE
        assert executions == []
    finally:
        agent.close()


@pytest.mark.parametrize(
    "preclaim",
    [None, "cancelled", "fault", "foreign_session", "cancel_after_call", "post_call_fault"],
)
def test_owned_worker_checkpoint_preserves_prefix_and_exact_once(
    tmp_path: Path, monkeypatch, preclaim: str | None
) -> None:
    agent = Agent(
        brain=SimpleBrain("unused"),
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.set_session_context("session-1", set())
    agent.apply_runtime_workspace_root(str(tmp_path))
    agent.set_runtime_state(
        mode="act", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    (tmp_path / "Makefile").write_text("check:\n\t@true\n")
    executions = []
    names = ["mwv_prefix", "mwv_pending", "mwv_next"]
    for name in names:
        agent.tool_registry.register(
            name,
            lambda request: executions.append(request.name)
            or ToolResult.success({"output": request.name}),
            enabled=True,
            capability="read",
            risk_classes=[] if name == "mwv_prefix" else ["network"],
            description="Diagnostic read",
            parameters_schema={"type": "object"},
        )
    packet = with_task_packet_hash(
        TaskPacket(
            task_id="task-1",
            session_id="session-1",
            trace_id="trace-1",
            goal="inspect",
            scope={"workspace_root": str(tmp_path)},
            steps=[
                TaskStepContract(
                    step_id=name,
                    title=name,
                    description=name,
                    allowed_tool_kinds=[name],
                    inputs={"operation": name, "tool_args": {}},
                )
                for name in names
            ],
        )
    )
    context = RunContext(
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
        safe_mode=True,
    )
    try:
        with pytest.raises(TaskPacketApprovalPending) as first:
            agent._mwv_worker_runner(packet, context)
        assert executions == ["mwv_prefix"]
        checkpoint = agent._mwv_checkpoints[first.value.checkpoint_id]
        assert checkpoint.packet == packet
        assert checkpoint.observations[0][0].name == "mwv_prefix"
        assert checkpoint.observations[0][1].data["output"] == "mwv_prefix"
        if preclaim == "cancelled":
            token = asyncio.Event()
            token.set()
            with pytest.raises(ValueError, match="not_started"):
                agent.resume_task_packet(
                    first.value.checkpoint_id, packet, cancellation_token=token
                )
        elif preclaim == "fault":
            original = agent._workspace_diff_pre_call

            def fail(request):
                raise RuntimeError("pre-dispatch fault")

            monkeypatch.setattr(agent, "_workspace_diff_pre_call", fail)
            with pytest.raises(ValueError, match="not_started"):
                agent.resume_task_packet(first.value.checkpoint_id, packet)
            monkeypatch.setattr(agent, "_workspace_diff_pre_call", original)
        elif preclaim == "foreign_session":
            agent.set_session_context("other-session", set())
            with pytest.raises(ValueError, match="unavailable"):
                agent.resume_task_packet(first.value.checkpoint_id, packet)
            agent.set_session_context("session-1", set())
        assert agent._mwv_checkpoints[first.value.checkpoint_id] is checkpoint
        assert executions == ["mwv_prefix"]
        if preclaim == "post_call_fault":

            def fail_after(request, result, context):
                raise RuntimeError("projection unavailable")

            monkeypatch.setattr(agent, "_workspace_diff_post_call", fail_after)
            failed = agent.resume_task_packet(first.value.checkpoint_id, packet)
            assert failed.work_result.status == WorkStatus.FAILURE
            assert len(failed.tool_observations) == 2
            assert failed.tool_observations[-1][1].data["output"] == "mwv_pending"
            assert executions == ["mwv_prefix", "mwv_pending"]
            assert agent._mwv_checkpoints == {}
            return
        if preclaim == "cancel_after_call":
            token = asyncio.Event()

            def cancel_after(request):
                executions.append(request.name)
                token.set()
                return ToolResult.success({"output": "retained-after-cancel"})

            agent.tool_registry.register(
                "mwv_pending",
                cancel_after,
                enabled=True,
                capability="read",
                risk_classes=["network"],
                description="Diagnostic read",
                parameters_schema={"type": "object"},
            )
            cancelled = agent.resume_task_packet(
                first.value.checkpoint_id, packet, cancellation_token=token
            )
            assert cancelled.work_result.status == WorkStatus.FAILURE
            assert cancelled.work_result.root_cause_tag == "cancelled"
            assert len(cancelled.tool_observations) == 2
            assert cancelled.tool_observations[-1][1].data["output"] == "retained-after-cancel"
            assert executions == ["mwv_prefix", "mwv_pending"]
            assert agent._mwv_checkpoints == {}
            return
        with pytest.raises(TaskPacketApprovalPending) as second:
            agent.resume_task_packet(first.value.checkpoint_id, packet)
        assert executions == ["mwv_prefix", "mwv_pending"]
        assert agent.approved_categories == set()
        next_checkpoint = agent._mwv_checkpoints[second.value.checkpoint_id]
        assert len(next_checkpoint.observations) == 2
        result = agent.resume_task_packet(second.value.checkpoint_id, packet)
        assert result.work_result.status == WorkStatus.SUCCESS
        with pytest.raises(ValueError, match="unavailable"):
            agent.resume_task_packet(second.value.checkpoint_id, packet)
        assert executions == names
        assert result.work_result.tool_calls_used == 3
        assert len(result.tool_observations) == 3
        assert packet.context == {}
        assert agent.approved_categories == set()
    finally:
        agent.close()


@pytest.mark.parametrize("retry_fault", [False, True])
def test_retry_checkpoint_uses_owned_revision_and_fault_attempt(tmp_path, monkeypatch, retry_fault):
    from dataclasses import replace

    agent = Agent(
        brain=SimpleBrain("unused"),
        enable_tools={"safe_mode": True},
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.set_session_context("session-1", set())
    agent.apply_runtime_workspace_root(str(tmp_path))
    agent.set_runtime_state(
        mode="act", active_plan=None, active_task=None, enforce_plan_guard=False
    )
    (tmp_path / "Makefile").write_text("check:\n\t@test -e verified || (touch verified; false)\n")
    calls = []
    agent.tool_registry.register(
        "retry_probe",
        lambda request: calls.append(request) or ToolResult.success({"output": "ok"}),
        enabled=True,
        capability="read",
        risk_classes=["network"],
        description="Probe",
        parameters_schema={"type": "object"},
    )
    packet = with_task_packet_hash(
        TaskPacket(
            task_id="task-1",
            session_id="session-1",
            trace_id="trace-1",
            goal="inspect",
            scope={"workspace_root": str(tmp_path)},
            budgets={"max_attempts": 2},
            verifier={"command": ["make", "check"]},
            steps=[
                TaskStepContract(
                    step_id="probe",
                    title="Probe",
                    description="Probe",
                    allowed_tool_kinds=["retry_probe"],
                    inputs={"operation": "retry_probe", "tool_args": {}},
                )
            ],
        )
    )
    context = RunContext(
        session_id="session-1",
        trace_id="trace-1",
        workspace_root=str(tmp_path),
        safe_mode=True,
        max_retries=1,
    )
    try:
        with pytest.raises(TaskPacketApprovalPending) as first:
            agent.run_task_packet(packet, context)
        with pytest.raises(TaskPacketApprovalPending) as second:
            agent.resume_task_packet(first.value.checkpoint_id, packet)
        checkpoint = agent._mwv_checkpoints[second.value.checkpoint_id]
        assert checkpoint.context.attempt == 2
        assert checkpoint.packet.packet_revision == 2
        assert len(calls) == 1
        tampered = with_task_packet_hash(replace(packet, goal="different"))
        with pytest.raises(ValueError, match="unavailable"):
            agent.resume_task_packet(second.value.checkpoint_id, tampered)
        if retry_fault:

            def post_fault(request, result, context):
                raise RuntimeError("retry projection fault")

            monkeypatch.setattr(agent, "_workspace_diff_post_call", post_fault)
        result = agent.resume_task_packet(second.value.checkpoint_id, packet)
        assert result.attempt == 2
        assert result.task == checkpoint.packet
        assert len(calls) == 2
        assert len(result.tool_observations) == 2
        assert result.work_result.status == (
            WorkStatus.FAILURE if retry_fault else WorkStatus.SUCCESS
        )
        assert agent._mwv_checkpoints == {}
    finally:
        agent.close()
