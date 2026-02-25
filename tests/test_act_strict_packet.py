from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.mwv.models import (
    RunContext,
    StopReasonCode,
    TaskPacket,
    TaskStepContract,
    WorkStatus,
    with_task_packet_hash,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class StaticBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        del messages, config
        return LLMResult(text="ok")


def _build_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=StaticBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    return agent


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        session_id="s",
        trace_id="trace",
        workspace_root=str(tmp_path),
        safe_mode=True,
    )


def test_act_strict_rejects_invalid_packet_hash(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    packet = TaskPacket(
        task_id="task-1",
        session_id="s",
        trace_id="trace",
        goal="g",
        packet_hash="broken",
        steps=[
            TaskStepContract(
                step_id="step-1",
                title="noop",
                description="noop",
                allowed_tool_kinds=[],
                inputs={},
            )
        ],
    )
    result = agent._mwv_worker_runner(packet, _context(tmp_path))
    assert result.status == WorkStatus.FAILURE
    assert result.diagnostics.get("stop_reason_code") == StopReasonCode.REPLAN_REQUIRED.value


def test_act_strict_rejects_operation_outside_allowed(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    packet = TaskPacket(
        task_id="task-1",
        session_id="s",
        trace_id="trace",
        goal="g",
        steps=[
            TaskStepContract(
                step_id="step-1",
                title="bad",
                description="bad",
                allowed_tool_kinds=["workspace_read"],
                inputs={"operation": "workspace_write"},
            )
        ],
    )
    packet = with_task_packet_hash(packet)
    result = agent._mwv_worker_runner(packet, _context(tmp_path))
    assert result.status == WorkStatus.FAILURE
    assert result.diagnostics.get("stop_reason_code") == StopReasonCode.REPLAN_REQUIRED.value
