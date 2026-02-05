from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.mwv.manager import MWVRunResult
from core.mwv.models import (
    ChangeType,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class StaticBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text=self.text)


def _prepare_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=StaticBrain("chat ok"),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def _assert_required_contract_fields(report: dict[str, object]) -> None:
    assert "plan_summary" in report
    assert isinstance(report["plan_summary"], str)
    assert report["plan_summary"].strip()
    assert "execution_summary" in report
    assert isinstance(report["execution_summary"], str)
    assert report["execution_summary"].strip()


def test_ux_contract_present_in_chat_response(tmp_path: Path) -> None:
    agent = _prepare_agent(tmp_path)
    response = agent.respond([LLMMessage(role="user", content="привет")])
    report = extract_report_block(response)
    assert report["route"] == "chat"
    _assert_required_contract_fields(report)


def test_ux_contract_present_in_command_lane_response(tmp_path: Path) -> None:
    agent = _prepare_agent(tmp_path)
    response = agent.handle_tool_command("/trace")
    report = extract_report_block(response)
    assert report["route"] == "command"
    _assert_required_contract_fields(report)


def test_ux_contract_present_in_mwv_response(tmp_path: Path) -> None:
    agent = _prepare_agent(tmp_path)
    run_result = MWVRunResult(
        task=TaskPacket(
            task_id="task-1",
            session_id="session-1",
            trace_id="trace-1",
            goal="fix tests",
        ),
        work_result=WorkResult(
            task_id="task-1",
            status=WorkStatus.SUCCESS,
            summary="1. ✅ Применить фикс",
            changes=[
                WorkChange(
                    path="src/app.py",
                    change_type=ChangeType.UPDATE,
                    summary="+3/-1",
                )
            ],
        ),
        verification_result=VerificationResult(
            status=VerificationStatus.PASSED,
            command=["scripts/check.sh"],
            exit_code=0,
            stdout="ok",
            stderr="",
            duration_seconds=0.1,
        ),
        attempt=1,
        max_attempts=1,
        retry_decision=None,
    )
    response = agent._format_mwv_response(run_result)
    report = extract_report_block(response)
    assert report["route"] == "mwv"
    _assert_required_contract_fields(report)
