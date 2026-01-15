from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class SimpleBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(
        brain=SimpleBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )


def test_command_lane_safe_command_passes(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    response = agent.handle_tool_command("/fs list")
    assert "Командный режим (без MWV)" in response
    assert "Что случилось" not in response
    report = extract_report_block(response)
    assert report["route"] == "command"
    assert report["stop_reason_code"] == "COMMAND_LANE_NOTICE"


def test_command_lane_dangerous_requires_approval(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    response = agent.handle_tool_command("/fs write project/demo.txt")
    lowered = response.lower()
    assert "что случилось" in lowered
    assert "подтверждение" in lowered
    assert "command_lane" in lowered
    report = extract_report_block(response)
    assert report["route"] == "command"
    assert report["stop_reason_code"] == "APPROVAL_REQUIRED"


def test_command_lane_dangerous_after_approve_passes(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    agent.set_session_context("s1", {"FS_DELETE_OVERWRITE"})
    response = agent.handle_tool_command("/fs write project/demo.txt")
    assert "Командный режим (без MWV)" in response
    assert "Файл записан" in response or "записан" in response.lower()
    report = extract_report_block(response)
    assert report["route"] == "command"
    assert report["stop_reason_code"] == "COMMAND_LANE_NOTICE"
