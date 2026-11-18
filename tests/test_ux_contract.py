from __future__ import annotations

from pathlib import Path

from core.agent import Agent
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
    response = agent.respond([LLMMessage(role="user", content="привет")]).text
    report = extract_report_block(response)
    assert report["route"] == "chat"
    _assert_required_contract_fields(report)


def test_ux_contract_present_in_command_lane_response(tmp_path: Path) -> None:
    agent = _prepare_agent(tmp_path)
    response = agent.handle_tool_command("/trace")
    report = extract_report_block(response)
    assert report["route"] == "command"
    _assert_required_contract_fields(report)
