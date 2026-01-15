from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class CountingBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def _prepare_agent(tmp_path: Path) -> tuple[Agent, CountingBrain]:
    main = CountingBrain("main")
    agent = Agent(
        brain=main,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent, main


def test_agent_chat_uses_main_brain(tmp_path: Path) -> None:
    agent, main = _prepare_agent(tmp_path)
    response = agent.respond([LLMMessage(role="user", content="привет")])
    assert response.startswith("main")
    assert main.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_agent_mwv_route_bypasses_brain(tmp_path: Path, monkeypatch) -> None:
    agent, main = _prepare_agent(tmp_path)

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        return "mwv"

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="исправь тесты")])
    assert response == "mwv"
    assert main.calls == 0
