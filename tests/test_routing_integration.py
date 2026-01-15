from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class DummyBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def test_routing_chat_path_uses_llm(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV path should not be used for chat input.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="какая погода")])
    assert response.startswith("chat")
    assert brain.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_routing_chat_path_for_explanation(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV path should not be used for chat input.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="объясни термин git")])
    assert response.startswith("chat")
    assert brain.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_routing_mwv_path_bypasses_llm(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain("chat")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        return "mwv"

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])
    assert response == "mwv"
    assert brain.calls == 0
