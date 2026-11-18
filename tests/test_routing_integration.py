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

    response = agent.respond([LLMMessage(role="user", content="какая погода")]).text
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

    response = agent.respond([LLMMessage(role="user", content="объясни термин git")]).text
    assert response.startswith("chat")
    assert brain.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"


def test_routing_code_input_uses_chat_loop(tmp_path: Path) -> None:
    # E01-T08: MWV-пути нет; запрос с кодовыми триггерами идёт в chat loop.
    brain = DummyBrain("chat")
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")]).text
    assert response.startswith("chat")
    assert brain.calls == 1
    report = extract_report_block(response)
    assert report["route"] == "chat"
