from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.dual_brain import DualBrain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class CountingBrain(Brain):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


def _prepare_agent(mode: str, tmp_path: Path) -> tuple[Agent, CountingBrain, CountingBrain]:
    main = CountingBrain("main")
    critic = CountingBrain("critic")
    brain = DualBrain(main, critic)
    brain.set_mode(mode)
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent, main, critic


def test_agent_dual_brain_ignored_for_chat(tmp_path: Path) -> None:
    agent, main, critic = _prepare_agent("dual", tmp_path)
    response = agent.respond([LLMMessage(role="user", content="привет")])
    assert response == "main"
    assert main.calls == 1
    assert critic.calls == 0


def test_agent_mwv_route_bypasses_brain(tmp_path: Path, monkeypatch) -> None:
    agent, main, critic = _prepare_agent("dual", tmp_path)

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        return "mwv"

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="исправь тесты")])
    assert response == "mwv"
    assert main.calls == 0
    assert critic.calls == 0
