from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class StaticBrain(Brain):
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        del messages, config
        return LLMResult(text=self._text)


def _build_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=StaticBrain("ok"),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
        canonical_atoms_db_path=str(tmp_path / "atoms.db"),
    )
    agent.runtime_mode = "ask"
    return agent


def test_ask_mode_does_not_write_memory_or_claims(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    object.__setattr__(agent.memory_config, "auto_save_dialogue", True)

    calls = {"save": 0, "claims": 0}

    def _save(_prompt: str, _answer: str) -> None:
        calls["save"] += 1

    def _capture(*args: object, **kwargs: object) -> list[dict[str, object]]:
        del args, kwargs
        calls["claims"] += 1
        return []

    agent.save_to_memory = _save  # type: ignore[method-assign]
    agent.capture_memory_claims_from_text = _capture  # type: ignore[method-assign]
    response = agent.respond([LLMMessage(role="user", content="Привет")])

    assert "ok" in response
    assert calls["save"] == 0
    assert calls["claims"] == 0


def test_ask_mode_vector_context_does_not_trigger_runtime_init(tmp_path: Path) -> None:
    agent = _build_agent(tmp_path)
    agent.memory.get_recent = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]

    calls = {"ensure_runtime_ready": 0}

    def _ensure_runtime_ready() -> None:
        calls["ensure_runtime_ready"] += 1
        raise AssertionError("ensure_runtime_ready should not be called in ask read-only path")

    agent.vectors.ensure_runtime_ready = _ensure_runtime_ready  # type: ignore[method-assign]
    response = agent.respond([LLMMessage(role="user", content="Покажи контекст проекта")])

    assert "ok" in response
    assert calls["ensure_runtime_ready"] == 0
