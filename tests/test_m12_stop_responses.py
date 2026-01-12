from __future__ import annotations

from pathlib import Path

import core.agent as agent_module
from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="chat")


def test_mwv_internal_error_returns_stop_response(tmp_path: Path, monkeypatch) -> None:
    agent = Agent(brain=DummyBrain(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("boom")

    monkeypatch.setattr(agent_module.ManagerRuntime, "run_flow", _boom)
    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])

    lowered = response.lower()
    assert "что случилось" in lowered
    assert "mwv internal error" in lowered
    assert "что делать дальше" in lowered
    assert "trace_id=" in response
    report = extract_report_block(response)
    assert report["route"] == "mwv"
    assert report["stop_reason_code"] == "MWV_INTERNAL_ERROR"
