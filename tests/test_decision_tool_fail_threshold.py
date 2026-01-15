from __future__ import annotations

import json
from pathlib import Path

from core.agent import SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD, Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, ToolRequest, ToolResult


class DummyBrain(Brain):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text="chat")


def _make_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def test_decision_packet_tool_fail_threshold(tmp_path: Path, monkeypatch) -> None:
    agent = _make_agent(tmp_path)

    def _fail_call(_: ToolRequest, *, bypass_safe_mode: bool = False) -> ToolResult:
        _ = bypass_safe_mode
        return ToolResult.failure("boom")

    monkeypatch.setattr(agent.tool_registry, "call", _fail_call)

    response = ""
    for _ in range(SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD):
        response = agent.respond([LLMMessage(role="user", content="/fs list")])

    payload = json.loads(response)
    assert payload["reason"] == "tool_fail"
    assert 3 <= len(payload["options"]) <= 5
    assert any(option["action"] == "retry" for option in payload["options"])
    assert agent.brain.calls == 0
