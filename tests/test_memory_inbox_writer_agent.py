from __future__ import annotations

from pathlib import Path

import core.agent as agent_module
from config.memory_config import MemoryConfig
from core.skills.index import SkillMatchDecision
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from memory.memory_inbox_writer import MemoryInboxWriter
from shared.memory_category_models import MemoryCategory
from shared.models import LLMMessage, ToolRequest, ToolResult


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> agent_module.Agent:
    agent = agent_module.Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    return agent


def test_unknown_request_writes_inbox(tmp_path: Path, monkeypatch) -> None:
    agent = _make_agent(tmp_path)

    def _no_match(_text: str) -> SkillMatchDecision:
        return SkillMatchDecision(
            status="no_match",
            match=None,
            alternatives=[],
            reason="no_pattern_match",
        )

    monkeypatch.setattr(agent.skill_index, "match_decision", _no_match)
    monkeypatch.setattr(agent, "_run_mwv_flow", lambda *a, **k: "ok")

    response = agent.respond([LLMMessage(role="user", content="поправь баг в коде")])
    assert response == "ok"

    inbox = agent._memory_inbox_store.list_items(MemoryCategory.INBOX, limit=10)
    assert len(inbox.items) == 1
    assert inbox.items[0].category == MemoryCategory.INBOX
    assert "Неизвестный запрос" in inbox.items[0].content

    notes = agent._memory_inbox_store.list_items(MemoryCategory.NOTES, limit=10)
    assert len(notes.items) == 0


def test_tool_fail_threshold_writes_inbox_once(tmp_path: Path, monkeypatch) -> None:
    agent = _make_agent(tmp_path)
    monkeypatch.setenv("SKILLS_CANDIDATES_DIR", str(tmp_path / "candidates"))
    agent._memory_inbox_writer = MemoryInboxWriter(
        agent._memory_inbox_store,
        MemoryConfig(
            inbox_max_items=200,
            inbox_ttl_days=30,
            inbox_writes_per_minute=1000,
        ),
    )

    def _fail_call(_: ToolRequest, *, bypass_safe_mode: bool = False) -> ToolResult:
        _ = bypass_safe_mode
        return ToolResult.failure("boom")

    monkeypatch.setattr(agent.tool_registry, "call", _fail_call)

    for _ in range(agent_module.SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD * 2):
        agent.respond([LLMMessage(role="user", content="/fs list")])

    inbox = agent._memory_inbox_store.list_items(MemoryCategory.INBOX, limit=10)
    assert len(inbox.items) == 1
