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


def _no_match_decision(_text: str) -> SkillMatchDecision:
    return SkillMatchDecision(
        status="no_match",
        match=None,
        alternatives=[],
        reason="no_pattern_match",
    )


def _assert_no_unknown_side_effects(agent, tmp_path: Path) -> None:
    inbox = agent._memory_inbox_store.list_items(MemoryCategory.INBOX, limit=10)
    assert len(inbox.items) == 0
    candidates_dir = tmp_path / "candidates"
    assert list(candidates_dir.glob("*.md")) == []


def test_unknown_request_does_not_write_inbox(tmp_path: Path, monkeypatch) -> None:
    # Codex P1, PR #162: ordinary chat ("привет", no skill match) in the
    # plan/act chat-fallback must NOT write to the Memory inbox or create
    # a skill candidate file. Memory writes stay behind an explicit
    # confirmed action.
    from core.skills.candidates import SkillCandidateWriter

    agent = _make_agent(tmp_path)
    agent.runtime_mode = "plan"
    agent._skill_candidate_writer = SkillCandidateWriter(  # noqa: SLF001
        candidates_dir=tmp_path / "candidates"
    )
    monkeypatch.setattr(agent.skill_index, "match_decision", _no_match_decision)

    response = agent.respond([LLMMessage(role="user", content="привет")]).text
    assert response.startswith("ok")

    _assert_no_unknown_side_effects(agent, tmp_path)


def test_unknown_request_does_not_write_inbox_stream(tmp_path: Path, monkeypatch) -> None:
    # Same as above, streaming branch of the chat-fallback.
    from core.skills.candidates import SkillCandidateWriter
    from llm.stream_model import Done

    agent = _make_agent(tmp_path)
    agent.runtime_mode = "plan"
    agent._skill_candidate_writer = SkillCandidateWriter(  # noqa: SLF001
        candidates_dir=tmp_path / "candidates"
    )
    monkeypatch.setattr(agent.skill_index, "match_decision", _no_match_decision)

    events = list(
        agent.respond_stream([LLMMessage(role="user", content="привет")], cancellation_token=None)
    )
    assert any(isinstance(event, Done) for event in events)

    _assert_no_unknown_side_effects(agent, tmp_path)


def test_disabled_command_lane_tool_does_not_write_tool_fail_inbox(
    tmp_path: Path, monkeypatch
) -> None:
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

    def _fail_call(
        _: ToolRequest,
        *,
        bypass_safe_mode: bool = False,
        confirmed_decision: bool = False,
    ) -> ToolResult:
        del bypass_safe_mode, confirmed_decision
        return ToolResult.failure("boom")

    monkeypatch.setattr(agent.tool_registry, "call", _fail_call)

    for _ in range(agent_module.SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD * 2):
        agent.respond([LLMMessage(role="user", content="/fs list")])

    inbox = agent._memory_inbox_store.list_items(MemoryCategory.INBOX, limit=10)
    assert len(inbox.items) == 0
