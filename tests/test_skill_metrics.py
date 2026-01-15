from __future__ import annotations

from pathlib import Path

from core.agent import Agent
from core.skills.index import SkillMatch, SkillMatchDecision
from core.skills.models import SkillEntry
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        _ = messages
        _ = config
        return LLMResult(text="ok")


def _entry(skill_id: str, *, deprecated: bool = False) -> SkillEntry:
    return SkillEntry(
        id=skill_id,
        version="1.0.0",
        title=skill_id,
        entrypoints=["tool"],
        patterns=[skill_id],
        requires=[],
        risk="low",
        tests=[],
        path=f"skills/{skill_id}/skill.md",
        content_hash="hash",
        deprecated=deprecated,
    )


def test_skill_metrics_increment(tmp_path: Path) -> None:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    match = SkillMatch(entry=_entry("alpha"), pattern="alpha")
    decision = SkillMatchDecision(
        status="matched",
        match=match,
        alternatives=[],
        reason="skill_match",
    )
    agent._apply_skill_decision(decision)  # noqa: SLF001
    assert agent._skill_metrics["skill_match_hit"] == 1  # noqa: SLF001

    miss = SkillMatchDecision(
        status="no_match",
        match=None,
        alternatives=[],
        reason="none",
    )
    agent._apply_skill_decision(miss)  # noqa: SLF001
    assert agent._skill_metrics["skill_match_miss"] == 1  # noqa: SLF001

    deprecated_match = SkillMatch(entry=_entry("legacy", deprecated=True), pattern="legacy")
    deprecated = SkillMatchDecision(
        status="deprecated",
        match=deprecated_match,
        alternatives=[],
        reason="deprecated",
        replaced_by="modern",
    )
    agent._apply_skill_decision(deprecated)  # noqa: SLF001
    assert agent._skill_metrics["deprecated_count"] == 1  # noqa: SLF001
