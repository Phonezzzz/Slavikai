from __future__ import annotations

import json
from pathlib import Path

from core.agent import Agent
from core.skills.index import SkillIndex
from core.skills.models import SkillEntry, SkillManifest
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class DummyBrain(Brain):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text="ok")


def _make_skill_index(entries: list[SkillEntry]) -> SkillIndex:
    manifest = SkillManifest(manifest_version=1, skills=entries)
    return SkillIndex(manifest)


def _skill_entry(skill_id: str, patterns: list[str]) -> SkillEntry:
    return SkillEntry(
        id=skill_id,
        version="1.0.0",
        title=f"{skill_id} title",
        entrypoints=["mwv"],
        patterns=patterns,
        requires=[],
        risk="low",
        tests=[],
        path=f"skills/{skill_id}/skill.md",
        content_hash=f"hash-{skill_id}",
        deprecated=False,
        replaced_by=None,
    )


def test_decision_packet_ambiguous_skill(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain()
    agent = Agent(brain=brain, memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.skill_index = _make_skill_index(
        [
            _skill_entry("alpha", ["alpha"]),
            _skill_entry("beta", ["alpha"]),
        ]
    )

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV should not be called for ambiguous skill.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="alpha request")])

    payload = json.loads(response)
    assert payload["reason"] == "ambiguous_skill"
    assert 3 <= len(payload["options"]) <= 5
    assert any(option["action"] == "select_skill" for option in payload["options"])
    assert brain.calls == 0
    assert agent.last_decision_packet is not None
