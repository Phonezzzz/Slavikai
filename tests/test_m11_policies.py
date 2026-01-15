from __future__ import annotations

import json
from pathlib import Path

from core.agent import Agent
from core.skills.index import SkillIndex
from core.skills.models import SkillEntry, SkillManifest
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage, PlanStep, TaskPlan
from tests.report_utils import extract_report_block


class DummyBrain(Brain):
    def __init__(self, text: str = "chat") -> None:
        self.text = text
        self.calls = 0

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text)


class StubPlanner:
    def build_plan(self, goal: str, brain=None, model_config=None) -> TaskPlan:
        _ = (brain, model_config)
        return TaskPlan(goal=goal, steps=[PlanStep(description="step-one")])


class StubExecutor:
    def __init__(self) -> None:
        self.run_called = False

    def run(self, plan: TaskPlan, tool_gateway=None) -> TaskPlan:  # noqa: ANN001
        _ = tool_gateway
        self.run_called = True
        return plan


def _make_skill_index(entries: list[SkillEntry]) -> SkillIndex:
    manifest = SkillManifest(manifest_version=1, skills=entries)
    return SkillIndex(manifest)


def _skill_entry(
    skill_id: str,
    patterns: list[str],
    *,
    deprecated: bool = False,
    replaced_by: str | None = None,
) -> SkillEntry:
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
        deprecated=deprecated,
        replaced_by=replaced_by,
    )


def test_command_lane_manual_mode_without_mwv(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.planner = StubPlanner()  # type: ignore[assignment]
    agent.executor = StubExecutor()  # type: ignore[assignment]

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV should not be called for / commands.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="/plan goal")])

    assert "без MWV" in response
    assert "step-one" in response
    assert brain.calls == 0


def test_skill_match_routes_to_mwv(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.memory.get_recent = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.memory.get_user_prefs = lambda: []  # type: ignore[attr-defined]
    agent.vectors.search = lambda *a, **k: []  # type: ignore[attr-defined]
    agent.skill_index = _make_skill_index([_skill_entry("alpha", ["alpha"])])

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        return "mwv"

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="use alpha skill")])

    assert response == "mwv"
    assert brain.calls == 0


def test_skill_ambiguous_blocks_with_instruction(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
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


def test_skill_deprecated_blocks_with_instruction(tmp_path: Path, monkeypatch) -> None:
    brain = DummyBrain()
    agent = Agent(
        brain=brain,
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    agent.skill_index = _make_skill_index(
        [
            _skill_entry("legacy", ["legacy"], deprecated=True, replaced_by="alpha"),
        ]
    )

    def _mwv_stub(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("MWV should not be called for deprecated skill.")

    monkeypatch.setattr(agent, "_run_mwv_flow", _mwv_stub)
    response = agent.respond([LLMMessage(role="user", content="legacy request")])

    lowered = response.lower()
    assert "что случилось" in lowered
    assert "deprecated" in lowered
    assert "что делать дальше" in lowered
    assert brain.calls == 0
    report = extract_report_block(response)
    assert report["route"] == "blocked"
    assert report["stop_reason_code"] == "BLOCKED_SKILL_DEPRECATED"
