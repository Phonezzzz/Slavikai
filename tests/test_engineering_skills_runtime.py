from __future__ import annotations

import pytest

from core.agent import Agent
from core.skills.index import SkillIndex, SkillResolutionError
from core.skills.models import SkillEntry, SkillManifest
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from tests.report_utils import extract_report_block


class _Brain(Brain):
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> LLMResult:
        del messages, config
        return LLMResult(text="ok")


def _entry(
    skill_id: str,
    *,
    dependencies: list[str] | None = None,
    supporting: bool = False,
) -> SkillEntry:
    return SkillEntry(
        id=skill_id,
        version="1.2.3",
        title=skill_id,
        entrypoints=[] if supporting else ["workspace_read"],
        patterns=[] if supporting else [skill_id],
        requires=[],
        risk="low",
        tests=[],
        path=f"skills/{skill_id}/skill.md",
        content_hash=f"hash-{skill_id}",
        instructions=f"Instructions for {skill_id}",
        dependencies=list(dependencies or []),
        supporting=supporting,
    )


def test_default_engineering_skill_resolves_body_and_supporting_skills() -> None:
    index = SkillIndex.load_default(dev_mode=False)
    engineering_ids = {
        entry.id for entry in index.manifest.skills if entry.path.startswith("skills/engineering/")
    }

    assert engineering_ids == {
        "code-review",
        "codebase-design",
        "diagnosing-bugs",
        "domain-modeling",
        "grill-with-docs",
        "grilling",
        "handoff",
        "implement",
        "improve-codebase-architecture",
        "prototype",
        "research",
        "resolving-merge-conflicts",
        "tdd",
        "to-spec",
        "to-tickets",
    }
    assert all(index.by_id[skill_id].instructions.strip() for skill_id in engineering_ids)

    decision = index.match_decision("implement spec for the runtime")

    assert decision.status == "matched"
    assert decision.match is not None
    resolution = index.resolve_match(decision.match)
    assert resolution.primary.id == "implement"
    assert [entry.id for entry in resolution.supporting] == ["codebase-design"]
    instruction = resolution.system_instruction()
    assert "Instructions for" not in instruction
    assert "codebase-design@1.0.0" in instruction
    assert "implement@1.0.0" in instruction
    assert "do not grant tools" in instruction


def test_supporting_skills_are_not_directly_matched() -> None:
    supporting = _entry("support", supporting=True)
    index = SkillIndex(SkillManifest(manifest_version=2, skills=[supporting]))

    assert index.match_decision("support").status == "no_match"


def test_dependency_cycle_is_rejected() -> None:
    alpha = _entry("alpha", dependencies=["beta"], supporting=True)
    beta = _entry("beta", dependencies=["alpha"], supporting=True)

    with pytest.raises(SkillResolutionError, match="dependency cycle"):
        SkillIndex(SkillManifest(manifest_version=2, skills=[alpha, beta]))


def test_missing_or_workflow_dependency_is_rejected() -> None:
    primary = _entry("primary", dependencies=["missing"])
    with pytest.raises(SkillResolutionError, match="missing skill"):
        SkillIndex(SkillManifest(manifest_version=2, skills=[primary]))

    workflow_dependency = _entry("workflow")
    primary = _entry("primary", dependencies=["workflow"])
    with pytest.raises(SkillResolutionError, match="is not supporting"):
        SkillIndex(SkillManifest(manifest_version=2, skills=[primary, workflow_dependency]))


def test_mwv_report_serializes_skill_runtime_observability(tmp_path) -> None:
    agent = Agent(
        brain=_Brain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    response = agent._append_report_block(  # noqa: SLF001
        "ok",
        route="auto",
        trace_id="trace",
        attempts=(1, 1),
        verifier=None,
        next_steps=[],
        stop_reason_code=None,
        skill={
            "status": "completed",
            "skill_id": "implement",
            "version": "1.2.3",
            "supporting_skills": [{"skill_id": "codebase-design", "version": "1.2.3"}],
        },
    )
    report = extract_report_block(response)

    assert report["skill"] == {
        "status": "completed",
        "skill_id": "implement",
        "version": "1.2.3",
        "supporting_skills": [{"skill_id": "codebase-design", "version": "1.2.3"}],
    }
