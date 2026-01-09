from __future__ import annotations

from core.mwv.routing import classify_request
from core.skills.index import SkillIndex
from core.skills.models import SkillEntry, SkillManifest


def _entry(skill_id: str, pattern: str, *, deprecated: bool = False) -> SkillEntry:
    return SkillEntry(
        id=skill_id,
        version="1.0.0",
        title=skill_id,
        entrypoints=["tool"],
        patterns=[pattern],
        requires=[],
        risk="low",
        tests=[],
        path=f"skills/{skill_id}/skill.md",
        content_hash="hash",
        deprecated=deprecated,
        replaced_by="modern" if deprecated else None,
    )


def test_skill_regression_default_manifest() -> None:
    index = SkillIndex.load_default(dev_mode=False)
    cases = [
        ("прочитай файл README.md", "workspace-files"),
        ("запиши файл notes.txt", "workspace-files"),
        ("применить патч для README", "workspace-files"),
        ("поиск в интернете про rust", "web-search"),
        ("найди в интернете про asyncio", "web-search"),
        ("web search fastapi", "web-search"),
        ("сгенерируй отчет о продажах", "doc-generator"),
        ("создай документ с итогами", "doc-generator"),
        ("сделай таблицу расходов", "doc-generator"),
        ("сгенерируй отчет по проекту", "doc-generator"),
    ]
    for text, expected in cases:
        decision = index.match_decision(text)
        assert decision.status == "matched"
        assert decision.match is not None
        assert decision.match.entry.id == expected


def test_skill_regression_candidate_route() -> None:
    empty_index = SkillIndex(SkillManifest(manifest_version=1, skills=[]))
    decision = classify_request(
        messages=[],
        user_input="исправь баг в коде",
        context=None,
        skill_index=empty_index,
    )
    assert decision.route == "mwv"
    assert decision.skill_decision is not None
    assert decision.skill_decision.status == "no_match"
    assert "skill_no_match" in decision.reason


def test_skill_regression_ambiguous_and_deprecated() -> None:
    manifest = SkillManifest(
        manifest_version=1,
        skills=[
            _entry("alpha", "build"),
            _entry("beta", "build"),
            _entry("legacy", "deploy", deprecated=True),
        ],
    )
    index = SkillIndex(manifest)

    ambiguous = index.match_decision("build pipeline")
    assert ambiguous.status == "ambiguous"
    assert ambiguous.match is None
    assert len(ambiguous.alternatives) == 2

    deprecated = index.match_decision("deploy service")
    assert deprecated.status == "deprecated"
    assert deprecated.match is not None
    assert deprecated.match.entry.id == "legacy"
