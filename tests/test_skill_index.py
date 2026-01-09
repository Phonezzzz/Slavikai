from __future__ import annotations

import json
from pathlib import Path

from core.skills.index import SkillIndex
from core.skills.manifest import load_manifest


def _write_manifest(path: Path, skills: list[dict[str, object]]) -> None:
    payload = {
        "manifest_version": 1,
        "skills": skills,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_skill_index_match(tmp_path: Path) -> None:
    manifest_path = tmp_path / "skills.manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "alpha",
                "version": "1.0.0",
                "title": "Alpha",
                "entrypoints": ["tool_a"],
                "patterns": ["alpha"],
                "requires": [],
                "risk": "low",
                "tests": [],
                "path": "skills/alpha/skill.md",
                "content_hash": "hash",
            },
            {
                "id": "beta",
                "version": "1.0.0",
                "title": "Beta",
                "entrypoints": ["tool_b"],
                "patterns": ["beta"],
                "requires": [],
                "risk": "medium",
                "tests": [],
                "path": "skills/beta/skill.md",
                "content_hash": "hash2",
            },
        ],
    )
    manifest = load_manifest(path=manifest_path, dev_mode=False)
    index = SkillIndex(manifest)

    decision = index.match_decision("Use ALPHA skill")
    assert decision.status == "matched"
    assert decision.match is not None
    assert decision.match.entry.id == "alpha"
    assert decision.match.pattern == "alpha"

    no_match = index.match_decision("unknown request")
    assert no_match.status == "no_match"
    assert no_match.match is None


def test_skill_index_ambiguous_same_score(tmp_path: Path) -> None:
    manifest_path = tmp_path / "skills.manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "alpha",
                "version": "1.0.0",
                "title": "Alpha",
                "entrypoints": ["tool_a"],
                "patterns": ["build"],
                "requires": [],
                "risk": "low",
                "tests": [],
                "path": "skills/alpha/skill.md",
                "content_hash": "hash",
            },
            {
                "id": "beta",
                "version": "1.0.0",
                "title": "Beta",
                "entrypoints": ["tool_b"],
                "patterns": ["build"],
                "requires": [],
                "risk": "medium",
                "tests": [],
                "path": "skills/beta/skill.md",
                "content_hash": "hash2",
            },
        ],
    )
    manifest = load_manifest(path=manifest_path, dev_mode=False)
    index = SkillIndex(manifest)

    decision = index.match_decision("build pipeline")
    assert decision.status == "ambiguous"
    ids = {match.entry.id for match in decision.alternatives}
    assert ids == {"alpha", "beta"}


def test_skill_index_deprecated(tmp_path: Path) -> None:
    manifest_path = tmp_path / "skills.manifest.json"
    _write_manifest(
        manifest_path,
        [
            {
                "id": "legacy",
                "version": "1.0.0",
                "title": "Legacy",
                "entrypoints": ["tool_a"],
                "patterns": ["deploy"],
                "requires": [],
                "risk": "low",
                "tests": [],
                "path": "skills/legacy/skill.md",
                "content_hash": "hash",
                "deprecated": True,
                "replaced_by": "modern",
            }
        ],
    )
    manifest = load_manifest(path=manifest_path, dev_mode=False)
    index = SkillIndex(manifest)

    decision = index.match_decision("deploy service")
    assert decision.status == "deprecated"
    assert decision.match is not None
    assert decision.match.entry.id == "legacy"
    assert decision.replaced_by == "modern"
