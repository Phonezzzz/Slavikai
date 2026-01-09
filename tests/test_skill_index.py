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

    match = index.match("Use ALPHA skill")
    assert match is not None
    assert match.entry.id == "alpha"
    assert match.pattern == "alpha"

    no_match = index.match("unknown request")
    assert no_match is None
