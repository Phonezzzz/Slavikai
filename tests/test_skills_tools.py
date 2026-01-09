from __future__ import annotations

import shutil
from pathlib import Path

from skills.tools.build_manifest import build_manifest
from skills.tools.lint_skills import lint_skills


def _write_skill(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """---
"""
        "id: example\n"
        "version: 1.0.0\n"
        "title: Example\n"
        "entrypoints:\n"
        "  - tool\n"
        "patterns:\n"
        "  - example\n"
        "requires: []\n"
        "risk: low\n"
        "tests: []\n"
        "---\n\n"
        "# Example skill\n",
        encoding="utf-8",
    )


def test_build_manifest_and_lint(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    skills_root = tmp_path / "skills"
    schema_src = repo_root / "skills" / "_schema" / "skill.schema.json"
    schema_dst = skills_root / "_schema" / "skill.schema.json"
    schema_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(schema_src, schema_dst)

    _write_skill(skills_root / "example" / "skill.md")

    manifest = build_manifest(skills_root=skills_root, schema_path=schema_dst)
    assert manifest["skills"][0]["id"] == "example"

    errors = lint_skills(skills_root, schema_dst)
    assert errors == []
