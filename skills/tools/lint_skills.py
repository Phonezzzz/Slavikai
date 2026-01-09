from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from skills.tools.skill_utils import (
    iter_skill_files,
    load_schema,
    parse_front_matter,
    validate_front_matter,
)

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = DEFAULT_SKILLS_ROOT / "_schema" / "skill.schema.json"


def lint_skills(skills_root: Path, schema_path: Path) -> list[str]:
    schema = load_schema(schema_path)
    errors: list[str] = []
    seen_ids: set[str] = set()
    repo_root = skills_root.parent

    candidate_skill_files = list((skills_root / "_candidates").rglob("skill.md"))
    if candidate_skill_files:
        for path in candidate_skill_files:
            errors.append(f"Candidate contains active skill.md: {path}")

    for path in iter_skill_files(skills_root):
        try:
            front = parse_front_matter(path)
            validate_front_matter(front, schema, path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
            continue
        if front.id in seen_ids:
            errors.append(f"Duplicate id: {front.id}")
        seen_ids.add(front.id)
        for test_path in front.tests:
            resolved = (repo_root / test_path).resolve()
            if not resolved.exists():
                errors.append(f"Missing test path for {front.id}: {test_path}")
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lint skills")
    parser.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    errors = lint_skills(args.skills_root, args.schema)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
