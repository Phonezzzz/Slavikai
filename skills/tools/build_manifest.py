from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ is None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from skills.tools.skill_utils import (
    SchemaData,
    SkillFrontMatter,
    iter_skill_files,
    load_schema,
    parse_front_matter,
    validate_front_matter,
)

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = DEFAULT_SKILLS_ROOT / "_schema" / "skill.schema.json"
DEFAULT_OUTPUT_PATH = DEFAULT_SKILLS_ROOT / "_generated" / "skills.manifest.json"
MANIFEST_VERSION = 1


def build_manifest(
    *,
    skills_root: Path,
    schema_path: Path,
) -> dict[str, object]:
    schema = load_schema(schema_path)
    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    repo_root = skills_root.parent
    for skill_path in iter_skill_files(skills_root):
        front = _load_front_matter(skill_path, schema)
        if front.id in seen_ids:
            raise ValueError(f"Duplicate skill id: {front.id}")
        seen_ids.add(front.id)
        entry: dict[str, object] = {
            "id": front.id,
            "version": front.version,
            "title": front.title,
            "entrypoints": front.entrypoints,
            "patterns": front.patterns,
            "requires": front.requires,
            "risk": front.risk,
            "tests": front.tests,
            "deprecated": front.deprecated,
            "content_hash": _hash_file(skill_path),
            "path": str(skill_path.relative_to(repo_root)),
        }
        if front.replaced_by is not None:
            entry["replaced_by"] = front.replaced_by
        entries.append(entry)
    entries.sort(key=lambda item: str(item.get("id", "")))
    return {
        "manifest_version": MANIFEST_VERSION,
        "skills": entries,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(f"{payload}\n", encoding="utf-8")


def check_manifest(path: Path, manifest: dict[str, object]) -> bool:
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    return existing == manifest


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_front_matter(path: Path, schema: SchemaData) -> SkillFrontMatter:
    front = parse_front_matter(path)
    validate_front_matter(front, schema, path)
    return front


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build skills manifest")
    parser.add_argument("--skills-root", type=Path, default=DEFAULT_SKILLS_ROOT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_manifest(skills_root=args.skills_root, schema_path=args.schema)
    if args.check:
        if check_manifest(args.output, manifest):
            return 0
        print("skills.manifest.json is out of date", file=sys.stderr)
        return 1
    write_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
