from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jsonschema
import yaml

SchemaData = dict[str, object]


@dataclass(frozen=True)
class SkillFrontMatter:
    id: str
    version: str
    title: str
    entrypoints: list[str]
    patterns: list[str]
    requires: list[str]
    risk: str
    tests: list[str]
    deprecated: bool
    replaced_by: str | None


def iter_skill_files(skills_root: Path) -> list[Path]:
    skill_paths: list[Path] = []
    for path in skills_root.rglob("skill.md"):
        if _is_ignored_path(path):
            continue
        skill_paths.append(path)
    return sorted(skill_paths)


def load_schema(schema_path: Path) -> SchemaData:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    data = schema_path.read_text(encoding="utf-8")
    raw_schema = yaml.safe_load(data)
    if not isinstance(raw_schema, dict):
        raise ValueError("Schema must be a JSON object")
    return _normalize_keys(raw_schema, schema_path)


def parse_front_matter(path: Path) -> SkillFrontMatter:
    raw = path.read_text(encoding="utf-8")
    front_text = _extract_front_matter(raw, path)
    front = yaml.safe_load(front_text)
    if not isinstance(front, dict):
        raise ValueError(f"Front matter must be a mapping: {path}")
    normalized = _normalize_keys(front, path)
    return _coerce_front_matter(normalized, path)


def validate_front_matter(front: SkillFrontMatter, schema: SchemaData, path: Path) -> None:
    payload: dict[str, object] = {
        "id": front.id,
        "version": front.version,
        "title": front.title,
        "entrypoints": front.entrypoints,
        "patterns": front.patterns,
        "requires": front.requires,
        "risk": front.risk,
        "tests": front.tests,
        "deprecated": front.deprecated,
    }
    if front.replaced_by is not None:
        payload["replaced_by"] = front.replaced_by
    jsonschema.validate(instance=payload, schema=schema)


def _extract_front_matter(text: str, path: Path) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"Front matter must start at top of file: {path}")
    end_index = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_index = idx
            break
    if end_index is None:
        raise ValueError(f"Front matter closing marker not found: {path}")
    return "\n".join(lines[1:end_index])


def _normalize_keys(raw: dict[object, object], path: Path) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ValueError(f"Front matter keys must be strings: {path}")
        normalized[key] = value
    return normalized


def _coerce_front_matter(front: dict[str, object], path: Path) -> SkillFrontMatter:
    return SkillFrontMatter(
        id=_require_str(front, "id", path),
        version=_require_str(front, "version", path),
        title=_require_str(front, "title", path),
        entrypoints=_require_str_list(front, "entrypoints", path),
        patterns=_require_str_list(front, "patterns", path),
        requires=_optional_str_list(front, "requires", path),
        risk=_require_str(front, "risk", path),
        tests=_optional_str_list(front, "tests", path),
        deprecated=_optional_bool(front, "deprecated", path, False),
        replaced_by=_optional_str(front, "replaced_by", path),
    )


def _require_str(front: dict[str, object], key: str, path: Path) -> str:
    value = front.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string: {path}")
    return value


def _require_str_list(front: dict[str, object], key: str, path: Path) -> list[str]:
    value = front.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} must be a non-empty list: {path}")
    items = [item for item in value if isinstance(item, str) and item]
    if len(items) != len(value):
        raise ValueError(f"{key} must contain only strings: {path}")
    return items


def _optional_str_list(front: dict[str, object], key: str, path: Path) -> list[str]:
    value = front.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list: {path}")
    items = [item for item in value if isinstance(item, str) and item]
    if len(items) != len(value):
        raise ValueError(f"{key} must contain only strings: {path}")
    return items


def _optional_bool(front: dict[str, object], key: str, path: Path, default: bool) -> bool:
    value = front.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean: {path}")
    return value


def _optional_str(front: dict[str, object], key: str, path: Path) -> str | None:
    value = front.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string: {path}")
    return value


def _is_ignored_path(path: Path) -> bool:
    parts = set(path.parts)
    if "_candidates" in parts or "_generated" in parts:
        return True
    if "tools" in parts or "_schema" in parts:
        return True
    return False
