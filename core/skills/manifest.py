from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core.skills.models import SkillEntry, SkillManifest, SkillRisk

DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parents[2] / "skills"
DEFAULT_MANIFEST_PATH = DEFAULT_SKILLS_ROOT / "_generated" / "skills.manifest.json"
DEFAULT_BUILD_SCRIPT = DEFAULT_SKILLS_ROOT / "tools" / "build_manifest.py"


class SkillManifestError(RuntimeError):
    pass


def load_manifest(
    path: Path | None = None,
    *,
    dev_mode: bool | None = None,
) -> SkillManifest:
    manifest_path = path or DEFAULT_MANIFEST_PATH
    if dev_mode is None:
        dev_mode = os.getenv("SKILLS_DEV_MODE", "").lower() in {"1", "true", "yes"}
    if dev_mode:
        _rebuild_manifest(manifest_path)
    if not manifest_path.exists():
        raise SkillManifestError(f"Skills manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return _parse_manifest(data)


def _rebuild_manifest(manifest_path: Path) -> None:
    repo_root = manifest_path.parents[2]
    if not DEFAULT_BUILD_SCRIPT.exists():
        raise SkillManifestError("build_manifest.py not found")
    result = subprocess.run(
        [sys.executable, str(DEFAULT_BUILD_SCRIPT)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise SkillManifestError(f"Failed to rebuild skills manifest: {message}")


def _parse_manifest(data: object) -> SkillManifest:
    if not isinstance(data, dict):
        raise SkillManifestError("Manifest payload must be an object")
    manifest_version = data.get("manifest_version")
    if not isinstance(manifest_version, int):
        raise SkillManifestError("manifest_version must be an integer")
    raw_skills = data.get("skills")
    if not isinstance(raw_skills, list):
        raise SkillManifestError("skills must be a list")
    skills: list[SkillEntry] = []
    for raw in raw_skills:
        if not isinstance(raw, dict):
            raise SkillManifestError("skill entry must be an object")
        skills.append(_parse_entry(raw))
    return SkillManifest(manifest_version=manifest_version, skills=skills)


def _parse_entry(raw: dict[str, object]) -> SkillEntry:
    skill_id = _require_str(raw, "id")
    version = _require_str(raw, "version")
    title = _require_str(raw, "title")
    entrypoints = _require_str_list(raw, "entrypoints")
    patterns = _require_str_list(raw, "patterns")
    requires = _require_str_list(raw, "requires")
    risk = _require_risk(raw, "risk")
    tests = _require_str_list(raw, "tests")
    path = _require_str(raw, "path")
    content_hash = _require_str(raw, "content_hash")
    deprecated = _optional_bool(raw, "deprecated", False)
    replaced_by = _optional_str(raw, "replaced_by")
    return SkillEntry(
        id=skill_id,
        version=version,
        title=title,
        entrypoints=entrypoints,
        patterns=patterns,
        requires=requires,
        risk=risk,
        tests=tests,
        path=path,
        content_hash=content_hash,
        deprecated=deprecated,
        replaced_by=replaced_by,
    )


def _require_str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SkillManifestError(f"{key} must be a non-empty string")
    return value


def _require_str_list(raw: dict[str, object], key: str) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise SkillManifestError(f"{key} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise SkillManifestError(f"{key} must be a list of strings")
    return [item for item in value if item]


def _require_risk(raw: dict[str, object], key: str) -> SkillRisk:
    value = raw.get(key)
    if value == "low":
        return "low"
    if value == "medium":
        return "medium"
    if value == "high":
        return "high"
    raise SkillManifestError("risk must be low, medium, or high")


def _optional_bool(raw: dict[str, object], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise SkillManifestError(f"{key} must be boolean")
    return value


def _optional_str(raw: dict[str, object], key: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SkillManifestError(f"{key} must be a non-empty string")
    return value
