from __future__ import annotations

from dataclasses import dataclass

from core.skills.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from core.skills.models import SkillEntry, SkillManifest


@dataclass(frozen=True)
class SkillMatch:
    entry: SkillEntry
    pattern: str


class SkillIndex:
    def __init__(self, manifest: SkillManifest) -> None:
        self._manifest = manifest
        self._by_id = {entry.id: entry for entry in manifest.skills}
        self._patterns = _build_pattern_index(manifest.skills)

    @property
    def manifest(self) -> SkillManifest:
        return self._manifest

    @property
    def by_id(self) -> dict[str, SkillEntry]:
        return dict(self._by_id)

    @classmethod
    def load_default(cls, *, dev_mode: bool | None = None) -> SkillIndex:
        manifest = load_manifest(DEFAULT_MANIFEST_PATH, dev_mode=dev_mode)
        return cls(manifest)

    def match(self, text: str) -> SkillMatch | None:
        normalized = text.strip().lower()
        if not normalized:
            return None
        for pattern, entry in self._patterns:
            if pattern in normalized:
                return SkillMatch(entry=entry, pattern=pattern)
        return None


def _build_pattern_index(skills: list[SkillEntry]) -> list[tuple[str, SkillEntry]]:
    ordered = sorted(skills, key=lambda entry: entry.id)
    indexed: list[tuple[str, SkillEntry]] = []
    for entry in ordered:
        for pattern in entry.patterns:
            normalized = pattern.strip().lower()
            if not normalized:
                continue
            indexed.append((normalized, entry))
    return indexed
