from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SkillRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class SkillEntry:
    id: str
    version: str
    title: str
    entrypoints: list[str]
    patterns: list[str]
    requires: list[str]
    risk: SkillRisk
    tests: list[str]
    path: str
    content_hash: str
    deprecated: bool = False
    replaced_by: str | None = None


@dataclass(frozen=True)
class SkillManifest:
    manifest_version: int
    skills: list[SkillEntry] = field(default_factory=list)
