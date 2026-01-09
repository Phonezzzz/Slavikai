from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.skills.manifest import DEFAULT_MANIFEST_PATH, load_manifest
from core.skills.models import SkillEntry, SkillManifest


@dataclass(frozen=True)
class SkillMatch:
    entry: SkillEntry
    pattern: str


SkillDecisionStatus = Literal["matched", "no_match", "deprecated", "ambiguous"]


@dataclass(frozen=True)
class SkillMatchDecision:
    status: SkillDecisionStatus
    match: SkillMatch | None
    alternatives: list[SkillMatch]
    reason: str
    replaced_by: str | None = None


@dataclass(frozen=True)
class _PatternCandidate:
    entry: SkillEntry
    pattern: str
    score: int


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
        decision = self.match_decision(text)
        if decision.status == "matched":
            return decision.match
        return None

    def match_decision(self, text: str) -> SkillMatchDecision:
        normalized = text.strip().lower()
        if not normalized:
            return SkillMatchDecision(
                status="no_match",
                match=None,
                alternatives=[],
                reason="empty_text",
            )
        candidates = _collect_candidates(normalized, self._patterns)
        if not candidates:
            return SkillMatchDecision(
                status="no_match",
                match=None,
                alternatives=[],
                reason="no_pattern_match",
            )

        best_by_entry: dict[str, _PatternCandidate] = {}
        for candidate in candidates:
            current = best_by_entry.get(candidate.entry.id)
            if current is None or candidate.score > current.score:
                best_by_entry[candidate.entry.id] = candidate

        ordered = sorted(best_by_entry.values(), key=lambda item: item.entry.id)
        top_score = max(item.score for item in ordered)
        top = [item for item in ordered if item.score == top_score]
        active = [item for item in top if not item.entry.deprecated]

        if active:
            if len(active) == 1:
                match = SkillMatch(entry=active[0].entry, pattern=active[0].pattern)
                return SkillMatchDecision(
                    status="matched",
                    match=match,
                    alternatives=[],
                    reason="skill_match",
                )
            alternatives = [SkillMatch(entry=item.entry, pattern=item.pattern) for item in active]
            return SkillMatchDecision(
                status="ambiguous",
                match=None,
                alternatives=alternatives,
                reason="ambiguous_score",
            )

        alternatives = [SkillMatch(entry=item.entry, pattern=item.pattern) for item in top]
        if len(top) == 1:
            match = alternatives[0]
            return SkillMatchDecision(
                status="deprecated",
                match=match,
                alternatives=[],
                reason="deprecated",
                replaced_by=match.entry.replaced_by,
            )
        return SkillMatchDecision(
            status="ambiguous",
            match=None,
            alternatives=alternatives,
            reason="ambiguous_score",
        )


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


def _collect_candidates(
    text: str,
    patterns: list[tuple[str, SkillEntry]],
) -> list[_PatternCandidate]:
    candidates: list[_PatternCandidate] = []
    for pattern, entry in patterns:
        if pattern in text:
            candidates.append(_PatternCandidate(entry=entry, pattern=pattern, score=len(pattern)))
    return candidates
