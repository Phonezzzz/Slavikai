from __future__ import annotations

import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from core.skills.models import SkillRisk

DEFAULT_CANDIDATES_DIR = Path(__file__).resolve().parents[2] / "skills" / "_candidates"
ENV_CANDIDATES_DIR = "SKILLS_CANDIDATES_DIR"
_MAX_TEXT_LENGTH = 200

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"https?://\S+")
_LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
_NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class CandidateDraft:
    title: str
    reason: str
    requests: list[str]
    patterns: list[str]
    entrypoints: list[str]
    expected_behavior: list[str]
    risk: SkillRisk
    notes: list[str] | None = None


class SkillCandidateWriter:
    def __init__(
        self,
        candidates_dir: Path | None = None,
        *,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if candidates_dir is None:
            override = os.getenv(ENV_CANDIDATES_DIR, "")
            candidates_dir = Path(override) if override else DEFAULT_CANDIDATES_DIR
        self._candidates_dir = candidates_dir
        self._created_keys: set[str] = set()
        self._now = now or _utc_now

    def write_once(self, key: str, draft: CandidateDraft) -> Path | None:
        if key in self._created_keys:
            return None
        path = self._write(draft)
        self._created_keys.add(key)
        return path

    def _write(self, draft: CandidateDraft) -> Path:
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(draft.title) or "candidate"
        suffix = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{slug}_{suffix}.md"
        path = self._candidates_dir / filename
        body = _render_candidate(draft, timestamp)
        path.write_text(body, encoding="utf-8")
        return path


def sanitize_text(text: str, *, max_length: int = _MAX_TEXT_LENGTH) -> str:
    cleaned = _URL_RE.sub("[redacted_url]", text)
    cleaned = _EMAIL_RE.sub("[redacted_email]", cleaned)
    cleaned = _LONG_NUMBER_RE.sub("[redacted_number]", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "..."
    return cleaned


def suggest_patterns(text: str, *, limit: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_-]{3,}", text.lower())
    filtered = [token for token in tokens if token not in _STOPWORDS]
    seen: list[str] = []
    for token in filtered:
        if token not in seen:
            seen.append(token)
        if len(seen) >= limit:
            break
    if seen:
        return seen
    fallback = sanitize_text(text).lower()
    return [fallback] if fallback else []


def _render_candidate(draft: CandidateDraft, timestamp: str) -> str:
    notes = draft.notes or []
    notes_lines = [f"- {line}" for line in notes] if notes else ["- none"]
    return "\n".join(
        [
            "---",
            f"id: candidate-{timestamp}",
            "version: 0.0.0",
            f"title: {draft.title}",
            f"risk: {draft.risk}",
            "entrypoints:",
            *_format_list(draft.entrypoints),
            "patterns:",
            *_format_list(draft.patterns),
            "requires: []",
            "tests: []",
            "status: candidate",
            "---",
            "",
            "# Candidate Skill Draft",
            "",
            f"Reason: {draft.reason}",
            "",
            "## Observed Requests",
            *_format_list(draft.requests),
            "",
            "## Expected Behavior",
            *_format_list(draft.expected_behavior),
            "",
            "## Notes",
            *notes_lines,
            "",
        ]
    )


def _format_list(items: list[str]) -> list[str]:
    if not items:
        return ["- none"]
    return [f"- {item}" for item in items]


def _slugify(text: str) -> str:
    lowered = text.strip().lower()
    slug = _NON_SLUG_RE.sub("-", lowered).strip("-")
    return slug


def _utc_now() -> datetime:
    return datetime.now(UTC)


_STOPWORDS = {
    "about",
    "after",
    "again",
    "all",
    "also",
    "and",
    "any",
    "are",
    "before",
    "because",
    "between",
    "but",
    "can",
    "could",
    "did",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "make",
    "more",
    "most",
    "need",
    "not",
    "only",
    "other",
    "please",
    "some",
    "that",
    "then",
    "this",
    "those",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "your",
}
