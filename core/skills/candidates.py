from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from core.skills.models import SkillRisk

DEFAULT_CANDIDATES_DIR = Path(__file__).resolve().parents[2] / "skills" / "_candidates"
ENV_CANDIDATES_DIR = "SKILLS_CANDIDATES_DIR"
_MAX_TEXT_LENGTH = 200
_DEFAULT_MAX_CANDIDATES = 200
_DEFAULT_TTL_DAYS = 30
_DEFAULT_COOLDOWN_SECONDS = 60 * 60
_ARCHIVE_DIR_NAME = "_archive"
_FRONT_MATTER_DELIM = "---"

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
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
        ttl_days: int = _DEFAULT_TTL_DAYS,
        cooldown_seconds: int = _DEFAULT_COOLDOWN_SECONDS,
    ) -> None:
        if candidates_dir is None:
            override = os.getenv(ENV_CANDIDATES_DIR, "")
            candidates_dir = Path(override) if override else DEFAULT_CANDIDATES_DIR
        self._candidates_dir = candidates_dir
        self._created_keys: set[str] = set()
        self._recent_fingerprints: dict[str, datetime] = {}
        self._now = now or _utc_now
        self._max_candidates = max(1, max_candidates)
        self._ttl = timedelta(days=max(1, ttl_days))
        self._cooldown_seconds = max(1, cooldown_seconds)

    def write_once(self, key: str, draft: CandidateDraft) -> Path | None:
        if key in self._created_keys:
            return None
        fingerprint = _fingerprint(draft)
        if self._is_in_cooldown(fingerprint):
            return None
        self._cleanup_candidates()
        path = self._write(draft, fingerprint)
        self._created_keys.add(key)
        self._recent_fingerprints[fingerprint] = self._now()
        return path

    def _write(self, draft: CandidateDraft, fingerprint: str) -> Path:
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self._now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(draft.title) or "candidate"
        suffix = uuid.uuid4().hex[:8]
        filename = f"{timestamp}_{slug}_{suffix}_{fingerprint[:8]}.md"
        path = self._candidates_dir / filename
        body = _render_candidate(draft, timestamp, fingerprint)
        path.write_text(body, encoding="utf-8")
        return path

    def _cleanup_candidates(self) -> None:
        self._candidates_dir.mkdir(parents=True, exist_ok=True)
        archive_dir = self._candidates_dir / _ARCHIVE_DIR_NAME
        archive_dir.mkdir(parents=True, exist_ok=True)
        now = self._now()
        cutoff = now - self._ttl
        candidates = sorted(self._candidates_dir.glob("*.md"), key=_file_mtime)
        for path in candidates:
            if _file_datetime(path) < cutoff:
                _archive_candidate(path, archive_dir)
        candidates = sorted(self._candidates_dir.glob("*.md"), key=_file_mtime)
        if len(candidates) <= self._max_candidates:
            return
        overflow = len(candidates) - self._max_candidates
        for path in candidates[:overflow]:
            _archive_candidate(path, archive_dir)

    def _is_in_cooldown(self, fingerprint: str) -> bool:
        now = self._now()
        recent = self._recent_fingerprints.get(fingerprint)
        if recent and (now - recent).total_seconds() < self._cooldown_seconds:
            return True
        for path in self._candidates_dir.glob("*.md"):
            if _read_fingerprint(path) != fingerprint:
                continue
            if (now - _file_datetime(path)).total_seconds() < self._cooldown_seconds:
                self._recent_fingerprints[fingerprint] = now
                return True
        return False


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


def _render_candidate(draft: CandidateDraft, timestamp: str, fingerprint: str) -> str:
    notes = draft.notes or []
    notes_lines = [f"- {line}" for line in notes] if notes else ["- none"]
    return "\n".join(
        [
            "---",
            f"id: candidate-{timestamp}",
            "version: 0.0.0",
            f"title: {draft.title}",
            f"risk: {draft.risk}",
            f"created_at: {timestamp}",
            f"fingerprint: {fingerprint}",
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


def _fingerprint(draft: CandidateDraft) -> str:
    payload = {
        "title": draft.title,
        "reason": draft.reason,
        "requests": draft.requests,
        "patterns": draft.patterns,
        "entrypoints": draft.entrypoints,
        "expected_behavior": draft.expected_behavior,
        "risk": draft.risk,
        "notes": draft.notes or [],
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _file_datetime(path: Path) -> datetime:
    return datetime.fromtimestamp(_file_mtime(path), UTC)


def _archive_candidate(path: Path, archive_dir: Path) -> None:
    target = archive_dir / path.name
    if target.exists():
        suffix = uuid.uuid4().hex[:6]
        target = archive_dir / f"{path.stem}_{suffix}{path.suffix}"
    try:
        path.rename(target)
    except OSError:
        return


def _read_fingerprint(path: Path) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    front = _extract_front_matter(raw)
    if front is None:
        return None
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("fingerprint")
    if not isinstance(value, str) or not value:
        return None
    return value


def _extract_front_matter(text: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIM:
        return None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FRONT_MATTER_DELIM:
            return "\n".join(lines[1:idx])
    return None


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
