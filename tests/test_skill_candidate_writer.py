from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from core.skills.candidates import CandidateDraft, SkillCandidateWriter, sanitize_text


def test_sanitize_text_redacts() -> None:
    text = "email test@example.com and https://example.com id 123456"
    cleaned = sanitize_text(text)
    assert "test@example.com" not in cleaned
    assert "https://example.com" not in cleaned
    assert "123456" not in cleaned


def test_candidate_writer_creates_file(tmp_path: Path) -> None:
    fixed_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    candidates_dir = tmp_path / "skills" / "_candidates"
    writer = SkillCandidateWriter(candidates_dir=candidates_dir, now=lambda: fixed_time)
    draft = CandidateDraft(
        title="Sample skill",
        reason="unknown_request",
        requests=["request"],
        patterns=["sample"],
        entrypoints=["unknown"],
        expected_behavior=["do something"],
        risk="low",
        notes=["note"],
    )
    path = writer.write_once("key", draft)
    assert path is not None
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "Sample skill" in content

    second = writer.write_once("key", draft)
    assert second is None


def test_candidate_writer_cooldown(tmp_path: Path) -> None:
    fixed_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    candidates_dir = tmp_path / "skills" / "_candidates"
    draft = CandidateDraft(
        title="Sample skill",
        reason="unknown_request",
        requests=["request"],
        patterns=["sample"],
        entrypoints=["unknown"],
        expected_behavior=["do something"],
        risk="low",
        notes=None,
    )
    writer = SkillCandidateWriter(
        candidates_dir=candidates_dir,
        now=lambda: fixed_time,
        cooldown_seconds=3600,
    )
    assert writer.write_once("key-a", draft) is not None
    second_writer = SkillCandidateWriter(
        candidates_dir=candidates_dir,
        now=lambda: fixed_time,
        cooldown_seconds=3600,
    )
    assert second_writer.write_once("key-b", draft) is None


def test_candidate_writer_archives_on_ttl_and_limit(tmp_path: Path) -> None:
    fixed_time = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
    candidates_dir = tmp_path / "skills" / "_candidates"
    writer = SkillCandidateWriter(
        candidates_dir=candidates_dir,
        now=lambda: fixed_time,
        max_candidates=1,
        ttl_days=1,
        cooldown_seconds=1,
    )
    draft = CandidateDraft(
        title="Sample skill",
        reason="unknown_request",
        requests=["request"],
        patterns=["sample"],
        entrypoints=["unknown"],
        expected_behavior=["do something"],
        risk="low",
        notes=None,
    )
    first = writer.write_once("key-a", draft)
    assert first is not None
    old_time = fixed_time - timedelta(days=2)
    os.utime(first, (old_time.timestamp(), old_time.timestamp()))

    later_time = fixed_time + timedelta(hours=2)
    writer_later = SkillCandidateWriter(
        candidates_dir=candidates_dir,
        now=lambda: later_time,
        max_candidates=1,
        ttl_days=1,
        cooldown_seconds=1,
    )
    second = writer_later.write_once("key-b", draft)
    assert second is not None
    remaining = list(candidates_dir.glob("*.md"))
    archive = list((candidates_dir / "_archive").glob("*.md"))
    assert len(remaining) == 1
    assert archive
