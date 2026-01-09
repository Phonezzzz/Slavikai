from __future__ import annotations

from datetime import UTC, datetime
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
