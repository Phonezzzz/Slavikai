from __future__ import annotations

from memory.feedback_manager import FeedbackManager


def test_feedback_saves_severity_and_hint(tmp_path) -> None:
    db = tmp_path / "fb.db"
    mgr = FeedbackManager(str(db))
    mgr.save_feedback("q", "a", "bad", severity="major", hint="avoid mistakes")
    hints = mgr.get_recent_hints(1)
    assert hints == ["avoid mistakes"]
    trends = mgr.analyze_trends()
    assert trends.get("bad") == 1.0


def test_feedback_stats_and_top_hints(tmp_path) -> None:
    db = tmp_path / "fb2.db"
    mgr = FeedbackManager(str(db))
    mgr.save_feedback("q1", "a1", "good", severity="minor")
    mgr.save_feedback("q2", "a2", "bad", severity="major", hint="fix facts")
    mgr.save_feedback("q3", "a3", "offtopic", severity="fatal", hint="stay on topic")
    mgr.save_feedback("q4", "a4", "bad", severity="major", hint="fix facts")
    stats = mgr.stats()
    assert stats["ratings"]["bad"] == 2
    assert stats["severity"]["major"] == 2
    assert stats["severity"]["fatal"] == 1
    top_hints = stats["top_hints"]
    assert top_hints[0]["hint"] == "fix facts"
    assert top_hints[0]["count"] == 2
    bad_records = mgr.get_recent_bad(5)
    assert len(bad_records) == 3
    assert all(rec["rating"] in {"bad", "offtopic"} for rec in bad_records)
