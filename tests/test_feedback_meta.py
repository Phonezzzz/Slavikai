from __future__ import annotations

import os
from tempfile import NamedTemporaryFile

from memory.feedback_manager import FeedbackManager


def test_feedback_hints_meta() -> None:
    with NamedTemporaryFile(delete=False) as tmp:
        path = tmp.name
    try:
        mgr = FeedbackManager(path)
        mgr.save_feedback("p", "a", "bad", severity="major", hint="fix it")
        hints = mgr.get_recent_hints_meta(1, severity_filter=["major"])
        assert hints and hints[0]["hint"] == "fix it"
        assert hints[0]["severity"] == "major"
    finally:
        os.remove(path)
