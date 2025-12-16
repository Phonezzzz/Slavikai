from __future__ import annotations

from memory.feedback_manager import FeedbackManager


def test_save_feedback_variants(tmp_path) -> None:
    from core.agent import Agent
    from llm.brain_base import Brain
    from llm.types import LLMResult, ModelConfig

    class BrainStub(Brain):
        def generate(self, messages, config: ModelConfig | None = None):  # type: ignore[override]
            return LLMResult(text="ok")

    agent = Agent(brain=BrainStub(), memory_companion_db_path=str(tmp_path / "mc.db"))
    agent.feedback = FeedbackManager(str(tmp_path / "fb.db"))

    agent.save_feedback("p", "a", "good", hint=None)
    agent.save_feedback("p2", "a2", "bad", hint="fix")
    agent.save_feedback("p3", "a3", "offtopic", hint=None)

    records = agent.feedback.get_recent_records(5)
    ratings = {r["rating"] for r in records}
    assert "good" in ratings and "bad" in ratings and "offtopic" in ratings
    majors = [r for r in records if r["severity"] in {"major", "fatal"}]
    assert majors, "major записи должны быть"
    hints = [r for r in records if r.get("hint")]
    assert any("fix" in r["hint"] for r in hints)
