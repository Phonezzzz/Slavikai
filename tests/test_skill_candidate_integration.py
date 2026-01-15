from __future__ import annotations

from pathlib import Path

from core.agent import SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD, Agent
from core.mwv.routing import RouteDecision
from core.skills.candidates import SkillCandidateWriter
from llm.brain_base import Brain
from llm.types import LLMResult
from shared.models import LLMMessage, ToolRequest, ToolResult


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config=None) -> LLMResult:  # type: ignore[override]
        _ = messages
        _ = config
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    agent = Agent(
        brain=DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )
    candidates_dir = tmp_path / "skills" / "_candidates"
    agent._skill_candidate_writer = SkillCandidateWriter(candidates_dir=candidates_dir)  # noqa: SLF001
    return agent


def test_unknown_request_creates_candidate(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    decision = RouteDecision(route="mwv", reason="trigger:tools", risk_flags=["tools"])
    agent._record_unknown_skill_candidate("fix tests", decision)  # noqa: SLF001
    files = list((tmp_path / "skills" / "_candidates").glob("*.md"))
    assert files


def test_unknown_request_deduped(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    decision = RouteDecision(route="mwv", reason="trigger:tools", risk_flags=["tools"])
    agent._record_unknown_skill_candidate("fix tests", decision)  # noqa: SLF001
    agent._record_unknown_skill_candidate("fix tests", decision)  # noqa: SLF001
    files = list((tmp_path / "skills" / "_candidates").glob("*.md"))
    assert len(files) == 1


def test_tool_error_candidate_after_threshold(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    request = ToolRequest(name="shell", args={})
    result = ToolResult.failure("boom")
    for _ in range(SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD):
        agent._track_tool_error(request, result)  # noqa: SLF001
    files = list((tmp_path / "skills" / "_candidates").glob("*.md"))
    assert files


def test_tool_error_candidate_deduped(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    request = ToolRequest(name="shell", args={})
    result = ToolResult.failure("boom")
    for _ in range(SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD):
        agent._track_tool_error(request, result)  # noqa: SLF001
    for _ in range(SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD):
        agent._track_tool_error(request, result)  # noqa: SLF001
    files = list((tmp_path / "skills" / "_candidates").glob("*.md"))
    assert len(files) == 1
