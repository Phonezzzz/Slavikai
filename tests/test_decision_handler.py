from __future__ import annotations

from datetime import UTC

from core.decision.handler import DecisionContext, DecisionEvent, DecisionHandler
from core.mwv.models import VerificationResult, VerificationStatus
from core.skills.index import SkillMatch, SkillMatchDecision
from core.skills.models import SkillEntry


def _ambiguous_decision() -> SkillMatchDecision:
    entry = SkillEntry(
        id="skill.alpha",
        version="1.0.0",
        title="Alpha",
        entrypoints=["run.py"],
        patterns=["alpha"],
        requires=[],
        risk="low",
        tests=[],
        path="skills/alpha",
        content_hash="hash",
    )
    match = SkillMatch(entry=entry, pattern="alpha")
    return SkillMatchDecision(
        status="ambiguous",
        match=None,
        alternatives=[match, match],
        reason="multiple candidates",
        replaced_by=None,
    )


def test_decision_handler_builds_tool_fail_packet() -> None:
    handler = DecisionHandler()

    packet = handler.evaluate(
        event=DecisionEvent.tool_fail(
            tool_name="workspace_write",
            error_text="boom",
            count=3,
            threshold=3,
            user_input="save file",
        )
    )

    assert packet is not None
    assert packet.reason.value == "tool_fail"
    assert packet.context["tool"] == "workspace_write"
    option_actions = [option.action.value for option in packet.options]
    assert "retry" not in option_actions
    assert option_actions == ["adjust_threshold", "create_candidate", "abort"]


def test_decision_handler_builds_verifier_fail_packet() -> None:
    handler = DecisionHandler()
    result = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["pytest", "-q"],
        exit_code=1,
        stdout="",
        stderr="line one\nline two\nline three",
        duration_seconds=0.2,
    )

    packet = handler.evaluate(
        event=DecisionEvent.verifier_fail(
            verification_result=result,
            task_id="task-1",
            trace_id="trace-1",
            attempt=1,
            max_attempts=3,
            retry_allowed=True,
        )
    )

    assert packet is not None
    assert packet.reason.value == "verifier_fail"
    assert "line one" in str(packet.context["excerpt"])
    assert packet.created_at.tzinfo == UTC


def test_decision_handler_builds_ambiguous_packet_from_context() -> None:
    handler = DecisionHandler()

    packet = handler.evaluate(
        context=DecisionContext(
            user_input="run alpha",
            route="mwv",
            routing_reason="skill",
            skill_decision=_ambiguous_decision(),
        )
    )

    assert packet is not None
    assert packet.reason.value == "ambiguous_skill"
    assert packet.created_at.tzinfo == UTC
