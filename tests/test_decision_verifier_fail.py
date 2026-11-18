from __future__ import annotations

from core.decision.handler import DecisionEvent, DecisionHandler
from core.decision.models import DecisionAction, DecisionReason
from core.decision.verifier_fail import build_verifier_fail_packet
from core.mwv.models import VerificationResult, VerificationStatus


def test_decision_packet_on_verifier_fail() -> None:
    # E01-T08: MWV-runner больше нет; verifier_fail-пакет строится напрямую
    # из события DecisionHandler (тот же путь, что у auto/plan loop).
    handler = DecisionHandler()
    packet = handler.evaluate(
        event=DecisionEvent.verifier_fail(
            verification_result=VerificationResult(
                status=VerificationStatus.FAILED,
                command=["check"],
                exit_code=1,
                stdout="",
                stderr="tests failed",
                duration_seconds=0.1,
                error=None,
            ),
            task_id="t1",
            trace_id="trace-1",
            attempt=1,
            max_attempts=3,
            retry_allowed=False,
        )
    )
    assert packet is not None
    assert packet.reason == DecisionReason.VERIFIER_FAIL
    assert 3 <= len(packet.options) <= 5
    actions = {option.action for option in packet.options}
    assert DecisionAction.RETRY not in actions
    assert {DecisionAction.ASK_USER, DecisionAction.PROCEED_SAFE, DecisionAction.ABORT} <= actions


def test_build_verifier_fail_packet_respects_retry_flag() -> None:
    result = VerificationResult(
        status=VerificationStatus.FAILED,
        command=["check"],
        exit_code=1,
        stdout="line1\nline2\nline3\nline4",
        stderr="tests failed",
        duration_seconds=0.1,
        error=None,
    )

    no_retry = build_verifier_fail_packet(
        result,
        task_id="t1",
        trace_id="trace-1",
        attempt=1,
        max_attempts=3,
        retry_allowed=False,
    )
    assert DecisionAction.RETRY not in [option.action for option in no_retry.options]

    with_retry = build_verifier_fail_packet(
        result,
        task_id="t1",
        trace_id="trace-1",
        attempt=1,
        max_attempts=3,
        retry_allowed=True,
    )
    retry_actions = [option.action for option in with_retry.options]
    assert retry_actions.count(DecisionAction.RETRY) == 1
