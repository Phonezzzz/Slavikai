"""E01-T05: canonical structured outcome (ADR-0002, not a rewrite of it).

Acceptance criteria (ROADMAP.md E01-T05):
(a) a state-changing dummy task without a confirmed postcondition cannot
    become full DONE;
(b) the mechanism separately preserves terminal reason and disposition for
    aborted+none, failed+accepted_partial, cancelled+accepted_partial and
    completed+accepted_partial; final_partial only for completed+accepted_partial;
(c) non-terminal blocked/waiting_approval are never presented as completion;
(d) the UI projection is fully verified (exhaustive matrix).

Negative criteria:
- tool.ok=true alone is not proof of completion;
- unknown verification becomes neither full success nor automatic failure
  and never rewrites terminal_reason.

Fail-on-old-code: the old code had no canonical outcome at all — the
user-visible outcome was prose derived ad hoc from work status +
verification status, and plan steps became DONE from a single tool.ok.
The naive mapping "completed + full -> DONE" (ignoring verification)
fails test_state_changing_task_without_confirmed_postcondition_is_not_done
and test_tool_ok_alone_is_not_proof_of_completion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.agent import Agent
from core.mwv.manager import MWVRunResult
from core.mwv.models import (
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkResult,
    WorkStatus,
)
from core.run_outcome import (
    Disposition,
    RunOutcome,
    RunPhase,
    TerminalReason,
    Verification,
    VerificationOutcome,
    outcome_from_loop_result,
    outcome_from_mwv,
)
from core.tool_loop import AgentToolLoopResult, ExecutedToolCall
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig, ToolCall
from shared.models import LLMMessage, ToolResult


def _passed_verification() -> VerificationResult:
    return VerificationResult(
        status=VerificationStatus.PASSED,
        command=["check"],
        exit_code=0,
        stdout="ok",
        stderr="",
        duration_seconds=0.1,
    )


def _verification(status: VerificationStatus) -> VerificationResult:
    return VerificationResult(
        status=status,
        command=["check"],
        exit_code=1 if status is VerificationStatus.FAILED else None,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        error="boom" if status is VerificationStatus.ERROR else None,
        excerpt="boom" if status is VerificationStatus.ERROR else None,
    )


def _outcome(
    reason: TerminalReason,
    disposition: Disposition,
    verification: VerificationOutcome = VerificationOutcome.PASSED,
    *,
    state_changing: bool = True,
    accepted_by: str | None = None,
) -> RunOutcome:
    return RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=reason,
        disposition=disposition,
        verification=Verification(outcome=verification),
        state_changing=state_changing,
        accepted_by=accepted_by,
    )


# --- Invariants: invalid terminal combinations cannot be constructed. ---


@pytest.mark.parametrize(
    ("reason", "disposition"),
    [
        (TerminalReason.COMPLETED, Disposition.NONE),
        (TerminalReason.FAILED, Disposition.FULL),
        (TerminalReason.CANCELLED, Disposition.FULL),
        (TerminalReason.ABORTED, Disposition.FULL),
    ],
)
def test_invalid_terminal_combinations_rejected(
    reason: TerminalReason, disposition: Disposition
) -> None:
    with pytest.raises(ValueError):
        _outcome(reason, disposition)


def test_terminal_phase_requires_reason_and_vice_versa() -> None:
    with pytest.raises(ValueError):
        RunOutcome(
            phase=RunPhase.TERMINAL,
            disposition=Disposition.NONE,
            verification=Verification.unknown(),
        )
    with pytest.raises(ValueError):
        RunOutcome(
            phase=RunPhase.RUNNING,
            terminal_reason=TerminalReason.COMPLETED,
            disposition=Disposition.FULL,
            verification=Verification.unknown(),
        )


def test_accepted_partial_requires_authority() -> None:
    with pytest.raises(ValueError):
        _outcome(TerminalReason.COMPLETED, Disposition.ACCEPTED_PARTIAL)
    with pytest.raises(ValueError):
        _outcome(
            TerminalReason.FAILED,
            Disposition.NONE,
            accepted_by="user",
        )
    outcome = _outcome(TerminalReason.COMPLETED, Disposition.ACCEPTED_PARTIAL, accepted_by="user")
    assert outcome.accepted_by == "user"


# --- ADR-0002 §3 mapping: one final per terminal cause + disposition. ---


@pytest.mark.parametrize(
    ("reason", "disposition", "expected"),
    [
        (TerminalReason.COMPLETED, Disposition.FULL, "final_success"),
        (TerminalReason.COMPLETED, Disposition.ACCEPTED_PARTIAL, "final_partial"),
        (TerminalReason.FAILED, Disposition.NONE, "final_failure"),
        (TerminalReason.FAILED, Disposition.ACCEPTED_PARTIAL, "final_failure"),
        (TerminalReason.CANCELLED, Disposition.NONE, "final_cancelled"),
        (TerminalReason.CANCELLED, Disposition.ACCEPTED_PARTIAL, "final_cancelled"),
        (TerminalReason.ABORTED, Disposition.NONE, "final_aborted"),
        (TerminalReason.ABORTED, Disposition.ACCEPTED_PARTIAL, "final_aborted"),
    ],
)
def test_communication_class_mapping(
    reason: TerminalReason, disposition: Disposition, expected: str
) -> None:
    accepted_by = "user" if disposition is Disposition.ACCEPTED_PARTIAL else None
    outcome = _outcome(reason, disposition, accepted_by=accepted_by)
    assert outcome.communication_class == expected


def test_final_partial_only_for_completed_accepted_partial() -> None:
    for reason in TerminalReason:
        for disposition in Disposition:
            try:
                outcome = _outcome(
                    reason,
                    disposition,
                    accepted_by="user" if disposition is Disposition.ACCEPTED_PARTIAL else None,
                )
            except ValueError:
                continue
            if (reason, disposition) == (
                TerminalReason.COMPLETED,
                Disposition.ACCEPTED_PARTIAL,
            ):
                assert outcome.communication_class == "final_partial"
            else:
                assert outcome.communication_class != "final_partial"


def test_reason_and_disposition_preserved_separately() -> None:
    cases = [
        (TerminalReason.ABORTED, Disposition.NONE),
        (TerminalReason.FAILED, Disposition.ACCEPTED_PARTIAL),
        (TerminalReason.CANCELLED, Disposition.ACCEPTED_PARTIAL),
        (TerminalReason.COMPLETED, Disposition.ACCEPTED_PARTIAL),
    ]
    for reason, disposition in cases:
        outcome = _outcome(
            reason,
            disposition,
            accepted_by="user" if disposition is Disposition.ACCEPTED_PARTIAL else None,
        )
        assert outcome.terminal_reason is reason
        assert outcome.disposition is disposition
        assert outcome.describe().startswith(f"{reason.value} + {disposition.value}")


# --- Headline acceptance: no full DONE without a confirmed postcondition. ---


def test_state_changing_task_without_confirmed_postcondition_is_not_done() -> None:
    # Verifier errored: the check did not run, so the postcondition was
    # not disproved — but the full claim stays UNVERIFIED, no final is
    # published, and the completed reason is preserved (not rewritten).
    error_outcome = outcome_from_mwv(
        WorkStatus.SUCCESS, _verification(VerificationStatus.ERROR), state_changing=True
    )
    assert error_outcome.terminal_reason is TerminalReason.COMPLETED
    assert error_outcome.disposition is Disposition.FULL
    assert error_outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert error_outcome.is_full_done is False
    assert error_outcome.ui_label == "UNVERIFIED"
    assert error_outcome.communication_class is None

    # Verifier ran and failed: the check disproved the claimed result,
    # so the run failed its acceptance — it is FAILED, not completed.
    failed_outcome = outcome_from_mwv(
        WorkStatus.SUCCESS, _verification(VerificationStatus.FAILED), state_changing=True
    )
    assert failed_outcome.terminal_reason is TerminalReason.FAILED
    assert failed_outcome.disposition is Disposition.NONE
    assert failed_outcome.is_full_done is False
    assert failed_outcome.ui_label == "FAILED"
    assert failed_outcome.communication_class == "final_failure"


def test_state_changing_task_with_passed_postcondition_is_done() -> None:
    outcome = outcome_from_mwv(WorkStatus.SUCCESS, _passed_verification(), state_changing=True)
    assert outcome.is_full_done is True
    assert outcome.ui_label == "DONE"
    assert outcome.communication_class == "final_success"


def test_mwv_success_passed_with_accepted_partial() -> None:
    # Same rule as the loop converter (fix 22a8a46): an accepted partial
    # result must not be upgraded to full DONE by the MWV converter.
    outcome = outcome_from_mwv(
        WorkStatus.SUCCESS, _passed_verification(), state_changing=True, accepted_by="user"
    )
    assert outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.disposition is Disposition.ACCEPTED_PARTIAL
    assert outcome.accepted_by == "user"
    assert outcome.is_full_done is False
    assert outcome.ui_label == "PARTIAL"
    assert outcome.communication_class == "final_partial"


def test_tool_ok_alone_is_not_proof_of_completion() -> None:
    call = ToolCall(id="write-0", name="write_file", arguments={})
    loop_result = AgentToolLoopResult(
        text="done",
        messages=[],
        tool_calls=[ExecutedToolCall(call=call, result=ToolResult.success({"output": "ok"}))],
        iterations=1,
    )
    outcome = outcome_from_loop_result(loop_result, state_changing=True)
    assert outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.is_full_done is False
    assert outcome.ui_label == "UNVERIFIED"
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN


def test_non_state_changing_task_may_complete_without_postcondition() -> None:
    call = ToolCall(id="ask-0", name="answer", arguments={})
    loop_result = AgentToolLoopResult(
        text="answer",
        messages=[],
        tool_calls=[ExecutedToolCall(call=call, result=ToolResult.success({}))],
        iterations=1,
    )
    outcome = outcome_from_loop_result(loop_result, state_changing=False)
    assert outcome.is_full_done is True
    assert outcome.ui_label == "DONE"


# --- Negative: unknown verification never rewrites the terminal reason. ---


@pytest.mark.parametrize(
    ("reason", "disposition", "label"),
    [
        (TerminalReason.FAILED, Disposition.NONE, "FAILED"),
        (TerminalReason.FAILED, Disposition.ACCEPTED_PARTIAL, "FAILED"),
        (TerminalReason.CANCELLED, Disposition.ACCEPTED_PARTIAL, "CANCELLED"),
        (TerminalReason.ABORTED, Disposition.NONE, "ABORTED"),
    ],
)
def test_unknown_verification_preserves_terminal_reason(
    reason: TerminalReason, disposition: Disposition, label: str
) -> None:
    outcome = _outcome(
        reason,
        disposition,
        VerificationOutcome.UNKNOWN,
        accepted_by="user" if disposition is Disposition.ACCEPTED_PARTIAL else None,
    )
    assert outcome.terminal_reason is reason
    assert outcome.disposition is disposition
    assert outcome.is_full_done is False
    assert outcome.ui_label == label


def test_unknown_verification_is_not_automatic_failure() -> None:
    outcome = _outcome(TerminalReason.COMPLETED, Disposition.FULL, VerificationOutcome.UNKNOWN)
    assert outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.ui_label == "UNVERIFIED"
    # No final_success is published for an unconfirmed full claim, and
    # unknown verification never turns into a failure final either.
    assert outcome.communication_class is None


# --- Non-terminal phases are never presented as completion. ---


@pytest.mark.parametrize(
    ("phase", "label"),
    [
        (RunPhase.RUNNING, "RUNNING"),
        (RunPhase.BLOCKED, "BLOCKED"),
        (RunPhase.WAITING_USER_INPUT, "WAITING"),
        (RunPhase.WAITING_APPROVAL, "WAITING"),
    ],
)
def test_non_terminal_never_presented_as_completion(phase: RunPhase, label: str) -> None:
    outcome = RunOutcome(
        phase=phase,
        disposition=Disposition.NONE,
        verification=Verification.unknown(),
    )
    assert outcome.ui_label == label
    assert outcome.communication_class is None
    assert outcome.is_full_done is False


# --- UI projection: exhaustive matrix, DONE iff is_full_done. ---


def test_ui_projection_matrix_is_total_and_done_iff_full_done() -> None:
    labels = {
        "RUNNING",
        "BLOCKED",
        "WAITING",
        "DONE",
        "PARTIAL",
        "FAILED",
        "CANCELLED",
        "ABORTED",
        "UNVERIFIED",
    }
    seen: set[str] = set()
    for phase in RunPhase:
        for reason in list(TerminalReason) + [None]:
            for disposition in Disposition:
                for verification in VerificationOutcome:
                    for state_changing in (False, True):
                        try:
                            outcome = RunOutcome(
                                phase=phase,
                                terminal_reason=reason,
                                disposition=disposition,
                                verification=Verification(outcome=verification),
                                state_changing=state_changing,
                                accepted_by="user"
                                if disposition is Disposition.ACCEPTED_PARTIAL
                                else None,
                            )
                        except ValueError:
                            continue
                        assert outcome.ui_label in labels
                        seen.add(outcome.ui_label)
                        assert (outcome.ui_label == "DONE") == outcome.is_full_done
    assert seen == labels


def test_completed_accepted_partial_is_partial_never_done() -> None:
    outcome = _outcome(
        TerminalReason.COMPLETED,
        Disposition.ACCEPTED_PARTIAL,
        VerificationOutcome.PASSED,
        accepted_by="user",
    )
    assert outcome.ui_label == "PARTIAL"
    assert outcome.is_full_done is False
    assert outcome.communication_class == "final_partial"


# --- Verification mapping keeps evidence. ---


def test_verification_from_verification_result() -> None:
    passed = Verification.from_verification_result(_passed_verification())
    assert passed.outcome is VerificationOutcome.PASSED
    assert passed.checks[0].passed is True
    assert "check" in passed.source

    failed = Verification.from_verification_result(_verification(VerificationStatus.FAILED))
    assert failed.outcome is VerificationOutcome.FAILED
    assert failed.checks[0].passed is False
    assert "exit_code=1" in failed.checks[0].detail

    error = Verification.from_verification_result(_verification(VerificationStatus.ERROR))
    assert error.outcome is VerificationOutcome.UNKNOWN
    assert error.checks[0].passed is False
    assert "boom" in error.checks[0].detail


# --- Derivation from loop signals. ---


def test_outcome_from_loop_result_cancelled_and_error() -> None:
    cancelled = AgentToolLoopResult(text="", messages=[], cancelled=True)
    outcome = outcome_from_loop_result(cancelled, state_changing=True)
    assert outcome.terminal_reason is TerminalReason.CANCELLED
    assert outcome.disposition is Disposition.NONE
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert outcome.ui_label == "CANCELLED"

    errored = AgentToolLoopResult(text="", messages=[], error="provider down")
    outcome = outcome_from_loop_result(errored, state_changing=True)
    assert outcome.terminal_reason is TerminalReason.FAILED
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert outcome.ui_label == "FAILED"


def test_outcome_from_loop_result_cancelled_with_accepted_partial() -> None:
    cancelled = AgentToolLoopResult(text="", messages=[], cancelled=True)
    outcome = outcome_from_loop_result(cancelled, state_changing=True, accepted_by="user")
    assert outcome.terminal_reason is TerminalReason.CANCELLED
    assert outcome.disposition is Disposition.ACCEPTED_PARTIAL
    assert outcome.communication_class == "final_cancelled"
    assert outcome.ui_label == "CANCELLED"


def test_outcome_from_loop_result_clean_finish_with_accepted_partial() -> None:
    # Codex review (PR #159, P2): the clean-finish branch dropped
    # accepted_by and always returned FULL, so an accepted partial
    # became DONE / final_success instead of PARTIAL / final_partial.
    loop_result = AgentToolLoopResult(text="partial", messages=[])
    outcome = outcome_from_loop_result(loop_result, state_changing=False, accepted_by="user")
    assert outcome.terminal_reason is TerminalReason.COMPLETED
    assert outcome.disposition is Disposition.ACCEPTED_PARTIAL
    assert outcome.accepted_by == "user"
    assert outcome.ui_label == "PARTIAL"
    assert outcome.communication_class == "final_partial"
    assert outcome.is_full_done is False

    # State-changing: still PARTIAL, never DONE — the user accepted the
    # partial result, and unknown verification does not rewrite it.
    outcome = outcome_from_loop_result(loop_result, state_changing=True, accepted_by="user")
    assert outcome.disposition is Disposition.ACCEPTED_PARTIAL
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert outcome.ui_label == "PARTIAL"
    assert outcome.communication_class == "final_partial"


def test_outcome_from_loop_result_policy_stop_is_aborted_not_failed() -> None:
    # Codex review (PR #159, P2): tool_policy_denied is an administrative
    # stop (ADR-0002 §1), not a subject failure — it must map to ABORTED.
    denied = AgentToolLoopResult(text="", messages=[], error="tool_policy_denied")
    outcome = outcome_from_loop_result(denied, state_changing=True)
    assert outcome.terminal_reason is TerminalReason.ABORTED
    assert outcome.disposition is Disposition.NONE
    assert outcome.verification.outcome is VerificationOutcome.UNKNOWN
    assert outcome.ui_label == "ABORTED"
    assert outcome.communication_class == "final_aborted"
    assert "tool_policy_denied" in outcome.verification.checks[0].detail

    # With an acceptance authority, the disposition is preserved too.
    outcome = outcome_from_loop_result(denied, state_changing=True, accepted_by="user")
    assert outcome.terminal_reason is TerminalReason.ABORTED
    assert outcome.disposition is Disposition.ACCEPTED_PARTIAL
    assert outcome.accepted_by == "user"

    # Any other loop error is still a subject failure.
    errored = AgentToolLoopResult(text="", messages=[], error="provider down")
    outcome = outcome_from_loop_result(errored, state_changing=True)
    assert outcome.terminal_reason is TerminalReason.FAILED
    assert outcome.ui_label == "FAILED"


# --- Production wiring: MWV response label derives from the canonical outcome. ---


class _DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="ok")


def _make_agent(tmp_path: Path) -> Agent:
    return Agent(
        brain=_DummyBrain(),
        memory_companion_db_path=str(tmp_path / "mc.db"),
        memory_inbox_db_path=str(tmp_path / "inbox.db"),
    )


def _mwv_result(work: WorkStatus, verification: VerificationStatus) -> MWVRunResult:
    task = TaskPacket(task_id="t", session_id="s", trace_id="trace", goal="g")
    return MWVRunResult(
        task=task,
        work_result=WorkResult(task_id="t", status=work, summary="summary"),
        verification_result=_verification(verification)
        if verification is not VerificationStatus.PASSED
        else _passed_verification(),
        attempt=1,
        max_attempts=3,
        retry_decision=None,
    )


def test_mwv_outcome_label_is_canonical_projection(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    ok = _mwv_result(WorkStatus.SUCCESS, VerificationStatus.PASSED)
    assert agent._mwv_outcome_label(ok) == "проверки пройдены"
    failed_work = _mwv_result(WorkStatus.FAILURE, VerificationStatus.PASSED)
    assert agent._mwv_outcome_label(failed_work) == "ошибка выполнения"
    unverified = _mwv_result(WorkStatus.SUCCESS, VerificationStatus.ERROR)
    assert agent._mwv_outcome_label(unverified) == "результат не подтверждён проверкой"
    failed_checks = _mwv_result(WorkStatus.SUCCESS, VerificationStatus.FAILED)
    assert agent._mwv_outcome_label(failed_checks) == "ошибка выполнения"


def test_mwv_response_keeps_pinned_success_line(tmp_path: Path) -> None:
    agent = _make_agent(tmp_path)
    ok = _mwv_result(WorkStatus.SUCCESS, VerificationStatus.PASSED)
    assert "Итог: проверки пройдены" in agent._format_mwv_response(ok)
