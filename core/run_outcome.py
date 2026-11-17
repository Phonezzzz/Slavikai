"""Canonical structured run outcome (E01-T05).

Implements the ADR-0002 mapping without rewriting ADR-0002
(docs/architecture/decisions/ADR-0002-terminal-outcome-and-partial-result.md):
terminal cause and result disposition are orthogonal, revision-bound facts;
``final_partial`` is a communication class for ``completed + accepted_partial``
only, never a lifecycle state.

Structure (ROADMAP.md, Outcome section):

- RunPhase: running | blocked | waiting_user_input | waiting_approval | terminal
- TerminalReason (terminal phase only): completed | failed | cancelled | aborted
- Disposition: full | accepted_partial | none
- Verification: passed | failed | unknown | not_applicable, with evidence
  (postcondition checks + observations).

Rules enforced here, by construction, not by convention:

- ``completed + none`` and non-completed ``+ full`` are invalid terminal
  combinations (ADR-0002 §3) and cannot be constructed.
- ``accepted_partial`` requires an explicit acceptance authority (ADR-0002 §2:
  only the authorized user, or criteria approved by that user before
  execution, can accept an incomplete result).
- ``tool.ok`` is never proof of completion: full DONE requires
  ``completed + full + verification=passed`` — or ``not_applicable`` for a
  task that changes no state. A state-changing task without a confirmed
  postcondition is UNVERIFIED, never DONE.
- Unknown verification never becomes full success, never automatically
  becomes failed, and never rewrites the terminal reason.
- No ``final_success`` is published for an unconfirmed full claim: the
  ADR-0002 §3 table still maps the (cause, disposition) combinations, but
  the §4 acceptance transaction withholds the final until the criteria are
  confirmed. This is a publication gate, not a change to the table:
  ``completed + accepted_partial`` still publishes ``final_partial``
  (the user's acceptance is the authority there), and failure /
  cancellation / abort finals are published on their cause alone.
- Non-terminal phases (running / blocked / waiting_*) never project as
  completion.
- UI labels (``ui_label``) are a pure projection of this structure, never
  the source of truth and never a substitute for ``TerminalReason``.

Derivation helpers (``outcome_from_mwv`` / ``outcome_from_loop_result``)
are the only sanctioned way to build an outcome from runtime signals;
they are mechanical mappings, they never infer acceptance or completion
from prose, budget exhaustion, artifact existence, or a single ``ok`` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from core.mwv.models import VerificationResult, VerificationStatus, WorkStatus
from core.tool_loop import AgentToolLoopResult


class TerminalReason(StrEnum):
    """Why a terminal run stopped (ADR-0002 §1). Terminal phase only."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"


class Disposition(StrEnum):
    """What of the task result stands (ADR-0002 §2)."""

    FULL = "full"
    ACCEPTED_PARTIAL = "accepted_partial"
    NONE = "none"


class RunPhase(StrEnum):
    """Where the run is. Only ``terminal`` carries a TerminalReason."""

    RUNNING = "running"
    BLOCKED = "blocked"
    WAITING_USER_INPUT = "waiting_user_input"
    WAITING_APPROVAL = "waiting_approval"
    TERMINAL = "terminal"


class VerificationOutcome(StrEnum):
    """Postcondition verification outcome, with evidence attached."""

    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class PostconditionCheck:
    """One checked postcondition: name, pass/fail, human-readable detail."""

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class Verification:
    """Typed verification evidence for a run outcome."""

    outcome: VerificationOutcome
    checks: tuple[PostconditionCheck, ...] = ()
    source: str = ""

    @classmethod
    def from_verification_result(cls, result: VerificationResult) -> Verification:
        """Map an MWV VerificationResult to canonical verification.

        PASSED -> passed, FAILED -> failed, ERROR -> unknown: a verifier
        that errored did not disprove the postcondition, it failed to
        check it. The check itself is never silently dropped: its command,
        exit code and excerpt are kept as evidence.
        """
        mapping = {
            VerificationStatus.PASSED: VerificationOutcome.PASSED,
            VerificationStatus.FAILED: VerificationOutcome.FAILED,
            VerificationStatus.ERROR: VerificationOutcome.UNKNOWN,
        }
        detail_parts = []
        if result.exit_code is not None:
            detail_parts.append(f"exit_code={result.exit_code}")
        if result.excerpt:
            detail_parts.append(result.excerpt)
        if result.error:
            detail_parts.append(f"error={result.error}")
        check = PostconditionCheck(
            name="verifier",
            passed=result.status is VerificationStatus.PASSED,
            detail="; ".join(detail_parts),
        )
        source = " ".join(result.command) if result.command else "verifier"
        return cls(outcome=mapping[result.status], checks=(check,), source=source)

    @classmethod
    def not_applicable(cls, *, source: str = "") -> Verification:
        """No postcondition applies (e.g. a pure answer task)."""
        return cls(outcome=VerificationOutcome.NOT_APPLICABLE, source=source)

    @classmethod
    def unknown(cls, *, source: str = "") -> Verification:
        """No verification ran (yet). Never a pass, never a fail."""
        return cls(outcome=VerificationOutcome.UNKNOWN, source=source)


@dataclass(frozen=True)
class RunOutcome:
    """One canonical outcome for a run.

    Invariants (ADR-0002 §3, enforced in __post_init__):

    - terminal phase <=> terminal_reason is set;
    - completed + none is invalid; failed/cancelled/aborted + full is invalid;
    - accepted_partial requires accepted_by (acceptance authority);
    - accepted_by is meaningless without accepted_partial.
    """

    phase: RunPhase
    disposition: Disposition
    verification: Verification
    state_changing: bool = False
    terminal_reason: TerminalReason | None = None
    accepted_by: str | None = None

    def __post_init__(self) -> None:
        if self.phase is RunPhase.TERMINAL and self.terminal_reason is None:
            raise ValueError("terminal outcome requires terminal_reason")
        if self.phase is not RunPhase.TERMINAL and self.terminal_reason is not None:
            raise ValueError("non-terminal outcome must not carry terminal_reason")
        if (
            self.terminal_reason is TerminalReason.COMPLETED
            and self.disposition is Disposition.NONE
        ):
            raise ValueError("completed + none is an invalid terminal combination")
        if (
            self.terminal_reason is not None
            and self.terminal_reason is not TerminalReason.COMPLETED
            and self.disposition is Disposition.FULL
        ):
            raise ValueError(
                f"{self.terminal_reason.value} + full is an invalid terminal combination"
            )
        if self.disposition is Disposition.ACCEPTED_PARTIAL and not self.accepted_by:
            raise ValueError("accepted_partial requires an acceptance authority")
        if self.disposition is not Disposition.ACCEPTED_PARTIAL and self.accepted_by:
            raise ValueError("accepted_by is only meaningful with accepted_partial")

    @property
    def communication_class(self) -> str | None:
        """ADR-0002 §3 mapping. None while the run is not terminal: no
        final is published for a run whose task remains open.

        Also None for an unconfirmed full claim: ``final_success`` is
        published only when the acceptance preconditions hold
        (``is_full_done``), per ADR-0002 §4 — the acceptance transaction
        validates the criteria before a final is published. The §3 table
        itself is unchanged: once confirmed, completed + full is
        final_success, and completed + accepted_partial is final_partial
        (the user's acceptance is the authority there, so it is published
        with its verification evidence attached).
        """
        if self.phase is not RunPhase.TERMINAL or self.terminal_reason is None:
            return None
        if self.terminal_reason is TerminalReason.COMPLETED:
            if self.disposition is Disposition.FULL:
                return "final_success" if self.is_full_done else None
            return "final_partial"
        return {
            TerminalReason.FAILED: "final_failure",
            TerminalReason.CANCELLED: "final_cancelled",
            TerminalReason.ABORTED: "final_aborted",
        }[self.terminal_reason]

    @property
    def is_full_done(self) -> bool:
        """Full DONE: completed + full + confirmed postcondition.

        For a task that changes no state, verification=not_applicable is
        an honest confirmation (there is nothing to check). For a
        state-changing task, only verification=passed confirms it.
        """
        if (
            self.phase is not RunPhase.TERMINAL
            or self.terminal_reason is not TerminalReason.COMPLETED
            or self.disposition is not Disposition.FULL
        ):
            return False
        if self.verification.outcome is VerificationOutcome.PASSED:
            return True
        return (
            not self.state_changing
            and self.verification.outcome is VerificationOutcome.NOT_APPLICABLE
        )

    @property
    def ui_label(self) -> str:
        """UI text label: a pure projection of this structure.

        DONE appears if and only if is_full_done. completed + accepted_partial
        is PARTIAL, never DONE. Unknown or failed verification on a completed
        full claim is UNVERIFIED with the terminal reason preserved — it is
        never rewritten to FAILED. Non-terminal phases never read as DONE.
        """
        if self.phase is RunPhase.RUNNING:
            return "RUNNING"
        if self.phase is RunPhase.BLOCKED:
            return "BLOCKED"
        if self.phase in (RunPhase.WAITING_USER_INPUT, RunPhase.WAITING_APPROVAL):
            return "WAITING"
        if self.terminal_reason is TerminalReason.COMPLETED:
            if self.disposition is Disposition.ACCEPTED_PARTIAL:
                return "PARTIAL"
            return "DONE" if self.is_full_done else "UNVERIFIED"
        if self.terminal_reason is TerminalReason.FAILED:
            return "FAILED"
        if self.terminal_reason is TerminalReason.CANCELLED:
            return "CANCELLED"
        return "ABORTED"

    def describe(self) -> str:
        """One-line human summary: cause first, disposition and
        verification preserved, never merged into a single flat word."""
        if self.phase is not RunPhase.TERMINAL or self.terminal_reason is None:
            return (
                f"{self.phase.value}, disposition={self.disposition.value}, "
                f"verification={self.verification.outcome.value}"
            )
        text = (
            f"{self.terminal_reason.value} + {self.disposition.value}, "
            f"verification={self.verification.outcome.value}"
        )
        final = self.communication_class
        if final is not None:
            text += f" -> {final}"
        else:
            text += " -> final withheld (verification not confirmed)"
        if self.disposition is Disposition.ACCEPTED_PARTIAL and self.accepted_by:
            text += f" (accepted by {self.accepted_by})"
        return text


def outcome_from_mwv(
    work_status: WorkStatus,
    verification_result: VerificationResult,
    *,
    state_changing: bool,
    accepted_by: str | None = None,
) -> RunOutcome:
    """Derive a terminal outcome from an MWV run (work + verifier).

    - work SUCCESS + verification PASSED -> completed + full.
    - work SUCCESS + verification FAILED -> failed + none: the check ran
      and disproved the claimed result, so the run failed its acceptance
      (consistent with ManagerRuntime._is_success, which requires both).
    - work SUCCESS + verification ERROR -> completed + full with
      verification unknown: the check did not run, the postcondition was
      not disproved, and unknown must not automatically become failed.
      The claim is preserved but UNVERIFIED, and no final is published.
    - work FAILURE -> failed, with disposition accepted_partial only
      under an explicit acceptance authority, else none.
    - Verification ERROR maps to unknown, never to failed: the check did
      not run, the postcondition was not disproved.
    """
    verification = Verification.from_verification_result(verification_result)
    if work_status is WorkStatus.SUCCESS and verification.outcome is VerificationOutcome.PASSED:
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.COMPLETED,
            disposition=Disposition.FULL,
            verification=verification,
            state_changing=state_changing,
        )
    if work_status is WorkStatus.SUCCESS and verification.outcome is VerificationOutcome.FAILED:
        disposition = Disposition.ACCEPTED_PARTIAL if accepted_by else Disposition.NONE
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.FAILED,
            disposition=disposition,
            verification=verification,
            state_changing=state_changing,
            accepted_by=accepted_by if disposition is Disposition.ACCEPTED_PARTIAL else None,
        )
    reason = (
        TerminalReason.COMPLETED if work_status is WorkStatus.SUCCESS else TerminalReason.FAILED
    )
    if reason is TerminalReason.COMPLETED:
        # Verification did not run (ERROR -> unknown): the full claim is
        # preserved but unverified. completed + none is invalid per
        # ADR-0002, so the disposition stays full — the label gate
        # (UNVERIFIED, no final published) is what prevents fake success.
        disposition = Disposition.ACCEPTED_PARTIAL if accepted_by else Disposition.FULL
    else:
        disposition = Disposition.ACCEPTED_PARTIAL if accepted_by else Disposition.NONE
    return RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=reason,
        disposition=disposition,
        verification=verification,
        state_changing=state_changing,
        accepted_by=accepted_by if disposition is Disposition.ACCEPTED_PARTIAL else None,
    )


# Loop errors that are administrative/policy stops, not subject failures
# (ADR-0002 §1: aborted, not failed). Produced by AgentToolLoop when a
# tool call is refused by policy (core/tool_loop.py).
_POLICY_STOP_ERRORS = frozenset({"tool_policy_denied"})


def outcome_from_loop_result(
    loop_result: AgentToolLoopResult,
    *,
    state_changing: bool,
    accepted_by: str | None = None,
) -> RunOutcome:
    """Derive a terminal outcome from an AgentToolLoopResult.

    - cancelled -> cancelled; a loop error -> failed, except a known
      policy stop (``tool_policy_denied``) -> aborted (ADR-0002 §1: an
      administrative stop without an asserted subject error). In all
      cases the verification is unknown: the loop stopping says nothing
      about the postcondition, and unknown never rewrites the reason.
    - A clean finish is completed + full, or completed + accepted_partial
      when the caller passes an acceptance authority (the user accepted
      an incomplete result). Full DONE still requires a confirmed
      postcondition: for a state-changing task the verification stays
      unknown (UNVERIFIED, not DONE) until a real check passes; for a
      task that changes no state, not_applicable confirms it.
    """
    disposition = Disposition.ACCEPTED_PARTIAL if accepted_by else Disposition.NONE
    if loop_result.cancelled:
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.CANCELLED,
            disposition=disposition,
            verification=Verification.unknown(source="tool_loop"),
            state_changing=state_changing,
            accepted_by=accepted_by if disposition is Disposition.ACCEPTED_PARTIAL else None,
        )
    if loop_result.error is not None:
        check = PostconditionCheck(name="loop_error", passed=False, detail=loop_result.error)
        reason = (
            TerminalReason.ABORTED
            if loop_result.error in _POLICY_STOP_ERRORS
            else TerminalReason.FAILED
        )
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=reason,
            disposition=disposition,
            verification=Verification(
                outcome=VerificationOutcome.UNKNOWN,
                checks=(check,),
                source="tool_loop",
            ),
            state_changing=state_changing,
            accepted_by=accepted_by if disposition is Disposition.ACCEPTED_PARTIAL else None,
        )
    verification = (
        Verification.unknown(source="tool_loop")
        if state_changing
        else Verification.not_applicable(source="tool_loop")
    )
    disposition = Disposition.ACCEPTED_PARTIAL if accepted_by else Disposition.FULL
    return RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=TerminalReason.COMPLETED,
        disposition=disposition,
        verification=verification,
        state_changing=state_changing,
        accepted_by=accepted_by if disposition is Disposition.ACCEPTED_PARTIAL else None,
    )
