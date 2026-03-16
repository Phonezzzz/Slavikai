from __future__ import annotations

from dataclasses import dataclass, field

from core.decision.ambiguous import build_ambiguous_skill_packet
from core.decision.models import DecisionPacket, DecisionReason
from core.decision.tool_fail import build_tool_fail_packet
from core.decision.verifier_fail import build_verifier_fail_packet
from core.mwv.models import VerificationResult
from core.skills.index import SkillMatchDecision


@dataclass(frozen=True)
class DecisionContext:
    user_input: str
    route: str
    routing_reason: str
    decision_reason: DecisionReason | None = None
    risk_flags: list[str] = field(default_factory=list)
    skill_decision: SkillMatchDecision | None = None


@dataclass(frozen=True)
class DecisionEvent:
    kind: str
    tool_name: str | None = None
    error_text: str | None = None
    count: int | None = None
    threshold: int | None = None
    verification_result: VerificationResult | None = None
    task_id: str | None = None
    trace_id: str | None = None
    attempt: int | None = None
    max_attempts: int | None = None
    retry_allowed: bool | None = None
    skill_decision: SkillMatchDecision | None = None
    user_input: str | None = None

    @classmethod
    def ambiguous_skill(
        cls,
        *,
        skill_decision: SkillMatchDecision,
        user_input: str,
    ) -> DecisionEvent:
        return cls(
            kind="ambiguous_skill",
            skill_decision=skill_decision,
            user_input=user_input,
        )

    @classmethod
    def tool_fail(
        cls,
        *,
        tool_name: str,
        error_text: str,
        count: int,
        threshold: int,
        user_input: str | None,
    ) -> DecisionEvent:
        return cls(
            kind="tool_fail",
            tool_name=tool_name,
            error_text=error_text,
            count=count,
            threshold=threshold,
            user_input=user_input,
        )

    @classmethod
    def verifier_fail(
        cls,
        *,
        verification_result: VerificationResult,
        task_id: str,
        trace_id: str | None,
        attempt: int,
        max_attempts: int,
        retry_allowed: bool,
    ) -> DecisionEvent:
        return cls(
            kind="verifier_fail",
            verification_result=verification_result,
            task_id=task_id,
            trace_id=trace_id,
            attempt=attempt,
            max_attempts=max_attempts,
            retry_allowed=retry_allowed,
        )


class DecisionHandler:
    def __init__(self) -> None:
        self._forced_packet: DecisionPacket | None = None

    def force_next(self, packet: DecisionPacket | None) -> None:
        self._forced_packet = packet

    def evaluate(
        self,
        context: DecisionContext | None = None,
        *,
        event: DecisionEvent | None = None,
    ) -> DecisionPacket | None:
        if self._forced_packet is None:
            if event is not None:
                return self._build_from_event(event)
            if (
                context is not None
                and context.skill_decision
                and context.skill_decision.status == "ambiguous"
            ):
                return self._build_from_event(
                    DecisionEvent.ambiguous_skill(
                        skill_decision=context.skill_decision,
                        user_input=context.user_input,
                    )
                )
            return None
        packet = self._forced_packet
        self._forced_packet = None
        return packet

    def _build_from_event(self, event: DecisionEvent) -> DecisionPacket:
        if event.kind == "ambiguous_skill":
            skill_decision = event.skill_decision
            user_input = event.user_input
            if skill_decision is None or user_input is None:
                raise ValueError("ambiguous_skill event требует skill_decision и user_input")
            return build_ambiguous_skill_packet(skill_decision, user_input=user_input)
        if event.kind == "tool_fail":
            if (
                event.tool_name is None
                or event.error_text is None
                or event.count is None
                or event.threshold is None
            ):
                raise ValueError("tool_fail event требует tool_name/error_text/count/threshold")
            return build_tool_fail_packet(
                tool_name=event.tool_name,
                error_text=event.error_text,
                count=event.count,
                threshold=event.threshold,
                user_input=event.user_input,
            )
        if event.kind == "verifier_fail":
            if (
                event.verification_result is None
                or event.task_id is None
                or event.attempt is None
                or event.max_attempts is None
                or event.retry_allowed is None
            ):
                raise ValueError(
                    "verifier_fail event требует "
                    "verification_result/task_id/attempt/max_attempts/retry_allowed"
                )
            return build_verifier_fail_packet(
                event.verification_result,
                task_id=event.task_id,
                trace_id=event.trace_id,
                attempt=event.attempt,
                max_attempts=event.max_attempts,
                retry_allowed=event.retry_allowed,
            )
        raise ValueError(f"Unsupported decision event kind: {event.kind}")


class DecisionRequired(Exception):
    def __init__(self, packet: DecisionPacket) -> None:
        super().__init__("Decision required")
        self.packet = packet
