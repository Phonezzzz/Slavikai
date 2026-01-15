from __future__ import annotations

from dataclasses import dataclass, field

from core.decision.ambiguous import build_ambiguous_skill_packet
from core.decision.models import DecisionPacket
from core.skills.index import SkillMatchDecision


@dataclass(frozen=True)
class DecisionContext:
    user_input: str
    route: str
    reason: str
    risk_flags: list[str] = field(default_factory=list)
    skill_decision: SkillMatchDecision | None = None


class DecisionHandler:
    def __init__(self) -> None:
        self._forced_packet: DecisionPacket | None = None

    def force_next(self, packet: DecisionPacket | None) -> None:
        self._forced_packet = packet

    def evaluate(self, context: DecisionContext) -> DecisionPacket | None:
        if self._forced_packet is None:
            if context.skill_decision and context.skill_decision.status == "ambiguous":
                return build_ambiguous_skill_packet(
                    context.skill_decision,
                    user_input=context.user_input,
                )
            return None
        packet = self._forced_packet
        self._forced_packet = None
        return packet


class DecisionRequired(Exception):
    def __init__(self, packet: DecisionPacket) -> None:
        super().__init__("Decision required")
        self.packet = packet
