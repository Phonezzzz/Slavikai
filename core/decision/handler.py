from __future__ import annotations

from dataclasses import dataclass, field

from core.decision.models import DecisionPacket


@dataclass(frozen=True)
class DecisionContext:
    user_input: str
    route: str
    reason: str
    risk_flags: list[str] = field(default_factory=list)
    skill_status: str | None = None


class DecisionHandler:
    def __init__(self) -> None:
        self._forced_packet: DecisionPacket | None = None

    def force_next(self, packet: DecisionPacket | None) -> None:
        self._forced_packet = packet

    def evaluate(self, context: DecisionContext) -> DecisionPacket | None:
        if self._forced_packet is None:
            return None
        packet = self._forced_packet
        self._forced_packet = None
        return packet
