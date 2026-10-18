from __future__ import annotations

import uuid
from datetime import UTC, datetime

from core.decision.models import DecisionAction, DecisionOption, DecisionPacket, DecisionReason
from shared.models import JSONValue


def build_session_end_packet() -> DecisionPacket:
    return DecisionPacket(
        id=f"decision-{uuid.uuid4().hex}",
        created_at=datetime.now(UTC),
        reason=DecisionReason.NEED_USER_INPUT,
        summary=(
            "Командный режим (без MWV). Сохранить важную информацию в Memory перед завершением?"
        ),
        context={"source_endpoint": "session.end"},
        options=[
            DecisionOption(id="yes", title="Да", action=DecisionAction.CONFIRM),
            DecisionOption(id="no", title="Нет", action=DecisionAction.REJECT),
            DecisionOption(id="cancel", title="Отмена", action=DecisionAction.ABORT),
        ],
        default_option_id="cancel",
        decision_type="end_session",
        policy={"require_explicit_memory_confirmation": True},
    )


def is_session_end_decision(decision: dict[str, JSONValue] | None) -> bool:
    if decision is None:
        return False
    context = decision.get("context")
    return decision.get("decision_type") == "end_session" or (
        decision.get("decision_type") == "memory_save"
        and isinstance(context, dict)
        and context.get("source_endpoint") == "session.end"
    )
