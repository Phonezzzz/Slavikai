from __future__ import annotations

import uuid
from datetime import UTC, datetime

from core.decision.models import (
    DecisionAction,
    DecisionOption,
    DecisionPacket,
    DecisionReason,
)
from shared.models import JSONValue


def build_memory_save_packet(
    proposed_action: dict[str, JSONValue],
) -> DecisionPacket:
    claims_raw = proposed_action.get("claims")
    claim_count = len(claims_raw) if isinstance(claims_raw, list) else 0
    return DecisionPacket(
        id=f"decision-{uuid.uuid4().hex}",
        created_at=datetime.now(UTC),
        reason=DecisionReason.MEMORY_SAVE_CONFIRMATION,
        summary=f"Сохранить предложенные изменения Memory ({claim_count})?",
        context={"source_endpoint": "memory.save"},
        options=[
            DecisionOption(
                id="confirm",
                title="Сохранить",
                action=DecisionAction.CONFIRM,
                risk="medium",
            ),
            DecisionOption(
                id="edit_and_confirm",
                title="Изменить и сохранить",
                action=DecisionAction.EDIT_AND_CONFIRM,
                risk="medium",
            ),
            DecisionOption(
                id="reject",
                title="Не сохранять",
                action=DecisionAction.REJECT,
                risk="low",
            ),
        ],
        default_option_id="reject",
        policy={"require_explicit_memory_confirmation": True},
        decision_type="memory_save",
        proposed_action=proposed_action,
    )
