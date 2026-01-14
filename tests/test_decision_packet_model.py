from __future__ import annotations

import json
from datetime import datetime

import pytest

from core.decision.models import (
    DecisionAction,
    DecisionOption,
    DecisionPacket,
    DecisionReason,
)


def _options() -> list[DecisionOption]:
    return [
        DecisionOption(
            id="select",
            title="Select a skill",
            action=DecisionAction.SELECT_SKILL,
            payload={"skill_ids": ["alpha", "beta"]},
            risk="low",
        ),
        DecisionOption(
            id="ask",
            title="Ask for clarification",
            action=DecisionAction.ASK_USER,
            payload={"question": "Which skill should be used?"},
            risk="low",
        ),
        DecisionOption(
            id="abort",
            title="Abort",
            action=DecisionAction.ABORT,
            payload={},
            risk="medium",
        ),
    ]


def test_decision_packet_to_dict_and_json() -> None:
    created = datetime(2025, 1, 1, 0, 0, 0)
    packet = DecisionPacket(
        id="dp-1",
        created_at=created,
        reason=DecisionReason.AMBIGUOUS_SKILL,
        summary="Multiple skills match the request.",
        context={"candidates": ["alpha", "beta"]},
        options=_options(),
        default_option_id="ask",
        ttl_seconds=600,
        policy={"require_user_choice": True},
    )

    data = packet.to_dict()
    assert data["id"] == "dp-1"
    assert data["reason"] == "ambiguous_skill"
    assert data["created_at"] == created.isoformat()
    assert data["default_option_id"] == "ask"
    assert data["ttl_seconds"] == 600

    options = data["options"]
    assert isinstance(options, list)
    assert len(options) == 3
    assert options[0]["action"] == "select_skill"
    assert options[1]["risk"] == "low"

    payload = json.loads(packet.to_json())
    assert payload["id"] == "dp-1"
    assert payload["reason"] == "ambiguous_skill"


def test_decision_packet_validation() -> None:
    created = datetime(2025, 1, 1, 0, 0, 0)
    with pytest.raises(ValueError):
        DecisionPacket(
            id="dp-2",
            created_at=created,
            reason=DecisionReason.TOOL_FAIL,
            summary="Too few options.",
            options=_options()[:2],
        )
    with pytest.raises(ValueError):
        DecisionPacket(
            id="dp-3",
            created_at=created,
            reason=DecisionReason.TOOL_FAIL,
            summary="Invalid default option.",
            options=_options(),
            default_option_id="missing",
        )
