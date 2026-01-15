from __future__ import annotations

import uuid
from datetime import datetime

from core.decision.models import (
    DecisionAction,
    DecisionOption,
    DecisionPacket,
    DecisionReason,
)
from shared.models import JSONValue


def build_tool_fail_packet(
    *,
    tool_name: str,
    error_text: str,
    count: int,
    threshold: int,
    user_input: str | None,
) -> DecisionPacket:
    summary = f"Инструмент {tool_name} не выполняется стабильно ({count}/{threshold})."
    options = [
        DecisionOption(
            id="retry",
            title="Повторить с тем же инструментом",
            action=DecisionAction.RETRY,
            payload={"tool": tool_name, "attempt": count, "threshold": threshold},
            risk="medium",
        ),
        DecisionOption(
            id="adjust_threshold",
            title="Временно увеличить порог ошибок",
            action=DecisionAction.ADJUST_THRESHOLD,
            payload={
                "tool": tool_name,
                "current_threshold": threshold,
                "suggested_threshold": threshold + 1,
            },
            risk="medium",
        ),
        DecisionOption(
            id="create_candidate",
            title="Создать candidate на основе ошибки",
            action=DecisionAction.CREATE_CANDIDATE,
            payload={"tool": tool_name, "error": error_text},
            risk="low",
        ),
        DecisionOption(
            id="abort",
            title="Остановить выполнение",
            action=DecisionAction.ABORT,
            payload={},
            risk="low",
        ),
    ]
    return DecisionPacket(
        id=_packet_id(),
        created_at=datetime.utcnow(),
        reason=DecisionReason.TOOL_FAIL,
        summary=summary,
        context=_build_context(tool_name, error_text, count, threshold, user_input),
        options=options,
        default_option_id=None,
        policy={"require_user_choice": True},
    )


def _packet_id() -> str:
    return f"decision-{uuid.uuid4()}"


def _build_context(
    tool_name: str,
    error_text: str,
    count: int,
    threshold: int,
    user_input: str | None,
) -> dict[str, JSONValue]:
    return {
        "tool": tool_name,
        "error": error_text,
        "count": count,
        "threshold": threshold,
        "user_input": user_input or "",
    }
