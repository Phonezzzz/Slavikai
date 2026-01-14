from __future__ import annotations

import uuid
from datetime import datetime

from core.decision.models import (
    DecisionAction,
    DecisionOption,
    DecisionPacket,
    DecisionReason,
)
from core.skills.index import SkillMatch, SkillMatchDecision
from shared.models import JSONValue


def build_ambiguous_skill_packet(
    decision: SkillMatchDecision,
    *,
    user_input: str,
) -> DecisionPacket:
    if decision.status != "ambiguous":
        raise ValueError("DecisionPacket для ambiguous_skill требует status=ambiguous")

    candidates = decision.alternatives[:3]
    payload_candidates = [_format_candidate(match) for match in candidates]
    options: list[DecisionOption] = [
        DecisionOption(
            id="select_skill",
            title="Выбрать нужный навык",
            action=DecisionAction.SELECT_SKILL,
            payload={"candidates": payload_candidates},
            risk="low",
        ),
        DecisionOption(
            id="ask_user",
            title="Уточнить у пользователя",
            action=DecisionAction.ASK_USER,
            payload={"prompt": _build_prompt(payload_candidates)},
            risk="low",
        ),
        DecisionOption(
            id="abort",
            title="Отменить запрос",
            action=DecisionAction.ABORT,
            payload={},
            risk="low",
        ),
    ]

    safe_skill = _select_safe_candidate(candidates)
    if safe_skill is not None:
        safe_skill_id = str(safe_skill["id"])
        options.insert(
            2,
            DecisionOption(
                id="proceed_safe",
                title=f"Продолжить безопасно: {safe_skill_id}",
                action=DecisionAction.PROCEED_SAFE,
                payload={"skill_id": safe_skill_id},
                risk="low",
            ),
        )

    return DecisionPacket(
        id=_packet_id(),
        created_at=datetime.utcnow(),
        reason=DecisionReason.AMBIGUOUS_SKILL,
        summary="Найдено несколько подходящих навыков. Нужен выбор.",
        context={
            "user_input": user_input,
            "reason": decision.reason,
            "candidates": payload_candidates,
        },
        options=options,
        default_option_id="ask_user",
        policy={"require_user_choice": True},
    )


def _packet_id() -> str:
    return f"decision-{uuid.uuid4()}"


def _format_candidate(match: SkillMatch) -> dict[str, JSONValue]:
    entry = match.entry
    return {
        "id": entry.id,
        "title": entry.title,
        "risk": entry.risk,
        "pattern": match.pattern,
    }


def _build_prompt(candidates: list[dict[str, JSONValue]]) -> str:
    ids: list[str] = []
    for candidate in candidates:
        value = candidate.get("id")
        if isinstance(value, str):
            ids.append(value)
    listed = ", ".join(ids) if ids else "skill_id"
    return f"Укажи нужный skill_id: {listed}."


def _select_safe_candidate(candidates: list[SkillMatch]) -> dict[str, JSONValue] | None:
    for match in candidates:
        entry = match.entry
        if entry.risk == "low":
            return _format_candidate(match)
    return None
