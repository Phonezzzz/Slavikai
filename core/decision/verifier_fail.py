from __future__ import annotations

import uuid
from datetime import UTC, datetime

from core.decision.models import (
    DecisionAction,
    DecisionOption,
    DecisionPacket,
    DecisionReason,
)
from core.mwv.models import VerificationResult
from core.mwv.verifier_summary import extract_verifier_excerpt, summarize_verifier_failure
from shared.models import JSONValue


def build_verifier_fail_packet(
    result: VerificationResult,
    *,
    task_id: str | None,
    trace_id: str | None,
    attempt: int,
    max_attempts: int,
    retry_allowed: bool,
) -> DecisionPacket:
    summary = summarize_verifier_failure(result)
    options = [
        DecisionOption(
            id="ask_user",
            title="Уточнить дальнейшие действия",
            action=DecisionAction.ASK_USER,
            payload={"summary": summary},
            risk="low",
        ),
        DecisionOption(
            id="proceed_safe",
            title="Продолжить без рискованных действий",
            action=DecisionAction.PROCEED_SAFE,
            payload={"note": "no_changes"},
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
    if retry_allowed:
        options.insert(
            2,
            DecisionOption(
                id="retry",
                title="Повторить проверку после исправления",
                action=DecisionAction.RETRY,
                payload={"attempt": attempt, "max_attempts": max_attempts},
                risk="medium",
            ),
        )
    return DecisionPacket(
        id=_packet_id(),
        created_at=datetime.now(UTC),
        reason=DecisionReason.VERIFIER_FAIL,
        summary=f"Проверка не прошла: {summary}",
        context=_build_context(result, task_id, trace_id, attempt, max_attempts),
        options=options,
        default_option_id="ask_user",
        policy={"require_user_choice": True},
    )


def _packet_id() -> str:
    return f"decision-{uuid.uuid4()}"


def _build_context(
    result: VerificationResult,
    task_id: str | None,
    trace_id: str | None,
    attempt: int,
    max_attempts: int,
) -> dict[str, JSONValue]:
    return {
        "task_id": task_id or "",
        "trace_id": trace_id or "",
        "status": result.status.value,
        "exit_code": result.exit_code,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "excerpt": extract_verifier_excerpt(result, max_lines=3, max_chars=300),
        "stderr": _truncate_text(result.stderr),
        "stdout": _truncate_text(result.stdout),
        "error": result.error or "",
    }


def _truncate_text(text: str, *, limit: int = 300) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    excerpt = "\n".join(lines[:3])
    if len(excerpt) <= limit:
        return excerpt
    return f"{excerpt[: limit - 3].rstrip()}..."
