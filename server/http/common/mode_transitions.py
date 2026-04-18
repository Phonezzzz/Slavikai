from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from shared.models import JSONValue

SessionMode = Literal["ask", "plan", "act", "auto"]
ReasonCode = Literal[
    "already_active",
    "mode_transition_not_allowed",
    "plan_not_approved",
    "auto_run_active",
]


_AUTO_ACTIVE_STATUSES = {
    "planning",
    "coding",
    "merging",
    "verifying",
    "waiting_approval",
}


def _target_payload(
    *,
    allowed: bool,
    reason_code: ReasonCode | None = None,
    requires_confirm: bool = False,
    message: str | None = None,
) -> dict[str, JSONValue]:
    return {
        "allowed": allowed,
        "reason_code": reason_code,
        "requires_confirm": requires_confirm,
        "message": message,
    }


def _message_for(
    *,
    current_mode: SessionMode,
    target_mode: SessionMode,
    reason_code: ReasonCode,
) -> str:
    if reason_code == "already_active":
        return "Режим уже активен."
    if reason_code == "plan_not_approved":
        return "Нужен approved план для перехода в act."
    if reason_code == "auto_run_active":
        return "Нельзя выйти из auto: auto-run ещё активен."
    if target_mode == "act":
        return "В act можно перейти только из plan-режима."
    if target_mode == "auto":
        return "Нельзя перейти в auto при активном plan/act workflow."
    if current_mode == "auto" and target_mode == "plan":
        return "Переход auto->plan запрещён до завершения auto-run."
    return "Переход между режимами сейчас недоступен."


def _auto_run_active(auto_state: Mapping[str, JSONValue] | None) -> bool:
    if auto_state is None:
        return False
    status_raw = auto_state.get("status")
    return isinstance(status_raw, str) and status_raw in _AUTO_ACTIVE_STATUSES


def _normalize_mode(value: object) -> SessionMode:
    if value == "plan":
        return "plan"
    if value == "act":
        return "act"
    if value == "auto":
        return "auto"
    return "ask"


def build_mode_transitions(
    *,
    current_mode: str,
    active_plan: Mapping[str, JSONValue] | None,
    active_task: Mapping[str, JSONValue] | None,
    auto_state: Mapping[str, JSONValue] | None,
) -> dict[str, JSONValue]:
    mode = _normalize_mode(current_mode)
    auto_run_active = _auto_run_active(auto_state)
    has_active_workflow = active_plan is not None or active_task is not None
    plan_status_raw = active_plan.get("status") if active_plan is not None else None
    plan_is_approved = isinstance(plan_status_raw, str) and plan_status_raw == "approved"

    targets: dict[str, JSONValue] = {}
    for target_mode in ("ask", "plan", "act", "auto"):
        if target_mode == mode:
            targets[target_mode] = _target_payload(
                allowed=False,
                reason_code="already_active",
                message=_message_for(
                    current_mode=mode,
                    target_mode=target_mode,
                    reason_code="already_active",
                ),
            )
            continue
        if target_mode == "act":
            if mode != "plan":
                reason_code: ReasonCode = "mode_transition_not_allowed"
                targets[target_mode] = _target_payload(
                    allowed=False,
                    reason_code=reason_code,
                    message=_message_for(
                        current_mode=mode,
                        target_mode=target_mode,
                        reason_code=reason_code,
                    ),
                )
                continue
            if not plan_is_approved:
                reason_code = "plan_not_approved"
                targets[target_mode] = _target_payload(
                    allowed=False,
                    reason_code=reason_code,
                    message=_message_for(
                        current_mode=mode,
                        target_mode=target_mode,
                        reason_code=reason_code,
                    ),
                )
                continue
            targets[target_mode] = _target_payload(
                allowed=True,
                requires_confirm=True,
            )
            continue
        if target_mode == "auto":
            if mode in {"plan", "act"} and has_active_workflow:
                reason_code = "mode_transition_not_allowed"
                targets[target_mode] = _target_payload(
                    allowed=False,
                    reason_code=reason_code,
                    message=_message_for(
                        current_mode=mode,
                        target_mode=target_mode,
                        reason_code=reason_code,
                    ),
                )
                continue
            targets[target_mode] = _target_payload(allowed=True)
            continue
        if target_mode == "ask":
            if mode == "auto" and auto_run_active:
                reason_code = "auto_run_active"
                targets[target_mode] = _target_payload(
                    allowed=False,
                    reason_code=reason_code,
                    message=_message_for(
                        current_mode=mode,
                        target_mode=target_mode,
                        reason_code=reason_code,
                    ),
                )
                continue
            targets[target_mode] = _target_payload(allowed=True)
            continue
        if target_mode == "plan":
            if mode == "auto" and auto_run_active:
                reason_code = "mode_transition_not_allowed"
                targets[target_mode] = _target_payload(
                    allowed=False,
                    reason_code=reason_code,
                    message=_message_for(
                        current_mode=mode,
                        target_mode=target_mode,
                        reason_code=reason_code,
                    ),
                )
                continue
            targets[target_mode] = _target_payload(allowed=True)
            continue

    return {"current_mode": mode, "targets": targets}
