from __future__ import annotations

from server.http.common.mode_transitions import build_mode_transitions


def test_mode_transitions_allow_ask_to_plan() -> None:
    contract = build_mode_transitions(
        current_mode="ask",
        active_plan=None,
        active_task=None,
        auto_state=None,
    )

    assert contract["current_mode"] == "ask"
    assert contract["targets"]["plan"]["allowed"] is True
    assert contract["targets"]["plan"]["reason_code"] is None


def test_mode_transitions_block_plan_to_act_without_approved_plan() -> None:
    contract = build_mode_transitions(
        current_mode="plan",
        active_plan={"status": "draft"},
        active_task=None,
        auto_state=None,
    )

    assert contract["targets"]["act"]["allowed"] is False
    assert contract["targets"]["act"]["reason_code"] == "plan_not_approved"
    assert contract["targets"]["act"]["requires_confirm"] is False


def test_mode_transitions_allow_plan_to_act_with_approved_plan() -> None:
    contract = build_mode_transitions(
        current_mode="plan",
        active_plan={"status": "approved"},
        active_task=None,
        auto_state=None,
    )

    assert contract["targets"]["act"]["allowed"] is True
    assert contract["targets"]["act"]["reason_code"] is None
    assert contract["targets"]["act"]["requires_confirm"] is True


def test_mode_transitions_block_plan_and_act_to_auto_with_active_workflow() -> None:
    plan_contract = build_mode_transitions(
        current_mode="plan",
        active_plan={"status": "approved"},
        active_task=None,
        auto_state=None,
    )
    act_contract = build_mode_transitions(
        current_mode="act",
        active_plan=None,
        active_task={"status": "running"},
        auto_state=None,
    )

    assert plan_contract["targets"]["auto"]["allowed"] is False
    assert plan_contract["targets"]["auto"]["reason_code"] == "mode_transition_not_allowed"
    assert act_contract["targets"]["auto"]["allowed"] is False
    assert act_contract["targets"]["auto"]["reason_code"] == "mode_transition_not_allowed"


def test_mode_transitions_block_auto_to_ask_and_plan_while_auto_run_active() -> None:
    contract = build_mode_transitions(
        current_mode="auto",
        active_plan=None,
        active_task=None,
        auto_state={"status": "planning"},
    )

    assert contract["targets"]["ask"]["allowed"] is False
    assert contract["targets"]["ask"]["reason_code"] == "auto_run_active"
    assert contract["targets"]["plan"]["allowed"] is False
    assert contract["targets"]["plan"]["reason_code"] == "mode_transition_not_allowed"


def test_mode_transitions_mark_same_mode_as_already_active() -> None:
    contract = build_mode_transitions(
        current_mode="plan",
        active_plan=None,
        active_task=None,
        auto_state=None,
    )

    assert contract["targets"]["plan"]["allowed"] is False
    assert contract["targets"]["plan"]["reason_code"] == "already_active"
