from __future__ import annotations

from core.critic_policy import decide_critic
from shared.models import LLMMessage


def test_dual_deps_triggers_critic() -> None:
    decision = decide_critic(
        mode="dual",
        messages=[
            LLMMessage(
                role="user", content="Установи зависимости: pip install requests"
            )
        ],
    )
    assert decision.should_run_critic is True
    assert "risk:deps" in decision.reasons


def test_dual_plain_question_no_critic() -> None:
    decision = decide_critic(
        mode="dual",
        messages=[LLMMessage(role="user", content="Что такое VectorIndex?")],
    )
    assert decision.should_run_critic is False
    assert "no_triggers" in decision.reasons


def test_single_always_disabled() -> None:
    decision = decide_critic(
        mode="single",
        messages=[LLMMessage(role="user", content="Проверь план")],
    )
    assert decision.should_run_critic is False
    assert "mode=single" in decision.reasons


def test_critic_only_always_runs() -> None:
    decision = decide_critic(
        mode="critic-only",
        messages=[LLMMessage(role="user", content="Оцени риски")],
    )
    assert decision.should_run_critic is True
    assert "mode=critic-only" in decision.reasons
