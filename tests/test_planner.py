from __future__ import annotations

from core.planner import MAX_STEPS, MIN_STEPS, Planner
from llm.types import LLMResult


class FakeBrain:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, messages, config=None):  # type: ignore[override]
        return LLMResult(text=self._text)


def test_classify_simple_vs_complex() -> None:
    planner = Planner()
    assert planner.classify_complexity("Привет, помоги").value == "simple"
    assert (
        planner.classify_complexity("Найди и проанализируй архитектуру проекта").value == "complex"
    )


def test_plan_parsing_validation_limits() -> None:
    planner = Planner()
    steps_text = "\n".join([f"{i + 1}. step {i + 1}" for i in range(MIN_STEPS + 1)])
    brain = FakeBrain(steps_text)
    parsed = planner._llm_plan("goal", brain, None)  # type: ignore[arg-type]
    assert parsed is not None
    assert MIN_STEPS <= len(parsed) <= MAX_STEPS

    too_many = "\n".join([f"{i + 1}. step {i + 1}" for i in range(MAX_STEPS + 5)])
    brain_many = FakeBrain(too_many)
    assert planner._llm_plan("goal", brain_many, None) is None  # type: ignore[arg-type]


def test_build_plan_fallback_on_invalid_llm_plan() -> None:
    planner = Planner()
    invalid_brain = FakeBrain("1. one\n")  # всего один шаг => будет отброшено
    plan = planner.build_plan("goal", brain=invalid_brain)  # type: ignore[arg-type]
    assert len(plan.steps) >= MIN_STEPS
