from __future__ import annotations

from dataclasses import dataclass

from shared.batch_review_models import IntentHypothesis, IntentKind, ParadoxFlag


@dataclass(frozen=True)
class IntentAnalysis:
    hypotheses: list[IntentHypothesis]


@dataclass(frozen=True)
class ParadoxDetection:
    flags: list[ParadoxFlag]


_QUESTION_WORDS = ("как", "почему", "зачем", "что", "где", "когда", "сколько", "какой")
_ACTION_WORDS = (
    "сделай",
    "добавь",
    "исправь",
    "поменяй",
    "удали",
    "настрой",
    "реализуй",
    "напиши",
    "сгенерируй",
    "покажи",
)
_PROBLEM_WORDS = ("ошибка", "не работает", "сломал", "неправильно", "баг", "fail", "crash")
_PRAISE_WORDS = ("спасибо", "круто", "полезно", "супер", "отлично", "спс", "thanks")
_CLARIFY_WORDS = ("уточню", "имею в виду", "точнее", "поясню")

_STYLE_SHORT = ("кратко", "короче", "коротко", "без воды", "brief")
_STYLE_LONG = ("подробно", "детально", "развернуто", "detailed")


def analyze_intent(text: str) -> IntentAnalysis:
    """
    Детерминированный анализ intent: возвращает несколько гипотез со score.
    Не выносит “вердикт” и не интерпретирует стиль (сарказм/мат) как оценку.
    """
    normalized = text.strip().lower()
    if not normalized:
        return IntentAnalysis(hypotheses=[IntentHypothesis(intent=IntentKind.OTHER, score=1.0)])

    scores: dict[IntentKind, float] = {k: 0.0 for k in IntentKind}
    scores[IntentKind.OTHER] = 0.2

    if "?" in normalized:
        scores[IntentKind.QUESTION] += 0.6
    if any(normalized.startswith(word + " ") or normalized == word for word in _QUESTION_WORDS):
        scores[IntentKind.QUESTION] += 0.4

    if any(word in normalized for word in _ACTION_WORDS):
        scores[IntentKind.REQUEST_ACTION] += 0.6

    if any(word in normalized for word in _PROBLEM_WORDS):
        scores[IntentKind.REPORT_PROBLEM] += 0.6

    if any(word in normalized for word in _PRAISE_WORDS):
        scores[IntentKind.PRAISE] += 0.6

    if any(word in normalized for word in _CLARIFY_WORDS):
        scores[IntentKind.CLARIFICATION] += 0.6

    total = sum(scores.values())
    if total <= 0.0:
        return IntentAnalysis(hypotheses=[IntentHypothesis(intent=IntentKind.OTHER, score=1.0)])

    normalized_scores = {k: v / total for k, v in scores.items() if v > 0.0}
    ordered = sorted(normalized_scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ordered[:3]
    return IntentAnalysis(
        hypotheses=[IntentHypothesis(intent=intent, score=score) for intent, score in top]
    )


def detect_paradox(text: str) -> ParadoxDetection:
    """
    Детерминированный детектор противоречий (paradox flags).
    Используется только для понижения уверенности выводов/кандидатов.
    """
    normalized = text.strip().lower()
    if not normalized:
        return ParadoxDetection(flags=[])

    flags: list[ParadoxFlag] = []

    if any(w in normalized for w in _STYLE_SHORT) and any(w in normalized for w in _STYLE_LONG):
        flags.append(ParadoxFlag.STYLE_CONFLICT)
    if "всегда" in normalized and "никогда" in normalized:
        flags.append(ParadoxFlag.ALWAYS_NEVER_CONFLICT)
    if "сделай" in normalized and "не делай" in normalized:
        flags.append(ParadoxFlag.DO_DONT_CONFLICT)

    return ParadoxDetection(flags=flags)
