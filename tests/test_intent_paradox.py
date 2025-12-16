from __future__ import annotations

from core.intent_paradox import analyze_intent, detect_paradox
from shared.batch_review_models import IntentKind, ParadoxFlag


def test_intent_analysis_returns_multiple_hypotheses() -> None:
    analysis = analyze_intent("Как исправить ошибку? Сделай кратко.")
    assert len(analysis.hypotheses) >= 2
    assert analysis.hypotheses[0].score >= analysis.hypotheses[1].score
    total = sum(h.score for h in analysis.hypotheses)
    assert abs(total - 1.0) < 1e-6
    assert any(h.intent == IntentKind.QUESTION for h in analysis.hypotheses)


def test_paradox_detection_flags_style_conflict() -> None:
    det = detect_paradox("Ответь кратко и подробно.")
    assert ParadoxFlag.STYLE_CONFLICT in det.flags
