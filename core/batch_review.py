from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from core.intent_paradox import analyze_intent, detect_paradox
from memory.memory_companion_store import MemoryCompanionStore
from shared.batch_review_models import (
    BatchReviewRun,
    CandidateStatus,
    EvidenceItem,
    IntentHypothesis,
    IntentKind,
    ParadoxFlag,
    PolicyRuleCandidate,
    Signal,
)
from shared.memory_companion_models import (
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
)
from shared.policy_models import (
    ActionAddInstruction,
    ActionSetResponseStyle,
    ResponseVerbosity,
    TriggerAlways,
    TriggerUserMessageContains,
)


@dataclass(frozen=True)
class BatchReviewResult:
    run: BatchReviewRun
    candidates: list[PolicyRuleCandidate]


_SHORT_STYLE_HINTS = ("кратко", "короче", "коротко", "без воды", "brief")
_SOURCES_HINTS = ("источник", "ссылка", "ссылки", "пруф", "sources", "source")
_SIMPLE_HINTS = ("простыми", "проще", "для новичка", "понятно", "easy")


def _format_ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _excerpt(text: str, limit: int = 140) -> str:
    cleaned = text.strip().replace("\n", " ")
    return cleaned[:limit]


def _label_to_signal(label: FeedbackLabel) -> Signal:
    return {
        FeedbackLabel.TOO_LONG: Signal.LABEL_TOO_LONG,
        FeedbackLabel.OFF_TOPIC: Signal.LABEL_OFF_TOPIC,
        FeedbackLabel.NO_SOURCES: Signal.LABEL_NO_SOURCES,
        FeedbackLabel.HALLUCINATION: Signal.LABEL_HALLUCINATION,
        FeedbackLabel.TOO_COMPLEX: Signal.LABEL_TOO_COMPLEX,
        FeedbackLabel.INCORRECT: Signal.LABEL_INCORRECT,
        FeedbackLabel.OTHER: Signal.LABEL_OTHER,
    }[label]


def _aggregate_intents(events: list[tuple[str, str | None]]) -> list[IntentHypothesis]:
    buckets: dict[IntentKind, float] = {k: 0.0 for k in IntentKind}
    samples = 0
    for prompt, free_text in events:
        analyses = [analyze_intent(prompt)]
        if free_text:
            analyses.append(analyze_intent(free_text))
        for analysis in analyses:
            for hyp in analysis.hypotheses:
                buckets[hyp.intent] += hyp.score
        samples += len(analyses)

    if samples <= 0:
        return [IntentHypothesis(intent=IntentKind.OTHER, score=1.0)]

    averaged = {k: v / samples for k, v in buckets.items() if v > 0.0}
    total = sum(averaged.values()) or 1.0
    normalized = {k: v / total for k, v in averaged.items()}
    ordered = sorted(normalized.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return [IntentHypothesis(intent=intent, score=score) for intent, score in ordered]


def _aggregate_paradox_flags(events: list[tuple[str, str | None]]) -> list[ParadoxFlag]:
    flags: list[ParadoxFlag] = []
    seen: set[ParadoxFlag] = set()
    for prompt, free_text in events:
        for src in (prompt, free_text or ""):
            det = detect_paradox(src)
            for flag in det.flags:
                if flag in seen:
                    continue
                seen.add(flag)
                flags.append(flag)
    return flags


def _confidence_suggestion(
    *,
    evidence_count: int,
    ratings: list[FeedbackRating],
    paradox_flags: list[ParadoxFlag],
    intent_hypotheses: list[IntentHypothesis],
    has_style_profanity_only: bool,
) -> float:
    base = min(0.2 + 0.15 * float(min(evidence_count, 5)), 0.75)
    if any(r is FeedbackRating.BAD for r in ratings):
        base += 0.05

    if paradox_flags:
        base *= 0.8

    top_score = intent_hypotheses[0].score if intent_hypotheses else 1.0
    if top_score < 0.55:
        base *= 0.85

    if has_style_profanity_only:
        base *= 0.9

    return max(0.05, min(base, 0.95))


def _propose_trigger(
    label: FeedbackLabel, prompts: list[str]
) -> TriggerAlways | TriggerUserMessageContains:
    normalized_prompts = " ".join(prompts).lower()
    keywords: list[str] = []

    if label is FeedbackLabel.TOO_LONG:
        keywords = [k for k in _SHORT_STYLE_HINTS if k in normalized_prompts]
    elif label is FeedbackLabel.NO_SOURCES:
        keywords = [k for k in _SOURCES_HINTS if k in normalized_prompts]
    elif label is FeedbackLabel.TOO_COMPLEX:
        keywords = [k for k in _SIMPLE_HINTS if k in normalized_prompts]

    unique = sorted({k.strip() for k in keywords if k.strip()})
    if unique:
        return TriggerUserMessageContains(substrings=unique, case_sensitive=False)
    return TriggerAlways()


def _candidate_action_for_label(
    label: FeedbackLabel,
) -> tuple[ActionAddInstruction | ActionSetResponseStyle, int]:
    if label is FeedbackLabel.TOO_LONG:
        return ActionSetResponseStyle(verbosity=ResponseVerbosity.CONCISE), 10
    if label is FeedbackLabel.TOO_COMPLEX:
        return (
            ActionAddInstruction(
                text="Объясняй проще: сначала коротко и понятно, затем детали по запросу."
            ),
            20,
        )
    if label is FeedbackLabel.NO_SOURCES:
        return (
            ActionAddInstruction(
                text=(
                    "Если утверждаешь факты — давай источники или явно помечай "
                    "допущения/неуверенность."
                )
            ),
            30,
        )
    if label is FeedbackLabel.HALLUCINATION:
        return (
            ActionAddInstruction(
                text=(
                    "Не выдумывай: если не уверен — скажи 'не знаю' "
                    "и предложи безопасный способ проверить."
                )
            ),
            50,
        )
    if label is FeedbackLabel.INCORRECT:
        return (
            ActionAddInstruction(
                text="Перепроверяй детали и делай self-check перед финальным ответом."
            ),
            40,
        )
    if label is FeedbackLabel.OFF_TOPIC:
        return (
            ActionAddInstruction(
                text=(
                    "Держись темы: сначала ответь строго по вопросу, затем "
                    "(если нужно) предложи расширение."
                )
            ),
            25,
        )
    return ActionAddInstruction(text="Учитывай фидбэк пользователя при формулировке ответа."), 5


def _contains_style_profanity_only(text: str) -> bool:
    normalized = text.lower()
    markers = ("blya", "бля", "fuck", "f*ck")
    return any(m in normalized for m in markers)


class BatchReviewer:
    def __init__(self, store: MemoryCompanionStore) -> None:
        self._store = store

    def run(self, *, user_id: str, period_days: int) -> BatchReviewResult:
        if period_days <= 0:
            raise ValueError("period_days должен быть > 0.")

        now = datetime.now()
        start = now - timedelta(days=period_days)
        period_start = _format_ts(start)
        period_end = _format_ts(now)

        interaction_count = self._store.count_chat_interactions(
            user_id=user_id, start_at=period_start, end_at=period_end
        )
        feedback_events = self._store.list_feedback_events_between(
            user_id=user_id, start_at=period_start, end_at=period_end
        )

        candidates = self._build_candidates(user_id=user_id, feedback_events=feedback_events)

        run_id = str(uuid.uuid4())
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        report_text = self._build_report_text(
            period_start=period_start,
            period_end=period_end,
            interaction_count=interaction_count,
            feedback_events=feedback_events,
            candidates=candidates,
        )
        run = BatchReviewRun(
            batch_review_run_id=run_id,
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            created_at=created_at,
            interaction_count=interaction_count,
            feedback_count=len(feedback_events),
            candidate_count=len(candidates),
            report_text=report_text,
        )

        stamped_candidates = [
            PolicyRuleCandidate(
                candidate_id=c.candidate_id,
                batch_review_run_id=run_id,
                user_id=c.user_id,
                proposed_trigger=c.proposed_trigger,
                proposed_action=c.proposed_action,
                priority_suggestion=c.priority_suggestion,
                confidence_suggestion=c.confidence_suggestion,
                evidence=c.evidence,
                signals=c.signals,
                intent_hypotheses=c.intent_hypotheses,
                paradox_flags=c.paradox_flags,
                status=CandidateStatus.PROPOSED,
                created_at=created_at,
                updated_at=created_at,
            )
            for c in candidates
        ]

        self._store.add_batch_review_run(run)
        self._store.add_policy_rule_candidates(stamped_candidates)

        return BatchReviewResult(run=run, candidates=stamped_candidates)

    def _build_candidates(
        self, *, user_id: str, feedback_events: list[FeedbackEvent]
    ) -> list[PolicyRuleCandidate]:
        grouped: dict[FeedbackLabel, list[FeedbackEvent]] = {}
        for event in feedback_events:
            for label in event.labels:
                grouped.setdefault(label, []).append(event)

        candidates: list[PolicyRuleCandidate] = []
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

        for label, events in grouped.items():
            evidence: list[EvidenceItem] = []
            prompt_texts: list[str] = []
            rating_list: list[FeedbackRating] = []
            signal_set: set[Signal] = set()
            style_profanity_only = False
            intent_inputs: list[tuple[str, str | None]] = []

            for ev in events[:10]:
                interaction = self._store.get_interaction(ev.interaction_id)
                prompt = ""
                if interaction is not None and interaction.interaction_kind == InteractionKind.CHAT:
                    prompt = interaction.raw_input
                excerpt = _excerpt(prompt or "[no prompt]")
                evidence.append(
                    EvidenceItem(
                        interaction_id=ev.interaction_id,
                        excerpt=excerpt,
                        feedback_id=ev.feedback_id,
                    )
                )
                prompt_texts.append(prompt or "")
                rating_list.append(ev.rating)
                intent_inputs.append((prompt or "", ev.free_text))
                if ev.free_text and _contains_style_profanity_only(ev.free_text):
                    style_profanity_only = True

                if ev.rating is FeedbackRating.BAD:
                    signal_set.add(Signal.DIRECT_NEGATIVE_FEEDBACK)
                elif ev.rating is FeedbackRating.GOOD:
                    signal_set.add(Signal.EXPLICIT_POSITIVE)
                else:
                    signal_set.add(Signal.RATING_OK)
                signal_set.add(_label_to_signal(label))

            if style_profanity_only:
                signal_set.add(Signal.STYLE_PROFANITY_ONLY)

            intent_hypotheses = _aggregate_intents(intent_inputs)
            paradox_flags = _aggregate_paradox_flags(intent_inputs)
            confidence = _confidence_suggestion(
                evidence_count=len(events),
                ratings=rating_list,
                paradox_flags=paradox_flags,
                intent_hypotheses=intent_hypotheses,
                has_style_profanity_only=style_profanity_only,
            )

            proposed_trigger = _propose_trigger(label, prompt_texts)
            action, priority = _candidate_action_for_label(label)
            if not isinstance(action, (ActionAddInstruction, ActionSetResponseStyle)):
                raise TypeError(f"Unsupported action: {type(action)}")

            candidates.append(
                PolicyRuleCandidate(
                    candidate_id=str(uuid.uuid4()),
                    batch_review_run_id="",
                    user_id=user_id,
                    proposed_trigger=proposed_trigger,
                    proposed_action=action,
                    priority_suggestion=priority,
                    confidence_suggestion=confidence,
                    evidence=evidence,
                    signals=sorted(signal_set, key=lambda s: s.value),
                    intent_hypotheses=intent_hypotheses,
                    paradox_flags=paradox_flags,
                    status=CandidateStatus.PROPOSED,
                    created_at=created_at,
                    updated_at=created_at,
                )
            )

        return candidates

    def _build_report_text(
        self,
        *,
        period_start: str,
        period_end: str,
        interaction_count: int,
        feedback_events: list[FeedbackEvent],
        candidates: list[PolicyRuleCandidate],
    ) -> str:
        rating_counts = Counter([e.rating.value for e in feedback_events])
        label_counts = Counter([label.value for e in feedback_events for label in e.labels])

        lines = [
            "BatchReview report (manual run)",
            f"period: {period_start} .. {period_end}",
            f"chat_interactions: {interaction_count}",
            f"feedback_events: {len(feedback_events)}",
            "ratings: " + ", ".join([f"{k}={v}" for k, v in sorted(rating_counts.items())])
            if rating_counts
            else "ratings: (none)",
            "labels: " + ", ".join([f"{k}={v}" for k, v in sorted(label_counts.items())])
            if label_counts
            else "labels: (none)",
            f"candidates_generated: {len(candidates)}",
            (
                "note: BatchReview only creates report + PolicyRuleCandidate[]; "
                "it does NOT modify PolicyRule/Memory."
            ),
        ]
        return "\n".join(lines)
