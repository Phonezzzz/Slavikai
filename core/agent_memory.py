from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING

from core.batch_review import BatchReviewer
from memory.memory_retrieval import build_memory_capsule as build_memory_capsule_payload
from shared.batch_review_models import (
    BatchReviewRun,
    CandidateStatus,
    PolicyRuleCandidate,
)
from shared.canonical_atom_models import (
    AtomStatus,
    CanonicalAtom,
    Claim,
    ClaimExtractionInput,
    ClaimType,
)
from shared.memory_companion_models import (
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionLog,
)
from shared.models import JSONValue, LLMMessage, MemoryKind
from shared.policy_models import (
    PolicyRule,
    PolicyScope,
    policy_action_from_json,
    policy_trigger_from_json,
)

if TYPE_CHECKING:
    from config.memory_config import MemoryConfig
    from core.tracer import Tracer
    from memory.atom_embedding_index import AtomEmbeddingIndex
    from memory.canonical_aggregator import CanonicalAggregator
    from memory.canonical_atom_store import CanonicalAtomStore
    from memory.claim_extractor import ClaimExtractor
    from memory.memory_companion_store import MemoryCompanionStore
    from memory.memory_manager import MemoryManager
    from memory.memory_retrieval import RetrievalConfig
    from memory.session_summarizer import SessionSummarizer
    from memory.vector_index import VectorIndex


_DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS = 30
_EXPLICIT_MEMORY_PREFIX = re.compile(r"^\s*(?:запомни|remember)\b[:\-\s]*", re.IGNORECASE)
_FEEDBACK_LABEL_HINTS: dict[FeedbackLabel, tuple[str, str]] = {
    FeedbackLabel.OFF_TOPIC: ("fatal", "Держись темы вопроса."),
    FeedbackLabel.HALLUCINATION: ("major", "Проверь факты и избегай галлюцинаций."),
    FeedbackLabel.INCORRECT: ("major", "Проверяй корректность ответа."),
    FeedbackLabel.NO_SOURCES: ("major", "Добавляй источники при необходимости."),
    FeedbackLabel.TOO_LONG: ("minor", "Делай ответ короче."),
    FeedbackLabel.TOO_COMPLEX: ("minor", "Упрощай объяснение."),
    FeedbackLabel.OTHER: ("minor", "Улучшай качество ответа."),
}


class AgentMemoryMixin:
    if TYPE_CHECKING:
        logger: logging.Logger
        tracer: Tracer
        user_id: str
        session_id: str | None
        conversation_id: str
        short_term: list[LLMMessage]
        memory_config: MemoryConfig
        memory: MemoryManager
        vectors: VectorIndex
        _interaction_store: MemoryCompanionStore
        _claim_extractor: ClaimExtractor
        _canonical_aggregator: CanonicalAggregator
        _canonical_store: CanonicalAtomStore
        _atom_embedding_index: AtomEmbeddingIndex
        _session_summarizer: SessionSummarizer
        _retrieval_config: RetrievalConfig
        workspace_file_path: str | None
        workspace_file_content: str | None
        workspace_selection: str | None
        last_context_text: str | None
        last_hints_used: list[str]
        last_hints_meta: list[dict[str, str]]

    def get_recent_feedback_events(self, limit: int = 50) -> list[FeedbackEvent]:
        return self._interaction_store.get_recent_feedback(user_id=self.user_id, limit=limit)

    def get_feedback_stats(self) -> dict[FeedbackRating, int]:
        return self._interaction_store.get_feedback_stats(user_id=self.user_id)

    def get_interaction_log(self, interaction_id: str) -> InteractionLog | None:
        return self._interaction_store.get_interaction(interaction_id)

    def run_batch_review(self, *, period_days: int) -> BatchReviewRun:
        reviewer = BatchReviewer(self._interaction_store)
        result = reviewer.run(user_id=self.user_id, period_days=period_days)
        self.tracer.log(
            "batch_review_completed",
            f"candidates={result.run.candidate_count}",
            {"run_id": result.run.batch_review_run_id, "period_days": period_days},
        )
        return result.run

    def get_recent_batch_review_runs(self, limit: int = 20) -> list[BatchReviewRun]:
        return self._interaction_store.get_recent_batch_review_runs(
            user_id=self.user_id,
            limit=limit,
        )

    def list_policy_rule_candidates(
        self,
        *,
        run_id: str | None = None,
        status: CandidateStatus | None = None,
        limit: int = 200,
    ) -> list[PolicyRuleCandidate]:
        return self._interaction_store.list_policy_rule_candidates(
            user_id=self.user_id,
            run_id=run_id,
            status=status,
            limit=limit,
        )

    def approve_policy_rule_candidate(
        self,
        *,
        candidate_id: str,
        scope: PolicyScope = PolicyScope.USER,
        decay_half_life_days: int = _DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS,
        override_trigger_json: str | None = None,
        override_action_json: str | None = None,
        override_priority: int | None = None,
        override_confidence: float | None = None,
    ) -> PolicyRule:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")
        if decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days должен быть > 0.")

        trigger = (
            policy_trigger_from_json(override_trigger_json)
            if override_trigger_json is not None
            else candidate.proposed_trigger
        )
        action = (
            policy_action_from_json(override_action_json)
            if override_action_json is not None
            else candidate.proposed_action
        )
        priority = (
            override_priority if override_priority is not None else candidate.priority_suggestion
        )
        confidence = (
            float(override_confidence)
            if override_confidence is not None
            else candidate.confidence_suggestion
        )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence должен быть в диапазоне 0..1.")

        feedback_ids = sorted({e.feedback_id for e in candidate.evidence if e.feedback_id})
        provenance = (
            f"batch_review_run_id:{candidate.batch_review_run_id};"
            f"candidate_id:{candidate.candidate_id}"
        )
        if len(feedback_ids) == 1:
            provenance += f";feedback_id:{feedback_ids[0]}"
        elif feedback_ids:
            provenance += f";feedback_ids:{','.join(feedback_ids)}"

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rule = PolicyRule(
            rule_id=str(uuid.uuid4()),
            user_id=self.user_id,
            scope=scope,
            trigger=trigger,
            action=action,
            priority=priority,
            confidence=confidence,
            decay_half_life_days=decay_half_life_days,
            provenance=provenance,
            created_at=now,
            updated_at=now,
        )

        self._interaction_store.approve_policy_rule_candidate(
            candidate_id=candidate_id,
            user_id=self.user_id,
            approved_rule=rule,
            final_trigger=trigger,
            final_action=action,
            final_priority=priority,
            final_confidence=confidence,
            updated_at=now,
        )
        self.tracer.log(
            "policy_rule_approved",
            rule.rule_id,
            {"candidate_id": candidate_id, "run_id": candidate.batch_review_run_id},
        )
        return rule

    def update_policy_rule_candidate_suggestion(
        self,
        *,
        candidate_id: str,
        proposed_trigger_json: str,
        proposed_action_json: str,
        priority_suggestion: int,
        confidence_suggestion: float,
    ) -> PolicyRuleCandidate:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")

        trigger = policy_trigger_from_json(proposed_trigger_json)
        action = policy_action_from_json(proposed_action_json)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self._interaction_store.update_policy_rule_candidate_suggestion(
            candidate_id=candidate_id,
            user_id=self.user_id,
            proposed_trigger=trigger,
            proposed_action=action,
            priority_suggestion=priority_suggestion,
            confidence_suggestion=confidence_suggestion,
            updated_at=now,
        )
        self.tracer.log("policy_candidate_updated", candidate_id)
        updated = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if updated is None:
            raise RuntimeError("Candidate missing after update (unexpected).")
        return updated

    def reject_policy_rule_candidate(self, *, candidate_id: str) -> None:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self._interaction_store.reject_policy_rule_candidate(
            candidate_id=candidate_id,
            user_id=self.user_id,
            updated_at=now,
        )
        self.tracer.log("policy_candidate_rejected", candidate_id)

    def capture_memory_claims_from_text(
        self,
        text: str,
        *,
        source_kind: str,
        source_id: str,
        lang_hint: str | None = None,
    ) -> list[dict[str, JSONValue]]:
        payload = ClaimExtractionInput(
            text=text,
            source_kind=source_kind,
            source_id=source_id,
            lang_hint=lang_hint,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        )
        claims = self._claim_extractor.extract(payload)
        applied: list[dict[str, JSONValue]] = []
        for claim in claims:
            if not self._should_promote_claim(claim):
                continue
            atom = self._canonical_aggregator.upsert_claim(claim)
            self._atom_embedding_index.sync_atom(atom)
            applied.append(
                {
                    "stable_key": atom.stable_key,
                    "status": atom.status.value,
                    "claim_type": atom.claim_type.value,
                    "confidence": atom.confidence,
                }
            )
        if applied:
            self.tracer.log(
                "memory_claims_applied",
                f"Applied {len(applied)} claims",
                {"claims": applied, "source_kind": source_kind, "source_id": source_id},
            )
        return applied

    def is_explicit_memory_request(self, text: str) -> bool:
        return _EXPLICIT_MEMORY_PREFIX.match(text.strip()) is not None

    def remember_explicit_text(
        self,
        text: str,
        *,
        source_kind: str,
        source_id: str | None = None,
        lang_hint: str | None = None,
    ) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "Нечего запомнить."

        capture_text = (
            cleaned if self.is_explicit_memory_request(cleaned) else f"remember {cleaned}"
        )
        resolved_source_id = source_id or self.session_id or self.conversation_id
        applied = self.capture_memory_claims_from_text(
            capture_text,
            source_kind=source_kind,
            source_id=resolved_source_id,
            lang_hint=lang_hint,
        )
        if not applied:
            return "Не удалось выделить факт для памяти."

        stable_keys = [
            str(item["stable_key"])
            for item in applied
            if isinstance(item.get("stable_key"), str) and item["stable_key"]
        ]
        if not stable_keys:
            return f"Запомнил: {len(applied)}"
        return f"Запомнил: {', '.join(stable_keys)}"

    def build_memory_capsule(
        self,
        query: str,
        *,
        for_mwv: bool = False,
        allow_vector_runtime_init: bool = True,
    ) -> dict[str, JSONValue]:
        return build_memory_capsule_payload(
            query=query,
            store=self._canonical_store,
            vector_index=self.vectors,
            for_mwv=for_mwv,
            config=self._retrieval_config,
            allow_vector_runtime_init=allow_vector_runtime_init,
        )

    def list_memory_conflicts(self, limit: int = 50) -> list[dict[str, JSONValue]]:
        atoms = self._canonical_store.list_conflicts(limit=limit)
        return [self._atom_to_payload(atom) for atom in atoms]

    def list_pinned_memory_atoms(self, limit: int = 20) -> list[dict[str, JSONValue]]:
        atoms = self._canonical_store.list_pinned(limit=limit)
        return [self._atom_to_payload(atom) for atom in atoms]

    def pin_memory_atom(self, stable_key: str) -> dict[str, JSONValue] | None:
        if not self._canonical_store.set_pinned(stable_key, True):
            return None
        atom = self._canonical_store.get_by_stable_key(stable_key)
        if atom is None:
            return None
        self.tracer.log("memory_atom_pinned", stable_key, {"stable_key": stable_key})
        return self._atom_to_payload(atom)

    def unpin_memory_atom(self, stable_key: str) -> dict[str, JSONValue] | None:
        if not self._canonical_store.set_pinned(stable_key, False):
            return None
        atom = self._canonical_store.get_by_stable_key(stable_key)
        if atom is None:
            return None
        self.tracer.log("memory_atom_unpinned", stable_key, {"stable_key": stable_key})
        return self._atom_to_payload(atom)

    def summarize_current_session(self) -> dict[str, JSONValue] | None:
        summary = self._session_summarizer.summarize(self.short_term)
        if summary is None:
            self.tracer.log("session_summary_skipped", "No session messages to summarize")
            return None

        atom = self._canonical_aggregator.upsert_claim(summary.claim)
        self._atom_embedding_index.sync_atom(atom)
        self.tracer.log(
            "session_summary_saved",
            atom.stable_key,
            {"stable_key": atom.stable_key, "chars": len(summary.text)},
        )
        return self._atom_to_payload(atom)

    def resolve_memory_conflict(
        self,
        *,
        stable_key: str,
        action: str,
        value_json: JSONValue | None = None,
    ) -> dict[str, JSONValue] | None:
        atom = self._canonical_aggregator.resolve_conflict(
            stable_key=stable_key,
            action=action,
            value_json=value_json,
        )
        if atom is None:
            return None
        self._atom_embedding_index.sync_atom(atom)
        self.tracer.log(
            "memory_conflict_resolved",
            stable_key,
            {"action": action, "status": atom.status.value},
        )
        return self._atom_to_payload(atom)

    def _should_promote_claim(self, claim: Claim) -> bool:
        if claim.is_explicit:
            return True
        return claim.claim_type in {
            ClaimType.PREFERENCE,
            ClaimType.ENVIRONMENT,
            ClaimType.FACT,
        }

    def _atom_to_payload(self, atom: CanonicalAtom) -> dict[str, JSONValue]:
        return {
            "atom_id": atom.atom_id,
            "stable_key": atom.stable_key,
            "claim_type": atom.claim_type.value,
            "value_json": atom.value_json,
            "confidence": atom.confidence,
            "support_count": atom.support_count,
            "contradict_count": atom.contradict_count,
            "last_seen_at": atom.last_seen_at,
            "status": atom.status.value,
            "summary_text": atom.summary_text,
            "pinned": atom.pinned,
        }

    def _build_context_messages(self, messages: list[LLMMessage], query: str) -> list[LLMMessage]:
        budget = self.memory_config.context_budget
        remaining = budget.total_chars
        filled_slots: list[str] = []
        slot_sizes: dict[str, int] = {}

        def _append_slot(name: str, raw_text: str, max_chars: int) -> None:
            nonlocal remaining
            text = raw_text.strip()
            if not text or max_chars <= 0:
                slot_sizes[name] = 0
                return
            if remaining <= 0:
                slot_sizes[name] = 0
                self.tracer.log("context_budget_exhausted", name, {"slot": name})
                return

            text = text[:max_chars].rstrip()
            prefix = f"[[SLOT:{name}]]\n"
            suffix = "\n[[/SLOT]]"
            separator_len = 2 if filled_slots else 0
            overhead = separator_len + len(prefix) + len(suffix)
            if remaining <= overhead:
                slot_sizes[name] = 0
                self.tracer.log(
                    "context_budget_exhausted",
                    name,
                    {"slot": name, "remaining": remaining},
                )
                return

            allowed_payload = min(len(text), remaining - overhead)
            text = text[:allowed_payload].rstrip()
            if not text:
                slot_sizes[name] = 0
                return
            slot = f"{prefix}{text}{suffix}"
            filled_slots.append(slot)
            slot_sizes[name] = len(slot)
            remaining -= len(slot) + separator_len

        pinned_atoms = self._canonical_store.list_pinned(limit=20)
        if pinned_atoms:
            pinned_parts = ["Закрепленная память:"]
            for atom in pinned_atoms:
                pinned_parts.append(
                    f"- [{atom.claim_type.value}] {atom.stable_key}: {atom.summary_text}"
                )
            _append_slot("pinned_atoms", "\n".join(pinned_parts), budget.pinned_atoms_chars)

        session_atoms = self._canonical_store.list_atoms(
            statuses={AtomStatus.ACTIVE},
            claim_types={ClaimType.FACT},
            stable_key_prefix="session:",
            limit=3,
        )
        if session_atoms:
            session_parts = ["Резюме прошлых сессий:"]
            for atom in session_atoms:
                summary_text = atom.summary_text
                value = atom.value_json
                if isinstance(value, dict):
                    raw_text = value.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        summary_text = raw_text.strip()
                session_parts.append(f"- {atom.stable_key}: {summary_text[:500]}")
            _append_slot(
                "session_summary",
                "\n".join(session_parts),
                budget.session_summary_chars,
            )

        recent_notes = self.memory.get_recent(
            max(1, budget.legacy_notes_chars // 200),
            kind=MemoryKind.NOTE,
        )
        if recent_notes:
            note_parts = ["Недавняя память:"]
            for note in recent_notes:
                note_parts.append(f"- {note.content[:200]}")
            _append_slot("legacy_notes", "\n".join(note_parts), budget.legacy_notes_chars)

        hints_meta = self._collect_feedback_hints(
            budget.feedback_max_items,
            severity_filter=["major", "fatal"],
        )
        if hints_meta:
            hint_parts = ["Подсказки от пользователя:"]
            for hint_meta in hints_meta:
                hint_parts.append(f"- ({hint_meta.get('severity')}) {hint_meta.get('hint')}")
            _append_slot("feedback", "\n".join(hint_parts), budget.feedback_chars)
            self.last_hints_used = [h["hint"] for h in hints_meta]
            self.last_hints_meta = hints_meta
            self.tracer.log("auto_hint_applied", "Использованы подсказки", {"hints": hints_meta})
        else:
            self.last_hints_used = []
            self.last_hints_meta = []

        prefs_all = self.memory.get_user_prefs()
        prefs = prefs_all[: budget.prefs_max_items]
        if len(prefs_all) > len(prefs):
            self.tracer.log(
                "prefs_truncated",
                f"{len(prefs_all)} -> {len(prefs)}",
                {"total": len(prefs_all), "kept": len(prefs)},
            )
        if prefs:
            pref_parts = ["Предпочтения пользователя:"]
            for pref in prefs:
                meta = pref.meta or {}
                pref_parts.append(f"- {meta.get('key')}: {meta.get('value')}")
            _append_slot("prefs", "\n".join(pref_parts), budget.prefs_chars)

        try:
            retrieval_config = replace(
                self._retrieval_config,
                max_context_chars=budget.canonical_memory_chars,
            )
            memory_capsule = build_memory_capsule_payload(
                query=query,
                store=self._canonical_store,
                vector_index=self.vectors,
                for_mwv=False,
                config=retrieval_config,
                allow_vector_runtime_init=True,
            )
            capsule_text = memory_capsule.get("text")
            if isinstance(capsule_text, str) and capsule_text.strip():
                canonical_text = "\n".join(["Каноническая память:", *capsule_text.splitlines()])
                _append_slot(
                    "canonical_memory",
                    canonical_text,
                    budget.canonical_memory_chars,
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Canonical memory retrieval failed: %s", exc)

        # Векторный поиск по проектному индексу (code + docs)
        try:
            code_top_k = max(1, budget.vector_code_chars // 400)
            docs_top_k = max(1, budget.vector_docs_chars // 400)
            vec_results_code = self.vectors.search(
                query,
                namespace="code",
                top_k=code_top_k,
                allow_runtime_init=True,
            )
            vec_results_docs = self.vectors.search(
                query,
                namespace="docs",
                top_k=docs_top_k,
                allow_runtime_init=True,
            )
            if vec_results_code:
                code_parts = ["Контекст проекта (code):"]
                for res in vec_results_code:
                    code_parts.append(f"- {res.path}: {res.snippet}")
                _append_slot("vectors_code", "\n".join(code_parts), budget.vector_code_chars)
            if vec_results_docs:
                docs_parts = ["Контекст проекта (docs):"]
                for res in vec_results_docs:
                    docs_parts.append(f"- {res.path}: {res.snippet}")
                _append_slot("vectors_docs", "\n".join(docs_parts), budget.vector_docs_chars)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Vector search failed: %s", exc)

        if self.workspace_file_path and self.workspace_file_content is not None:
            workspace_parts = ["Текущий файл:", f"- path: {self.workspace_file_path}"]
            if self.workspace_selection:
                selection_snippet = self.workspace_selection[: budget.workspace_file_chars]
                workspace_parts.append(f"- выделение:\n{selection_snippet}")
            content_snippet = self.workspace_file_content
            if len(content_snippet) > budget.workspace_file_chars:
                content_snippet = content_snippet[: budget.workspace_file_chars]
            workspace_parts.append(f"- содержимое:\n{content_snippet}")
            _append_slot("workspace_file", "\n".join(workspace_parts), budget.workspace_file_chars)

        if filled_slots:
            context_msg = "\n\n".join(filled_slots)
            self.last_context_text = context_msg
            self.tracer.log(
                "context_built",
                f"total_chars={len(context_msg)}",
                {"total_chars": len(context_msg), "slots": slot_sizes},
            )
            return [LLMMessage(role="system", content=context_msg), *messages]
        self.last_context_text = None
        return messages

    def _collect_feedback_hints(
        self,
        limit: int,
        severity_filter: list[str] | None = None,
    ) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        scan_limit = max(limit * 5, limit)
        events = self._interaction_store.get_recent_feedback(user_id=self.user_id, limit=scan_limit)
        hints: list[dict[str, str]] = []
        for event in events:
            meta = self._feedback_event_to_hint(event)
            if not meta:
                continue
            severity = meta.get("severity")
            if severity_filter and severity not in severity_filter:
                continue
            hints.append(meta)
            if len(hints) >= limit:
                break
        return hints

    def _feedback_event_to_hint(self, event: FeedbackEvent) -> dict[str, str] | None:
        if event.rating is FeedbackRating.GOOD:
            return None
        free_text = event.free_text.strip() if event.free_text else ""
        label_hint = self._best_label_hint(event.labels)
        severity = label_hint[0] if label_hint else "minor"
        hint = label_hint[1] if label_hint else ""
        if event.rating is FeedbackRating.BAD:
            severity = self._max_severity(severity, "major")
            if not hint:
                hint = "Проверь факты и избегай галлюцинаций."
        if free_text:
            hint = free_text
        if not hint:
            return None
        return {
            "severity": severity,
            "hint": hint,
            "timestamp": event.created_at,
            "feedback_id": event.feedback_id,
            "interaction_id": event.interaction_id,
            "rating": event.rating.value,
        }

    def _best_label_hint(self, labels: list[FeedbackLabel]) -> tuple[str, str] | None:
        best: tuple[str, str] | None = None
        best_rank = -1
        for label in labels:
            severity, hint = _FEEDBACK_LABEL_HINTS.get(label, ("minor", "Улучшай качество ответа."))
            rank = self._severity_rank(severity)
            if rank > best_rank:
                best = (severity, hint)
                best_rank = rank
        return best

    def _max_severity(self, current: str, incoming: str) -> str:
        if self._severity_rank(incoming) > self._severity_rank(current):
            return incoming
        return current

    def _severity_rank(self, severity: str) -> int:
        ranks = {"minor": 0, "major": 1, "fatal": 2}
        return ranks.get(severity, 0)
