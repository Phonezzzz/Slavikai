from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Final

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
    BlockedReason,
    ChatInteractionLog,
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionLog,
    InteractionMode,
    ToolInteractionLog,
    ToolStatus,
)
from shared.models import JSONValue
from shared.policy_models import (
    PolicyAction,
    PolicyRule,
    PolicyScope,
    PolicyTrigger,
    policy_action_from_json,
    policy_action_to_json,
    policy_trigger_from_json,
    policy_trigger_to_json,
)

DEFAULT_DB_PATH: Final[Path] = Path("memory/memory_companion.db")
SCHEMA_VERSION: Final[int] = 4
_SCHEMA_KEY: Final[str] = "schema_version"
_MAX_TOOL_OUTPUT_PREVIEW_CHARS: Final[int] = 2_000


class MemoryCompanionSchemaError(RuntimeError):
    pass


class SchemaVersionMismatchError(MemoryCompanionSchemaError):
    def __init__(self, path: Path, expected: int, actual: int) -> None:
        message = (
            "MemoryCompanion DB schema version mismatch: "
            f"expected={expected}, actual={actual}, path={path}"
        )
        super().__init__(message)
        self.path = path
        self.expected = expected
        self.actual = actual


class InvalidMemoryCompanionDbError(MemoryCompanionSchemaError):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(f"Invalid MemoryCompanion DB at {path}: {message}")
        self.path = path


def _json_default(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _coerce_json_value(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object key must be a string.")
            out[key] = _coerce_json_value(item)
        return out
    raise ValueError(f"Unsupported JSON value type: {type(value)}")


def _loads_json_value(text: str) -> JSONValue:
    parsed = json.loads(text)
    return _coerce_json_value(parsed)


def _loads_str_list(text: str) -> list[str]:
    value = _loads_json_value(text)
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("Expected JSON list[str].")
    return value


def _loads_dict(text: str) -> dict[str, JSONValue]:
    value = _loads_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object.")
    return value


def _loads_feedback_labels(text: str) -> list[FeedbackLabel]:
    raw = _loads_str_list(text)
    return [FeedbackLabel(item) for item in raw]


def _dumps_feedback_labels(labels: list[FeedbackLabel]) -> str:
    return _dumps([label.value for label in labels])


def _loads_signals(text: str) -> list[Signal]:
    raw = _loads_str_list(text)
    return [Signal(item) for item in raw]


def _dumps_signals(signals: list[Signal]) -> str:
    return _dumps([signal.value for signal in signals])


def _loads_paradox_flags(text: str) -> list[ParadoxFlag]:
    raw = _loads_str_list(text)
    return [ParadoxFlag(item) for item in raw]


def _dumps_paradox_flags(flags: list[ParadoxFlag]) -> str:
    return _dumps([flag.value for flag in flags])


def _loads_intent_hypotheses(text: str) -> list[IntentHypothesis]:
    value = _loads_json_value(text)
    if not isinstance(value, list):
        raise ValueError("Expected JSON list for intent hypotheses.")
    result: list[IntentHypothesis] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Intent hypothesis must be an object.")
        intent_raw = item.get("intent")
        score_raw = item.get("score")
        if not isinstance(intent_raw, str):
            raise ValueError("Intent hypothesis.intent must be a string.")
        if not isinstance(score_raw, (int, float)):
            raise ValueError("Intent hypothesis.score must be a number.")
        result.append(IntentHypothesis(intent=IntentKind(intent_raw), score=float(score_raw)))
    return result


def _dumps_intent_hypotheses(items: list[IntentHypothesis]) -> str:
    return _dumps([{"intent": h.intent.value, "score": h.score} for h in items])


def _loads_evidence_items(text: str) -> list[EvidenceItem]:
    value = _loads_json_value(text)
    if not isinstance(value, list):
        raise ValueError("Expected JSON list for evidence.")
    items: list[EvidenceItem] = []
    for obj in value:
        if not isinstance(obj, dict):
            raise ValueError("Evidence item must be an object.")
        interaction_id = obj.get("interaction_id")
        excerpt = obj.get("excerpt")
        feedback_id = obj.get("feedback_id")
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("EvidenceItem.interaction_id must be a non-empty string.")
        if not isinstance(excerpt, str):
            raise ValueError("EvidenceItem.excerpt must be a string.")
        if feedback_id is not None and not isinstance(feedback_id, str):
            raise ValueError("EvidenceItem.feedback_id must be string or null.")
        items.append(
            EvidenceItem(
                interaction_id=interaction_id,
                excerpt=excerpt,
                feedback_id=feedback_id if isinstance(feedback_id, str) and feedback_id else None,
            )
        )
    return items


def _dumps_evidence_items(items: list[EvidenceItem]) -> str:
    return _dumps(
        [
            {
                "interaction_id": item.interaction_id,
                "excerpt": item.excerpt,
                "feedback_id": item.feedback_id,
            }
            for item in items
        ]
    )


class MemoryCompanionStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not self.db_path.exists()
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

        if is_new:
            self._create_schema()
        else:
            self._verify_schema()

    def close(self) -> None:
        self.conn.close()

    def log_interaction(self, log: InteractionLog) -> None:
        if isinstance(log, ChatInteractionLog):
            self._insert_chat(log)
            return
        if isinstance(log, ToolInteractionLog):
            self._insert_tool(log)
            return
        raise TypeError(f"Unsupported InteractionLog: {type(log)}")

    def get_recent(self, limit: int = 50) -> list[InteractionLog]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM interaction_log ORDER BY created_at DESC, interaction_id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        result: list[InteractionLog] = []
        for row in rows:
            kind_raw = row["interaction_kind"]
            if kind_raw == InteractionKind.CHAT.value:
                result.append(self._row_to_chat(row))
            elif kind_raw == InteractionKind.TOOL.value:
                result.append(self._row_to_tool(row))
            else:
                raise InvalidMemoryCompanionDbError(
                    self.db_path, f"Unknown interaction_kind: {kind_raw!r}"
                )
        return result

    def get_interaction(self, interaction_id: str) -> InteractionLog | None:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM interaction_log WHERE interaction_id = ?", (interaction_id,))
        row = cur.fetchone()
        if row is None:
            return None
        kind_raw = row["interaction_kind"]
        if kind_raw == InteractionKind.CHAT.value:
            return self._row_to_chat(row)
        if kind_raw == InteractionKind.TOOL.value:
            return self._row_to_tool(row)
        raise InvalidMemoryCompanionDbError(self.db_path, f"Unknown interaction_kind: {kind_raw!r}")

    def add_policy_rule(self, rule: PolicyRule) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO policy_rule (
                    rule_id, user_id, scope,
                    trigger_json, action_json,
                    priority, confidence, decay_half_life_days,
                    provenance,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.rule_id,
                    rule.user_id,
                    rule.scope.value,
                    policy_trigger_to_json(rule.trigger),
                    policy_action_to_json(rule.action),
                    rule.priority,
                    rule.confidence,
                    rule.decay_half_life_days,
                    rule.provenance,
                    rule.created_at,
                    rule.updated_at,
                ),
            )

    def list_policy_rules(self, user_id: str) -> list[PolicyRule]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM policy_rule
            WHERE scope = 'global' OR (scope = 'user' AND user_id = ?)
            ORDER BY
                priority DESC,
                CASE WHEN scope = 'user' THEN 1 ELSE 0 END DESC,
                confidence DESC,
                updated_at DESC,
                rule_id ASC
            """,
            (user_id,),
        )
        rows = cur.fetchall()
        return [self._row_to_policy_rule(row) for row in rows]

    def add_feedback_event(self, event: FeedbackEvent) -> None:
        if not event.feedback_id.strip():
            raise ValueError("feedback_id не должен быть пустым.")
        if not event.interaction_id.strip():
            raise ValueError("interaction_id не должен быть пустым.")
        if not event.user_id.strip():
            raise ValueError("user_id не должен быть пустым.")
        if not event.created_at.strip():
            raise ValueError("created_at обязателен.")
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO feedback_event (
                        feedback_id, interaction_id, user_id,
                        rating, labels, free_text, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.feedback_id,
                        event.interaction_id,
                        event.user_id,
                        event.rating.value,
                        _dumps_feedback_labels(event.labels),
                        event.free_text,
                        event.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"feedback_event не может ссылаться на interaction_id={event.interaction_id!r}"
            ) from exc

    def get_recent_feedback(self, *, user_id: str, limit: int = 50) -> list[FeedbackEvent]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM feedback_event
            WHERE user_id = ?
            ORDER BY created_at DESC, feedback_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [self._row_to_feedback_event(row) for row in rows]

    def get_feedback_stats(self, *, user_id: str) -> dict[FeedbackRating, int]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT rating, COUNT(*) FROM feedback_event WHERE user_id = ? GROUP BY rating",
            (user_id,),
        )
        counts: dict[FeedbackRating, int] = {r: 0 for r in FeedbackRating}
        for rating_raw, count_raw in cur.fetchall():
            rating = FeedbackRating(str(rating_raw))
            counts[rating] = int(count_raw)
        return counts

    def count_chat_interactions(self, *, user_id: str, start_at: str, end_at: str) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*) FROM interaction_log
            WHERE user_id = ?
              AND interaction_kind = 'chat'
              AND created_at >= ?
              AND created_at <= ?
            """,
            (user_id, start_at, end_at),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def list_feedback_events_between(
        self,
        *,
        user_id: str,
        start_at: str,
        end_at: str,
        limit: int = 1000,
    ) -> list[FeedbackEvent]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM feedback_event
            WHERE user_id = ?
              AND created_at >= ?
              AND created_at <= ?
            ORDER BY created_at ASC, feedback_id ASC
            LIMIT ?
            """,
            (user_id, start_at, end_at, limit),
        )
        rows = cur.fetchall()
        return [self._row_to_feedback_event(row) for row in rows]

    def add_batch_review_run(self, run: BatchReviewRun) -> None:
        if not run.batch_review_run_id.strip():
            raise ValueError("batch_review_run_id не должен быть пустым.")
        if not run.user_id.strip():
            raise ValueError("user_id не должен быть пустым.")
        if not run.period_start.strip() or not run.period_end.strip():
            raise ValueError("period_start/period_end обязательны.")
        if not run.created_at.strip():
            raise ValueError("created_at обязателен.")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO batch_review_run (
                    batch_review_run_id, user_id,
                    period_start, period_end, created_at,
                    interaction_count, feedback_count, candidate_count,
                    report_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.batch_review_run_id,
                    run.user_id,
                    run.period_start,
                    run.period_end,
                    run.created_at,
                    run.interaction_count,
                    run.feedback_count,
                    run.candidate_count,
                    run.report_text,
                ),
            )

    def get_recent_batch_review_runs(
        self, *, user_id: str, limit: int = 20
    ) -> list[BatchReviewRun]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM batch_review_run
            WHERE user_id = ?
            ORDER BY created_at DESC, batch_review_run_id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cur.fetchall()
        return [self._row_to_batch_review_run(row) for row in rows]

    def add_policy_rule_candidates(self, candidates: list[PolicyRuleCandidate]) -> None:
        with self.conn:
            for candidate in candidates:
                self.conn.execute(
                    """
                    INSERT INTO policy_rule_candidate (
                        candidate_id, batch_review_run_id, user_id,
                        proposed_trigger_json, proposed_action_json,
                        priority_suggestion, confidence_suggestion,
                        evidence_json, signals_json,
                        intent_hypotheses_json, paradox_flags_json,
                        status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.candidate_id,
                        candidate.batch_review_run_id,
                        candidate.user_id,
                        policy_trigger_to_json(candidate.proposed_trigger),
                        policy_action_to_json(candidate.proposed_action),
                        candidate.priority_suggestion,
                        candidate.confidence_suggestion,
                        _dumps_evidence_items(candidate.evidence),
                        _dumps_signals(candidate.signals),
                        _dumps_intent_hypotheses(candidate.intent_hypotheses),
                        _dumps_paradox_flags(candidate.paradox_flags),
                        candidate.status.value,
                        candidate.created_at,
                        candidate.updated_at,
                    ),
                )

    def list_policy_rule_candidates(
        self,
        *,
        user_id: str,
        run_id: str | None = None,
        status: CandidateStatus | None = None,
        limit: int = 200,
    ) -> list[PolicyRuleCandidate]:
        clauses = ["user_id = ?"]
        params: list[object] = [user_id]
        if run_id is not None:
            clauses.append("batch_review_run_id = ?")
            params.append(run_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)

        where = " AND ".join(clauses)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT * FROM policy_rule_candidate
            WHERE {where}
            ORDER BY created_at DESC, candidate_id DESC
            LIMIT ?
            """,
            (*params, limit),
        )
        rows = cur.fetchall()
        return [self._row_to_policy_rule_candidate(row) for row in rows]

    def get_policy_rule_candidate(self, *, candidate_id: str) -> PolicyRuleCandidate | None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM policy_rule_candidate WHERE candidate_id = ?",
            (candidate_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_policy_rule_candidate(row)

    def update_policy_rule_candidate_suggestion(
        self,
        *,
        candidate_id: str,
        user_id: str,
        proposed_trigger: PolicyTrigger,
        proposed_action: PolicyAction,
        priority_suggestion: int,
        confidence_suggestion: float,
        updated_at: str,
    ) -> None:
        if not candidate_id.strip():
            raise ValueError("candidate_id не должен быть пустым.")
        if not user_id.strip():
            raise ValueError("user_id не должен быть пустым.")
        if not updated_at.strip():
            raise ValueError("updated_at обязателен.")
        if not (0.0 <= confidence_suggestion <= 1.0):
            raise ValueError("confidence_suggestion должен быть в диапазоне 0..1.")

        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE policy_rule_candidate
                SET
                    proposed_trigger_json = ?,
                    proposed_action_json = ?,
                    priority_suggestion = ?,
                    confidence_suggestion = ?,
                    updated_at = ?
                WHERE candidate_id = ?
                  AND user_id = ?
                  AND status = 'proposed'
                """,
                (
                    policy_trigger_to_json(proposed_trigger),
                    policy_action_to_json(proposed_action),
                    priority_suggestion,
                    confidence_suggestion,
                    updated_at,
                    candidate_id,
                    user_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError(
                    "Не удалось обновить candidate: "
                    "возможно он отсутствует, принадлежит другому пользователю или уже не proposed."
                )

    def reject_policy_rule_candidate(
        self,
        *,
        candidate_id: str,
        user_id: str,
        updated_at: str,
    ) -> None:
        if not candidate_id.strip():
            raise ValueError("candidate_id не должен быть пустым.")
        if not user_id.strip():
            raise ValueError("user_id не должен быть пустым.")
        if not updated_at.strip():
            raise ValueError("updated_at обязателен.")

        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE policy_rule_candidate
                SET status = 'rejected', updated_at = ?
                WHERE candidate_id = ?
                  AND user_id = ?
                  AND status = 'proposed'
                """,
                (updated_at, candidate_id, user_id),
            )
            if cur.rowcount != 1:
                raise ValueError(
                    "Не удалось reject candidate: "
                    "возможно он отсутствует, принадлежит другому пользователю или уже не proposed."
                )

    def approve_policy_rule_candidate(
        self,
        *,
        candidate_id: str,
        user_id: str,
        approved_rule: PolicyRule,
        final_trigger: PolicyTrigger,
        final_action: PolicyAction,
        final_priority: int,
        final_confidence: float,
        updated_at: str,
    ) -> None:
        if not candidate_id.strip():
            raise ValueError("candidate_id не должен быть пустым.")
        if not user_id.strip():
            raise ValueError("user_id не должен быть пустым.")
        if not updated_at.strip():
            raise ValueError("updated_at обязателен.")
        if approved_rule.user_id != user_id:
            raise ValueError("approved_rule.user_id должен совпадать с user_id.")
        if approved_rule.trigger != final_trigger:
            raise ValueError("approved_rule.trigger должен совпадать с final_trigger.")
        if approved_rule.action != final_action:
            raise ValueError("approved_rule.action должен совпадать с final_action.")
        if approved_rule.priority != final_priority:
            raise ValueError("approved_rule.priority должен совпадать с final_priority.")
        if approved_rule.confidence != final_confidence:
            raise ValueError("approved_rule.confidence должен совпадать с final_confidence.")
        if not (0.0 <= final_confidence <= 1.0):
            raise ValueError("final_confidence должен быть в диапазоне 0..1.")

        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE policy_rule_candidate
                SET
                    proposed_trigger_json = ?,
                    proposed_action_json = ?,
                    priority_suggestion = ?,
                    confidence_suggestion = ?,
                    status = 'approved',
                    updated_at = ?
                WHERE candidate_id = ?
                  AND user_id = ?
                  AND status = 'proposed'
                """,
                (
                    policy_trigger_to_json(final_trigger),
                    policy_action_to_json(final_action),
                    final_priority,
                    final_confidence,
                    updated_at,
                    candidate_id,
                    user_id,
                ),
            )
            if cur.rowcount != 1:
                raise ValueError(
                    "Не удалось approve candidate: "
                    "возможно он отсутствует, принадлежит другому пользователю или уже не proposed."
                )

            self.conn.execute(
                """
                INSERT INTO policy_rule (
                    rule_id, user_id, scope,
                    trigger_json, action_json,
                    priority, confidence, decay_half_life_days,
                    provenance,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    approved_rule.rule_id,
                    approved_rule.user_id,
                    approved_rule.scope.value,
                    policy_trigger_to_json(approved_rule.trigger),
                    policy_action_to_json(approved_rule.action),
                    approved_rule.priority,
                    approved_rule.confidence,
                    approved_rule.decay_half_life_days,
                    approved_rule.provenance,
                    approved_rule.created_at,
                    approved_rule.updated_at,
                ),
            )

    def _create_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        cur.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
            (_SCHEMA_KEY, str(SCHEMA_VERSION)),
        )
        cur.execute(
            """
            CREATE TABLE interaction_log (
                interaction_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                interaction_kind TEXT NOT NULL CHECK (interaction_kind IN ('chat','tool')),
                raw_input TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('standard','memory_companion')),
                created_at TEXT NOT NULL,

                -- chat
                retrieved_memory_ids TEXT NOT NULL DEFAULT '[]',
                applied_policy_ids TEXT NOT NULL DEFAULT '[]',
                response_text TEXT,

                -- tool
                tool_name TEXT,
                tool_args TEXT,
                tool_status TEXT CHECK (tool_status IN ('ok','error','blocked')),
                blocked_reason TEXT,
                tool_output_preview TEXT,
                tool_error TEXT,
                tool_meta TEXT
            )
            """
        )
        cur.execute("CREATE INDEX idx_interaction_log_created_at ON interaction_log(created_at)")
        cur.execute("CREATE INDEX idx_interaction_log_kind ON interaction_log(interaction_kind)")

        cur.execute(
            """
            CREATE TABLE policy_rule (
                rule_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                scope TEXT NOT NULL CHECK (scope IN ('global','user')),
                trigger_json TEXT NOT NULL,
                action_json TEXT NOT NULL,
                priority INTEGER NOT NULL,
                confidence REAL NOT NULL,
                decay_half_life_days INTEGER NOT NULL,
                provenance TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX idx_policy_rule_user_id ON policy_rule(user_id)")
        cur.execute("CREATE INDEX idx_policy_rule_scope ON policy_rule(scope)")
        cur.execute("CREATE INDEX idx_policy_rule_priority ON policy_rule(priority)")

        cur.execute(
            """
            CREATE TABLE feedback_event (
                feedback_id TEXT PRIMARY KEY,
                interaction_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                rating TEXT NOT NULL CHECK (rating IN ('good','ok','bad')),
                labels TEXT NOT NULL DEFAULT '[]',
                free_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (interaction_id)
                    REFERENCES interaction_log(interaction_id)
                    ON DELETE CASCADE
            )
            """
        )
        cur.execute("CREATE INDEX idx_feedback_event_user_id ON feedback_event(user_id)")
        cur.execute("CREATE INDEX idx_feedback_event_created_at ON feedback_event(created_at)")
        cur.execute(
            "CREATE INDEX idx_feedback_event_interaction_id ON feedback_event(interaction_id)"
        )

        cur.execute(
            """
            CREATE TABLE batch_review_run (
                batch_review_run_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                created_at TEXT NOT NULL,
                interaction_count INTEGER NOT NULL,
                feedback_count INTEGER NOT NULL,
                candidate_count INTEGER NOT NULL,
                report_text TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX idx_batch_review_run_user_id ON batch_review_run(user_id)")
        cur.execute("CREATE INDEX idx_batch_review_run_created_at ON batch_review_run(created_at)")

        cur.execute(
            """
            CREATE TABLE policy_rule_candidate (
                candidate_id TEXT PRIMARY KEY,
                batch_review_run_id TEXT NOT NULL,
                user_id TEXT NOT NULL,

                proposed_trigger_json TEXT NOT NULL,
                proposed_action_json TEXT NOT NULL,

                priority_suggestion INTEGER NOT NULL,
                confidence_suggestion REAL NOT NULL,

                evidence_json TEXT NOT NULL,
                signals_json TEXT NOT NULL,
                intent_hypotheses_json TEXT NOT NULL,
                paradox_flags_json TEXT NOT NULL,

                status TEXT NOT NULL CHECK (status IN ('proposed','approved','rejected')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (batch_review_run_id)
                    REFERENCES batch_review_run(batch_review_run_id)
                    ON DELETE CASCADE
            )
            """
        )
        cur.execute(
            "CREATE INDEX idx_policy_rule_candidate_run_id "
            "ON policy_rule_candidate(batch_review_run_id)"
        )
        cur.execute(
            "CREATE INDEX idx_policy_rule_candidate_user_id ON policy_rule_candidate(user_id)"
        )
        cur.execute(
            "CREATE INDEX idx_policy_rule_candidate_status ON policy_rule_candidate(status)"
        )
        cur.execute(
            "CREATE INDEX idx_policy_rule_candidate_created_at ON policy_rule_candidate(created_at)"
        )
        self.conn.commit()

    def _verify_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_meta'")
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "schema_meta table is missing.")
        cur.execute("SELECT value FROM schema_meta WHERE key = ?", (_SCHEMA_KEY,))
        row = cur.fetchone()
        if row is None or row[0] is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "schema_version is missing.")
        try:
            actual = int(row[0])
        except Exception as exc:  # noqa: BLE001
            raise InvalidMemoryCompanionDbError(
                self.db_path, f"schema_version is not int: {row[0]!r}"
            ) from exc
        if actual != SCHEMA_VERSION:
            raise SchemaVersionMismatchError(self.db_path, expected=SCHEMA_VERSION, actual=actual)

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='interaction_log'")
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "interaction_log table is missing.")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='policy_rule'")
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule table is missing.")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_event'")
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event table is missing.")
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='batch_review_run'")
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(self.db_path, "batch_review_run table is missing.")
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='policy_rule_candidate'"
        )
        if cur.fetchone() is None:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate table is missing."
            )

    def _insert_chat(self, log: ChatInteractionLog) -> None:
        if log.interaction_kind is not InteractionKind.CHAT:
            raise ValueError("ChatInteractionLog.interaction_kind must be InteractionKind.CHAT.")
        if not log.response_text:
            raise ValueError("ChatInteractionLog.response_text is required.")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO interaction_log (
                    interaction_id, user_id, interaction_kind, raw_input, mode, created_at,
                    retrieved_memory_ids, applied_policy_ids, response_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.interaction_id,
                    log.user_id,
                    log.interaction_kind.value,
                    log.raw_input,
                    log.mode.value,
                    log.created_at,
                    _dumps(log.retrieved_memory_ids),
                    _dumps(log.applied_policy_ids),
                    log.response_text,
                ),
            )

    def _insert_tool(self, log: ToolInteractionLog) -> None:
        if log.interaction_kind is not InteractionKind.TOOL:
            raise ValueError("ToolInteractionLog.interaction_kind must be InteractionKind.TOOL.")
        if not log.tool_name:
            raise ValueError("ToolInteractionLog.tool_name is required.")
        if log.tool_status is ToolStatus.BLOCKED and log.blocked_reason is None:
            raise ValueError("blocked_reason обязателен, когда tool_status='blocked'.")
        if log.tool_status is not ToolStatus.BLOCKED and log.blocked_reason is not None:
            raise ValueError("blocked_reason допустим только когда tool_status='blocked'.")
        preview = log.tool_output_preview
        if preview is not None and len(preview) > _MAX_TOOL_OUTPUT_PREVIEW_CHARS:
            preview = preview[:_MAX_TOOL_OUTPUT_PREVIEW_CHARS] + "…[truncated]"
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO interaction_log (
                    interaction_id, user_id, interaction_kind, raw_input, mode, created_at,
                    tool_name, tool_args, tool_status, blocked_reason,
                    tool_output_preview, tool_error, tool_meta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    log.interaction_id,
                    log.user_id,
                    log.interaction_kind.value,
                    log.raw_input,
                    log.mode.value,
                    log.created_at,
                    log.tool_name,
                    _dumps(log.tool_args),
                    log.tool_status.value,
                    log.blocked_reason.value if log.blocked_reason else None,
                    preview,
                    log.tool_error,
                    _dumps(log.tool_meta) if log.tool_meta is not None else None,
                ),
            )

    def _row_to_chat(self, row: sqlite3.Row) -> ChatInteractionLog:
        response_text = row["response_text"]
        if not isinstance(response_text, str):
            raise InvalidMemoryCompanionDbError(self.db_path, "chat response_text missing.")
        return ChatInteractionLog(
            interaction_id=str(row["interaction_id"]),
            user_id=str(row["user_id"]),
            interaction_kind=InteractionKind.CHAT,
            raw_input=str(row["raw_input"]),
            mode=InteractionMode(str(row["mode"])),
            created_at=str(row["created_at"]),
            response_text=response_text,
            retrieved_memory_ids=_loads_str_list(str(row["retrieved_memory_ids"])),
            applied_policy_ids=_loads_str_list(str(row["applied_policy_ids"])),
        )

    def _row_to_tool(self, row: sqlite3.Row) -> ToolInteractionLog:
        tool_name = row["tool_name"]
        tool_args = row["tool_args"]
        tool_status = row["tool_status"]
        if not isinstance(tool_name, str) or not tool_name:
            raise InvalidMemoryCompanionDbError(self.db_path, "tool_name missing.")
        if not isinstance(tool_args, str):
            raise InvalidMemoryCompanionDbError(self.db_path, "tool_args missing.")
        if not isinstance(tool_status, str):
            raise InvalidMemoryCompanionDbError(self.db_path, "tool_status missing.")
        blocked_reason = row["blocked_reason"]
        tool_meta = row["tool_meta"]
        return ToolInteractionLog(
            interaction_id=str(row["interaction_id"]),
            user_id=str(row["user_id"]),
            interaction_kind=InteractionKind.TOOL,
            raw_input=str(row["raw_input"]),
            mode=InteractionMode(str(row["mode"])),
            created_at=str(row["created_at"]),
            tool_name=tool_name,
            tool_args=_loads_dict(tool_args),
            tool_status=ToolStatus(tool_status),
            blocked_reason=BlockedReason(str(blocked_reason))
            if isinstance(blocked_reason, str) and blocked_reason
            else None,
            tool_output_preview=str(row["tool_output_preview"])
            if row["tool_output_preview"] is not None
            else None,
            tool_error=str(row["tool_error"]) if row["tool_error"] is not None else None,
            tool_meta=_loads_dict(tool_meta) if isinstance(tool_meta, str) else None,
        )

    def _row_to_policy_rule(self, row: sqlite3.Row) -> PolicyRule:
        rule_id = row["rule_id"]
        user_id = row["user_id"]
        scope_raw = row["scope"]
        trigger_json = row["trigger_json"]
        action_json = row["action_json"]
        priority = row["priority"]
        confidence = row["confidence"]
        decay_half_life_days = row["decay_half_life_days"]
        provenance = row["provenance"]
        created_at = row["created_at"]
        updated_at = row["updated_at"]

        if not isinstance(rule_id, str) or not rule_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.rule_id missing.")
        if not isinstance(user_id, str) or not user_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.user_id missing.")
        if not isinstance(scope_raw, str) or not scope_raw:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.scope missing.")
        if not isinstance(trigger_json, str) or not trigger_json:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.trigger_json missing.")
        if not isinstance(action_json, str) or not action_json:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.action_json missing.")
        if not isinstance(priority, int):
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.priority missing.")
        if not isinstance(confidence, (int, float)):
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.confidence missing.")
        if not isinstance(decay_half_life_days, int):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule.decay_half_life_days missing."
            )
        if not isinstance(provenance, str) or not provenance:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.provenance missing.")
        if not isinstance(created_at, str) or not created_at:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.created_at missing.")
        if not isinstance(updated_at, str) or not updated_at:
            raise InvalidMemoryCompanionDbError(self.db_path, "policy_rule.updated_at missing.")

        return PolicyRule(
            rule_id=rule_id,
            user_id=user_id,
            scope=PolicyScope(scope_raw),
            trigger=policy_trigger_from_json(trigger_json),
            action=policy_action_from_json(action_json),
            priority=priority,
            confidence=float(confidence),
            decay_half_life_days=decay_half_life_days,
            provenance=provenance,
            created_at=created_at,
            updated_at=updated_at,
        )

    def _row_to_feedback_event(self, row: sqlite3.Row) -> FeedbackEvent:
        feedback_id = row["feedback_id"]
        interaction_id = row["interaction_id"]
        user_id = row["user_id"]
        rating_raw = row["rating"]
        labels_raw = row["labels"]
        free_text = row["free_text"]
        created_at = row["created_at"]

        if not isinstance(feedback_id, str) or not feedback_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event.feedback_id missing.")
        if not isinstance(interaction_id, str) or not interaction_id:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "feedback_event.interaction_id missing."
            )
        if not isinstance(user_id, str) or not user_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event.user_id missing.")
        if not isinstance(rating_raw, str) or not rating_raw:
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event.rating missing.")
        if not isinstance(labels_raw, str):
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event.labels missing.")
        if not isinstance(created_at, str) or not created_at:
            raise InvalidMemoryCompanionDbError(self.db_path, "feedback_event.created_at missing.")

        try:
            labels = _loads_feedback_labels(labels_raw)
        except Exception as exc:  # noqa: BLE001
            raise InvalidMemoryCompanionDbError(
                self.db_path, f"feedback_event.labels invalid: {labels_raw!r}"
            ) from exc

        return FeedbackEvent(
            feedback_id=feedback_id,
            interaction_id=interaction_id,
            user_id=user_id,
            rating=FeedbackRating(rating_raw),
            created_at=created_at,
            labels=labels,
            free_text=str(free_text) if isinstance(free_text, str) and free_text else None,
        )

    def _row_to_batch_review_run(self, row: sqlite3.Row) -> BatchReviewRun:
        run_id = row["batch_review_run_id"]
        user_id = row["user_id"]
        period_start = row["period_start"]
        period_end = row["period_end"]
        created_at = row["created_at"]
        interaction_count = row["interaction_count"]
        feedback_count = row["feedback_count"]
        candidate_count = row["candidate_count"]
        report_text = row["report_text"]

        if not isinstance(run_id, str) or not run_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "batch_review_run_id missing.")
        if not isinstance(user_id, str) or not user_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "batch_review_run.user_id missing.")
        if not isinstance(period_start, str) or not period_start:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.period_start missing."
            )
        if not isinstance(period_end, str) or not period_end:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.period_end missing."
            )
        if not isinstance(created_at, str) or not created_at:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.created_at missing."
            )
        if not isinstance(interaction_count, int):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.interaction_count missing."
            )
        if not isinstance(feedback_count, int):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.feedback_count missing."
            )
        if not isinstance(candidate_count, int):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.candidate_count missing."
            )
        if not isinstance(report_text, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "batch_review_run.report_text missing."
            )

        return BatchReviewRun(
            batch_review_run_id=run_id,
            user_id=user_id,
            period_start=period_start,
            period_end=period_end,
            created_at=created_at,
            interaction_count=interaction_count,
            feedback_count=feedback_count,
            candidate_count=candidate_count,
            report_text=report_text,
        )

    def _row_to_policy_rule_candidate(self, row: sqlite3.Row) -> PolicyRuleCandidate:
        candidate_id = row["candidate_id"]
        run_id = row["batch_review_run_id"]
        user_id = row["user_id"]
        trigger_json = row["proposed_trigger_json"]
        action_json = row["proposed_action_json"]
        priority = row["priority_suggestion"]
        confidence = row["confidence_suggestion"]
        evidence_json = row["evidence_json"]
        signals_json = row["signals_json"]
        intents_json = row["intent_hypotheses_json"]
        paradox_json = row["paradox_flags_json"]
        status_raw = row["status"]
        created_at = row["created_at"]
        updated_at = row["updated_at"]

        if not isinstance(candidate_id, str) or not candidate_id:
            raise InvalidMemoryCompanionDbError(self.db_path, "candidate_id missing.")
        if not isinstance(run_id, str) or not run_id:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.batch_review_run_id missing."
            )
        if not isinstance(user_id, str) or not user_id:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.user_id missing."
            )
        if not isinstance(trigger_json, str) or not trigger_json:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.proposed_trigger_json missing."
            )
        if not isinstance(action_json, str) or not action_json:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.proposed_action_json missing."
            )
        if not isinstance(priority, int):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.priority_suggestion missing."
            )
        if not isinstance(confidence, (int, float)):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.confidence_suggestion missing."
            )
        if not isinstance(evidence_json, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.evidence_json missing."
            )
        if not isinstance(signals_json, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.signals_json missing."
            )
        if not isinstance(intents_json, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.intent_hypotheses_json missing."
            )
        if not isinstance(paradox_json, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.paradox_flags_json missing."
            )
        if not isinstance(status_raw, str):
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.status missing."
            )
        if not isinstance(created_at, str) or not created_at:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.created_at missing."
            )
        if not isinstance(updated_at, str) or not updated_at:
            raise InvalidMemoryCompanionDbError(
                self.db_path, "policy_rule_candidate.updated_at missing."
            )

        try:
            evidence = _loads_evidence_items(evidence_json)
            signals = _loads_signals(signals_json)
            intents = _loads_intent_hypotheses(intents_json)
            paradox_flags = _loads_paradox_flags(paradox_json)
        except Exception as exc:  # noqa: BLE001
            raise InvalidMemoryCompanionDbError(
                self.db_path, f"policy_rule_candidate JSON invalid: {exc}"
            ) from exc

        return PolicyRuleCandidate(
            candidate_id=candidate_id,
            batch_review_run_id=run_id,
            user_id=user_id,
            proposed_trigger=policy_trigger_from_json(trigger_json),
            proposed_action=policy_action_from_json(action_json),
            priority_suggestion=priority,
            confidence_suggestion=float(confidence),
            evidence=evidence,
            signals=signals,
            intent_hypotheses=intents,
            paradox_flags=paradox_flags,
            status=CandidateStatus(status_raw),
            created_at=created_at,
            updated_at=updated_at,
        )
