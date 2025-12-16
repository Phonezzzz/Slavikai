from __future__ import annotations

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from shared.batch_review_models import BatchReviewRun, CandidateStatus, PolicyRuleCandidate
from shared.memory_companion_models import ChatInteractionLog, InteractionKind
from shared.policy_models import policy_action_to_json, policy_trigger_to_json
from ui.policy_candidate_dialog import PolicyCandidateApproveDialog, PolicyCandidateEditDialog


class FeedbackPanel(QWidget):
    """Просмотр FeedbackEvent (Memory Companion)."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        self._runs: list[BatchReviewRun] = []
        self._candidates_by_id: dict[str, PolicyRuleCandidate] = {}

        layout = QVBoxLayout()
        self.stats_label = QLabel()
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setMinimumHeight(110)

        self.text = QTextEdit()
        self.text.setReadOnly(True)

        self.refresh_btn = QPushButton("↻ Обновить")
        self.refresh_btn.clicked.connect(self.refresh)

        self.analyze_btn = QPushButton("🧪 Анализ данных")
        self.analyze_btn.clicked.connect(self._run_batch_review)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.analyze_btn)

        layout.addWidget(self.stats_label)
        layout.addWidget(QLabel("BatchReview report:"))
        layout.addWidget(self.report)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("PolicyRule candidates:"))
        filters = QHBoxLayout()
        self.run_combo = QComboBox()
        self.run_combo.currentIndexChanged.connect(self._refresh_candidates_list)
        self.status_combo = QComboBox()
        self.status_combo.addItem("proposed", CandidateStatus.PROPOSED)
        self.status_combo.addItem("approved", CandidateStatus.APPROVED)
        self.status_combo.addItem("rejected", CandidateStatus.REJECTED)
        self.status_combo.addItem("all", None)
        self.status_combo.currentIndexChanged.connect(self._refresh_candidates_list)
        filters.addWidget(QLabel("Run:"))
        filters.addWidget(self.run_combo)
        filters.addWidget(QLabel("Status:"))
        filters.addWidget(self.status_combo)
        layout.addLayout(filters)

        self.candidates_list = QListWidget()
        self.candidates_list.currentItemChanged.connect(self._on_candidate_selected)
        layout.addWidget(self.candidates_list)

        self.candidate_details = QTextEdit()
        self.candidate_details.setReadOnly(True)
        self.candidate_details.setMinimumHeight(140)
        layout.addWidget(self.candidate_details)

        cand_btns = QHBoxLayout()
        self.edit_candidate_btn = QPushButton("✎ Edit")
        self.edit_candidate_btn.clicked.connect(self._edit_selected_candidate)
        self.approve_candidate_btn = QPushButton("✅ Approve")
        self.approve_candidate_btn.clicked.connect(self._approve_selected_candidate)
        self.reject_candidate_btn = QPushButton("🚫 Reject")
        self.reject_candidate_btn.clicked.connect(self._reject_selected_candidate)
        cand_btns.addWidget(self.edit_candidate_btn)
        cand_btns.addWidget(self.approve_candidate_btn)
        cand_btns.addWidget(self.reject_candidate_btn)
        layout.addLayout(cand_btns)

        layout.addWidget(QLabel("Feedback events:"))
        layout.addWidget(self.text)
        self.setLayout(layout)

        self.refresh()

    def refresh(self) -> None:
        stats = self.agent.get_feedback_stats()
        stats_text = " | ".join([f"{k.value}: {v}" for k, v in stats.items()])
        self.stats_label.setText(stats_text)

        events = self.agent.get_recent_feedback_events(20)
        self.text.clear()
        for ev in events:
            labels = ", ".join([label.value for label in ev.labels]) if ev.labels else "-"
            interaction = self.agent.get_interaction_log(ev.interaction_id)
            prompt = ""
            if (
                interaction is not None
                and interaction.interaction_kind == InteractionKind.CHAT
                and isinstance(interaction, ChatInteractionLog)
            ):
                prompt = interaction.raw_input.strip().replace("\n", " ")[:80]
            comment = ev.free_text.strip().replace("\n", " ") if ev.free_text else ""

            line = (
                f"[{ev.created_at}] {ev.rating.value} labels=[{labels}] "
                f"id={ev.interaction_id} input='{prompt}'"
            )
            if comment:
                line += f" comment='{comment}'"
            self.text.append(line)

        self._refresh_candidates(prefer_run_id=None)

    def _run_batch_review(self) -> None:
        days, ok = QInputDialog.getInt(
            self,
            "Анализ данных (BatchReview)",
            "Период, дней:",
            7,
            1,
            365,
            1,
        )
        if not ok:
            return
        try:
            run = self.agent.run_batch_review(period_days=int(days))
        except Exception as exc:  # noqa: BLE001
            self.report.setPlainText(f"[Ошибка BatchReview: {exc}]")
            return

        self.report.setPlainText(run.report_text)
        self._refresh_candidates(prefer_run_id=run.batch_review_run_id)
        self.refresh()

    def _refresh_candidates(self, *, prefer_run_id: str | None) -> None:
        runs = self.agent.get_recent_batch_review_runs(limit=20)
        self._runs = runs

        current_run_id = self._selected_run_id()
        preferred = prefer_run_id or current_run_id

        self.run_combo.blockSignals(True)
        try:
            self.run_combo.clear()
            for run in runs:
                title = (
                    f"{run.created_at} | candidates={run.candidate_count} | "
                    f"id={run.batch_review_run_id[:8]}"
                )
                self.run_combo.addItem(title, run.batch_review_run_id)
            if preferred:
                idx = self.run_combo.findData(preferred)
                if idx >= 0:
                    self.run_combo.setCurrentIndex(idx)
        finally:
            self.run_combo.blockSignals(False)

        self._refresh_candidates_list()

    def _selected_run_id(self) -> str | None:
        run_id = self.run_combo.currentData()
        return str(run_id) if isinstance(run_id, str) and run_id else None

    def _selected_status_filter(self) -> CandidateStatus | None:
        data = self.status_combo.currentData()
        if data is None:
            return None
        if isinstance(data, CandidateStatus):
            return data
        return None

    def _refresh_candidates_list(self, _index: int | None = None) -> None:
        run_id = self._selected_run_id()
        status = self._selected_status_filter()
        if run_id is None:
            self.candidates_list.clear()
            self.candidate_details.setPlainText("[No BatchReview runs yet]")
            self._candidates_by_id = {}
            return

        candidates = self.agent.list_policy_rule_candidates(run_id=run_id, status=status, limit=200)
        self._candidates_by_id = {c.candidate_id: c for c in candidates}

        self.candidates_list.blockSignals(True)
        try:
            current = self._selected_candidate_id()
            self.candidates_list.clear()
            for c in candidates:
                signals = ",".join([s.value for s in c.signals[:3]]) if c.signals else "-"
                title = (
                    f"{c.status.value} "
                    f"prio={c.priority_suggestion} "
                    f"conf={c.confidence_suggestion:.2f} "
                    f"signals={signals} "
                    f"id={c.candidate_id[:8]}"
                )
                item = QListWidgetItem(title)
                item.setData(Qt.ItemDataRole.UserRole, c.candidate_id)
                self.candidates_list.addItem(item)
            if current:
                self._select_candidate(current)
        finally:
            self.candidates_list.blockSignals(False)

        if candidates and self.candidates_list.currentItem() is None:
            self.candidates_list.setCurrentRow(0)
        if not candidates:
            self.candidate_details.setPlainText("[No candidates for selected filters]")

    def _selected_candidate_id(self) -> str | None:
        item = self.candidates_list.currentItem()
        if item is None:
            return None
        candidate_id = item.data(Qt.ItemDataRole.UserRole)
        return str(candidate_id) if isinstance(candidate_id, str) and candidate_id else None

    def _select_candidate(self, candidate_id: str) -> None:
        for idx in range(self.candidates_list.count()):
            item = self.candidates_list.item(idx)
            if item.data(Qt.ItemDataRole.UserRole) == candidate_id:
                self.candidates_list.setCurrentItem(item)
                return

    def _on_candidate_selected(
        self, _current: QListWidgetItem | None, _previous: QListWidgetItem | None
    ) -> None:
        candidate_id = self._selected_candidate_id()
        if not candidate_id:
            self.candidate_details.clear()
            return
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            self.candidate_details.setPlainText("[Candidate not found in list]")
            return
        self.candidate_details.setPlainText(self._format_candidate_details(candidate))

    def _format_candidate_details(self, c: PolicyRuleCandidate) -> str:
        trigger_json = self._pretty_json(policy_trigger_to_json(c.proposed_trigger))
        action_json = self._pretty_json(policy_action_to_json(c.proposed_action))
        signals = ", ".join([s.value for s in c.signals]) if c.signals else "-"
        paradox = ", ".join([p.value for p in c.paradox_flags]) if c.paradox_flags else "-"
        intents = (
            ", ".join([f"{h.intent.value}:{h.score:.2f}" for h in c.intent_hypotheses])
            if c.intent_hypotheses
            else "-"
        )
        evidence_lines = []
        for e in c.evidence:
            feedback = f" feedback_id={e.feedback_id}" if e.feedback_id else ""
            evidence_lines.append(
                f"- interaction_id={e.interaction_id}{feedback} excerpt={e.excerpt}"
            )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "-"

        return (
            f"candidate_id: {c.candidate_id}\n"
            f"run_id: {c.batch_review_run_id}\n"
            f"status: {c.status.value}\n"
            f"priority_suggestion: {c.priority_suggestion}\n"
            f"confidence_suggestion: {c.confidence_suggestion:.2f}\n"
            f"signals: {signals}\n"
            f"intent_hypotheses: {intents}\n"
            f"paradox_flags: {paradox}\n\n"
            f"trigger:\n{trigger_json}\n\n"
            f"action:\n{action_json}\n\n"
            f"evidence:\n{evidence_text}\n"
        )

    def _pretty_json(self, text: str) -> str:
        try:
            parsed = json.loads(text)
        except Exception:  # noqa: BLE001
            return text
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, indent=2)

    def _edit_selected_candidate(self) -> None:
        candidate_id = self._selected_candidate_id()
        if not candidate_id:
            return
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return

        dialog = PolicyCandidateEditDialog(
            candidate_id=candidate_id,
            initial_trigger_json=self._pretty_json(
                policy_trigger_to_json(candidate.proposed_trigger)
            ),
            initial_action_json=self._pretty_json(policy_action_to_json(candidate.proposed_action)),
            initial_priority=candidate.priority_suggestion,
            initial_confidence=candidate.confidence_suggestion,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.get_result()
        try:
            _ = self.agent.update_policy_rule_candidate_suggestion(
                candidate_id=candidate_id,
                proposed_trigger_json=result.trigger_json,
                proposed_action_json=result.action_json,
                priority_suggestion=result.priority,
                confidence_suggestion=result.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Edit failed", str(exc))
            return

        self._refresh_candidates(prefer_run_id=self._selected_run_id())
        self._select_candidate(candidate_id)

    def _approve_selected_candidate(self) -> None:
        candidate_id = self._selected_candidate_id()
        if not candidate_id:
            return
        candidate = self._candidates_by_id.get(candidate_id)
        if candidate is None:
            return

        dialog = PolicyCandidateApproveDialog(
            candidate_id=candidate_id,
            initial_trigger_json=self._pretty_json(
                policy_trigger_to_json(candidate.proposed_trigger)
            ),
            initial_action_json=self._pretty_json(policy_action_to_json(candidate.proposed_action)),
            initial_priority=candidate.priority_suggestion,
            initial_confidence=candidate.confidence_suggestion,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.get_result()
        try:
            rule = self.agent.approve_policy_rule_candidate(
                candidate_id=candidate_id,
                scope=result.scope,
                decay_half_life_days=result.decay_half_life_days,
                override_trigger_json=result.trigger_json,
                override_action_json=result.action_json,
                override_priority=result.priority,
                override_confidence=result.confidence,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Approve failed", str(exc))
            return

        QMessageBox.information(self, "Approved", f"PolicyRule created: {rule.rule_id}")
        self._refresh_candidates(prefer_run_id=self._selected_run_id())

    def _reject_selected_candidate(self) -> None:
        candidate_id = self._selected_candidate_id()
        if not candidate_id:
            return
        confirm = QMessageBox.question(
            self,
            "Reject candidate",
            f"Reject candidate {candidate_id[:8]}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self.agent.reject_policy_rule_candidate(candidate_id=candidate_id)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Reject failed", str(exc))
            return
        self._refresh_candidates(prefer_run_id=self._selected_run_id())
