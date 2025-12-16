from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from shared.policy_models import PolicyScope


@dataclass(frozen=True)
class PolicyCandidateEditResult:
    trigger_json: str
    action_json: str
    priority: int
    confidence: float


@dataclass(frozen=True)
class PolicyCandidateApproveResult(PolicyCandidateEditResult):
    scope: PolicyScope
    decay_half_life_days: int


class PolicyCandidateEditDialog(QDialog):
    def __init__(
        self,
        *,
        candidate_id: str,
        initial_trigger_json: str,
        initial_action_json: str,
        initial_priority: int,
        initial_confidence: float,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit policy candidate")
        self.setModal(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"candidate_id: {candidate_id}"))

        form = QFormLayout()
        self._priority = QSpinBox()
        self._priority.setRange(-100, 100)
        self._priority.setValue(int(initial_priority))
        form.addRow("priority:", self._priority)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.0, 1.0)
        self._confidence.setSingleStep(0.05)
        self._confidence.setDecimals(2)
        self._confidence.setValue(float(initial_confidence))
        form.addRow("confidence:", self._confidence)
        layout.addLayout(form)

        layout.addWidget(QLabel("trigger (JSON):"))
        self._trigger = QPlainTextEdit()
        self._trigger.setPlainText(initial_trigger_json)
        self._trigger.setMinimumHeight(90)
        layout.addWidget(self._trigger)

        layout.addWidget(QLabel("action (JSON):"))
        self._action = QPlainTextEdit()
        self._action.setPlainText(initial_action_json)
        self._action.setMinimumHeight(90)
        layout.addWidget(self._action)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_result(self) -> PolicyCandidateEditResult:
        return PolicyCandidateEditResult(
            trigger_json=self._trigger.toPlainText().strip(),
            action_json=self._action.toPlainText().strip(),
            priority=int(self._priority.value()),
            confidence=float(self._confidence.value()),
        )


class PolicyCandidateApproveDialog(QDialog):
    def __init__(
        self,
        *,
        candidate_id: str,
        initial_trigger_json: str,
        initial_action_json: str,
        initial_priority: int,
        initial_confidence: float,
        initial_scope: PolicyScope = PolicyScope.USER,
        initial_decay_half_life_days: int = 30,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Approve policy candidate")
        self.setModal(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"candidate_id: {candidate_id}"))
        layout.addWidget(QLabel("This will create an Approved PolicyRule."))

        form = QFormLayout()
        self._scope = QComboBox()
        self._scope.addItem(PolicyScope.USER.value, PolicyScope.USER)
        self._scope.addItem(PolicyScope.GLOBAL.value, PolicyScope.GLOBAL)
        self._scope.setCurrentIndex(0 if initial_scope is PolicyScope.USER else 1)
        form.addRow("scope:", self._scope)

        self._decay = QSpinBox()
        self._decay.setRange(1, 3650)
        self._decay.setValue(int(initial_decay_half_life_days))
        form.addRow("decay_half_life_days:", self._decay)

        self._priority = QSpinBox()
        self._priority.setRange(-100, 100)
        self._priority.setValue(int(initial_priority))
        form.addRow("priority:", self._priority)

        self._confidence = QDoubleSpinBox()
        self._confidence.setRange(0.0, 1.0)
        self._confidence.setSingleStep(0.05)
        self._confidence.setDecimals(2)
        self._confidence.setValue(float(initial_confidence))
        form.addRow("confidence:", self._confidence)

        layout.addLayout(form)

        layout.addWidget(QLabel("trigger (JSON):"))
        self._trigger = QPlainTextEdit()
        self._trigger.setPlainText(initial_trigger_json)
        self._trigger.setMinimumHeight(90)
        layout.addWidget(self._trigger)

        layout.addWidget(QLabel("action (JSON):"))
        self._action = QPlainTextEdit()
        self._action.setPlainText(initial_action_json)
        self._action.setMinimumHeight(90)
        layout.addWidget(self._action)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Approve")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_result(self) -> PolicyCandidateApproveResult:
        scope_data = self._scope.currentData()
        if not isinstance(scope_data, PolicyScope):
            raise RuntimeError("Invalid scope selected.")
        return PolicyCandidateApproveResult(
            trigger_json=self._trigger.toPlainText().strip(),
            action_json=self._action.toPlainText().strip(),
            priority=int(self._priority.value()),
            confidence=float(self._confidence.value()),
            scope=scope_data,
            decay_half_life_days=int(self._decay.value()),
        )
