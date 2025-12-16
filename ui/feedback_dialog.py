from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from shared.memory_companion_models import FeedbackLabel, FeedbackRating

_LABEL_TITLES: dict[FeedbackLabel, str] = {
    FeedbackLabel.TOO_LONG: "Слишком длинно",
    FeedbackLabel.OFF_TOPIC: "Не по делу / оффтоп",
    FeedbackLabel.NO_SOURCES: "Нет источников / ссылок",
    FeedbackLabel.HALLUCINATION: "Придумал / галлюцинация",
    FeedbackLabel.TOO_COMPLEX: "Слишком сложно",
    FeedbackLabel.INCORRECT: "Ошибка / неверно",
    FeedbackLabel.OTHER: "Другое",
}


@dataclass(frozen=True)
class FeedbackDialogResult:
    labels: list[FeedbackLabel]
    free_text: str | None


class FeedbackDialog(QDialog):
    def __init__(self, *, rating: FeedbackRating, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Feedback")
        self.setModal(True)

        layout = QVBoxLayout()
        rating_text = {"good": "👍 good", "ok": "😐 ok", "bad": "👎 bad"}[rating.value]
        layout.addWidget(QLabel(f"Оценка: {rating_text}"))

        layout.addWidget(QLabel("Labels (опционально):"))
        self._checkboxes: dict[FeedbackLabel, QCheckBox] = {}
        for label in FeedbackLabel:
            title = _LABEL_TITLES.get(label, label.value)
            cb = QCheckBox(title)
            cb.setChecked(False)
            self._checkboxes[label] = cb
            layout.addWidget(cb)

        layout.addWidget(QLabel("Комментарий (опционально):"))
        self._free_text = QTextEdit()
        self._free_text.setPlaceholderText("Коротко опишите, что улучшить/что было полезно…")
        self._free_text.setMinimumHeight(80)
        layout.addWidget(self._free_text)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_result(self) -> FeedbackDialogResult:
        labels = [label for label, cb in self._checkboxes.items() if cb.isChecked()]
        free_text = self._free_text.toPlainText().strip() or None
        return FeedbackDialogResult(labels=labels, free_text=free_text)
