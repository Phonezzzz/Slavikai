from __future__ import annotations

from typing import cast

from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import QComboBox, QLabel, QMessageBox, QVBoxLayout, QWidget

from core.agent import Agent
from llm.dual_brain import DualBrain


class ModePanel(QWidget):
    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Режим моделей"))
        self.mode_select = QComboBox()
        self.mode_select.addItems(["single", "dual", "critic-only"])
        self.status_label = QLabel()
        current = (
            getattr(self.agent.brain, "mode", "single")
            if hasattr(self.agent, "brain")
            else "single"
        )
        idx = self.mode_select.findText(current)
        if idx >= 0:
            self.mode_select.setCurrentIndex(idx)
        self._update_availability()
        self.mode_select.currentTextChanged.connect(self._on_mode_changed)
        layout.addWidget(self.mode_select)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def _on_mode_changed(self, mode: str) -> None:
        try:
            self.agent.set_mode(mode)
            self._update_status()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Ошибка смены режима", str(exc))
            # revert selection to previous
            current = (
                getattr(self.agent.brain, "mode", "single")
                if hasattr(self.agent, "brain")
                else "single"
            )
            idx = self.mode_select.findText(current)
            if idx >= 0:
                self.mode_select.blockSignals(True)
                self.mode_select.setCurrentIndex(idx)
                self.mode_select.blockSignals(False)
        self._update_availability()
        self._update_status()

    def _update_availability(self) -> None:
        has_critic = isinstance(getattr(self.agent, "brain", None), DualBrain)
        model = cast(QStandardItemModel, self.mode_select.model())
        if not has_critic:
            for mode in ("dual", "critic-only"):
                idx = self.mode_select.findText(mode)
                if idx >= 0:
                    model.item(idx).setEnabled(False)
        else:
            for mode in ("dual", "critic-only"):
                idx = self.mode_select.findText(mode)
                if idx >= 0:
                    model.item(idx).setEnabled(True)
        self._update_status()

    def _update_status(self) -> None:
        mode = (
            getattr(self.agent.brain, "mode", "single")
            if hasattr(self.agent, "brain")
            else "single"
        )
        has_critic = isinstance(getattr(self.agent, "brain", None), DualBrain)
        critic_state = "есть" if has_critic else "нет"
        self.status_label.setText(f"Текущий режим: {mode} | критик: {critic_state}")
