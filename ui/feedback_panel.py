from __future__ import annotations

from typing import Any, cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from memory.feedback_manager import FeedbackManager


class FeedbackPanel(QWidget):
    """Просмотр и повторное применение подсказок фидбека."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        self.manager = FeedbackManager("memory/feedback.db")
        layout = QVBoxLayout()
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.stats_label = QLabel()
        self.last_applied = QListWidget()
        self.last_applied.setMinimumHeight(60)
        self.bad_list = QListWidget()
        self.bad_list.setMinimumHeight(140)
        self.apply_btn = QPushButton("📌 Применить последний major hint")
        self.apply_btn.clicked.connect(self.apply_last_hint)
        layout.addWidget(self.text)
        layout.addWidget(QLabel("Последние авто-подсказки:"))
        layout.addWidget(self.last_applied)
        layout.addWidget(QLabel("Недавние bad/offtopic:"))
        layout.addWidget(self.bad_list)
        layout.addWidget(self.stats_label)
        layout.addWidget(self.apply_btn)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        records = self.manager.get_recent_records(10)
        self.text.clear()
        for rec in records:
            sev = rec.get("severity", "")
            hint = rec.get("hint", "")
            sev_icon = {"major": "⚠️", "fatal": "🛑", "minor": "ℹ️"}.get(sev, "•")
            self.text.append(
                f"{sev_icon}[{rec.get('rating', '')}/{sev}] {rec.get('prompt', '')[:80]} -> "
                f"{rec.get('answer', '')[:80]} {('hint: ' + hint) if hint else ''}"
            )

        # Статус качества
        stats_raw = self.manager.stats()
        stats = stats_raw if isinstance(stats_raw, dict) else {}
        rating_counts_raw = stats.get("ratings", {})
        rating_counts = (
            cast(dict[str, Any], rating_counts_raw) if isinstance(rating_counts_raw, dict) else {}
        )
        severity_counts_raw = stats.get("severity", {})
        severity_counts = (
            cast(dict[str, Any], severity_counts_raw)
            if isinstance(severity_counts_raw, dict)
            else {}
        )
        top_hints_raw = stats.get("top_hints", [])
        top_hints: list[dict[str, Any]] = (
            [hint for hint in top_hints_raw if isinstance(hint, dict)]
            if isinstance(top_hints_raw, list)
            else []
        )
        stats_parts = [
            f"good: {rating_counts.get('good', 0)}",
            f"bad: {rating_counts.get('bad', 0)}",
            f"offtopic: {rating_counts.get('offtopic', 0)}",
            f"major/fatal: {severity_counts.get('major', 0)} / {severity_counts.get('fatal', 0)}",
        ]
        if top_hints:
            best = top_hints[0]
            stats_parts.append(f"топ hint: {best['hint']} ({best['count']})")
        self.stats_label.setText(" | ".join(stats_parts))

        # Авто-применённые подсказки
        self.last_applied.clear()
        for meta in getattr(self.agent, "last_hints_meta", []):
            sev = meta.get("severity", "major")
            text = f"[{sev}] {meta.get('hint', '')}"
            item = QListWidgetItem(text)
            color = Qt.GlobalColor.red if sev in {"fatal", "major"} else Qt.GlobalColor.black
            item.setForeground(color)
            self.last_applied.addItem(item)

        # История bad/offtopic
        self.bad_list.clear()
        for rec in self.manager.get_recent_bad(10):
            sev = rec.get("severity", "")
            hint = rec.get("hint", "")
            prefix = f"[{rec.get('rating')}/{sev}]"
            item = QListWidgetItem(f"{prefix} {rec.get('prompt', '')[:60]} | {hint}")
            if sev in {"fatal", "major"}:
                item.setForeground(Qt.GlobalColor.red)
            self.bad_list.addItem(item)

    def apply_last_hint(self) -> None:
        hints = self.manager.get_recent_hints(1)
        if not hints:
            return
        hint = hints[0]
        # сохранить как новый major feedback для применения в контекст
        self.agent.save_feedback("[manual_hint]", "[manual_hint]", "bad", hint=hint)
        self.refresh()
