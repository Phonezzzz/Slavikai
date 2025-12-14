from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from shared.models import JSONValue, ToolCallRecord


class ToolLogsView(QWidget):
    """Панель просмотра истории вызовов инструментов."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Фильтр по названию/ошибке/meta…")
        self.errors_only = QCheckBox("Только ошибки")
        self.refresh_btn = QPushButton("🔄 Обновить вызовы tools")
        self.refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(self.filter_input)
        controls.addWidget(self.errors_only)
        controls.addWidget(self.refresh_btn)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addLayout(controls)
        layout.addWidget(self.text)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        calls = self.agent.get_recent_tool_calls(limit=80)
        self.text.clear()
        filtered = self._filter_calls(calls)
        for record in filtered:
            self.text.append(self._format_record(record))

    def _format_record(self, record: ToolCallRecord) -> str:
        status = "✅" if record.ok else "❌"
        error = f" — {record.error}" if record.error else ""
        meta = ""
        if record.meta:
            meta = f"\n   meta: {self._short_meta(record.meta)}"
        args = ""
        if record.args:
            args = f"\n   args: {self._short_meta(record.args)}"
        return f"[{record.timestamp}] {status} {record.tool}{error}{meta}{args}"

    def _short_meta(self, meta: dict[str, JSONValue]) -> str:
        items = [f"{k}={v}" for k, v in meta.items() if v is not None]
        if not items:
            return ""
        return "; ".join(items[:5])

    def _filter_calls(self, calls: list[ToolCallRecord]) -> list[ToolCallRecord]:
        query = self.filter_input.text().strip().lower()
        only_errors = self.errors_only.isChecked()

        def matches(record: ToolCallRecord) -> bool:
            if only_errors and record.ok:
                return False
            if not query:
                return True
            haystack = " ".join(
                [
                    record.tool,
                    record.error or "",
                    self._short_meta(record.meta or {}),
                    self._short_meta(record.args or {}),
                ]
            ).lower()
            return query in haystack

        return [rec for rec in calls if matches(rec)]
