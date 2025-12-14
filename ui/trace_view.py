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

from core.tracer import Tracer, TraceRecord
from shared.models import JSONValue


class TraceView(QWidget):
    """Вкладка GUI для просмотра reasoning-трейсов и планов."""

    def __init__(self) -> None:
        super().__init__()
        self.tracer = Tracer()
        layout = QVBoxLayout()
        controls = QHBoxLayout()
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Фильтр по событию/сообщению")
        self.errors_only = QCheckBox("Только ошибки")
        self.refresh_btn = QPushButton("🔄 Обновить трейсы / планы")
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
        logs = self.tracer.read_recent(60)
        self.text.clear()
        filtered = self._filter(logs)
        for log in filtered:
            msg = f"[{log['timestamp']}] {log['event']} → {log['message']}"
            if str(log.get("event", "")).startswith("step"):
                msg = f"   🔹 {msg}"
            elif str(log.get("event", "")).startswith("planning"):
                msg = f"📋 {msg}"
            meta = log.get("meta") or {}
            if meta:
                meta_short = self._short_meta(meta)
                if meta_short:
                    msg = f"{msg}\n   meta: {meta_short}"
            self.text.append(msg)

    def _filter(self, logs: list[TraceRecord]) -> list[TraceRecord]:
        query = self.filter_input.text().strip().lower()
        only_errors = self.errors_only.isChecked()

        def matches(log: TraceRecord) -> bool:
            event = str(log.get("event", "")).lower()
            message = str(log.get("message", "")).lower()
            if only_errors and "error" not in event:
                return False
            if not query:
                return True
            meta_str = self._short_meta(log.get("meta") or {}).lower()
            return query in event or query in message or query in meta_str

        return [log for log in logs if matches(log)]

    def _short_meta(self, meta: dict[str, JSONValue]) -> str:
        parts = [f"{k}={v}" for k, v in meta.items() if v is not None]
        return "; ".join(parts[:5])
