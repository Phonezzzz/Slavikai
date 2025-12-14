from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from shared.models import PlanStepStatus


class ReasoningPanel(QWidget):
    """Панель отображения плана и шагов исполнения с результатами/tools."""

    def __init__(self, agent: Agent) -> None:
        super().__init__()
        self.agent = agent
        layout = QVBoxLayout()

        controls = QHBoxLayout()
        self.show_original = QCheckBox("Оригинальный план")
        self.show_original.setChecked(True)
        self.show_context = QCheckBox("Контекст")
        self.show_context.setChecked(True)
        self.show_tool_calls = QCheckBox("Tool calls")
        self.show_tool_calls.setChecked(True)
        controls.addWidget(self.show_original)
        controls.addWidget(self.show_context)
        controls.addWidget(self.show_tool_calls)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.refresh_btn = QPushButton("🔄 Обновить reasoning")
        self.refresh_btn.clicked.connect(self.refresh)

        layout.addLayout(controls)
        layout.addWidget(self.text)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)
        self.refresh()

    def refresh(self) -> None:
        self.text.clear()
        plan = getattr(self.agent, "last_plan", None)
        original = getattr(self.agent, "last_plan_original", None)
        if not plan:
            self.text.append("План ещё не выполнялся.")
        else:
            self.text.append(f"<b>Цель:</b> {plan.goal}<br>")
            if original and original != plan and self.show_original.isChecked():
                self.text.append("<b>Критик переписал план (оригинал):</b>")
                for idx, step in enumerate(original.steps, start=1):
                    self.text.append(f"   {idx}. {step.description}")
                self.text.append("<b>Исполняемый план:</b>")
            for idx, step in enumerate(plan.steps, start=1):
                icon = {
                    PlanStepStatus.PENDING: "⏳",
                    PlanStepStatus.IN_PROGRESS: "🔄",
                    PlanStepStatus.DONE: "✅",
                    PlanStepStatus.ERROR: "❌",
                }.get(step.status, "•")
                color = {
                    PlanStepStatus.PENDING: "#888",
                    PlanStepStatus.IN_PROGRESS: "#0055aa",
                    PlanStepStatus.DONE: "#0a8a0a",
                    PlanStepStatus.ERROR: "#c1121f",
                }.get(step.status, "#444")
                result = f"\n   ↳ {step.result}" if step.result else ""
                self.text.append(
                    f'<span style="color:{color}">{idx}. {icon} {step.description}</span>{result}'
                )
        hints_meta = getattr(self.agent, "last_hints_meta", [])
        if hints_meta:
            self.text.append("\nАвто-подсказки (major/fatal):")
            for item in hints_meta:
                sev = item.get("severity", "unknown")
                hint = item.get("hint", "")
                color = "#c1121f" if sev in {"fatal", "major"} else "#444"
                self.text.append(f'<span style="color:{color}">- [{sev}] {hint}</span>')
        ctx = getattr(self.agent, "last_context_text", None)
        if ctx and self.show_context.isChecked():
            self.text.append("\n<b>Контекст для LLM:</b>")
            self.text.append(ctx)
        # tool calls (последние)
        if self.show_tool_calls.isChecked() and hasattr(self.agent, "tool_registry"):
            calls = self.agent.tool_registry.read_recent_calls(10)
            if calls:
                self.text.append("\n🛠 Tool calls:")
                for call in calls:
                    status = "✅" if call.ok else "❌"
                    meta = f" meta={call.meta}" if call.meta else ""
                    args = f" args={call.args}" if call.args else ""
                    prefix = ""
                    if not call.ok and call.error and "safe mode" in call.error.lower():
                        prefix = "[SAFE MODE BLOCK] "
                    message = (
                        f"{prefix}[{call.timestamp}] {status} {call.tool} "
                        f"{call.error or ''}{meta}{args}"
                    )
                    self.text.append(message)
