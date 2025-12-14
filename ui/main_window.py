from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QSplitter, QVBoxLayout, QWidget

from config.model_store import load_model_configs
from core.agent import Agent
from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage
from ui.audio_player import ChatAudioPlayer
from ui.chat_view import ChatView
from ui.docs_panel import DocsPanel
from ui.feedback_panel import FeedbackPanel
from ui.logs_view import LogsView
from ui.memory_view import MemoryView
from ui.mode_panel import ModePanel
from ui.reasoning_panel import ReasoningPanel
from ui.settings_dialog import SettingsDialog
from ui.tool_logs_view import ToolLogsView
from ui.tools_panel import ToolsPanel
from ui.trace_view import TraceView
from ui.workspace_panel import WorkspacePanel


class DummyBrain(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text=f"Ответ модели на: {messages[-1].content}")


class DummyCritic(Brain):
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        return LLMResult(text="Критик: ответ корректен.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SlavikAI Core v0.8 — Observability Layer")
        self.resize(1200, 700)

        saved_main, saved_critic = load_model_configs()
        if saved_main is not None:
            self.agent = Agent(main_config=saved_main, critic_config=saved_critic)
        else:
            self.agent = Agent(DummyBrain(), critic=DummyCritic())
        self.audio_player = ChatAudioPlayer()

        self.chat = ChatView(
            agent=self.agent,
            audio_player=self.audio_player,
            on_send_callback=self.handle_message,
            on_feedback_callback=self.handle_feedback,
        )
        self.tools = ToolsPanel(
            on_change=self.handle_tools_toggle, initial_state=self.agent.tools_enabled
        )
        self.logs = LogsView()
        self.tool_logs = ToolLogsView(agent=self.agent)
        self.memory_view = MemoryView()
        self.docs_panel = DocsPanel()
        self.trace_view = TraceView()
        self.reasoning_panel = ReasoningPanel(agent=self.agent)
        self.feedback_panel = FeedbackPanel(agent=self.agent)
        self.workspace_panel = WorkspacePanel(
            agent=self.agent, on_ask_ai=self._log_workspace_answer
        )
        self.model_status = QLabel()
        self.model_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.mode_panel = ModePanel(self.agent)

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.model_status)
        splitter_row = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.mode_panel)
        splitter.addWidget(self.tools)
        splitter.addWidget(self.chat)
        splitter.addWidget(self.logs)
        splitter.addWidget(self.tool_logs)
        splitter.addWidget(self.memory_view)
        splitter.addWidget(self.trace_view)
        splitter.addWidget(self.reasoning_panel)
        splitter.addWidget(self.docs_panel)
        splitter.addWidget(self.feedback_panel)
        splitter.addWidget(self.workspace_panel)
        splitter_row.addWidget(splitter)
        layout.addLayout(splitter_row)
        central.setLayout(layout)
        self.setCentralWidget(central)

        settings_action = QAction("⚙️ Настройки", self)
        settings_action.triggered.connect(self.open_settings)
        menu = self.menuBar().addMenu("Файл")
        menu.addAction(settings_action)
        self.refresh_model_status()

    def handle_message(self, text: str) -> None:
        msg = [LLMMessage(role="user", content=text)]
        response = self.agent.respond(msg)
        self.chat.append_response(response)
        self.logs.append_log(f"[User]: {text}\n[Response]: {response}")

    def handle_feedback(
        self, prompt: str, answer: str, rating: str, hint: str | None = None
    ) -> None:
        self.agent.save_feedback(prompt, answer, rating, hint=hint)
        self.logs.append_log(f"[FEEDBACK] {rating} — {prompt[:30]}")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.agent)
        dialog.exec()
        self.refresh_model_status()

    def handle_tools_toggle(self, state: dict[str, bool]) -> None:
        self.agent.update_tools_enabled(state)
        self.refresh_model_status()

    def refresh_model_status(self) -> None:
        main_cfg = getattr(self.agent, "main_config", None)
        critic_cfg = getattr(self.agent, "critic_config", None)
        mode = getattr(self.agent.brain, "mode", "single")

        main_text = f"{main_cfg.provider}/{main_cfg.model}" if main_cfg else "не настроена"
        critic_text = f"{critic_cfg.provider}/{critic_cfg.model}" if critic_cfg else "—"
        enabled_tools = [k for k, v in self.agent.tools_enabled.items() if v and k != "safe_mode"]
        tools_text = ", ".join(enabled_tools) or "нет"
        safe_mode_text = "ON" if self.agent.tools_enabled.get("safe_mode") else "off"
        self.model_status.setText(
            f"Main: {main_text} | Critic: {critic_text} | Mode: {mode} | "
            f"Tools: {tools_text} | SafeMode: {safe_mode_text}"
        )

    def _log_workspace_answer(self, question: str, answer: str) -> None:
        self.logs.append_log(f"[Workspace Q]: {question}\n[Answer]: {answer}")
