from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QCheckBox, QLabel, QVBoxLayout, QWidget


class ToolsPanel(QWidget):
    def __init__(
        self,
        on_change: Callable[[dict[str, bool]], None] | None = None,
        initial_state: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.on_change = on_change
        state = initial_state or {}

        layout = QVBoxLayout()
        layout.addWidget(QLabel("🧰 Инструменты:"))
        self.web_tool = QCheckBox("Интернет")
        self.fs_tool = QCheckBox("Файлы")
        self.shell_tool = QCheckBox("Shell")
        self.img_tool = QCheckBox("Картинки")
        self.tts_tool = QCheckBox("TTS (озвучка)")
        self.stt_tool = QCheckBox("STT (распознавание)")
        self.safe_mode = QCheckBox("Safe mode (блок web/shell)")

        for widget, key in (
            (self.web_tool, "web"),
            (self.fs_tool, "fs"),
            (self.shell_tool, "shell"),
            (self.img_tool, "img"),
            (self.tts_tool, "tts"),
            (self.stt_tool, "stt"),
            (self.safe_mode, "safe_mode"),
        ):
            widget.setChecked(state.get(key, False))
            widget.stateChanged.connect(self._emit_state)
            layout.addWidget(widget)

        self.safe_mode.setToolTip("Safe mode выключает web/shell/project/tts/stt")
        self._update_safe_indicator()

        self.setLayout(layout)

    def _emit_state(self) -> None:
        state = {
            "web": self.web_tool.isChecked(),
            "fs": self.fs_tool.isChecked(),
            "shell": self.shell_tool.isChecked(),
            "img": self.img_tool.isChecked(),
            "tts": self.tts_tool.isChecked(),
            "stt": self.stt_tool.isChecked(),
            "safe_mode": self.safe_mode.isChecked(),
        }
        if self.on_change:
            self.on_change(state)
        self._update_safe_indicator()

    def set_state(self, state: dict[str, bool]) -> None:
        self.web_tool.setChecked(state.get("web", False))
        self.fs_tool.setChecked(state.get("fs", False))
        self.shell_tool.setChecked(state.get("shell", False))
        self.img_tool.setChecked(state.get("img", False))
        self.tts_tool.setChecked(state.get("tts", False))
        self.stt_tool.setChecked(state.get("stt", False))
        self.safe_mode.setChecked(state.get("safe_mode", False))

    def _update_safe_indicator(self) -> None:
        if self.safe_mode.isChecked():
            self.safe_mode.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.safe_mode.setStyleSheet("")
