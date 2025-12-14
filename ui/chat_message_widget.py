from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget


class ChatMessageWidget(QWidget):
    """Сообщение в чате с опцией озвучки для сообщений ассистента."""

    tts_requested = Signal(object)

    def __init__(self, text: str, is_assistant: bool, spoken_text: str | None = None) -> None:
        super().__init__()
        self.text = text
        self.spoken_text = spoken_text or text
        self.is_assistant = is_assistant
        self.tts_file_path: str | None = None
        self.tts_button: QPushButton | None = None
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout()
        text_label = QLabel(self.text)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        if self.is_assistant:
            self.tts_button = QPushButton("▶ Озвучить")
            self.tts_button.clicked.connect(self._emit_tts)
            layout.addWidget(self.tts_button)

        self.setLayout(layout)

    def _emit_tts(self) -> None:
        self.tts_requested.emit(self)

    def get_spoken_text(self) -> str:
        return self.spoken_text

    def set_tts_file(self, file_path: str) -> None:
        self.tts_file_path = file_path
        if self.tts_button:
            self.tts_button.setText("▶ Воспроизвести")

    def set_tts_busy(self, busy: bool) -> None:
        if self.tts_button:
            self.tts_button.setEnabled(not busy)
            self.tts_button.setText("⏳ Озвучка..." if busy else "▶ Озвучить")
