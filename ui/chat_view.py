from __future__ import annotations

import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from core.agent import Agent
from shared.memory_companion_models import FeedbackLabel, FeedbackRating
from shared.models import ToolResult
from ui.audio_player import ChatAudioPlayer, ChatAudioRecorder
from ui.chat_message_widget import ChatMessageWidget
from ui.feedback_dialog import FeedbackDialog

SANDBOX_AUDIO = Path("sandbox/audio")
SANDBOX_AUDIO.mkdir(parents=True, exist_ok=True)


class RecordState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class ToolWorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)

    def __init__(self) -> None:
        super().__init__()


class ToolWorker(QObject, QRunnable):
    def __init__(self, fn: Callable[[], ToolResult]) -> None:
        QObject.__init__(self)
        QRunnable.__init__(self)
        self.fn = fn
        self.signals = ToolWorkerSignals()

    def run(self) -> None:
        try:
            result = self.fn()
            self.signals.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.signals.error.emit(str(exc))


class ChatView(QWidget):
    def __init__(
        self,
        agent: Agent,
        audio_player: ChatAudioPlayer,
        on_send_callback: Callable[[str], None],
        on_feedback_callback: Callable[[str, FeedbackRating, list[FeedbackLabel], str | None], None]
        | None = None,
    ):
        super().__init__()
        self.agent = agent
        self.audio_player = audio_player
        self.audio_recorder = ChatAudioRecorder()
        self.on_send_callback = on_send_callback
        self.on_feedback_callback = on_feedback_callback
        self.record_state = RecordState.IDLE
        self.last_prompt = ""
        self.last_response = ""
        self.last_interaction_id: str | None = None
        self.thread_pool = QThreadPool.globalInstance()
        self._message_widgets: list[ChatMessageWidget] = []
        self._current_record_path: Path | None = None
        self.init_ui()

    def init_ui(self) -> None:
        layout = QVBoxLayout()

        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_container.setLayout(self.messages_layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.messages_container)

        self.feedback_layout = QHBoxLayout()
        self.good_btn = QPushButton("👍")
        self.ok_btn = QPushButton("😐")
        self.bad_btn = QPushButton("👎")

        for btn, tag in [
            (self.good_btn, FeedbackRating.GOOD),
            (self.ok_btn, FeedbackRating.OK),
            (self.bad_btn, FeedbackRating.BAD),
        ]:
            btn.clicked.connect(lambda _, t=tag: self.rate_response(t))
            self.feedback_layout.addWidget(btn)

        input_row = QHBoxLayout()
        self.input_field = QLineEdit()
        self.send_btn = QPushButton("Отправить")
        self.send_btn.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.clicked.connect(self.toggle_recording)
        input_row.addWidget(self.input_field)
        input_row.addWidget(self.mic_btn)
        input_row.addWidget(self.send_btn)

        self.status_label = QLabel()
        self.status_label.setFrameShape(QFrame.Shape.NoFrame)
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")

        layout.addWidget(self.scroll_area)
        layout.addLayout(self.feedback_layout)
        layout.addLayout(input_row)
        layout.addWidget(self.status_label)
        self.setLayout(layout)

    def send_message(self) -> None:
        text = self.input_field.text().strip()
        if not text:
            return
        self.last_prompt = text
        self._add_message(f"🧍‍♂️ Вы: {text}", is_assistant=False)
        self.on_send_callback(text)
        self.input_field.clear()

    def append_response(self, response: str) -> None:
        self.last_response = response
        self.last_interaction_id = self.agent.last_chat_interaction_id
        self._add_message(f"🤖 AI: {response}", is_assistant=True, spoken_text=response)
        hints_meta = getattr(self.agent, "last_hints_meta", [])
        if hints_meta:
            applied = "; ".join(
                f"[{item.get('severity')}] {item.get('hint')}" for item in hints_meta
            )
            self._set_status(f"Применены подсказки: {applied}")

    def _add_message(self, text: str, is_assistant: bool, spoken_text: str | None = None) -> None:
        widget = ChatMessageWidget(text, is_assistant=is_assistant, spoken_text=spoken_text)
        if is_assistant and widget.tts_button:
            widget.tts_requested.connect(self.handle_tts_request)
        self.messages_layout.addWidget(widget)
        self._message_widgets.append(widget)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def rate_response(self, rating: FeedbackRating) -> None:
        if not self.on_feedback_callback:
            return
        if not self.last_response:
            self._set_status("Нет ответа для оценки.")
            return
        if not self.last_interaction_id:
            self._set_status("Нет interaction_id для оценки (InteractionLog не записан).")
            return

        dialog = FeedbackDialog(rating=rating, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        data = dialog.get_result()
        self.on_feedback_callback(
            self.last_interaction_id,
            rating,
            data.labels,
            data.free_text,
        )
        self._set_status(f"💬 Feedback сохранён: {rating.value}")

    def handle_tts_request(self, widget: ChatMessageWidget) -> None:
        if widget.tts_file_path:
            self.audio_player.play_file(widget.tts_file_path)
            return
        widget.set_tts_busy(True)
        worker = ToolWorker(lambda: self.agent.synthesize_speech(widget.get_spoken_text()))
        worker.signals.finished.connect(lambda result, w=widget: self._on_tts_finished(w, result))
        worker.signals.error.connect(lambda err, w=widget: self._on_tts_error(w, err))
        self.thread_pool.start(worker)

    def _on_tts_finished(self, widget: ChatMessageWidget, result: ToolResult) -> None:
        widget.set_tts_busy(False)
        if not result.ok:
            self._set_status(result.error or "Ошибка TTS.")
            return
        file_path = str(result.data.get("file_path") or "")
        if not file_path:
            self._set_status("TTS не вернул файл.")
            return
        widget.set_tts_file(file_path)
        played = self.audio_player.play_file(file_path)
        if played:
            self._set_status("Озвучка воспроизводится.")
        else:
            self._set_status("Файл озвучки не найден.")

    def _on_tts_error(self, widget: ChatMessageWidget, error: str) -> None:
        widget.set_tts_busy(False)
        self._set_status(f"Ошибка TTS: {error}")

    def toggle_recording(self) -> None:
        if self.record_state == RecordState.IDLE:
            self._start_recording()
        elif self.record_state == RecordState.RECORDING:
            self._stop_and_transcribe()
        else:
            self._set_status("Идёт обработка записи, подождите.")

    def _start_recording(self) -> None:
        file_path = SANDBOX_AUDIO / f"rec_{int(time.time())}.wav"
        try:
            self.audio_recorder.start(file_path)
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Не удалось начать запись: {exc}")
            return
        self.record_state = RecordState.RECORDING
        self._current_record_path = file_path
        self.mic_btn.setText("■ Стоп")
        self._set_status("Запись... нажмите ещё раз, чтобы остановить.")

    def _stop_and_transcribe(self) -> None:
        if not self._current_record_path:
            self._set_status("Нет активной записи.")
            return
        try:
            saved_path = self.audio_recorder.stop()
        except Exception as exc:  # noqa: BLE001
            self._set_status(f"Ошибка при остановке записи: {exc}")
            self.record_state = RecordState.IDLE
            self.mic_btn.setText("🎤")
            self._current_record_path = None
            return
        self.record_state = RecordState.PROCESSING
        self.mic_btn.setText("⏳")
        worker = ToolWorker(lambda: self.agent.transcribe_audio(str(saved_path), language="ru"))
        worker.signals.finished.connect(self._on_stt_finished)
        worker.signals.error.connect(self._on_stt_error)
        self.thread_pool.start(worker)

    def _on_stt_finished(self, result: ToolResult) -> None:
        self.record_state = RecordState.IDLE
        self.mic_btn.setText("🎤")
        self._current_record_path = None
        if not result.ok:
            self._set_status(result.error or "Ошибка STT.")
            return
        text = str(result.data.get("text") or result.data.get("output") or "")
        if text:
            self.input_field.setText(text)
            self._set_status("Распознавание готово. Проверьте текст и отправьте.")
        else:
            self._set_status("STT вернул пустой текст.")

    def _on_stt_error(self, error: str) -> None:
        self.record_state = RecordState.IDLE
        self.mic_btn.setText("🎤")
        self._current_record_path = None
        self._set_status(f"Ошибка STT: {error}")

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)
