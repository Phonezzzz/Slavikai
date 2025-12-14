from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtMultimedia import (
    QAudioInput,
    QAudioOutput,
    QMediaCaptureSession,
    QMediaPlayer,
    QMediaRecorder,
)


class ChatAudioPlayer:
    """Глобальный аудиоплеер, без наложения треков."""

    def __init__(self) -> None:
        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

    def play_file(self, file_path: str) -> bool:
        path = Path(file_path)
        if not path.exists():
            return False
        # остановить, если что-то играет
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.player.play()
        return True


class ChatAudioRecorder:
    """Мини-обёртка над Qt-рекордером для записи микрофона в файл."""

    def __init__(self) -> None:
        self.audio_input = QAudioInput()
        self.recorder = QMediaRecorder()
        self.capture_session = QMediaCaptureSession()
        self.capture_session.setAudioInput(self.audio_input)
        self.capture_session.setRecorder(self.recorder)
        self._output_path: Path | None = None

    def start(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self.recorder.setOutputLocation(QUrl.fromLocalFile(str(output_path.resolve())))
        self.recorder.record()

    def stop(self) -> Path:
        if not self._output_path:
            raise RuntimeError("Запись не запущена.")
        self.recorder.stop()
        return self._output_path

    def is_recording(self) -> bool:
        return self.recorder.recorderState() == QMediaRecorder.RecorderState.RecordingState
