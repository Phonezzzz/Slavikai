from __future__ import annotations

import hashlib
import time
from pathlib import Path

from config.tts_config import TtsConfig
from shared.models import ToolRequest, ToolResult
from tools.http_client import HttpClient, HttpResult

SANDBOX_AUDIO = Path("sandbox/audio")
SANDBOX_AUDIO.mkdir(parents=True, exist_ok=True)


class TtsTool:
    def __init__(self, http_client: HttpClient, config: TtsConfig | None = None) -> None:
        self.http = http_client
        self.config = config or TtsConfig.from_ui_settings()

    def handle(self, request: ToolRequest) -> ToolResult:
        text = str(request.args.get("text") or "").strip()
        if not text:
            return ToolResult.failure("Текст для озвучки пуст.")
        voice_id = str(request.args.get("voice_id") or self.config.resolve_voice_id() or "").strip()
        if not voice_id:
            return ToolResult.failure("voice_id не задан. Укажите голос или настройте в конфиге.")
        fmt = str(request.args.get("format") or self.config.format).lower()
        if fmt not in {"mp3", "wav"}:
            return ToolResult.failure("Формат должен быть mp3 или wav.")
        api_key = self.config.resolve_api_key()
        if not api_key:
            return ToolResult.failure("TTS API key не задан (TTS_API_KEY).")

        url = f"{self.config.endpoint}/{voice_id}"
        headers = {"xi-api-key": api_key, "Accept": f"audio/{fmt}"}
        payload = {"text": text, "model_id": "eleven_multilingual_v2"}

        result: HttpResult = self.http.post_bytes(
            url, json=payload, headers=headers, timeout=self.config.timeout
        )
        if not result.ok or not isinstance(result.data, (bytes, bytearray)):
            return ToolResult.failure(result.error or "Ошибка TTS сервиса.")

        file_name = self._build_filename(text, fmt)
        file_path = SANDBOX_AUDIO / file_name
        try:
            file_path.write_bytes(result.data)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Ошибка записи файла озвучки: {exc}")

        return ToolResult.success(
            {
                "output": "Аудио сгенерировано",
                "file_path": str(file_path),
                "format": fmt,
                "voice_id": voice_id,
            }
        )

    def _build_filename(self, text: str, fmt: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"tts_{int(time.time())}_{digest}.{fmt}"
