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
        self.config = config or TtsConfig()

    def handle(self, request: ToolRequest) -> ToolResult:
        text = str(request.args.get("text") or "").strip()
        if not text:
            return ToolResult.failure("Текст для озвучки пуст.")
        if len(text) > self.config.max_input_chars:
            return ToolResult.failure(
                f"Текст для озвучки превышает лимит {self.config.max_input_chars} символов."
            )
        voice = str(
            request.args.get("voice")
            or request.args.get("voice_id")
            or self.config.resolve_voice()
            or ""
        ).strip()
        if not voice:
            return ToolResult.failure(
                "voice не задан. Укажите голос или настройте OPENAI_TTS_VOICE."
            )
        fmt = str(request.args.get("format") or self.config.resolve_format()).lower()
        if fmt not in {"mp3", "wav"}:
            return ToolResult.failure("Формат должен быть mp3 или wav.")
        model = str(request.args.get("model") or self.config.resolve_model()).strip()
        if not model:
            return ToolResult.failure(
                "TTS model не задан. Укажите модель или настройте OPENAI_TTS_MODEL."
            )
        api_key = self.config.resolve_api_key()
        if not api_key:
            return ToolResult.failure("OpenAI API key не задан для TTS (env OPENAI_API_KEY).")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": f"audio/{fmt}",
        }
        payload = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": fmt,
        }

        result: HttpResult = self.http.post_bytes(
            self.config.endpoint,
            json=payload,
            headers=headers,
            timeout=self.config.timeout,
        )
        if (
            not result.ok
            or not isinstance(result.data, (bytes, bytearray))
            or len(result.data) == 0
        ):
            return ToolResult.failure(result.error or "Ошибка OpenAI TTS сервиса.")

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
                "voice": voice,
                "voice_id": voice,
                "model": model,
            }
        )

    def _build_filename(self, text: str, fmt: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"tts_{int(time.time())}_{digest}.{fmt}"
