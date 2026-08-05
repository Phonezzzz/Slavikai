from __future__ import annotations

import hashlib
import time
import wave
from io import BytesIO
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
            return ToolResult.failure(
                "OpenAI API key не задан для TTS (env или Settings → API Keys)."
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": f"audio/{fmt}",
        }

        chunks = self._split_text(text, self.config.max_request_chars)
        audio_segments: list[bytes] = []
        for index, chunk in enumerate(chunks, start=1):
            payload = {
                "model": model,
                "voice": voice,
                "input": chunk,
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
                chunk_hint = f" (chunk {index}/{len(chunks)})" if len(chunks) > 1 else ""
                return ToolResult.failure(result.error or f"Ошибка OpenAI TTS сервиса{chunk_hint}.")
            audio_segments.append(bytes(result.data))

        try:
            audio_data = self._join_audio_segments(audio_segments, fmt)
        except ValueError as exc:
            return ToolResult.failure(str(exc))

        file_name = self._build_filename(text, fmt)
        file_path = SANDBOX_AUDIO / file_name
        try:
            file_path.write_bytes(audio_data)
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
                "chunks": len(chunks),
            }
        )

    def _build_filename(self, text: str, fmt: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"tts_{int(time.time())}_{digest}.{fmt}"

    def _split_text(self, text: str, max_chars: int) -> list[str]:
        limit = max(1, max_chars)
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len(remaining) > limit:
            cut = self._best_split_index(remaining, limit)
            chunk = remaining[:cut].strip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[cut:].strip()
        if remaining:
            chunks.append(remaining)
        return chunks

    def _best_split_index(self, text: str, limit: int) -> int:
        candidates = [
            text.rfind("\n\n", 0, limit + 1),
            text.rfind("\n", 0, limit + 1),
            text.rfind(". ", 0, limit + 1),
            text.rfind("! ", 0, limit + 1),
            text.rfind("? ", 0, limit + 1),
            text.rfind(" ", 0, limit + 1),
        ]
        best = max(candidates)
        if best < max(1, limit // 2):
            return limit
        if text[best : best + 1] in {".", "!", "?"}:
            return best + 1
        return best

    def _join_audio_segments(self, segments: list[bytes], fmt: str) -> bytes:
        if not segments:
            raise ValueError("TTS не вернул аудио.")
        if len(segments) == 1:
            return segments[0]
        if fmt == "mp3":
            return b"".join(segments)
        if fmt == "wav":
            return self._join_wav_segments(segments)
        raise ValueError("Формат должен быть mp3 или wav.")

    def _join_wav_segments(self, segments: list[bytes]) -> bytes:
        output = BytesIO()
        params: wave._wave_params | None = None
        frames: list[bytes] = []
        for segment in segments:
            try:
                with wave.open(BytesIO(segment), "rb") as reader:
                    current_params = reader.getparams()
                    if params is None:
                        params = current_params
                    elif current_params[:3] != params[:3]:
                        raise ValueError("TTS WAV chunks have incompatible audio parameters.")
                    frames.append(reader.readframes(reader.getnframes()))
            except (EOFError, wave.Error) as exc:
                raise ValueError(f"Не удалось объединить WAV chunks: {exc}") from exc
        if params is None:
            raise ValueError("TTS не вернул WAV-аудио.")
        with wave.open(output, "wb") as writer:
            writer.setparams(params)
            for frame in frames:
                writer.writeframes(frame)
        return output.getvalue()
