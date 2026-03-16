from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TtsConfig:
    api_key: str | None = None
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    endpoint: str = "https://api.openai.com/v1/audio/speech"
    format: str = "mp3"
    timeout: int = 20
    max_input_chars: int = 4096

    def resolve_api_key(self) -> str | None:
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        return env_key or None

    def resolve_model(self) -> str:
        env_model = os.getenv("OPENAI_TTS_MODEL", "").strip()
        if env_model:
            return env_model
        return self.model

    def resolve_voice(self) -> str:
        env_voice = os.getenv("OPENAI_TTS_VOICE", "").strip()
        if env_voice:
            return env_voice
        return self.voice

    def resolve_format(self) -> str:
        env_format = os.getenv("OPENAI_TTS_FORMAT", "").strip().lower()
        if env_format:
            return env_format
        return self.format.lower()
