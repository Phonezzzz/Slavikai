from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TtsConfig:
    api_key: str | None = None
    voice_id: str | None = None
    endpoint: str = "https://api.elevenlabs.io/v1/text-to-speech"
    format: str = "mp3"
    timeout: int = 20

    def resolve_api_key(self) -> str | None:
        return self.api_key or os.getenv("TTS_API_KEY")

    def resolve_voice_id(self) -> str | None:
        return self.voice_id or os.getenv("TTS_VOICE_ID")
