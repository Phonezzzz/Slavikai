from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SttConfig:
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/audio/transcriptions"
    model: str = "whisper-1"
    language: str = "ru"
    timeout: int = 20

    def resolve_api_key(self) -> str | None:
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key
        legacy_key = os.getenv("STT_API_KEY", "").strip()
        return legacy_key or None
