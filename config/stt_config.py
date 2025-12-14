from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class SttConfig:
    api_key: str | None = None
    endpoint: str = "https://api.elevenlabs.io/v1/speech-to-text"
    language: str = "ru"
    timeout: int = 20

    def resolve_api_key(self) -> str | None:
        return self.api_key or os.getenv("STT_API_KEY")
