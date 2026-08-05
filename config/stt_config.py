from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from config.api_keys import DEFAULT_API_KEYS_PATH, load_api_keys


@dataclass
class SttConfig:
    api_key: str | None = None
    endpoint: str = "https://api.openai.com/v1/audio/transcriptions"
    model: str = "whisper-1"
    language: str = "ru"
    timeout: int = 20
    api_keys_path: Path = DEFAULT_API_KEYS_PATH

    def resolve_api_key(self) -> str | None:
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key
        legacy_key = os.getenv("STT_API_KEY", "").strip()
        if legacy_key:
            return legacy_key
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        return load_api_keys(path=self.api_keys_path).get("openai")
