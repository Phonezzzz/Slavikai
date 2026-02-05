from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

UI_SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".run" / "ui_settings.json"


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
        ui_key = _load_ui_openai_api_key()
        if ui_key:
            return ui_key
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key
        legacy_key = os.getenv("STT_API_KEY", "").strip()
        return legacy_key or None


def _load_ui_openai_api_key() -> str | None:
    if not UI_SETTINGS_PATH.exists():
        return None
    try:
        raw_payload = json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(raw_payload, dict):
        return None
    providers_raw = raw_payload.get("providers")
    if not isinstance(providers_raw, dict):
        return None
    openai_raw = providers_raw.get("openai")
    if not isinstance(openai_raw, dict):
        return None
    api_key_raw = openai_raw.get("api_key")
    if not isinstance(api_key_raw, str):
        return None
    normalized = api_key_raw.strip()
    return normalized or None
