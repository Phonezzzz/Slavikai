from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

UI_SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".run" / "ui_settings.json"

# Доступные голоса OpenAI TTS
OPENAI_VOICES = frozenset(
    {
        "alloy",
        "echo",
        "fable",
        "onyx",
        "nova",
        "shimmer",
        "ash",
        "ballad",
        "coral",
        "sage",
        "verse",
    }
)

# Доступные модели OpenAI TTS
OPENAI_MODELS = frozenset({"tts-1", "tts-1-hd", "gpt-4o-mini-tts"})


def _load_ui_openai_api_key() -> str | None:
    """Загружает OpenAI API key из UI настроек (providers.openai.api_key)."""
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


def load_ui_tts_settings() -> dict[str, str]:
    """Загружает TTS settings из UI json (audio section)."""
    from typing import TypedDict

    class AudioUI(TypedDict):
        provider: str
        voice: str
        model: str

    if not UI_SETTINGS_PATH.exists():
        return {
            "provider": "elevenlabs",
            "voice": "",
            "model": "tts-1",
        }
    try:
        raw_payload: dict[str, object] = json.loads(UI_SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {
            "provider": "elevenlabs",
            "voice": "",
            "model": "tts-1",
        }
    if not isinstance(raw_payload, dict):
        return {
            "provider": "elevenlabs",
            "voice": "",
            "model": "tts-1",
        }
    audio_raw = raw_payload.get("audio")
    if not isinstance(audio_raw, dict):
        return {
            "provider": "elevenlabs",
            "voice": "",
            "model": "tts-1",
        }
    provider_raw = audio_raw.get("provider", "elevenlabs")
    voice_raw = audio_raw.get("voice", "")
    model_raw = audio_raw.get("model", "tts-1")
    provider = provider_raw.strip().lower() if isinstance(provider_raw, str) else "elevenlabs"
    if provider not in ("openai", "elevenlabs"):
        provider = "elevenlabs"
    voice = voice_raw.strip() if isinstance(voice_raw, str) else ""
    model = model_raw.strip() if isinstance(model_raw, str) else "tts-1"
    return {
        "provider": provider,
        "voice": voice,
        "model": model,
    }


@dataclass
class TtsConfig:
    # Общие настройки
    provider: str = "elevenlabs"  # "openai" | "elevenlabs"
    format: str = "mp3"
    timeout: int = 20

    # ElevenLabs настройки
    api_key: str | None = None  # ElevenLabs API key
    voice_id: str | None = None
    endpoint: str = "https://api.elevenlabs.io/v1/text-to-speech"

    # OpenAI TTS настройки
    openai_voice: str = "alloy"
    openai_model: str = "tts-1"
    openai_endpoint: str = "https://api.openai.com/v1/audio/speech"

    def resolve_api_key(self) -> str | None:
        """Возвращает API key в зависимости от провайдера."""
        if self.provider == "openai":
            return self._resolve_openai_api_key()
        return self._resolve_elevenlabs_api_key()

    def _resolve_openai_api_key(self) -> str | None:
        """Получает OpenAI API key из тех же источников что и STT."""
        env_key = os.getenv("OPENAI_API_KEY", "").strip()
        if env_key:
            return env_key
        ui_key = _load_ui_openai_api_key()
        if ui_key:
            return ui_key
        return None

    def _resolve_elevenlabs_api_key(self) -> str | None:
        """Получает ElevenLabs API key."""
        if self.api_key and self.api_key.strip():
            return self.api_key.strip()
        return os.getenv("TTS_API_KEY") or None

    def resolve_voice_id(self) -> str | None:
        """Возвращает voice_id для ElevenLabs."""
        return self.voice_id or os.getenv("TTS_VOICE_ID")

    def resolve_openai_voice(self) -> str:
        """Возвращает голос для OpenAI TTS с валидацией."""
        voice = self.openai_voice.lower()
        if voice not in OPENAI_VOICES:
            return "alloy"
        return voice

    def resolve_openai_model(self) -> str:
        """Возвращает модель для OpenAI TTS с валидацией."""
        model = self.openai_model.lower()
        if model not in OPENAI_MODELS:
            return "tts-1"
        return model

    def get_endpoint(self) -> str:
        """Возвращает endpoint в зависимости от провайдера."""
        if self.provider == "openai":
            return self.openai_endpoint
        return self.endpoint

    def get_voice_param(self) -> str | None:
        """Возвращает параметр голоса в зависимости от провайдера."""
        if self.provider == "openai":
            return self.resolve_openai_voice()
        return self.resolve_voice_id() or ""

    def get_model_param(self) -> str:
        """Возвращает параметр модели в зависимости от провайдера."""
        if self.provider == "openai":
            return self.resolve_openai_model()
        return "eleven_multilingual_v2"

    @classmethod
    def from_ui_settings(cls) -> TtsConfig:
        """Создаёт config из UI settings (fallback на defaults)."""
        ui = load_ui_tts_settings()
        provider = ui["provider"]
        voice = ui["voice"]
        model = ui["model"]
        return cls(
            provider=provider,
            voice_id=voice if provider == "elevenlabs" else None,
            openai_voice=voice if provider == "openai" else "alloy",
            openai_model=model if provider == "openai" else "tts-1",
        )
