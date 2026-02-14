from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

EmbeddingsProvider = Literal["local", "openai"]

DEFAULT_UI_SETTINGS_PATH: Final[Path] = (
    Path(__file__).resolve().parent.parent / ".run" / "ui_settings.json"
)
DEFAULT_EMBEDDINGS_PROVIDER: Final[EmbeddingsProvider] = "local"
DEFAULT_LOCAL_EMBEDDINGS_MODEL: Final[str] = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDINGS_MODEL: Final[str] = "text-embedding-3-small"


@dataclass(frozen=True)
class UIEmbeddingsSettings:
    provider: EmbeddingsProvider = DEFAULT_EMBEDDINGS_PROVIDER
    local_model: str = DEFAULT_LOCAL_EMBEDDINGS_MODEL
    openai_model: str = DEFAULT_OPENAI_EMBEDDINGS_MODEL


def _load_ui_settings_blob(path: Path = DEFAULT_UI_SETTINGS_PATH) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_ui_settings_blob(
    payload: dict[str, object],
    path: Path = DEFAULT_UI_SETTINGS_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_provider(value: object) -> EmbeddingsProvider:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"local", "openai"}:
            return cast(EmbeddingsProvider, normalized)
    return DEFAULT_EMBEDDINGS_PROVIDER


def _normalize_model(value: object, *, default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip()
        if normalized:
            return normalized
    return default


def load_ui_embeddings_settings(path: Path = DEFAULT_UI_SETTINGS_PATH) -> UIEmbeddingsSettings:
    payload = _load_ui_settings_blob(path)
    memory_raw = payload.get("memory")
    if not isinstance(memory_raw, dict):
        return UIEmbeddingsSettings()

    embeddings_raw = memory_raw.get("embeddings")
    if isinstance(embeddings_raw, dict):
        provider = _normalize_provider(embeddings_raw.get("provider"))
        local_model = _normalize_model(
            embeddings_raw.get("local_model"),
            default=DEFAULT_LOCAL_EMBEDDINGS_MODEL,
        )
        openai_model = _normalize_model(
            embeddings_raw.get("openai_model"),
            default=DEFAULT_OPENAI_EMBEDDINGS_MODEL,
        )
        return UIEmbeddingsSettings(
            provider=provider,
            local_model=local_model,
            openai_model=openai_model,
        )

    # Backward compatibility with legacy settings.memory.embeddings_model
    legacy_model = _normalize_model(
        memory_raw.get("embeddings_model"),
        default=DEFAULT_LOCAL_EMBEDDINGS_MODEL,
    )
    return UIEmbeddingsSettings(
        provider="local",
        local_model=legacy_model,
        openai_model=DEFAULT_OPENAI_EMBEDDINGS_MODEL,
    )


def save_ui_embeddings_settings(
    settings: UIEmbeddingsSettings,
    path: Path = DEFAULT_UI_SETTINGS_PATH,
) -> None:
    payload = _load_ui_settings_blob(path)
    memory_raw = payload.get("memory")
    memory_payload: dict[str, object]
    if isinstance(memory_raw, dict):
        memory_payload = dict(memory_raw)
    else:
        memory_payload = {}
    memory_payload["embeddings"] = {
        "provider": settings.provider,
        "local_model": settings.local_model,
        "openai_model": settings.openai_model,
    }
    memory_payload.pop("embeddings_model", None)
    payload["memory"] = memory_payload
    _save_ui_settings_blob(payload, path)


def resolve_openai_api_key(path: Path = DEFAULT_UI_SETTINGS_PATH) -> str | None:
    payload = _load_ui_settings_blob(path)
    providers_raw = payload.get("providers")
    if isinstance(providers_raw, dict):
        openai_raw = providers_raw.get("openai")
        key_raw: object | None = None
        if isinstance(openai_raw, dict):
            key_raw = openai_raw.get("api_key")
        elif isinstance(openai_raw, str):
            key_raw = openai_raw
        if isinstance(key_raw, str):
            normalized = key_raw.strip()
            if normalized:
                return normalized

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    return env_key or None
