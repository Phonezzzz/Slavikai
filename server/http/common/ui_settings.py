from __future__ import annotations

import difflib
import json
import os
from pathlib import Path
from typing import Final, Literal

import requests

from config.memory_config import load_memory_config
from config.tools_config import (
    DEFAULT_TOOLS_STATE,
    ToolsConfig,
    load_tools_config,
    save_tools_config,
)
from config.ui_embeddings_settings import (
    UIEmbeddingsSettings,
    load_ui_embeddings_settings,
    save_ui_embeddings_settings,
)
from llm.local_http_brain import DEFAULT_LOCAL_ENDPOINT
from llm.types import ModelConfig
from shared.models import JSONValue

SUPPORTED_MODEL_PROVIDERS: Final[set[str]] = {"xai", "openrouter", "local"}
API_KEY_SETTINGS_PROVIDERS: Final[set[str]] = {"xai", "openrouter", "local", "openai"}
PROVIDER_API_KEY_ENV: Final[dict[str, str]] = {
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "local": "LOCAL_LLM_API_KEY",
    "openai": "OPENAI_API_KEY",
}
XAI_MODELS_ENDPOINT: Final[str] = "https://api.x.ai/v1/models"
OPENROUTER_MODELS_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/models"
OPENAI_STT_ENDPOINT: Final[str] = "https://api.openai.com/v1/audio/transcriptions"
MODEL_FETCH_TIMEOUT: Final[int] = 20
UI_SETTINGS_PATH: Final[Path] = Path(__file__).resolve().parents[3] / ".run" / "ui_settings.json"
DEFAULT_UI_TONE: Final[str] = "balanced"
INDEX_ENABLED_ENV: Final[str] = "SLAVIK_INDEX_ENABLED"
DEFAULT_LONG_PASTE_TO_FILE_ENABLED: Final[bool] = True
DEFAULT_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 12_000
MIN_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 1_000
MAX_LONG_PASTE_THRESHOLD_CHARS: Final[int] = 80_000
POLICY_PROFILES: Final[set[str]] = {"sandbox", "index", "yolo"}
DEFAULT_POLICY_PROFILE: Final[str] = "sandbox"
UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS: Final[set[str]] = {
    "personalization",
    "composer",
    "memory",
    "providers",
}
UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS: Final[set[str]] = {
    "tools",
    "policy",
    "risk",
    "security",
    "risk_categories",
    "security_categories",
    "approval_categories",
    "approved_categories",
    "safe_mode",
}


def _normalize_provider(raw_provider: str) -> str | None:
    normalized = raw_provider.strip().lower()
    if normalized in SUPPORTED_MODEL_PROVIDERS:
        return normalized
    return None


def _build_model_config(provider: str, model_id: str) -> ModelConfig:
    if provider == "xai":
        return ModelConfig(provider="xai", model=model_id)
    if provider == "openrouter":
        return ModelConfig(provider="openrouter", model=model_id)
    if provider == "local":
        return ModelConfig(provider="local", model=model_id)
    raise ValueError(f"Неизвестный провайдер: {provider}")


def _closest_model_suggestion(model_id: str, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    matches = difflib.get_close_matches(model_id, candidates, n=1, cutoff=0.4)
    if not matches:
        return None
    return matches[0]


def _local_models_endpoint() -> str:
    base_url = os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_ENDPOINT).strip()
    if not base_url:
        base_url = DEFAULT_LOCAL_ENDPOINT
    if base_url.endswith("/chat/completions"):
        return f"{base_url.removesuffix('/chat/completions')}/models"
    return f"{base_url.rstrip('/')}/models"


def _provider_models_endpoint(provider: str) -> str:
    if provider == "xai":
        return XAI_MODELS_ENDPOINT
    if provider == "openrouter":
        return OPENROUTER_MODELS_ENDPOINT
    if provider == "local":
        return _local_models_endpoint()
    raise ValueError(f"Неизвестный провайдер: {provider}")


def _provider_auth_headers(
    provider: str,
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> tuple[dict[str, str], str | None]:
    if provider == "xai":
        api_key = _resolve_provider_api_key("xai", ui_settings_path=ui_settings_path)
        if not api_key:
            return {}, "Не задан XAI_API_KEY (env или UI settings)."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "openrouter":
        api_key = _resolve_provider_api_key("openrouter", ui_settings_path=ui_settings_path)
        if not api_key:
            return {}, "Не задан OPENROUTER_API_KEY (env или UI settings)."
        return {"Authorization": f"Bearer {api_key}"}, None
    if provider == "local":
        api_key = _resolve_provider_api_key("local", ui_settings_path=ui_settings_path)
        if not api_key:
            return {}, None
        return {"Authorization": f"Bearer {api_key}"}, None
    return {}, f"Неизвестный провайдер: {provider}"


def _parse_models_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    models: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if isinstance(model_id, str) and model_id.strip():
            models.append(model_id.strip())
    unique: list[str] = []
    for model_id in models:
        if model_id not in unique:
            unique.append(model_id)
    return unique


def _fetch_provider_models(
    provider: str,
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> tuple[list[str], str | None]:
    try:
        url = _provider_models_endpoint(provider)
    except ValueError as exc:
        return [], str(exc)
    headers, auth_error = _provider_auth_headers(provider, ui_settings_path=ui_settings_path)
    if auth_error:
        return [], auth_error
    try:
        response = requests.get(url, headers=headers, timeout=MODEL_FETCH_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        if provider == "local":
            fallback = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "local-default").strip()
            if fallback:
                return [fallback], None
        return [], f"Не удалось получить список моделей провайдера {provider}: {exc}"
    models = _parse_models_payload(response.json())
    if not models and provider == "local":
        fallback = os.getenv("LOCAL_LLM_DEFAULT_MODEL", "local-default").strip()
        if fallback:
            models = [fallback]
    return models, None


def _load_ui_settings_blob(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> dict[str, object]:
    if not ui_settings_path.exists():
        return {}
    try:
        raw = json.loads(ui_settings_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_ui_settings_blob(
    payload: dict[str, object],
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    ui_settings_path.parent.mkdir(parents=True, exist_ok=True)
    ui_settings_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_personalization_settings(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> tuple[str, str]:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    personalization_raw = payload.get("personalization")
    if not isinstance(personalization_raw, dict):
        return DEFAULT_UI_TONE, ""
    tone_raw = personalization_raw.get("tone")
    prompt_raw = personalization_raw.get("system_prompt")
    tone = tone_raw.strip() if isinstance(tone_raw, str) and tone_raw.strip() else DEFAULT_UI_TONE
    system_prompt = prompt_raw if isinstance(prompt_raw, str) else ""
    return tone, system_prompt


def _save_personalization_settings(
    *,
    tone: str,
    system_prompt: str,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    payload["personalization"] = {
        "tone": tone.strip() or DEFAULT_UI_TONE,
        "system_prompt": system_prompt,
    }
    _save_ui_settings_blob(payload, ui_settings_path=ui_settings_path)


def _load_embeddings_settings(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> UIEmbeddingsSettings:
    return load_ui_embeddings_settings(path=ui_settings_path)


def _save_embeddings_settings(
    settings: UIEmbeddingsSettings,
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    save_ui_embeddings_settings(settings, path=ui_settings_path)


def _load_composer_settings(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> tuple[bool, int]:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    composer_raw = payload.get("composer")
    if not isinstance(composer_raw, dict):
        return DEFAULT_LONG_PASTE_TO_FILE_ENABLED, DEFAULT_LONG_PASTE_THRESHOLD_CHARS
    enabled_raw = composer_raw.get("long_paste_to_file_enabled")
    threshold_raw = composer_raw.get("long_paste_threshold_chars")
    enabled = enabled_raw if isinstance(enabled_raw, bool) else DEFAULT_LONG_PASTE_TO_FILE_ENABLED
    threshold = (
        threshold_raw
        if isinstance(threshold_raw, int) and not isinstance(threshold_raw, bool)
        else DEFAULT_LONG_PASTE_THRESHOLD_CHARS
    )
    if threshold < MIN_LONG_PASTE_THRESHOLD_CHARS:
        threshold = MIN_LONG_PASTE_THRESHOLD_CHARS
    if threshold > MAX_LONG_PASTE_THRESHOLD_CHARS:
        threshold = MAX_LONG_PASTE_THRESHOLD_CHARS
    return enabled, threshold


def _save_composer_settings(
    *,
    long_paste_to_file_enabled: bool,
    long_paste_threshold_chars: int,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    payload["composer"] = {
        "long_paste_to_file_enabled": long_paste_to_file_enabled,
        "long_paste_threshold_chars": long_paste_threshold_chars,
    }
    _save_ui_settings_blob(payload, ui_settings_path=ui_settings_path)


def _user_plane_forbidden_settings_key(payload: dict[str, object]) -> str | None:
    invalid = sorted(
        {
            normalized
            for key in payload
            if (
                (normalized := str(key).strip()) in UI_SETTINGS_CONTROL_TOP_LEVEL_KEYS
                or normalized not in UI_SETTINGS_USER_ALLOWED_TOP_LEVEL_KEYS
            )
        }
    )
    if not invalid:
        return None
    return invalid[0]


def _normalize_policy_profile(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in POLICY_PROFILES:
            return normalized
    return DEFAULT_POLICY_PROFILE


def _load_policy_settings(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> tuple[str, bool, str | None]:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    policy_raw = payload.get("policy")
    if not isinstance(policy_raw, dict):
        return DEFAULT_POLICY_PROFILE, False, None
    profile = _normalize_policy_profile(policy_raw.get("profile"))
    yolo_armed_raw = policy_raw.get("yolo_armed")
    yolo_armed = yolo_armed_raw if isinstance(yolo_armed_raw, bool) else False
    yolo_armed_at_raw = policy_raw.get("yolo_armed_at")
    yolo_armed_at = (
        yolo_armed_at_raw.strip()
        if isinstance(yolo_armed_at_raw, str) and yolo_armed_at_raw.strip()
        else None
    )
    if not yolo_armed:
        yolo_armed_at = None
    return profile, yolo_armed, yolo_armed_at


def _save_policy_settings(
    *,
    profile: str,
    yolo_armed: bool,
    yolo_armed_at: str | None,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    payload["policy"] = {
        "profile": _normalize_policy_profile(profile),
        "yolo_armed": yolo_armed,
        "yolo_armed_at": yolo_armed_at if yolo_armed else None,
    }
    _save_ui_settings_blob(payload, ui_settings_path=ui_settings_path)


def _tools_state_for_profile(profile: str, current: dict[str, bool]) -> dict[str, bool]:
    normalized = _normalize_policy_profile(profile)
    if normalized == "sandbox":
        return {
            **current,
            "fs": True,
            "project": False,
            "shell": False,
            "web": False,
            "safe_mode": True,
        }
    if normalized == "index":
        return {
            **current,
            "fs": True,
            "project": True,
            "shell": False,
            "web": False,
            "safe_mode": True,
        }
    return {
        **current,
        "safe_mode": True,
    }


def _load_tools_state() -> dict[str, bool]:
    try:
        return load_tools_config().to_dict()
    except Exception:  # noqa: BLE001
        return dict(DEFAULT_TOOLS_STATE)


def _save_tools_state(state: dict[str, bool]) -> None:
    payload: dict[str, object] = {key: value for key, value in state.items()}
    save_tools_config(ToolsConfig.from_dict(payload))


def _load_provider_api_keys(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> dict[str, str]:
    del ui_settings_path
    return {}


def _save_provider_api_keys(
    api_keys: dict[str, str],
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    del api_keys
    _drop_legacy_provider_api_keys(ui_settings_path=ui_settings_path)


def _drop_legacy_provider_api_keys(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> bool:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    if "providers" not in payload:
        return False
    payload.pop("providers", None)
    _save_ui_settings_blob(payload, ui_settings_path=ui_settings_path)
    return True


def _load_provider_runtime_checks(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> dict[str, dict[str, JSONValue]]:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    checks_raw = payload.get("provider_runtime_checks")
    if not isinstance(checks_raw, dict):
        return {}
    checks: dict[str, dict[str, JSONValue]] = {}
    for provider in API_KEY_SETTINGS_PROVIDERS:
        item_raw = checks_raw.get(provider)
        if not isinstance(item_raw, dict):
            continue
        valid_raw = item_raw.get("api_key_valid")
        error_raw = item_raw.get("last_check_error")
        checked_at_raw = item_raw.get("last_checked_at")
        checks[provider] = {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_at_raw if isinstance(checked_at_raw, str) else None,
        }
    return checks


def _save_provider_runtime_checks(
    checks: dict[str, dict[str, JSONValue]],
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> None:
    payload = _load_ui_settings_blob(ui_settings_path=ui_settings_path)
    serialized: dict[str, object] = {}
    for provider in sorted(API_KEY_SETTINGS_PROVIDERS):
        item = checks.get(provider)
        if not isinstance(item, dict):
            continue
        valid_raw = item.get("api_key_valid")
        error_raw = item.get("last_check_error")
        checked_at_raw = item.get("last_checked_at")
        serialized[provider] = {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_at_raw if isinstance(checked_at_raw, str) else None,
        }
    if serialized:
        payload["provider_runtime_checks"] = serialized
    else:
        payload.pop("provider_runtime_checks", None)
    _save_ui_settings_blob(payload, ui_settings_path=ui_settings_path)


def _load_provider_env_api_key(provider: str) -> str | None:
    env_name = PROVIDER_API_KEY_ENV.get(provider)
    if env_name is None:
        return None
    key_raw = os.getenv(env_name, "")
    normalized = key_raw.strip()
    return normalized or None


def _resolve_provider_api_key(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> str | None:
    del settings_api_keys, ui_settings_path
    return _load_provider_env_api_key(provider)


def _provider_api_key_source(
    provider: str,
    *,
    settings_api_keys: dict[str, str] | None = None,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> Literal["env", "missing"]:
    del settings_api_keys, ui_settings_path
    if _load_provider_env_api_key(provider) is not None:
        return "env"
    return "missing"


def _provider_settings_payload(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> list[dict[str, JSONValue]]:
    local_endpoint = (
        os.getenv("LOCAL_LLM_URL", DEFAULT_LOCAL_ENDPOINT).strip() or DEFAULT_LOCAL_ENDPOINT
    )
    runtime_checks = _load_provider_runtime_checks(ui_settings_path=ui_settings_path)

    def _runtime_status(provider: str) -> dict[str, JSONValue]:
        item = runtime_checks.get(provider, {})
        valid_raw = item.get("api_key_valid")
        error_raw = item.get("last_check_error")
        checked_raw = item.get("last_checked_at")
        return {
            "api_key_valid": valid_raw if isinstance(valid_raw, bool) else None,
            "last_check_error": error_raw if isinstance(error_raw, str) else None,
            "last_checked_at": checked_raw if isinstance(checked_raw, str) else None,
        }

    return [
        {
            "provider": "xai",
            "api_key_env": "XAI_API_KEY",
            "api_key_set": _resolve_provider_api_key("xai", ui_settings_path=ui_settings_path)
            is not None,
            "api_key_source": _provider_api_key_source(
                "xai",
                ui_settings_path=ui_settings_path,
            ),
            "endpoint": XAI_MODELS_ENDPOINT,
            **_runtime_status("xai"),
        },
        {
            "provider": "openrouter",
            "api_key_env": "OPENROUTER_API_KEY",
            "api_key_set": _resolve_provider_api_key(
                "openrouter",
                ui_settings_path=ui_settings_path,
            )
            is not None,
            "api_key_source": _provider_api_key_source(
                "openrouter",
                ui_settings_path=ui_settings_path,
            ),
            "endpoint": OPENROUTER_MODELS_ENDPOINT,
            **_runtime_status("openrouter"),
        },
        {
            "provider": "local",
            "api_key_env": "LOCAL_LLM_API_KEY",
            "api_key_set": _resolve_provider_api_key(
                "local",
                ui_settings_path=ui_settings_path,
            )
            is not None,
            "api_key_source": _provider_api_key_source(
                "local",
                ui_settings_path=ui_settings_path,
            ),
            "endpoint": local_endpoint,
            **_runtime_status("local"),
        },
        {
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "api_key_set": _resolve_provider_api_key(
                "openai",
                ui_settings_path=ui_settings_path,
            )
            is not None,
            "api_key_source": _provider_api_key_source(
                "openai",
                ui_settings_path=ui_settings_path,
            ),
            "endpoint": OPENAI_STT_ENDPOINT,
            **_runtime_status("openai"),
        },
    ]


def _build_settings_payload(
    *,
    ui_settings_path: Path = UI_SETTINGS_PATH,
) -> dict[str, JSONValue]:
    tone, system_prompt = _load_personalization_settings(ui_settings_path=ui_settings_path)
    long_paste_to_file_enabled, long_paste_threshold_chars = _load_composer_settings(
        ui_settings_path=ui_settings_path
    )
    policy_profile, yolo_armed, yolo_armed_at = _load_policy_settings(
        ui_settings_path=ui_settings_path
    )
    memory_config = load_memory_config()
    embeddings_settings = _load_embeddings_settings(ui_settings_path=ui_settings_path)
    tools_state = _load_tools_state()
    tools_registry = {key: value for key, value in tools_state.items() if key != "safe_mode"}
    return {
        "settings": {
            "personalization": {"tone": tone, "system_prompt": system_prompt},
            "composer": {
                "long_paste_to_file_enabled": long_paste_to_file_enabled,
                "long_paste_threshold_chars": long_paste_threshold_chars,
            },
            "memory": {
                "auto_save_dialogue": memory_config.auto_save_dialogue,
                "inbox_max_items": memory_config.inbox_max_items,
                "inbox_ttl_days": memory_config.inbox_ttl_days,
                "inbox_writes_per_minute": memory_config.inbox_writes_per_minute,
                "embeddings": {
                    "provider": embeddings_settings.provider,
                    "local_model": embeddings_settings.local_model,
                    "openai_model": embeddings_settings.openai_model,
                },
            },
            "tools": {
                "state": tools_state,
                "registry": tools_registry,
            },
            "policy": {
                "profile": policy_profile,
                "yolo_armed": yolo_armed,
                "yolo_armed_at": yolo_armed_at,
            },
            "providers": _provider_settings_payload(ui_settings_path=ui_settings_path),
        },
    }
