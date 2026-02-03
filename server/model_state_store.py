from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, cast

from llm.types import ModelConfig

_DEFAULT_MODEL_CONFIG_PATH = Path("config/model_config.json")


def normalize_main_config(cfg: ModelConfig) -> ModelConfig:
    return ModelConfig(
        provider=cast(Literal["openrouter", "local"], "xai"),
        model=cfg.model,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        extra_headers=dict(cfg.extra_headers),
        system_prompt=cfg.system_prompt,
        mode=cfg.mode,
        thinking_enabled=cfg.thinking_enabled,
    )


def _config_from_dict(data: dict[str, object]) -> ModelConfig | None:
    model_raw = data.get("model")
    if not isinstance(model_raw, str) or not model_raw.strip():
        return None
    provider_raw = data.get("provider")
    provider = (
        cast(Literal["openrouter", "local"], provider_raw)
        if isinstance(provider_raw, str) and provider_raw in {"openrouter", "local"}
        else cast(Literal["openrouter", "local"], "xai")
    )
    temperature_raw = data.get("temperature", 0.7)
    temperature = float(temperature_raw) if isinstance(temperature_raw, (int, float)) else 0.7
    top_p_raw = data.get("top_p")
    max_tokens_raw = data.get("max_tokens")
    base_url_raw = data.get("base_url")
    api_key_raw = data.get("api_key")
    extra_headers_raw = data.get("extra_headers")
    system_prompt_raw = data.get("system_prompt")
    return normalize_main_config(
        ModelConfig(
            provider=provider,
            model=model_raw.strip(),
            temperature=temperature,
            top_p=float(top_p_raw) if isinstance(top_p_raw, (int, float)) else None,
            max_tokens=int(max_tokens_raw) if isinstance(max_tokens_raw, int) else None,
            base_url=base_url_raw if isinstance(base_url_raw, str) else None,
            api_key=api_key_raw if isinstance(api_key_raw, str) else None,
            extra_headers=(
                {
                    str(key): str(value)
                    for key, value in extra_headers_raw.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
                if isinstance(extra_headers_raw, dict)
                else {}
            ),
            system_prompt=system_prompt_raw if isinstance(system_prompt_raw, str) else None,
            mode="planner" if data.get("mode") == "planner" else "default",
            thinking_enabled=bool(data.get("thinking_enabled", False)),
        ),
    )


def _config_to_dict(config: ModelConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "provider": config.provider,
        "model": config.model,
        "temperature": config.temperature,
        "mode": config.mode,
        "thinking_enabled": config.thinking_enabled,
    }
    if config.top_p is not None:
        payload["top_p"] = config.top_p
    if config.max_tokens is not None:
        payload["max_tokens"] = config.max_tokens
    if config.base_url is not None:
        payload["base_url"] = config.base_url
    if config.api_key is not None:
        payload["api_key"] = config.api_key
    if config.extra_headers:
        payload["extra_headers"] = dict(config.extra_headers)
    if config.system_prompt is not None:
        payload["system_prompt"] = config.system_prompt
    return payload


class ModelStateStore:
    def __init__(self, main: ModelConfig | None = None) -> None:
        self._main = normalize_main_config(main) if main is not None else None
        self._lock = asyncio.Lock()

    async def get_main(self) -> ModelConfig | None:
        async with self._lock:
            return self._main

    async def set_main(self, main: ModelConfig | None) -> None:
        async with self._lock:
            self._main = normalize_main_config(main) if main is not None else None


class FileBackedModelStateStore(ModelStateStore):
    def __init__(self, path: Path = _DEFAULT_MODEL_CONFIG_PATH) -> None:
        self._path = path
        super().__init__(self._load_main(path))

    async def set_main(self, main: ModelConfig | None) -> None:
        async with self._lock:
            self._main = normalize_main_config(main) if main is not None else None
            self._save_main(self._main, self._path)

    def _load_main(self, path: Path) -> ModelConfig | None:
        if not path.exists():
            return None
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(parsed, dict):
            return None
        main_raw = parsed.get("main")
        if not isinstance(main_raw, dict):
            return None
        return _config_from_dict(main_raw)

    def _save_main(self, main: ModelConfig | None, path: Path) -> None:
        payload: dict[str, object] = {}
        if main is not None:
            payload["main"] = _config_to_dict(main)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
