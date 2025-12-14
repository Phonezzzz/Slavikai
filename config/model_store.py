from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from llm.types import ModelConfig

MODEL_CONFIG_PATH = Path("config/model_config.json")


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    data = asdict(config)
    return {k: v for k, v in data.items() if v is not None}


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    return ModelConfig(
        provider=data["provider"],
        model=data["model"],
        temperature=float(data.get("temperature", 0.7)),
        top_p=data.get("top_p"),
        max_tokens=data.get("max_tokens"),
        base_url=data.get("base_url"),
        api_key=data.get("api_key"),
        extra_headers=data.get("extra_headers", {}),
        system_prompt=data.get("system_prompt"),
        mode=data.get("mode", "default"),
    )


def load_model_configs(
    path: Path = MODEL_CONFIG_PATH,
) -> tuple[ModelConfig | None, ModelConfig | None]:
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    main_cfg = model_config_from_dict(data["main"]) if "main" in data else None
    critic_cfg = model_config_from_dict(data["critic"]) if "critic" in data else None
    return main_cfg, critic_cfg


def save_model_configs(
    main: ModelConfig | None,
    critic: ModelConfig | None = None,
    path: Path = MODEL_CONFIG_PATH,
) -> None:
    payload: dict[str, Any] = {}
    if main:
        payload["main"] = model_config_to_dict(main)
    if critic:
        payload["critic"] = model_config_to_dict(critic)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cli_example() -> None:  # pragma: no cover
    """Пример использования из CLI."""
    sample = {
        "main": {
            "provider": "openrouter",
            "model": "gpt-4o-mini",
            "temperature": 0.4,
        }
    }
    save_model_configs(model_config_from_dict(sample["main"]))
