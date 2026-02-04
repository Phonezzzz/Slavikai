from __future__ import annotations

from llm.brain_base import Brain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.types import ModelConfig
from llm.xai_brain import XAiBrain


def create_brain(config: ModelConfig, api_key: str | None = None) -> Brain:
    if config.provider == "openrouter":
        return OpenRouterBrain(api_key=api_key or config.api_key, default_config=config)
    if config.provider == "xai":
        return XAiBrain(api_key=api_key or config.api_key, default_config=config)
    if config.provider == "local":
        return LocalHttpBrain(
            default_config=config,
            base_url=config.base_url,
            api_key=api_key or config.api_key,
        )
    raise ValueError(f"Неизвестный провайдер модели: {config.provider}")
