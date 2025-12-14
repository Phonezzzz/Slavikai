from __future__ import annotations

import os
from typing import Final

import requests

from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig
from shared.models import JSONValue, LLMMessage

OPENROUTER_ENDPOINT: Final[str] = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT: Final[int] = 30


class OpenRouterBrain(Brain):
    """Клиент OpenRouter, совместимый с интерфейсом Brain."""

    def __init__(self, api_key: str | None, default_config: ModelConfig) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.default_config = default_config

    def _resolve_config(self, override: ModelConfig | None) -> ModelConfig:
        if override:
            return override
        return self.default_config

    def _build_headers(self, config: ModelConfig) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Не задан OpenRouter API key (env OPENROUTER_API_KEY).")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(config.extra_headers)
        return headers

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload = {
            "model": cfg.model,
            "messages": [message.__dict__ for message in self._inject_system(messages, cfg)],
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p

        response = requests.post(
            OPENROUTER_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise RuntimeError("Некорректный ответ OpenRouter.")
        data: dict[str, JSONValue] = data_json
        choices_raw = data.get("choices")
        if not isinstance(choices_raw, list) or not choices_raw:
            raise RuntimeError("Пустой или некорректный ответ OpenRouter.")
        first_choice = choices_raw[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Некорректный формат choices.")
        message_raw = first_choice.get("message")
        if not isinstance(message_raw, dict):
            raise RuntimeError("Некорректный формат message.")
        content = str(message_raw.get("content", ""))

        usage: LLMUsage | None = None
        usage_block = data.get("usage")
        if isinstance(usage_block, dict):
            usage = LLMUsage(
                prompt_tokens=int(usage_block.get("prompt_tokens", 0)),
                completion_tokens=int(usage_block.get("completion_tokens", 0)),
                total_tokens=int(usage_block.get("total_tokens", 0)),
            )

        return LLMResult(text=content, usage=usage, raw=data)

    def _inject_system(self, messages: list[LLMMessage], config: ModelConfig) -> list[LLMMessage]:
        if config.system_prompt and (not messages or messages[0].role != "system"):
            return [LLMMessage(role="system", content=config.system_prompt), *messages]
        return messages
