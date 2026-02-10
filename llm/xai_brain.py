from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Final

import requests

from config.system_prompts import THINKING_PROMPT
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig
from shared.models import JSONValue, LLMMessage

XAI_ENDPOINT: Final[str] = "https://api.x.ai/v1/chat/completions"
DEFAULT_TIMEOUT: Final[int] = 30


def _extract_stream_delta(data: dict[str, JSONValue]) -> str:
    choices_raw = data.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        return ""
    first_choice = choices_raw[0]
    if not isinstance(first_choice, dict):
        return ""
    delta_raw = first_choice.get("delta")
    if not isinstance(delta_raw, dict):
        return ""
    content_raw = delta_raw.get("content")
    if isinstance(content_raw, str):
        return content_raw
    if isinstance(content_raw, list):
        parts: list[str] = []
        for item in content_raw:
            if not isinstance(item, dict):
                continue
            text_raw = item.get("text")
            if isinstance(text_raw, str):
                parts.append(text_raw)
        return "".join(parts)
    return ""


class XAiBrain(Brain):
    """Клиент xAI, совместимый с интерфейсом Brain."""

    def __init__(self, api_key: str | None, default_config: ModelConfig) -> None:
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.default_config = default_config
        self.base_url = default_config.base_url or os.getenv("XAI_API_URL") or XAI_ENDPOINT

    def _resolve_config(self, override: ModelConfig | None) -> ModelConfig:
        if override:
            return override
        return self.default_config

    def _build_headers(self, config: ModelConfig) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Не задан xAI API key (env XAI_API_KEY).")
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
            self.base_url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise RuntimeError("Некорректный ответ xAI.")
        data: dict[str, JSONValue] = data_json
        choices_raw = data.get("choices")
        if not isinstance(choices_raw, list) or not choices_raw:
            raise RuntimeError("Пустой или некорректный ответ xAI.")
        first_choice = choices_raw[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Некорректный формат choices.")
        message_raw = first_choice.get("message")
        if not isinstance(message_raw, dict):
            raise RuntimeError("Некорректный формат message.")
        content = str(message_raw.get("content", ""))
        reasoning_raw = message_raw.get("reasoning")
        reasoning = (
            str(reasoning_raw).strip()
            if isinstance(reasoning_raw, str) and reasoning_raw.strip()
            else None
        )

        usage: LLMUsage | None = None
        usage_block = data.get("usage")
        if isinstance(usage_block, dict):
            usage = LLMUsage(
                prompt_tokens=int(usage_block.get("prompt_tokens", 0)),
                completion_tokens=int(usage_block.get("completion_tokens", 0)),
                total_tokens=int(usage_block.get("total_tokens", 0)),
            )

        return LLMResult(text=content, reasoning=reasoning, usage=usage, raw=data)

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> Iterator[str]:
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload = {
            "model": cfg.model,
            "messages": [message.__dict__ for message in self._inject_system(messages, cfg)],
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p

        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        response.encoding = "utf-8"

        for raw_line in response.iter_lines(decode_unicode=True):
            if raw_line is None:
                continue
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_part = line.removeprefix("data:").strip()
            if not data_part or data_part == "[DONE]":
                continue
            try:
                parsed = json.loads(data_part)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            delta = _extract_stream_delta(parsed)
            if delta:
                yield delta

    def _inject_system(self, messages: list[LLMMessage], config: ModelConfig) -> list[LLMMessage]:
        system_messages: list[LLMMessage] = []
        if config.thinking_enabled:
            system_messages.append(LLMMessage(role="system", content=THINKING_PROMPT))
        if config.system_prompt:
            system_messages.append(LLMMessage(role="system", content=config.system_prompt))
        if not system_messages:
            return messages
        return [*system_messages, *messages]
