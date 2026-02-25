from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Final

import requests

from config.system_prompts import THINKING_PROMPT
from llm.brain_base import Brain
from llm.types import LLMResult, LLMStreamChunk, LLMStreamChunkMode, LLMUsage, ModelConfig
from shared.models import JSONValue, LLMMessage

DEFAULT_API_BASE: Final[str] = "https://api.inceptionlabs.ai/v1"
DEFAULT_TIMEOUT: Final[int] = 30


def _build_completions_endpoint(base_url: str) -> str:
    normalized = base_url.strip()
    if not normalized:
        normalized = DEFAULT_API_BASE
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized.rstrip('/')}/chat/completions"


def _extract_stream_delta(data: dict[str, JSONValue]) -> str:
    choices_raw = data.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        return ""
    for choice in choices_raw:
        if not isinstance(choice, dict):
            continue
        delta_raw = choice.get("delta")
        if not isinstance(delta_raw, dict):
            continue
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
            if parts:
                return "".join(parts)
    return ""


class InceptionBrain(Brain):
    """Клиент Inception (OpenAI-compatible chat completions)."""

    def __init__(self, api_key: str | None, default_config: ModelConfig) -> None:
        self.api_key = api_key or os.getenv("INCEPTION_API_KEY")
        self.default_config = default_config
        configured_base = (
            default_config.base_url or os.getenv("INCEPTION_API_URL") or DEFAULT_API_BASE
        )
        self.base_url = _build_completions_endpoint(configured_base)

    def _resolve_config(self, override: ModelConfig | None) -> ModelConfig:
        if override:
            return override
        return self.default_config

    def _build_headers(self, config: ModelConfig) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Не задан Inception API key (env INCEPTION_API_KEY).")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        headers.update(config.extra_headers)
        return headers

    def _build_reasoning_payload(self, config: ModelConfig) -> dict[str, JSONValue]:
        effort = config.reasoning_effort or "instant"
        summary = True if config.reasoning_summary is None else bool(config.reasoning_summary)
        summary_wait = (
            False if config.reasoning_summary_wait is None else bool(config.reasoning_summary_wait)
        )
        return {
            "reasoning_effort": effort,
            "reasoning_summary": summary,
            "reasoning_summary_wait": summary_wait,
        }

    def _build_payload(
        self,
        messages: list[LLMMessage],
        config: ModelConfig,
        *,
        stream: bool,
    ) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "model": config.model,
            "messages": [message.__dict__ for message in self._inject_system(messages, config)],
            "temperature": config.temperature,
            "stream": stream,
            **self._build_reasoning_payload(config),
        }
        if config.max_tokens is not None:
            payload["max_tokens"] = config.max_tokens
        if config.top_p is not None:
            payload["top_p"] = config.top_p
        if config.diffusing is not None:
            payload["diffusing"] = bool(config.diffusing)
        return payload

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload = self._build_payload(messages, cfg, stream=False)
        response = requests.post(
            self.base_url,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise RuntimeError("Некорректный ответ Inception.")
        data: dict[str, JSONValue] = data_json
        choices_raw = data.get("choices")
        if not isinstance(choices_raw, list) or not choices_raw:
            raise RuntimeError("Пустой или некорректный ответ Inception.")
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

    def generate_stream_chunks(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> Iterator[LLMStreamChunk]:
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload = self._build_payload(messages, cfg, stream=True)
        stream_mode: LLMStreamChunkMode = "replace" if cfg.diffusing else "append"
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
            if not delta:
                continue
            yield LLMStreamChunk(text=delta, mode=stream_mode)

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> Iterator[str]:
        for chunk in self.generate_stream_chunks(messages, config=config):
            if chunk.text:
                yield chunk.text

    def _inject_system(self, messages: list[LLMMessage], config: ModelConfig) -> list[LLMMessage]:
        system_messages: list[LLMMessage] = []
        if config.thinking_enabled:
            system_messages.append(LLMMessage(role="system", content=THINKING_PROMPT))
        if config.system_prompt:
            system_messages.append(LLMMessage(role="system", content=config.system_prompt))
        if not system_messages:
            return messages
        return [*system_messages, *messages]
