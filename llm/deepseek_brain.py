from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Final

import requests

from llm.brain_base import Brain
from llm.stream_model import StreamEvent, iter_openai_sse_events
from llm.types import LLMResult, LLMUsage, ModelConfig, ToolCall, ToolSpec
from shared.models import JSONValue, LLMMessage

DEEPSEEK_ENDPOINT: Final[str] = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODELS_ENDPOINT: Final[str] = "https://api.deepseek.com/models"
DEFAULT_TIMEOUT: Final[int] = 30


def _parse_tool_arguments(value: JSONValue) -> dict[str, JSONValue]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): item for key, item in parsed.items()}


def _parse_tool_calls(message: dict[str, JSONValue]) -> list[ToolCall]:
    calls_raw = message.get("tool_calls")
    if not isinstance(calls_raw, list):
        return []
    calls: list[ToolCall] = []
    for index, item in enumerate(calls_raw):
        if not isinstance(item, dict):
            continue
        function_raw = item.get("function")
        if not isinstance(function_raw, dict):
            continue
        name_raw = function_raw.get("name")
        if not isinstance(name_raw, str) or not name_raw.strip():
            continue
        raw_arguments = function_raw.get("arguments")
        arguments = _parse_tool_arguments(raw_arguments)
        call_id_raw = item.get("id")
        call_id = (
            call_id_raw.strip() if isinstance(call_id_raw, str) and call_id_raw.strip() else ""
        )
        calls.append(
            ToolCall(
                id=call_id or f"deepseek-tool-call-{index}",
                name=name_raw.strip(),
                arguments=arguments,
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
            )
        )
    return calls


def _tool_spec_to_provider_dict(tool: ToolSpec) -> dict[str, JSONValue]:
    parameters: dict[str, JSONValue] = tool.parameters_schema or {
        "type": "object",
        "properties": {},
    }
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _message_content_to_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


class DeepSeekBrain(Brain):
    """Прямой клиент DeepSeek API (OpenAI-compatible)."""

    supports_native_tools = True
    supports_streaming_tools = True

    def __init__(
        self,
        api_key: str | None = None,
        default_config: ModelConfig | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.default_config = default_config

    def _resolve_config(self, override: ModelConfig | None) -> ModelConfig:
        if override:
            return override
        if self.default_config:
            return self.default_config
        raise RuntimeError(
            "DeepSeekBrain: нужен ModelConfig (передайте или задайте default_config)."
        )

    def _build_headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY не задан.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _inject_system(self, messages: list[LLMMessage], config: ModelConfig) -> list[LLMMessage]:
        system_messages: list[LLMMessage] = []
        if config.system_prompt:
            system_messages.append(LLMMessage(role="system", content=config.system_prompt))
        if not system_messages:
            return messages
        return [*system_messages, *messages]

    def _build_thinking(self, config: ModelConfig) -> dict[str, JSONValue] | None:
        if config.thinking_enabled:
            return {"type": "enabled"}
        return {"type": "disabled"}

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        cfg = self._resolve_config(config)
        headers = self._build_headers()
        payload: dict[str, JSONValue] = {
            "model": cfg.model,
            "messages": [
                message.to_provider_dict() for message in self._inject_system(messages, cfg)
            ],
            "temperature": cfg.temperature,
        }
        thinking = self._build_thinking(cfg)
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = [_tool_spec_to_provider_dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p

        response = requests.post(
            DEEPSEEK_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise RuntimeError("Некорректный ответ DeepSeek API.")
        data: dict[str, JSONValue] = data_json
        choices_raw = data.get("choices")
        if not isinstance(choices_raw, list) or not choices_raw:
            raise RuntimeError("Пустой ответ DeepSeek API.")
        first_choice = choices_raw[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("Некорректный формат choices.")
        message_raw = first_choice.get("message")
        if not isinstance(message_raw, dict):
            raise RuntimeError("Некорректный формат message.")
        content = _message_content_to_text(message_raw.get("content"))
        tool_calls = _parse_tool_calls(message_raw)
        reasoning_raw = message_raw.get("reasoning_content")
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

        return LLMResult(
            text=content,
            reasoning=reasoning,
            usage=usage,
            raw=data,
            tool_calls=tool_calls,
        )

    def generate_stream_events(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> Iterator[StreamEvent]:
        cfg = self._resolve_config(config)
        headers = self._build_headers()
        payload: dict[str, JSONValue] = {
            "model": cfg.model,
            "messages": [
                message.to_provider_dict() for message in self._inject_system(messages, cfg)
            ],
            "temperature": cfg.temperature,
            "stream": True,
        }
        thinking = self._build_thinking(cfg)
        if thinking is not None:
            payload["thinking"] = thinking
        if tools:
            payload["tools"] = [_tool_spec_to_provider_dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p

        response = requests.post(
            DEEPSEEK_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        yield from iter_openai_sse_events(response.iter_lines(decode_unicode=True))
