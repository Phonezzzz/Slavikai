from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Generator, Iterator
from contextlib import closing
from typing import Final

import requests

from config.system_prompts import THINKING_PROMPT
from llm.brain_base import Brain
from llm.cancellation import (
    GenerationCancelled,
    cancellation_requested,
    iter_cancellable,
)
from llm.capabilities import CapabilityEvidence, NativeToolCapability, NativeToolQualification
from llm.provider_http import post_plain_completion, post_stream_lines, proxies_for_provider_url
from llm.retry import request_with_retry
from llm.stream_model import (
    Done,
    StreamEvent,
    iter_openai_sse_events,
    stream_events_from_result,
)
from llm.types import LLMResult, ModelConfig, ToolCall, ToolSpec, is_custom_provider_id
from llm.usage import parse_reported_usage
from shared.models import JSONValue, LLMMessage

DEFAULT_LOCAL_ENDPOINT: Final[str] = "http://localhost:11434/v1/chat/completions"
DEFAULT_TIMEOUT: Final[int] = 30


class ChatResponseError(RuntimeError):
    """Успешный HTTP response не соответствует chat completion protocol."""


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
                id=call_id or f"local-tool-call-{index}",
                name=name_raw.strip(),
                arguments=arguments,
                raw_arguments=raw_arguments if isinstance(raw_arguments, str) else None,
            )
        )
    return calls


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


def _message_content_to_text(value: JSONValue) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _parse_chat_result(data_json: object) -> LLMResult:
    if not isinstance(data_json, dict):
        raise ChatResponseError("Некорректный ответ локального LLM.")
    data: dict[str, JSONValue] = data_json
    choices_raw = data.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        raise ChatResponseError("Пустой ответ локального LLM.")
    first_choice = choices_raw[0]
    if not isinstance(first_choice, dict):
        raise ChatResponseError("Некорректный формат choices.")
    message_raw = first_choice.get("message")
    if not isinstance(message_raw, dict):
        raise ChatResponseError("Некорректный формат message.")
    content = _message_content_to_text(message_raw.get("content"))
    tool_calls = _parse_tool_calls(message_raw)
    reasoning_raw = message_raw.get("reasoning")
    reasoning = (
        str(reasoning_raw).strip()
        if isinstance(reasoning_raw, str) and reasoning_raw.strip()
        else None
    )

    usage = parse_reported_usage(data.get("usage"))

    return LLMResult(
        text=content,
        reasoning=reasoning,
        usage=usage,
        raw=data,
        tool_calls=tool_calls,
    )


class LocalHttpBrain(Brain):
    """Клиент для локальных LLM-эндпоинтов совместимых с OpenAI API (Ollama/LM Studio/MSTI)."""

    supports_native_tools = True
    supports_streaming_tools = True

    def __init__(
        self,
        default_config: ModelConfig,
        base_url: str | None = None,
        api_key: str | None = None,
        native_tools: bool = True,
        qualification: NativeToolQualification | None = None,
    ) -> None:
        self.supports_native_tools = native_tools
        self.supports_streaming_tools = native_tools
        self.default_config = default_config
        self._qualification = qualification
        self.base_url = (
            base_url
            or default_config.base_url
            or os.getenv("LOCAL_LLM_URL")
            or DEFAULT_LOCAL_ENDPOINT
        )
        self.api_key = api_key or default_config.api_key
        if self.api_key is None and not is_custom_provider_id(default_config.provider):
            self.api_key = os.getenv("LOCAL_LLM_API_KEY")

    def request_binding(self, config: ModelConfig | None = None) -> str:
        cfg = self._resolve_config(config)
        material = [cfg.provider, cfg.model, self.base_url, cfg.base_url, self._build_headers(cfg)]
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def native_tool_capability(self) -> NativeToolCapability:
        return self.native_tool_capability_for()

    def can_stream_native_tools(self) -> bool:
        return self.native_tool_capability().allowed

    def native_tool_capability_for(self, config: ModelConfig | None = None) -> NativeToolCapability:
        if self._qualification is not None:
            if self._qualification.valid_for(self.request_binding(config)):
                return NativeToolCapability(
                    adapter=CapabilityEvidence("supported", "adapter_declaration"),
                    model=CapabilityEvidence(self._qualification.state, "inert_protocol_v1"),
                    eligibility=(
                        "qualified" if self._qualification.state == "supported" else "unsupported"
                    ),
                )
        elif self.supports_native_tools:
            return super().native_tool_capability()
        return NativeToolCapability(
            adapter=CapabilityEvidence(state="supported", source="adapter_declaration"),
            model=CapabilityEvidence(state="unknown", source="not_observed"),
            eligibility="unqualified",
        )

    def _resolve_config(self, override: ModelConfig | None) -> ModelConfig:
        if override:
            return override
        return self.default_config

    def _build_headers(self, config: ModelConfig) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(config.extra_headers)
        return headers

    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        if tools and not self.native_tool_capability_for(config).allowed:
            raise RuntimeError("native_tools_required")
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload = self._build_chat_payload(cfg, messages, tools)
        # Never send the API key through a system proxy for loopback providers.
        proxies = proxies_for_provider_url(self.base_url)

        def send_request() -> requests.Response:
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=DEFAULT_TIMEOUT,
                allow_redirects=False,
                proxies=proxies,
            )
            if 300 <= response.status_code < 400:
                response.close()
                raise requests.HTTPError("Provider redirect rejected", response=response)
            return response

        response = request_with_retry(send_request, provider=cfg.provider)
        return _parse_chat_result(response.json())

    def _build_chat_payload(
        self,
        cfg: ModelConfig,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None,
    ) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "model": cfg.model,
            "messages": [
                message.to_provider_dict() for message in self._inject_system(messages, cfg)
            ],
            "temperature": cfg.temperature,
        }
        if tools:
            payload["tools"] = [_tool_spec_to_provider_dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        return payload

    def generate_stream_events(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> Generator[StreamEvent, None, None]:
        if tools and not self.native_tool_capability_for(config).allowed:
            raise RuntimeError("native_tools_required")
        if not self.native_tool_capability_for(config).allowed:
            # Unqualified dynamic endpoint: SSE support is not verified, so
            # never claim native streaming. Use a plain (non-SSE) chat
            # completion, cancellable through the standard token, and convert
            # it into stream events.
            yield from self._stream_events_from_plain_completion(
                messages,
                config=config,
                cancellation_token=cancellation_token,
            )
            return
        if cancellation_requested(cancellation_token):
            yield from iter_cancellable((), cancellation_token=cancellation_token)
            return
        cfg = self._resolve_config(config)
        headers = self._build_headers(cfg)
        payload: dict[str, JSONValue] = {
            "model": cfg.model,
            "messages": [
                message.to_provider_dict() for message in self._inject_system(messages, cfg)
            ],
            "temperature": cfg.temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = [_tool_spec_to_provider_dict(tool) for tool in tools]
            payload["tool_choice"] = "auto"
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p

        try:
            with closing(
                post_stream_lines(
                    url=self.base_url,
                    payload=payload,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT,
                    cancellation_token=cancellation_token,
                )
            ) as lines:
                yield from iter_cancellable(
                    iter_openai_sse_events(lines), cancellation_token=cancellation_token
                )
        except GenerationCancelled:
            yield Done(finish_reason="cancelled")

    def _stream_events_from_plain_completion(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None,
        cancellation_token: asyncio.Event | None,
    ) -> Iterator[StreamEvent]:
        """Stream events for dynamic endpoints without verified SSE support.

        Runs a plain (non-streaming) chat completion through the standard
        cancellation token and converts the result into stream events.
        """
        if cancellation_requested(cancellation_token):
            yield Done(finish_reason="cancelled")
            return
        try:
            result = self.generate_cancellable(
                messages, config=config, cancellation_token=cancellation_token
            )
        except GenerationCancelled:
            yield Done(finish_reason="cancelled")
            return
        if cancellation_requested(cancellation_token):
            yield Done(finish_reason="cancelled")
            return
        for event in stream_events_from_result(result):
            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return
            yield event

    def generate_cancellable(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None,
        cancellation_token: asyncio.Event | None,
        tools: list[ToolSpec] | None = None,
        max_attempts: int = 3,
    ) -> LLMResult:
        if tools and not self.native_tool_capability_for(config).allowed:
            raise RuntimeError("native_tools_required")
        cfg = self._resolve_config(config)
        if cancellation_token is None:
            return self.generate(messages, config=config, tools=tools)
        response = post_plain_completion(
            url=self.base_url,
            headers=self._build_headers(cfg),
            payload=self._build_chat_payload(cfg, messages, tools),
            provider=cfg.provider,
            timeout=DEFAULT_TIMEOUT,
            cancellation_token=cancellation_token,
            max_attempts=max_attempts,
        )
        return _parse_chat_result(response.json())

    def _inject_system(self, messages: list[LLMMessage], config: ModelConfig) -> list[LLMMessage]:
        system_messages: list[LLMMessage] = []
        if config.thinking_enabled:
            system_messages.append(LLMMessage(role="system", content=THINKING_PROMPT))
        if config.system_prompt:
            system_messages.append(LLMMessage(role="system", content=config.system_prompt))
        if not system_messages:
            return messages
        return [*system_messages, *messages]
