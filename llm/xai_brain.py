from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from typing import Final

import requests

from config.system_prompts import THINKING_PROMPT
from llm.brain_base import Brain
from llm.types import LLMResult, LLMUsage, ModelConfig, WebSearchEvidence
from shared.models import JSONValue, LLMMessage

XAI_ENDPOINT: Final[str] = "https://api.x.ai/v1/chat/completions"
XAI_RESPONSES_ENDPOINT: Final[str] = "https://api.x.ai/v1/responses"
DEFAULT_TIMEOUT: Final[int] = 30
logger = logging.getLogger("SlavikAI.XAiBrain")


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


def _responses_endpoint_for(chat_endpoint: str) -> str:
    normalized = chat_endpoint.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return f"{normalized.removesuffix('/chat/completions')}/responses"
    if normalized.endswith("/v1"):
        return f"{normalized}/responses"
    if normalized == XAI_ENDPOINT.rstrip("/"):
        return XAI_RESPONSES_ENDPOINT
    return f"{normalized}/responses"


def _responses_input(messages: list[LLMMessage]) -> list[dict[str, JSONValue]]:
    return [{"role": message.role, "content": message.content} for message in messages]


def _extract_responses_text(data: dict[str, JSONValue]) -> str:
    output_text_raw = data.get("output_text")
    if isinstance(output_text_raw, str) and output_text_raw:
        return output_text_raw

    parts: list[str] = []
    output_raw = data.get("output")
    if isinstance(output_raw, list):
        for item in output_raw:
            if not isinstance(item, dict):
                continue
            item_output_text = item.get("output_text")
            if isinstance(item_output_text, str):
                parts.append(item_output_text)
            content_raw = item.get("content")
            if isinstance(content_raw, list):
                for block in content_raw:
                    if isinstance(block, str):
                        parts.append(block)
                        continue
                    if not isinstance(block, dict):
                        continue
                    text_raw = block.get("text")
                    if isinstance(text_raw, str):
                        parts.append(text_raw)
    return "".join(parts)


def _extract_usage(data: dict[str, JSONValue]) -> LLMUsage | None:
    usage_block = data.get("usage")
    if not isinstance(usage_block, dict):
        return None
    prompt_tokens_raw = usage_block.get("prompt_tokens", usage_block.get("input_tokens", 0))
    completion_tokens_raw = usage_block.get(
        "completion_tokens",
        usage_block.get("output_tokens", 0),
    )
    total_tokens_raw = usage_block.get("total_tokens", 0)
    return LLMUsage(
        prompt_tokens=int(prompt_tokens_raw or 0),
        completion_tokens=int(completion_tokens_raw or 0),
        total_tokens=int(total_tokens_raw or 0),
    )


def _extract_citations(data: dict[str, JSONValue]) -> list[JSONValue] | None:
    citations_raw = data.get("citations")
    if not isinstance(citations_raw, list):
        return None
    return list(citations_raw)


def _append_xai_citation_candidate(
    citations: list[JSONValue],
    candidate: JSONValue,
) -> None:
    if isinstance(candidate, str) and candidate.strip():
        citations.append(candidate)
        return
    if not isinstance(candidate, dict):
        return
    direct_url = candidate.get("url")
    if isinstance(direct_url, str) and direct_url.strip():
        citations.append(candidate)
        return
    for key in ("web_citation", "x_citation", "source"):
        nested = candidate.get(key)
        if not isinstance(nested, dict):
            continue
        nested_url = nested.get("url")
        if isinstance(nested_url, str) and nested_url.strip():
            citations.append(nested)
            return


def _extract_xai_citation_urls(data: dict[str, JSONValue]) -> list[JSONValue]:
    citations: list[JSONValue] = []
    top_level_citations = data.get("citations")
    if isinstance(top_level_citations, list):
        for citation in top_level_citations:
            _append_xai_citation_candidate(citations, citation)

    output_raw = data.get("output")
    if not isinstance(output_raw, list):
        return citations
    for item in output_raw:
        if not isinstance(item, dict):
            continue
        action_raw = item.get("action")
        if isinstance(action_raw, dict):
            sources_raw = action_raw.get("sources")
            if isinstance(sources_raw, list):
                for source in sources_raw:
                    _append_xai_citation_candidate(citations, source)
        content_raw = item.get("content")
        if not isinstance(content_raw, list):
            continue
        for block in content_raw:
            if not isinstance(block, dict):
                continue
            annotations_raw = block.get("annotations")
            if not isinstance(annotations_raw, list):
                continue
            for annotation in annotations_raw:
                _append_xai_citation_candidate(citations, annotation)
    return citations


def _xai_tool_name_matches(value: JSONValue) -> bool:
    return isinstance(value, str) and "web_search" in value.lower()


def _xai_server_side_usage_seen(data: dict[str, JSONValue]) -> bool:
    usage_raw = data.get("server_side_tool_usage")
    if isinstance(usage_raw, dict):
        for key, value in usage_raw.items():
            if "web_search" not in key.lower():
                continue
            if isinstance(value, bool):
                return value
            if isinstance(value, int | float):
                return value > 0
            if isinstance(value, list):
                return len(value) > 0
            if isinstance(value, dict):
                return len(value) > 0
    if isinstance(usage_raw, list):
        for item in usage_raw:
            if isinstance(item, str) and "web_search" in item.lower():
                return True
            if not isinstance(item, dict):
                continue
            if _xai_tool_name_matches(item.get("type")) or _xai_tool_name_matches(item.get("name")):
                return True
    return False


def _xai_tool_calls_seen(data: dict[str, JSONValue]) -> bool:
    tool_calls_raw = data.get("tool_calls")
    if not isinstance(tool_calls_raw, list):
        return False
    for item in tool_calls_raw:
        if not isinstance(item, dict):
            continue
        if _xai_tool_name_matches(item.get("type")) or _xai_tool_name_matches(item.get("name")):
            return True
        function_raw = item.get("function")
        if isinstance(function_raw, dict) and _xai_tool_name_matches(function_raw.get("name")):
            return True
    return False


def _xai_web_search_tool_call_seen(data: dict[str, JSONValue]) -> bool:
    output_raw = data.get("output")
    if isinstance(output_raw, list):
        for item in output_raw:
            if not isinstance(item, dict):
                continue
            if _xai_tool_name_matches(item.get("type")) or _xai_tool_name_matches(item.get("name")):
                return True
    if _xai_tool_calls_seen(data):
        return True
    return _xai_server_side_usage_seen(data)


def _xai_output_item_types(data: dict[str, JSONValue]) -> list[str]:
    output_raw = data.get("output")
    if not isinstance(output_raw, list):
        return []
    item_types: list[str] = []
    for item in output_raw:
        if not isinstance(item, dict):
            continue
        type_raw = item.get("type")
        if isinstance(type_raw, str):
            item_types.append(type_raw)
    return item_types


def _has_server_side_tool_usage(data: dict[str, JSONValue]) -> bool:
    usage_raw = data.get("server_side_tool_usage")
    if isinstance(usage_raw, dict | list):
        return len(usage_raw) > 0
    return usage_raw is not None


def _has_top_level_citations(data: dict[str, JSONValue]) -> bool:
    citations_raw = data.get("citations")
    if isinstance(citations_raw, list):
        return len(citations_raw) > 0
    return citations_raw is not None


def _has_xai_web_search_evidence(
    *,
    tool_call_seen: bool,
    citations_count: int,
) -> bool:
    return tool_call_seen or citations_count > 0


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
        if cfg.web_search_enabled:
            return self._generate_with_responses_web_search(messages, cfg, headers)

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

        usage = _extract_usage(data)

        return LLMResult(text=content, reasoning=reasoning, usage=usage, raw=data)

    def _generate_with_responses_web_search(
        self,
        messages: list[LLMMessage],
        cfg: ModelConfig,
        headers: dict[str, str],
    ) -> LLMResult:
        payload: dict[str, JSONValue] = {
            "model": cfg.model,
            "input": _responses_input(self._inject_system(messages, cfg)),
            "tools": [{"type": "web_search"}],
            "include": ["web_search_call.action.sources"],
        }
        endpoint = _responses_endpoint_for(self.base_url)
        logger.debug(
            "xai_web_search_request",
            extra={
                "provider": cfg.provider,
                "model": cfg.model,
                "web_required": True,
                "web_mode": "xai_native",
                "endpoint": endpoint,
                "tools_sent": ["web_search"],
            },
        )
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data_json = response.json()
        if not isinstance(data_json, dict):
            raise RuntimeError("Некорректный ответ xAI Responses API.")
        data: dict[str, JSONValue] = data_json
        content = _extract_responses_text(data)
        if not content:
            raise RuntimeError("Пустой или некорректный ответ xAI Responses API.")
        citations = _extract_citations(data)
        evidence_citations = _extract_xai_citation_urls(data)
        citations_count = len(evidence_citations)
        tool_call_seen = _xai_web_search_tool_call_seen(data)
        evidence_seen = _has_xai_web_search_evidence(
            tool_call_seen=tool_call_seen,
            citations_count=citations_count,
        )
        evidence = WebSearchEvidence(
            requested=True,
            executed=evidence_seen,
            provider="xai_native",
            tool_call_seen=tool_call_seen,
            citations_count=citations_count,
            local_result_seen=False,
            error=None if evidence_seen else "xAI response contained no web search evidence",
        )
        logger.debug(
            "xai_web_search_response",
            extra={
                "provider": cfg.provider,
                "model": cfg.model,
                "web_required": True,
                "web_mode": "xai_native",
                "endpoint": endpoint,
                "tools_sent": ["web_search"],
                "tool_call_seen": evidence.tool_call_seen,
                "citations_count": evidence.citations_count,
                "local_result_seen": evidence.local_result_seen,
                "output_item_types": _xai_output_item_types(data),
                "has_top_level_citations": _has_top_level_citations(data),
                "has_server_side_tool_usage": _has_server_side_tool_usage(data),
                "extracted_source_count": citations_count,
                "error": evidence.error,
            },
        )
        return LLMResult(
            text=content,
            usage=_extract_usage(data),
            raw=data,
            citations=citations,
            web_search_evidence=evidence,
        )

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> Iterator[str]:
        cfg = self._resolve_config(config)
        if cfg.web_search_enabled:
            result = self.generate(messages, config=cfg)
            chunk_size = 80
            for idx in range(0, len(result.text), chunk_size):
                yield result.text[idx : idx + chunk_size]
            return

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
