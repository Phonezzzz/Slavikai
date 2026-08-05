from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import ClassVar

from llm.cancellation import cancellation_requested
from llm.stream_model import Done, StreamEvent, stream_events_from_result
from llm.types import LLMResult, ModelConfig, ToolSpec
from shared.models import LLMMessage


class Brain(ABC):
    """Абстракция для всех моделей (OpenRouter, xAI, Local)."""

    supports_native_tools: ClassVar[bool] = False
    supports_streaming_tools: ClassVar[bool] = False

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        """Сгенерировать ответ на основе списка сообщений."""
        raise NotImplementedError

    def generate_stream_events(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Типизированный fallback для провайдеров без нативного стриминга."""
        if cancellation_requested(cancellation_token):
            yield Done(finish_reason="cancelled")
            return
        if tools is None:
            result = self.generate(messages, config=config)
        else:
            result = self.generate(messages, config=config, tools=tools)
        if cancellation_requested(cancellation_token):
            yield Done(finish_reason="cancelled")
            return
        for event in stream_events_from_result(result):
            if cancellation_requested(cancellation_token):
                yield Done(finish_reason="cancelled")
                return
            yield event

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> Iterator[StreamEvent]:
        """Каноническая точка входа в потоковую генерацию."""
        yield from self.generate_stream_events(
            messages,
            config=config,
            tools=tools,
            cancellation_token=cancellation_token,
        )
