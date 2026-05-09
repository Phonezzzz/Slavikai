from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from llm.types import LLMResult, LLMStreamChunk, ModelConfig, ToolSpec
from shared.models import LLMMessage


class Brain(ABC):
    """Абстракция для всех моделей (OpenRouter, xAI, Local)."""

    @abstractmethod
    def generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        """Сгенерировать ответ на основе списка сообщений."""
        raise NotImplementedError

    def generate_stream(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> Iterator[str]:
        """
        Потоковая генерация.
        Дефолтный fallback: обычный generate() и chunked stream, чтобы UI
        отображал поэтапный вывод даже без нативного streaming у провайдера.
        """
        if tools is None:
            result = self.generate(messages, config=config)
        else:
            result = self.generate(messages, config=config, tools=tools)
        if not result.text:
            return
        chunk_size = 80
        for idx in range(0, len(result.text), chunk_size):
            yield result.text[idx : idx + chunk_size]

    def generate_stream_chunks(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> Iterator[LLMStreamChunk]:
        if tools is None:
            stream = self.generate_stream(messages, config=config)
        else:
            stream = self.generate_stream(messages, config=config, tools=tools)
        for delta in stream:
            if not delta:
                continue
            yield LLMStreamChunk(text=delta, mode="append")
