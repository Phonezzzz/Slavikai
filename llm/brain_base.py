from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class Brain(ABC):
    """Абстракция для всех моделей (OpenRouter, xAI, Local)."""

    @abstractmethod
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        """Сгенерировать ответ на основе списка сообщений."""
        raise NotImplementedError

    def stream_generate(
        self,
        messages: list[LLMMessage],
        config: ModelConfig | None = None,
    ) -> Iterator[str]:
        """Потоковая генерация. По умолчанию отдаёт полный ответ одним чанком."""
        result = self.generate(messages, config=config)
        if result.text:
            yield result.text
