from __future__ import annotations

from abc import ABC, abstractmethod

from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage


class Brain(ABC):
    """Абстракция для всех моделей (OpenRouter, xAI, Local)."""

    @abstractmethod
    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        """Сгенерировать ответ на основе списка сообщений."""
        raise NotImplementedError
