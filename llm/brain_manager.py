from __future__ import annotations

from dataclasses import dataclass

from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.types import ModelConfig


@dataclass
class BrainManager:
    """Строит и хранит основной Brain по конфигам."""

    main_config: ModelConfig
    main_api_key: str | None = None

    def build(self) -> Brain:
        return create_brain(self.main_config, api_key=self.main_api_key)
