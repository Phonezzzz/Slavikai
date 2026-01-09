from __future__ import annotations

import logging
from dataclasses import dataclass

from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.types import ModelConfig


@dataclass
class BrainManager:
    """Строит и хранит основной и критический Brain по конфигам."""

    main_config: ModelConfig
    critic_config: ModelConfig | None = None
    main_api_key: str | None = None
    critic_api_key: str | None = None

    def build(self) -> Brain:
        logger = logging.getLogger("SlavikAI.BrainManager")
        main_brain = create_brain(self.main_config, api_key=self.main_api_key)
        if self.critic_config:
            logger.warning("Deprecated: critic_config ignored; DualBrain disabled in MWV runtime.")
        return main_brain
