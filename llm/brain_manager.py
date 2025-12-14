from __future__ import annotations

from dataclasses import dataclass

from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.dual_brain import DualBrain
from llm.types import ModelConfig


@dataclass
class BrainManager:
    """Строит и хранит основной и критический Brain по конфигам."""

    main_config: ModelConfig
    critic_config: ModelConfig | None = None
    main_api_key: str | None = None
    critic_api_key: str | None = None

    def build(self) -> Brain:
        main_brain = create_brain(self.main_config, api_key=self.main_api_key)
        if self.critic_config:
            critic_brain = create_brain(self.critic_config, api_key=self.critic_api_key)
            return DualBrain(main_brain, critic_brain)
        return main_brain
