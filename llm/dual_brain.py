from __future__ import annotations

from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage

DualMode = str  # simple alias to avoid Literal explosion here


class DualBrain(Brain):
    """Контейнер для двух моделей: main + critic.

    Координация dual-логики выполняется на уровне Agent/сервисов.
    """

    def __init__(self, main_brain: Brain, critic_brain: Brain):
        self.main = main_brain
        self.critic = critic_brain
        self.mode: DualMode = "dual"  # single | dual | critic-only

    def set_mode(self, mode: DualMode) -> None:
        if mode not in {"single", "dual", "critic-only"}:
            raise ValueError("Некорректный режим DualBrain")
        self.mode = mode

    def generate(
        self, messages: list[LLMMessage], config: ModelConfig | None = None
    ) -> LLMResult:
        if self.mode == "critic-only":
            return self.critic.generate(messages, config)
        return self.main.generate(messages, config)
