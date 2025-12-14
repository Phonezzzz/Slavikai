from __future__ import annotations

from llm.brain_base import Brain
from llm.types import LLMResult, ModelConfig
from shared.models import LLMMessage

DualMode = str  # simple alias to avoid Literal explosion here


class DualBrain(Brain):
    """
    Две модели: основная отвечает, критик проверяет ответ.
    """

    def __init__(self, main_brain: Brain, critic_brain: Brain):
        self.main = main_brain
        self.critic = critic_brain
        self.mode: DualMode = "dual"  # single | dual | critic-only

    def set_mode(self, mode: DualMode) -> None:
        if mode not in {"single", "dual", "critic-only"}:
            raise ValueError("Некорректный режим DualBrain")
        self.mode = mode

    def generate(self, messages: list[LLMMessage], config: ModelConfig | None = None) -> LLMResult:
        if self.mode == "critic-only":
            return self.critic.generate(messages, config)

        main_reply = self.main.generate(messages, config)

        if self.mode == "single":
            return main_reply

        review_prompt = [
            LLMMessage(role="system", content="Ты — критик и рецензент."),
            LLMMessage(
                role="user",
                content=f"Проверь ответ модели:\n{main_reply.text}\nи предложи улучшения.",
            ),
        ]
        critic_reply = self.critic.generate(review_prompt, config)
        combined = f"💬 Ответ:\n{main_reply.text}\n\n🧠 Критик:\n{critic_reply.text}"
        return LLMResult(
            text=combined,
            usage=main_reply.usage,
            raw={"main": main_reply.raw, "critic": critic_reply.raw},
        )
