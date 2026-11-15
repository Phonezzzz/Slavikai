from __future__ import annotations

from llm.brain_base import Brain
from llm.capabilities import NativeToolQualification
from llm.chatgpt_plan import ChatGPTPlan, ChatGPTPlanError
from llm.chatgpt_plan_brain import ChatGPTPlanBrain
from llm.deepseek_brain import DeepSeekBrain
from llm.inception_brain import InceptionBrain
from llm.local_http_brain import LocalHttpBrain
from llm.openrouter_brain import OpenRouterBrain
from llm.types import ModelConfig, is_chatgpt_plan_provider, is_custom_provider_id
from llm.xai_brain import XAiBrain


def create_brain(
    config: ModelConfig,
    api_key: str | None = None,
    *,
    qualification: NativeToolQualification | None = None,
    chatgpt_plan: ChatGPTPlan | None = None,
    principal: str | None = None,
) -> Brain:
    if is_chatgpt_plan_provider(config.provider):
        if chatgpt_plan is None or principal is None:
            raise ChatGPTPlanError(
                "chatgpt_plan_unavailable",
                "Подключение подписки ChatGPT недоступно для этой сессии.",
                status=503,
            )
        return ChatGPTPlanBrain(chatgpt_plan, principal, config)
    if config.provider == "openrouter":
        return OpenRouterBrain(api_key=api_key or config.api_key, default_config=config)
    if config.provider == "xai":
        return XAiBrain(api_key=api_key or config.api_key, default_config=config)
    if config.provider == "local":
        return LocalHttpBrain(
            default_config=config,
            base_url=config.base_url,
            api_key=api_key or config.api_key,
        )
    if config.provider == "inception":
        return InceptionBrain(api_key=api_key or config.api_key, default_config=config)
    if config.provider == "deepseek":
        return DeepSeekBrain(api_key=api_key or config.api_key, default_config=config)
    if config.base_url:
        # Base URL не разрешает инструменты. Формат custom ID допускает только
        # подключение Chat-адаптера, а не подтверждает capabilities модели.
        if not is_custom_provider_id(config.provider):
            raise ValueError(f"Неизвестный провайдер модели: {config.provider}")
        return LocalHttpBrain(
            default_config=config,
            base_url=config.base_url,
            api_key=api_key or config.api_key,
            native_tools=False,
            qualification=qualification,
        )
    raise ValueError(f"Неизвестный провайдер модели: {config.provider}")
