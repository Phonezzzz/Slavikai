from __future__ import annotations

from config.settings import AppSettings
from llm.types import ModelConfig


def test_app_settings_defaults() -> None:
    main = ModelConfig(
        provider="openrouter",
        model="gpt-test",
        temperature=0.1,
        max_tokens=100,
    )
    settings = AppSettings(main_config=main)
    assert settings.main_config.model == "gpt-test"
    assert settings.shell_timeout_seconds == 10
    assert settings.tools_enabled == {}
