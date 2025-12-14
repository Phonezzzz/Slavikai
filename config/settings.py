from __future__ import annotations

from dataclasses import dataclass, field

from llm.types import ModelConfig


@dataclass
class AppSettings:
    """Настройки выбора моделей и ключей."""

    main_config: ModelConfig
    critic_config: ModelConfig | None = None
    main_api_key: str | None = None
    critic_api_key: str | None = None
    shell_config_path: str = "config/shell_config.json"
    sandbox_root: str = "sandbox"
    tools_enabled: dict[str, bool] = field(default_factory=dict)
    shell_allowed: list[str] = field(default_factory=list)
    shell_timeout_seconds: int = 10
    shell_max_output_chars: int = 6000
