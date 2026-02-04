from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shared.models import JSONValue, LLMMessage


@dataclass(frozen=True)
class ModelConfig:
    provider: Literal["openrouter", "local", "xai"]
    model: str
    temperature: float = 0.7
    top_p: float | None = None
    max_tokens: int | None = None
    base_url: str | None = None
    api_key: str | None = None
    extra_headers: dict[str, str] = field(default_factory=dict)
    system_prompt: str | None = None
    mode: Literal["default", "planner"] = "default"
    thinking_enabled: bool = False


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class LLMResult:
    text: str
    reasoning: str | None = None
    usage: LLMUsage | None = None
    raw: dict[str, JSONValue] | None = None


@dataclass(frozen=True)
class BrainRequest:
    messages: list[LLMMessage]
    config: ModelConfig
