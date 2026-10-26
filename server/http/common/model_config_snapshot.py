from __future__ import annotations

import math

from llm.types import ModelConfig
from shared.models import JSONValue


def build_model_config_snapshot(config: ModelConfig | None) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {
        "basis": "loaded_default_model_at_refresh",
        "source": "Saved default model configuration with loader defaults",
        "state": "unavailable" if config is None else "available",
    }
    if config is None:
        return result
    values: dict[str, JSONValue] = {}
    unavailable: list[JSONValue] = []
    for key in ("temperature", "top_p", "max_tokens"):
        raw = getattr(config, key)
        valid = (
            isinstance(raw, (float, int))
            and not isinstance(raw, bool)
            and abs(raw) <= 2**53 - 1
            and math.isfinite(raw)
            and (key != "max_tokens" or isinstance(raw, int))
        )
        values[key] = raw if valid else None
        if raw is not None and not valid:
            unavailable.append(key)
    for key, allowed in [
        ("mode", {"default", "planner"}),
        ("reasoning_effort", {"instant", "low", "medium", "high"}),
    ]:
        raw = getattr(config, key)
        valid = isinstance(raw, str) and raw in allowed
        values[key] = raw if valid else None
        if raw is not None and not valid:
            unavailable.append(key)
    for key in (
        "thinking_enabled",
        "reasoning_summary",
        "reasoning_summary_wait",
        "diffusing",
        "web_search_enabled",
    ):
        raw = getattr(config, key)
        values[key] = raw if isinstance(raw, bool) else None
        if not isinstance(raw, bool) and (raw is not None or key == "web_search_enabled"):
            unavailable.append(key)
    presence: dict[str, JSONValue] = {}
    for key in ("api_key", "base_url", "system_prompt"):
        raw = getattr(config, key)
        presence[key] = (
            bool(raw.strip()) if isinstance(raw, str) else False if raw is None else None
        )
    presence["extra_headers"] = (
        bool(config.extra_headers) if isinstance(config.extra_headers, dict) else None
    )
    result.update(
        {
            "identity": {"provider": config.provider, "model": config.model},
            "values": values,
            "presence": presence,
            "unavailable_fields": unavailable,
            "unsupported_fields": [],
        }
    )
    return result
