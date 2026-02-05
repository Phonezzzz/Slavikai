from __future__ import annotations

import os
from typing import Final

MODEL_WHITELIST: Final[set[str]] = {"slavik"}
MODEL_PROVIDER_WHITELIST: Final[set[str]] = {"local", "xai"}
MODEL_WHITELIST_ENV: Final[str] = "SLAVIK_MODEL_WHITELIST"
_PROVIDER_RULE_PREFIX: Final[str] = "provider:"


class ModelNotAllowedError(RuntimeError):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Модель '{model_id}' не входит в whitelist.")


def _normalize_provider(provider: str | None) -> str | None:
    if provider is None:
        return None
    normalized = provider.strip().lower()
    return normalized or None


def _resolve_allow_rules() -> tuple[set[str], set[str]]:
    allowed_models = {item.strip() for item in MODEL_WHITELIST if item.strip()}
    allowed_providers = {item.strip().lower() for item in MODEL_PROVIDER_WHITELIST if item.strip()}
    env_value = os.getenv(MODEL_WHITELIST_ENV, "").strip()
    if not env_value:
        return allowed_models, allowed_providers
    for part in env_value.split(","):
        rule = part.strip()
        if not rule:
            continue
        lowered = rule.lower()
        if lowered.startswith(_PROVIDER_RULE_PREFIX):
            provider = lowered.removeprefix(_PROVIDER_RULE_PREFIX).strip()
            if provider:
                allowed_providers.add(provider)
            continue
        allowed_models.add(rule)
    return allowed_models, allowed_providers


def is_model_allowed(model_id: str, provider: str | None = None) -> bool:
    allowed_models, allowed_providers = _resolve_allow_rules()
    normalized_provider = _normalize_provider(provider)
    if normalized_provider and normalized_provider in allowed_providers:
        return True
    for entry in allowed_models:
        if entry.endswith("*"):
            prefix = entry[:-1]
            if model_id.startswith(prefix):
                return True
        if model_id == entry:
            return True
    return False


def ensure_model_allowed(model_id: str, provider: str | None = None) -> None:
    if not is_model_allowed(model_id, provider):
        raise ModelNotAllowedError(model_id)
