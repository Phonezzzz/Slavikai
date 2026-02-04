from __future__ import annotations

import os
from typing import Final

MODEL_WHITELIST: Final[set[str]] = {"slavik"}
MODEL_WHITELIST_ENV: Final[str] = "SLAVIK_MODEL_WHITELIST"


class ModelNotAllowedError(RuntimeError):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Модель '{model_id}' не входит в whitelist.")


def is_model_allowed(model_id: str) -> bool:
    allowed = {item.strip() for item in MODEL_WHITELIST}
    env_value = os.getenv(MODEL_WHITELIST_ENV, "").strip()
    if env_value:
        allowed.update(part.strip() for part in env_value.split(",") if part.strip())
    for entry in allowed:
        if entry.endswith("*"):
            prefix = entry[:-1]
            if model_id.startswith(prefix):
                return True
        if model_id == entry:
            return True
    return False


def ensure_model_allowed(model_id: str) -> None:
    if not is_model_allowed(model_id):
        raise ModelNotAllowedError(model_id)
