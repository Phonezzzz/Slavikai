from __future__ import annotations

from typing import Final

MODEL_WHITELIST: Final[set[str]] = {"slavik"}


class ModelNotAllowedError(RuntimeError):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        super().__init__(f"Модель '{model_id}' не входит в whitelist.")


def is_model_allowed(model_id: str) -> bool:
    return model_id in MODEL_WHITELIST


def ensure_model_allowed(model_id: str) -> None:
    if not is_model_allowed(model_id):
        raise ModelNotAllowedError(model_id)
