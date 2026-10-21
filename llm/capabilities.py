from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CapabilityState = Literal["unknown", "supported", "unsupported"]
CapabilitySource = Literal["adapter_declaration", "not_observed"]


@dataclass(frozen=True)
class CapabilityEvidence:
    state: CapabilityState
    source: CapabilitySource


@dataclass(frozen=True)
class NativeToolCapability:
    """Снимок admission; декларация адаптера не является проверкой модели."""

    adapter: CapabilityEvidence
    model: CapabilityEvidence
    eligibility: Literal["legacy_adapter", "unqualified", "unsupported"]

    @property
    def allowed(self) -> bool:
        return self.eligibility == "legacy_adapter"

    @property
    def reason(self) -> str:
        if self.eligibility == "unqualified":
            return "Поддержка native tool calls выбранным endpoint и моделью не проверена."
        if self.eligibility == "unsupported":
            return "Адаптер выбранной модели не поддерживает native tool calls."
        return "Native tools разрешены по декларации адаптера; модель отдельно не проверена."
