from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from hmac import compare_digest
from typing import Literal

CapabilityState = Literal["unknown", "supported", "unsupported"]
CapabilitySource = Literal["adapter_declaration", "not_observed", "inert_protocol_v1"]


@dataclass(frozen=True)
class CapabilityEvidence:
    state: CapabilityState
    source: CapabilitySource


@dataclass(frozen=True)
class NativeToolCapability:
    """Снимок admission; декларация адаптера не является проверкой модели."""

    adapter: CapabilityEvidence
    model: CapabilityEvidence
    eligibility: Literal["legacy_adapter", "unqualified", "unsupported", "qualified"]

    @property
    def allowed(self) -> bool:
        return self.eligibility in {"legacy_adapter", "qualified"}

    @property
    def reason(self) -> str:
        if self.eligibility == "qualified":
            return "Native tools проверены явным inert protocol check текущей модели."
        if self.eligibility == "unqualified":
            return "Поддержка native tool calls выбранным endpoint и моделью не проверена."
        if self.eligibility == "unsupported" and self.model.state == "unsupported":
            return "Выбранная модель не прошла inert tool protocol check."
        if self.eligibility == "unsupported":
            return "Адаптер выбранной модели не поддерживает native tool calls."
        return "Native tools разрешены по декларации адаптера; модель отдельно не проверена."


@dataclass(frozen=True)
class NativeToolQualification:
    """Transient server-owned evidence; не является grant для host tools."""

    binding: str = field(repr=False)
    is_current: Callable[[], bool] = field(repr=False, compare=False)
    state: Literal["supported", "unsupported"] = "supported"

    def valid_for(self, binding: str) -> bool:
        return compare_digest(self.binding, binding) and self.is_current()
