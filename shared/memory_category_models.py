from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

from shared.models import JSONValue

MemorySource = Literal["agent", "user", "triage"]


class MemoryCategory(str, Enum):
    INBOX = "inbox"
    NOTES = "notes"
    FACTS = "facts"
    PREFERENCES = "preferences"
    RULES = "rules"
    GLOSSARY = "glossary"


@dataclass(frozen=True)
class MemoryItem:
    id: str
    category: MemoryCategory
    created_at: datetime
    updated_at: datetime | None
    title: str | None
    content: str
    tags: list[str] = field(default_factory=list)
    source: MemorySource = "agent"
    fingerprint: str = ""
    meta: dict[str, JSONValue] = field(default_factory=dict)
    triaged_from: str | None = None
    triaged_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("MemoryItem.id должен быть непустым")
        if not self.content.strip():
            raise ValueError("MemoryItem.content должен быть непустым")
        if not self.fingerprint:
            raise ValueError("MemoryItem.fingerprint должен быть непустым")
        if not isinstance(self.tags, list) or not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("MemoryItem.tags должен быть list[str]")
