from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path("config/memory.json")
DEFAULT_INBOX_MAX_ITEMS = 200
DEFAULT_INBOX_TTL_DAYS = 30
DEFAULT_INBOX_WRITES_PER_MINUTE = 6


@dataclass(frozen=True)
class ContextBudgetConfig:
    total_chars: int = 12000
    prefs_max_items: int = 10
    prefs_chars: int = 800
    legacy_notes_chars: int = 600
    feedback_max_items: int = 2
    feedback_chars: int = 400
    canonical_memory_chars: int = 1800
    vector_code_chars: int = 1200
    vector_docs_chars: int = 1200
    workspace_file_chars: int = 2000
    pinned_atoms_chars: int = 1200
    session_summary_chars: int = 1500

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ContextBudgetConfig:
        defaults = cls()
        return cls(
            total_chars=_read_int(data, "total_chars", defaults.total_chars),
            prefs_max_items=_read_int(data, "prefs_max_items", defaults.prefs_max_items),
            prefs_chars=_read_int(data, "prefs_chars", defaults.prefs_chars),
            legacy_notes_chars=_read_int(
                data,
                "legacy_notes_chars",
                defaults.legacy_notes_chars,
            ),
            feedback_max_items=_read_int(
                data,
                "feedback_max_items",
                defaults.feedback_max_items,
            ),
            feedback_chars=_read_int(data, "feedback_chars", defaults.feedback_chars),
            canonical_memory_chars=_read_int(
                data,
                "canonical_memory_chars",
                defaults.canonical_memory_chars,
            ),
            vector_code_chars=_read_int(data, "vector_code_chars", defaults.vector_code_chars),
            vector_docs_chars=_read_int(data, "vector_docs_chars", defaults.vector_docs_chars),
            workspace_file_chars=_read_int(
                data,
                "workspace_file_chars",
                defaults.workspace_file_chars,
            ),
            pinned_atoms_chars=_read_int(
                data,
                "pinned_atoms_chars",
                defaults.pinned_atoms_chars,
            ),
            session_summary_chars=_read_int(
                data,
                "session_summary_chars",
                defaults.session_summary_chars,
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "total_chars": self.total_chars,
            "prefs_max_items": self.prefs_max_items,
            "prefs_chars": self.prefs_chars,
            "legacy_notes_chars": self.legacy_notes_chars,
            "feedback_max_items": self.feedback_max_items,
            "feedback_chars": self.feedback_chars,
            "canonical_memory_chars": self.canonical_memory_chars,
            "vector_code_chars": self.vector_code_chars,
            "vector_docs_chars": self.vector_docs_chars,
            "workspace_file_chars": self.workspace_file_chars,
            "pinned_atoms_chars": self.pinned_atoms_chars,
            "session_summary_chars": self.session_summary_chars,
        }


@dataclass(frozen=True)
class MemoryConfig:
    inbox_max_items: int = DEFAULT_INBOX_MAX_ITEMS
    inbox_ttl_days: int = DEFAULT_INBOX_TTL_DAYS
    inbox_writes_per_minute: int = DEFAULT_INBOX_WRITES_PER_MINUTE
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MemoryConfig:
        inbox_max_items = _read_int(data, "inbox_max_items", DEFAULT_INBOX_MAX_ITEMS)
        inbox_ttl_days = _read_int(data, "inbox_ttl_days", DEFAULT_INBOX_TTL_DAYS)
        inbox_writes_per_minute = _read_int(
            data,
            "inbox_writes_per_minute",
            DEFAULT_INBOX_WRITES_PER_MINUTE,
        )
        context_budget = _read_context_budget(data.get("context_budget"))
        return cls(
            inbox_max_items=inbox_max_items,
            inbox_ttl_days=inbox_ttl_days,
            inbox_writes_per_minute=inbox_writes_per_minute,
            context_budget=context_budget,
        )

    def to_dict(self) -> dict[str, int | bool | dict[str, int]]:
        return {
            "inbox_max_items": self.inbox_max_items,
            "inbox_ttl_days": self.inbox_ttl_days,
            "inbox_writes_per_minute": self.inbox_writes_per_minute,
            "context_budget": self.context_budget.to_dict(),
        }


def load_memory_config(path: Path = DEFAULT_PATH) -> MemoryConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return MemoryConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("memory.json должен содержать объект.")
        return MemoryConfig.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка чтения конфигурации memory: {exc}") from exc


def save_memory_config(config: MemoryConfig, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _read_context_budget(raw: object) -> ContextBudgetConfig:
    if raw is None:
        return ContextBudgetConfig()
    if not isinstance(raw, dict):
        raise ValueError("context_budget должен быть объектом")
    context_data = {key: value for key, value in raw.items() if isinstance(key, str)}
    return ContextBudgetConfig.from_dict(context_data)


def _read_int(data: dict[str, object], key: str, default: int) -> int:
    raw = data.get(key, default)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} должен быть int")
    if raw <= 0:
        raise ValueError(f"{key} должен быть положительным")
    return raw
