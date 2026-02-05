from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("config/memory.json")
# policies-first: автосохранение диалога выключено по умолчанию
# и включается только явно через config
DEFAULT_AUTO_SAVE_DIALOGUE = False
DEFAULT_INBOX_MAX_ITEMS = 200
DEFAULT_INBOX_TTL_DAYS = 30
DEFAULT_INBOX_WRITES_PER_MINUTE = 6


@dataclass(frozen=True)
class MemoryConfig:
    auto_save_dialogue: bool = DEFAULT_AUTO_SAVE_DIALOGUE
    inbox_max_items: int = DEFAULT_INBOX_MAX_ITEMS
    inbox_ttl_days: int = DEFAULT_INBOX_TTL_DAYS
    inbox_writes_per_minute: int = DEFAULT_INBOX_WRITES_PER_MINUTE

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MemoryConfig:
        raw_value = data.get("auto_save_dialogue", DEFAULT_AUTO_SAVE_DIALOGUE)
        if not isinstance(raw_value, bool):
            raise ValueError("auto_save_dialogue должен быть bool")
        inbox_max_items = _read_int(data, "inbox_max_items", DEFAULT_INBOX_MAX_ITEMS)
        inbox_ttl_days = _read_int(data, "inbox_ttl_days", DEFAULT_INBOX_TTL_DAYS)
        inbox_writes_per_minute = _read_int(
            data,
            "inbox_writes_per_minute",
            DEFAULT_INBOX_WRITES_PER_MINUTE,
        )
        return cls(
            auto_save_dialogue=raw_value,
            inbox_max_items=inbox_max_items,
            inbox_ttl_days=inbox_ttl_days,
            inbox_writes_per_minute=inbox_writes_per_minute,
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "auto_save_dialogue": self.auto_save_dialogue,
            "inbox_max_items": self.inbox_max_items,
            "inbox_ttl_days": self.inbox_ttl_days,
            "inbox_writes_per_minute": self.inbox_writes_per_minute,
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


def _read_int(data: dict[str, object], key: str, default: int) -> int:
    raw = data.get(key, default)
    if not isinstance(raw, int):
        raise ValueError(f"{key} должен быть int")
    if raw <= 0:
        raise ValueError(f"{key} должен быть положительным")
    return raw
