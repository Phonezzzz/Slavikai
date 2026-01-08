from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path("config/memory.json")
DEFAULT_AUTO_SAVE_DIALOGUE = False


@dataclass(frozen=True)
class MemoryConfig:
    auto_save_dialogue: bool = DEFAULT_AUTO_SAVE_DIALOGUE

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> MemoryConfig:
        raw_value = data.get("auto_save_dialogue", DEFAULT_AUTO_SAVE_DIALOGUE)
        if not isinstance(raw_value, bool):
            raise ValueError("auto_save_dialogue должен быть bool")
        return cls(auto_save_dialogue=raw_value)

    def to_dict(self) -> dict[str, bool]:
        return {"auto_save_dialogue": self.auto_save_dialogue}


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
