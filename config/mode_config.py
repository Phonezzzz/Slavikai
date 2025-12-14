from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODE = "single"
DEFAULT_PATH = Path("config/mode.json")
ALLOWED_MODES = {"single", "dual", "critic-only"}


def load_mode(path: Path = DEFAULT_PATH) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return DEFAULT_MODE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            raw_mode = data.get("mode")
            if isinstance(raw_mode, str) and raw_mode in ALLOWED_MODES:
                return raw_mode
        return DEFAULT_MODE
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка чтения режима: {exc}") from exc


def save_mode(mode: str, path: Path = DEFAULT_PATH) -> None:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Недопустимый режим: {mode}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"mode": mode}, ensure_ascii=False, indent=2), encoding="utf-8")
