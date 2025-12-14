from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_TOOLS_STATE: dict[str, bool] = {
    "fs": True,
    "shell": False,
    "web": False,
    "project": True,
    "img": False,
    "tts": False,
    "stt": False,
    "safe_mode": True,
}


@dataclass
class ToolsConfig:
    fs: bool = DEFAULT_TOOLS_STATE["fs"]
    shell: bool = DEFAULT_TOOLS_STATE["shell"]
    web: bool = DEFAULT_TOOLS_STATE["web"]
    project: bool = DEFAULT_TOOLS_STATE["project"]
    img: bool = DEFAULT_TOOLS_STATE["img"]
    tts: bool = DEFAULT_TOOLS_STATE["tts"]
    stt: bool = DEFAULT_TOOLS_STATE["stt"]
    safe_mode: bool = DEFAULT_TOOLS_STATE["safe_mode"]

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ToolsConfig:
        def _get_bool(key: str) -> bool:
            val = data.get(key, DEFAULT_TOOLS_STATE[key])
            if isinstance(val, bool):
                return val
            raise ValueError(f"Некорректное значение для {key}: {val}")

        return cls(
            fs=_get_bool("fs"),
            shell=_get_bool("shell"),
            web=_get_bool("web"),
            project=_get_bool("project"),
            img=_get_bool("img"),
            tts=_get_bool("tts"),
            stt=_get_bool("stt"),
            safe_mode=_get_bool("safe_mode"),
        )


DEFAULT_PATH = Path("config/tools.json")


def load_tools_config(path: Path = DEFAULT_PATH) -> ToolsConfig:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return ToolsConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("tools.json должен содержать объект.")
        return ToolsConfig.from_dict(data)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Ошибка чтения конфигурации tools: {exc}") from exc


def save_tools_config(config: ToolsConfig, path: Path = DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
