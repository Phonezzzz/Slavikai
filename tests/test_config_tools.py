from __future__ import annotations

from pathlib import Path

from config.mode_config import load_mode, save_mode
from config.tools_config import ToolsConfig, load_tools_config, save_tools_config


def test_tools_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tools.json"
    cfg = ToolsConfig(fs=False, shell=True, web=True, project=False, img=True, tts=True, stt=False)
    save_tools_config(cfg, path)

    loaded = load_tools_config(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_mode_config_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mode.json"
    save_mode("single", path)
    assert load_mode(path) == "single"
