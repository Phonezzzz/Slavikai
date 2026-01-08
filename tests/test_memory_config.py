from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.memory_config import MemoryConfig, load_memory_config, save_memory_config


def test_memory_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    config = load_memory_config(path)
    assert config.auto_save_dialogue is False


def test_memory_config_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    save_memory_config(MemoryConfig(auto_save_dialogue=True), path)
    loaded = load_memory_config(path)
    assert loaded.auto_save_dialogue is True


def test_memory_config_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"auto_save_dialogue": "yes"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps(["bad"]), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)
