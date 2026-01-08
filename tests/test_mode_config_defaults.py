from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.mode_config import DEFAULT_MODE, load_mode, save_mode


def test_load_mode_default_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "mode.json"
    assert load_mode(path) == DEFAULT_MODE


def test_load_mode_falls_back_on_invalid_value(tmp_path: Path) -> None:
    path = tmp_path / "mode.json"
    path.write_text(json.dumps({"mode": "unknown"}), encoding="utf-8")
    assert load_mode(path) == DEFAULT_MODE


def test_save_mode_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        save_mode("bad-mode")
