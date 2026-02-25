from __future__ import annotations

import json
from pathlib import Path

from config.model_store import (
    load_model_configs,
    model_config_from_dict,
    model_config_to_dict,
    save_model_configs,
)
from llm.types import ModelConfig


def test_model_config_roundtrip(tmp_path: Path) -> None:
    cfg = ModelConfig(
        provider="openrouter",
        model="gpt-test",
        temperature=0.5,
        max_tokens=128,
        system_prompt="hi",
    )
    path = tmp_path / "model.json"
    save_model_configs(cfg, path)
    loaded_main = load_model_configs(path)
    assert loaded_main and loaded_main.model == "gpt-test"

    data = model_config_to_dict(cfg)
    roundtrip = model_config_from_dict(data)
    assert roundtrip.model == cfg.model
    assert json.loads(path.read_text())["main"]["model"] == "gpt-test"


def test_load_model_configs_returns_none_for_missing_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-model.json"
    assert load_model_configs(missing_path) is None


def test_load_model_configs_returns_none_when_main_missing(tmp_path: Path) -> None:
    path = tmp_path / "model-without-main.json"
    path.write_text(json.dumps({"legacy": {"provider": "local", "model": "x"}}), encoding="utf-8")
    assert load_model_configs(path) is None
