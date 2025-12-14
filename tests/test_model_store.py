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
    save_model_configs(cfg, None, path)
    loaded_main, loaded_critic = load_model_configs(path)
    assert loaded_critic is None
    assert loaded_main and loaded_main.model == "gpt-test"

    data = model_config_to_dict(cfg)
    roundtrip = model_config_from_dict(data)
    assert roundtrip.model == cfg.model
    assert json.loads(path.read_text())["main"]["model"] == "gpt-test"
