from __future__ import annotations

import pytest

import config.shell_config as shell_config_module
from config.shell_config import ShellConfig, load_shell_config, save_shell_config


def test_shell_config_defaults(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    cfg = load_shell_config("missing.json")
    assert cfg.allowed_commands
    cfg.timeout_seconds = 5
    cfg.max_output_chars = 100
    cfg.sandbox_root = "tmp_sandbox"
    save_shell_config(cfg, "saved.json")
    loaded = load_shell_config("saved.json")
    assert loaded.timeout_seconds == 5
    assert loaded.max_output_chars == 100
    assert loaded.sandbox_root == "tmp_sandbox"
    assert (tmp_path / "saved.json").exists()
    assert not (tmp_path / "config").exists()


def test_shell_config_rejects_invalid_sandbox_root_on_save_and_load(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(shell_config_module, "SHELL_CONFIG_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="Некорректный sandbox_root"):
        save_shell_config(ShellConfig(sandbox_root="../outside"), "invalid.json")

    invalid_path = tmp_path / "invalid_load.json"
    invalid_path.write_text('{"sandbox_root":"../outside"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Некорректный sandbox_root"):
        load_shell_config("invalid_load.json")
