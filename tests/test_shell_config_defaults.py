from __future__ import annotations

from config.shell_config import load_shell_config, save_shell_config


def test_shell_config_defaults(tmp_path) -> None:
    cfg = load_shell_config(tmp_path / "missing.json")
    assert cfg.allowed_commands
    cfg.timeout_seconds = 5
    cfg.max_output_chars = 100
    cfg.sandbox_root = "tmp_sandbox"
    save_path = tmp_path / "saved.json"
    save_shell_config(cfg, save_path)
    loaded = load_shell_config(save_path)
    assert loaded.timeout_seconds == 5
    assert loaded.max_output_chars == 100
    assert loaded.sandbox_root == "tmp_sandbox"
