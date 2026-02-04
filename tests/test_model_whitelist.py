from __future__ import annotations

from config.model_whitelist import is_model_allowed


def test_model_whitelist_allows_default_entries() -> None:
    assert is_model_allowed("slavik")
    assert not is_model_allowed("xai-model")


def test_model_whitelist_supports_env_extension(monkeypatch) -> None:
    monkeypatch.setenv("SLAVIK_MODEL_WHITELIST", "custom-model,another-*")
    assert is_model_allowed("custom-model")
    assert is_model_allowed("another-v2")
