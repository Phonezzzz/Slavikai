from __future__ import annotations

import pytest

from config.model_whitelist import is_model_allowed


def test_model_whitelist_allows_default_entries() -> None:
    assert is_model_allowed("slavik")
    assert not is_model_allowed("xai-model")
    assert is_model_allowed("any-local-model", provider="local")
    assert is_model_allowed("grok-4-1212", provider="xai")
    assert not is_model_allowed("openrouter-model", provider="openrouter")


def test_model_whitelist_supports_env_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAVIK_MODEL_WHITELIST", "custom-model,another-*,provider:openrouter")
    assert is_model_allowed("custom-model")
    assert is_model_allowed("another-v2")
    assert is_model_allowed("any-openrouter", provider="openrouter")
