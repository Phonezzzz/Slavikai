from __future__ import annotations

import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("enabled", [True, False])
def test_saved_web_search_bool_survives_file_reload(tmp_path: Path, enabled: bool) -> None:
    path = tmp_path / "model.json"
    save_model_configs(
        ModelConfig(provider="xai", model="fixture", web_search_enabled=enabled), path
    )
    before = path.read_bytes()
    for _ in range(2):
        loaded = load_model_configs(path)
        assert loaded is not None and loaded.web_search_enabled is enabled
    assert path.read_bytes() == before


def test_missing_web_search_flag_remains_disabled() -> None:
    loaded = model_config_from_dict({"provider": "xai", "model": "fixture"})
    assert loaded.web_search_enabled is False


@pytest.mark.parametrize("invalid", [None, "false", "true", 0, 1, [], {}])
def test_non_bool_saved_web_search_flag_is_rejected_without_value_echo(invalid) -> None:
    with pytest.raises(ValueError, match="^web_search_enabled must be a boolean$"):
        model_config_from_dict(
            {"provider": "xai", "model": "fixture", "web_search_enabled": invalid}
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_loaded_web_flag_reaches_existing_adapter_without_changing_override_precedence(
    monkeypatch, tmp_path: Path, enabled: bool
) -> None:
    import asyncio

    from llm.brain_factory import create_brain
    from server.http.common.runtime_model_state import (
        RuntimeModelResolver,
        build_runtime_model_state_from_persisted,
    )
    from shared.models import LLMMessage
    from tests.test_llm_clients import _mock_response

    path = tmp_path / "models.json"
    save_model_configs(
        ModelConfig(provider="xai", model="fixture", web_search_enabled=enabled), path
    )
    store = build_runtime_model_state_from_persisted(
        load_model_configs_fn=lambda: load_model_configs(path)
    )
    # Changing the saved file does not mutate the already hydrated runtime config.
    save_model_configs(
        ModelConfig(provider="xai", model="fixture", web_search_enabled=not enabled), path
    )
    calls = []

    def transport(url, json, headers, timeout):
        calls.append((url, json))
        return _mock_response(
            {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "fixture"}]}
                ],
                "choices": [{"message": {"content": "fixture"}}],
            }
        )

    monkeypatch.setattr("llm.xai_brain.requests.post", transport)

    async def run():
        resolver = RuntimeModelResolver(store)
        config = await resolver.resolve_main("no-model-selection")
        assert config is not None and config.web_search_enabled is enabled
        brain = create_brain(config, api_key="inert-fixture-key")
        assert (
            brain.generate([LLMMessage(role="user", content="fixture request")]).text == "fixture"
        )
        await store.set_session_override("overridden", ModelConfig(provider="xai", model="fixture"))
        override = await resolver.resolve_main("overridden")
        assert override is not None and override.web_search_enabled is False
        assert (
            create_brain(override, api_key="inert-fixture-key")
            .generate([LLMMessage(role="user", content="fixture override")])
            .text
            == "fixture"
        )

    asyncio.run(run())
    assert len(calls) == 2
    if enabled:
        assert calls[0][0] == "https://api.x.ai/v1/responses"
        assert calls[0][1]["tools"] == [{"type": "web_search"}]
    else:
        assert calls[0][0] == "https://api.x.ai/v1/chat/completions"
        assert "tools" not in calls[0][1]
    assert calls[1][0] == "https://api.x.ai/v1/chat/completions" and "tools" not in calls[1][1]
