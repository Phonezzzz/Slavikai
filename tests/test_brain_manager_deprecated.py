from __future__ import annotations

import logging

import pytest

import llm.brain_manager as brain_manager_module
from llm.brain_base import Brain
from llm.brain_manager import BrainManager
from llm.types import ModelConfig


class DummyBrain(Brain):
    def generate(self, messages, config: ModelConfig | None = None):  # noqa: ANN001
        raise AssertionError("should not be called")


def test_brain_manager_ignores_critic_config(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    created = []

    def _create_brain(config: ModelConfig, api_key: str | None = None) -> Brain:
        created.append((config, api_key))
        return DummyBrain()

    monkeypatch.setattr(brain_manager_module, "create_brain", _create_brain)

    main_cfg = ModelConfig(provider="local", model="main")
    critic_cfg = ModelConfig(provider="local", model="critic")
    manager = BrainManager(main_config=main_cfg, critic_config=critic_cfg)

    caplog.set_level(logging.WARNING, logger="SlavikAI.BrainManager")
    brain = manager.build()

    assert isinstance(brain, DummyBrain)
    assert created == [(main_cfg, None)]
    assert "critic_config ignored" in caplog.text
