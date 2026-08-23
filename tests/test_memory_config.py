from __future__ import annotations

import json
from pathlib import Path

import pytest

from config.memory_config import (
    ContextBudgetConfig,
    MemoryConfig,
    load_memory_config,
    save_memory_config,
)


def test_memory_config_defaults(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    config = load_memory_config(path)
    assert config.inbox_max_items == 200
    assert config.inbox_ttl_days == 30
    assert config.inbox_writes_per_minute == 6
    assert config.context_budget == ContextBudgetConfig()
    assert config.context_budget.total_chars == 12000
    assert config.context_budget.workspace_file_chars == 2000


def test_memory_config_save_and_load(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    save_memory_config(
        MemoryConfig(
            inbox_max_items=100,
            inbox_ttl_days=14,
            inbox_writes_per_minute=12,
            context_budget=ContextBudgetConfig(
                total_chars=14000,
                workspace_file_chars=3000,
            ),
        ),
        path,
    )
    loaded = load_memory_config(path)
    assert loaded.inbox_max_items == 100
    assert loaded.inbox_ttl_days == 14
    assert loaded.inbox_writes_per_minute == 12
    assert loaded.context_budget.total_chars == 14000
    assert loaded.context_budget.workspace_file_chars == 3000


def test_memory_config_context_budget_from_dict_defaults_and_overrides() -> None:
    default_config = MemoryConfig.from_dict({})
    assert default_config.context_budget == ContextBudgetConfig()

    config = MemoryConfig.from_dict(
        {
            "context_budget": {
                "prefs_chars": 900,
                "unknown_future_field": 12345,
            }
        }
    )

    assert config.context_budget.prefs_chars == 900
    assert config.context_budget.total_chars == ContextBudgetConfig().total_chars
    assert (
        config.context_budget.session_summary_chars == ContextBudgetConfig().session_summary_chars
    )


def test_memory_config_to_dict_includes_context_budget() -> None:
    payload = MemoryConfig(
        context_budget=ContextBudgetConfig(vector_code_chars=2222),
    ).to_dict()

    assert "auto_save_dialogue" not in payload
    context_budget = payload["context_budget"]
    assert isinstance(context_budget, dict)
    assert context_budget["vector_code_chars"] == 2222
    assert context_budget["prefs_max_items"] == 10


def test_memory_config_invalid_payload(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"auto_save_dialogue": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="auto_save_dialogue удалён"):
        load_memory_config(path)

    path.write_text(json.dumps({"inbox_max_items": -1}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps({"inbox_max_items": True}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps({"context_budget": "bad"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps({"context_budget": {"total_chars": "12000"}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps({"context_budget": {"total_chars": True}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps({"context_budget": {"total_chars": 0}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)

    path.write_text(json.dumps(["bad"]), encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_memory_config(path)
