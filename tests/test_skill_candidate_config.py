from __future__ import annotations

import pytest

from config.skills_config import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_TTL_DAYS,
    load_skill_candidate_config,
)


def test_skill_candidate_config_defaults() -> None:
    config = load_skill_candidate_config({})
    assert config.max_candidates == DEFAULT_MAX_CANDIDATES
    assert config.ttl_days == DEFAULT_TTL_DAYS
    assert config.cooldown_seconds == DEFAULT_COOLDOWN_SECONDS


def test_skill_candidate_config_env_override() -> None:
    config = load_skill_candidate_config(
        {
            "SKILLS_CANDIDATE_MAX": "10",
            "SKILLS_CANDIDATE_TTL_DAYS": "7",
            "SKILLS_CANDIDATE_COOLDOWN_SECONDS": "3600",
        }
    )
    assert config.max_candidates == 10
    assert config.ttl_days == 7
    assert config.cooldown_seconds == 3600


def test_skill_candidate_config_invalid() -> None:
    with pytest.raises(ValueError):
        load_skill_candidate_config({"SKILLS_CANDIDATE_MAX": "0"})
