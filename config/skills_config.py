from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

# Candidate retention defaults: keep recent drafts while limiting disk usage and noise.
DEFAULT_MAX_CANDIDATES = 200
DEFAULT_TTL_DAYS = 30
DEFAULT_COOLDOWN_SECONDS = 60 * 60


@dataclass(frozen=True)
class SkillCandidateConfig:
    max_candidates: int
    ttl_days: int
    cooldown_seconds: int


def load_skill_candidate_config(
    env: Mapping[str, str] | None = None,
) -> SkillCandidateConfig:
    env = env or os.environ
    max_candidates = _read_int(env, "SKILLS_CANDIDATE_MAX", DEFAULT_MAX_CANDIDATES)
    ttl_days = _read_int(env, "SKILLS_CANDIDATE_TTL_DAYS", DEFAULT_TTL_DAYS)
    cooldown_seconds = _read_int(
        env,
        "SKILLS_CANDIDATE_COOLDOWN_SECONDS",
        DEFAULT_COOLDOWN_SECONDS,
    )
    return SkillCandidateConfig(
        max_candidates=max_candidates,
        ttl_days=ttl_days,
        cooldown_seconds=cooldown_seconds,
    )


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be >= 1")
    return value
