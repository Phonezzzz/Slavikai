from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

ALLOWED_OPERATIONS: Sequence[str] = (
    "web",
    "fs",
    "shell",
    "project",
    "tts",
    "stt",
    "image_analyze",
    "image_generate",
    "workspace_read",
    "workspace_write",
    "workspace_patch",
    "workspace_run",
)


@dataclass(frozen=True)
class PlanStepSchema:
    description: str
    status: Literal["pending", "in_progress", "done", "error"] = "pending"
    operation: str | None = None
    result: str | None = None

    def is_valid(self) -> bool:
        if len(self.description.strip()) < 3:
            return False
        if self.operation is None:
            return True
        return self.operation in ALLOWED_OPERATIONS


@dataclass(frozen=True)
class TaskPlanSchema:
    goal: str
    steps: list[PlanStepSchema]

    def is_valid(self, min_steps: int, max_steps: int) -> bool:
        if len(self.goal.strip()) < 3:
            return False
        if len(self.steps) < min_steps or len(self.steps) > max_steps:
            return False
        return all(step.is_valid() for step in self.steps)
