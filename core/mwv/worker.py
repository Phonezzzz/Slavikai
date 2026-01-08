from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from core.mwv.models import RunContext, TaskPacket, WorkResult

WorkerRunner = Callable[[TaskPacket, RunContext], WorkResult]


@dataclass(frozen=True)
class WorkerRuntime:
    runner: WorkerRunner

    def run(self, task: TaskPacket, context: RunContext) -> WorkResult:
        return self.runner(task, context)
