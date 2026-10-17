from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from core.auto_runtime import AutoOrchestrator, AutoRunOutcome
from core.skills.index import SkillResolution
from core.tracer import Tracer
from shared.models import JSONValue

if TYPE_CHECKING:
    from core.agent import Agent


class AutoAgent:
    """Фасад канонического Auto runtime с typed outcome и native continuation."""

    def __init__(self, parent_agent: Agent) -> None:
        self.parent = parent_agent
        self.tracer = Tracer()
        self._progress_callback: Callable[[dict[str, JSONValue]], None] | None = None
        self.orchestrator = AutoOrchestrator(
            parent_agent,
            progress_callback=self._on_progress,
        )

    def set_progress_callback(
        self,
        callback: Callable[[dict[str, JSONValue]], None] | None,
    ) -> None:
        self._progress_callback = callback

    def _on_progress(self, state: dict[str, JSONValue]) -> None:
        callback = self._progress_callback
        if callback is None:
            return
        callback(dict(state))

    def run_outcome(
        self,
        goal: str,
        *,
        skill_resolution: SkillResolution | None = None,
    ) -> AutoRunOutcome:
        self.tracer.log("auto_invoke", f"AgentToolLoop->Verifier: {goal}")
        return self.orchestrator.run_v1(goal, skill_resolution=skill_resolution)

    def resume_outcome(
        self,
        run_id: str,
        *,
        cancellation_token: asyncio.Event | None = None,
    ) -> AutoRunOutcome:
        return self.orchestrator.resume(run_id, cancellation_token=cancellation_token)

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "cancelled_by_user",
    ) -> dict[str, JSONValue] | None:
        return self.orchestrator.cancel(run_id, reason=reason)
