from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from core.auto_runtime import AutoOrchestrator, AutoRunOutcome
from core.tracer import Tracer
from shared.models import JSONValue

if TYPE_CHECKING:
    from core.agent import Agent


class AutoAgent:
    """Role-based auto orchestrator adapter."""

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

    def run_outcome(self, goal: str) -> AutoRunOutcome:
        self.tracer.log("auto_invoke", f"Planner->Coder->Verifier: {goal}")
        return self.orchestrator.run(goal)

    def resume_outcome(self, run_id: str) -> AutoRunOutcome | None:
        return self.orchestrator.resume(run_id)

    def cancel_run(
        self,
        run_id: str,
        *,
        reason: str = "cancelled_by_user",
    ) -> dict[str, JSONValue] | None:
        return self.orchestrator.cancel(run_id, reason=reason)
