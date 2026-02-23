from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import TYPE_CHECKING

from core.auto_runtime import AutoOrchestrator, AutoRunOutcome
from core.tracer import Tracer
from llm.types import LLMResult
from shared.models import JSONValue, LLMMessage

if TYPE_CHECKING:
    from core.agent import Agent

MAX_SUBTASKS = 6


class AutoAgent:
    """Compatibility wrapper: legacy helpers + role-based auto orchestrator."""

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

    # Legacy compatibility helpers (used by existing tests).
    def generate_subtasks(self, goal: str) -> list[str]:
        goal_clean = goal.strip()
        parts = [part.strip() for part in goal_clean.split("и") if part.strip()]
        subtasks = [p.capitalize() for p in parts if len(p) > 3][:MAX_SUBTASKS]
        if not subtasks:
            subtasks = [
                f"Анализировать задачу: {goal_clean}",
                f"Реализовать решение для: {goal_clean}",
                "Проверить корректность результата",
            ]
        self.tracer.log("auto_subtasks", f"Создано {len(subtasks)} подзадач", {"tasks": subtasks})
        return subtasks

    def run_parallel(self, subtasks: list[str]) -> list[tuple[str, str]]:
        results: list[tuple[str, str]] = []
        if not subtasks:
            return results
        self.tracer.log("auto_start", f"Параллельное выполнение {len(subtasks)} подзадач")
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(subtasks))) as executor:
            future_to_task = {executor.submit(self.run_subagent, task): task for task in subtasks}
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append((task, result))
                    self.tracer.log("auto_step_done", task, {"result": result[:100]})
                except Exception as exc:  # noqa: BLE001
                    self.tracer.log("auto_step_error", task, {"error": str(exc)})
        self.tracer.log("auto_end", "Параллельное выполнение завершено")
        return results

    def run_subagent(self, task: str) -> str:
        brain = self.parent.brain
        try:
            prompt = LLMMessage(role="user", content=f"Подзадача: {task}")
            result = brain.generate([prompt])
            return result.text if isinstance(result, LLMResult) else str(result)
        except Exception as exc:  # noqa: BLE001
            return f"[Ошибка мини-агента: {exc}]"

    def auto_execute(self, goal: str) -> str:
        outcome = self.run_outcome(goal)
        return outcome.text

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
