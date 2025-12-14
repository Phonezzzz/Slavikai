from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from core.tracer import Tracer
from llm.types import LLMResult
from shared.models import LLMMessage

if TYPE_CHECKING:
    from core.agent import Agent

MAX_SUBTASKS = 6


class AutoAgent:
    """Создаёт подагентов, распределяет им задачи и собирает результаты."""

    def __init__(self, parent_agent: Agent) -> None:
        self.parent = parent_agent
        self.tracer = Tracer()

    def generate_subtasks(self, goal: str) -> list[str]:
        """Создаёт список подзадач из общей цели."""
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
        """Выполняет подзадачи параллельно."""
        results: list[tuple[str, str]] = []
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
        """Исполняет задачу отдельным “мини-агентом”."""
        brain = self.parent.brain
        try:
            prompt = LLMMessage(role="user", content=f"Подзадача: {task}")
            result = brain.generate([prompt])
            return result.text if isinstance(result, LLMResult) else str(result)
        except Exception as exc:  # noqa: BLE001
            return f"[Ошибка мини-агента: {exc}]"

    def auto_execute(self, goal: str) -> str:
        """Основной интерфейс: планирует и запускает мини-агентов."""
        subtasks = self.generate_subtasks(goal)
        results = self.run_parallel(subtasks)
        summary = "\n".join([f"🔹 {task} → {result[:120]}" for task, result in results])
        final = f"🧩 Итог ({len(results)} подзадач):\n{summary}"
        self.tracer.log("auto_summary", final)
        return final
