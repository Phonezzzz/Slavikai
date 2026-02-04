from __future__ import annotations

from collections.abc import Callable
from typing import Final

from config.system_prompts import PLANNER_PROMPT
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.types import ModelConfig
from shared.models import (
    LLMMessage,
    PlanStep,
    PlanStepStatus,
    TaskComplexity,
    TaskPlan,
    ToolRequest,
)
from shared.plan_models import PlanStepSchema, TaskPlanSchema

MIN_STEPS: Final[int] = 2
MAX_STEPS: Final[int] = 8
COMPLEX_KEYWORDS: Final[list[str]] = [
    "analyze",
    "analysis",
    "debug",
    "refactor",
    "deploy",
    "database",
    "мног",
    "анализ",
    "проект",
    "архитект",
    "план",
    "vector",
    "вектор",
    "модель",
    "pipeline",
]
FILE_HINTS: Final[tuple[str, ...]] = (
    "файл",
    "file",
    "workspace",
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
)
READ_HINTS: Final[tuple[str, ...]] = (
    "прочита",
    "read",
    "содержим",
    "open",
    "посмотр",
    "покаж",
)
WRITE_HINTS: Final[tuple[str, ...]] = (
    "запиш",
    "записа",
    "write",
    "save",
    "созда",
    "перезапиш",
    "сохра",
)
PATCH_HINTS: Final[tuple[str, ...]] = (
    "patch",
    "патч",
    "измен",
    "обнов",
    "исправ",
    "редакт",
    "замен",
)


class Planner:
    """Строит и исполняет пошаговые планы для выполнения сложных задач."""

    def __init__(self) -> None:
        self.tracer = Tracer()

    def build_plan(
        self,
        goal: str,
        brain: Brain | None = None,
        model_config: ModelConfig | None = None,
    ) -> TaskPlan:
        """Создаёт план; если передан brain — пробует LLM, иначе эвристика."""
        cleaned_goal = goal.strip()
        self.tracer.log("planning_start", f"Создание плана для: {cleaned_goal}")
        steps = self._llm_plan(cleaned_goal, brain, model_config) if brain else None
        if not steps:
            steps = self._heuristic_plan(cleaned_goal)
        plan = self._build_task_plan(cleaned_goal, steps)
        plan = self._assign_operations(plan)
        if not self._is_plan_valid(plan):
            self.tracer.log(
                "planning_invalid_plan", "План не прошёл валидацию, fallback к эвристике"
            )
            fallback_steps = self._heuristic_plan(cleaned_goal)
            plan = self._build_task_plan(cleaned_goal, fallback_steps)
            plan = self._assign_operations(plan)
        self.tracer.log("planning_done", f"Сформирован план из {len(plan.steps)} шагов")
        return plan

    def execute_plan(
        self,
        plan: TaskPlan,
        agent_callback: Callable[[PlanStep], str] | None = None,
        tool_gateway: ToolGateway | None = None,
    ) -> TaskPlan:
        """Выполняет шаги плана последовательно."""
        self.tracer.log("execution_start", f"Начато выполнение плана ({len(plan.steps)} шагов)")
        for index, step in enumerate(plan.steps, start=1):
            self.tracer.log("step_start", f"Шаг {index}: {step.description}")
            step.status = PlanStepStatus.IN_PROGRESS
            try:
                if agent_callback:
                    result = agent_callback(step)
                    step.result = result
                elif tool_gateway:
                    step.result = self._execute_with_tools(step, plan, tool_gateway)
                else:
                    step.result = f"Выполнен: {step.description}"
                step.status = PlanStepStatus.DONE
                self.tracer.log("step_done", step.description, {"result": str(step.result)[:200]})
            except Exception as exc:  # noqa: BLE001
                step.status = PlanStepStatus.ERROR
                step.result = str(exc)
                self.tracer.log("step_error", f"Ошибка: {exc}")
                break
        self.tracer.log("execution_end", "План выполнен.")
        return plan

    def classify_complexity(self, goal: str) -> TaskComplexity:
        normalized = goal.lower()
        tokens = normalized.split()
        if any(keyword in normalized for keyword in COMPLEX_KEYWORDS):
            return TaskComplexity.COMPLEX
        if len(tokens) > 16:
            return TaskComplexity.COMPLEX
        if len(tokens) <= 6:
            return TaskComplexity.SIMPLE
        return TaskComplexity.SIMPLE

    def _heuristic_plan(self, goal: str) -> list[str]:
        normalized = goal.lower()
        if self._is_workspace_patch_goal(normalized):
            return [
                "Определить целевой путь в workspace",
                "Прочитать текущую версию файла",
                "Применить patch к файлу",
                "Проверить итоговое содержимое",
            ]
        if self._is_workspace_write_goal(normalized):
            return [
                "Определить целевой путь в workspace",
                "Подготовить содержимое",
                "Записать файл в workspace",
                "Проверить результат записи",
            ]
        if self._is_workspace_read_goal(normalized):
            return [
                "Определить целевой путь в workspace",
                "Прочитать содержимое файла",
                "Сформировать вывод по содержимому",
            ]
        if "analyze" in goal or "анализ" in goal:
            return [
                "Понять контекст задачи",
                "Выявить ключевые требования",
                "Сформировать краткое резюме",
            ]
        if "file" in goal or "data" in goal or "файл" in goal:
            return [
                "Найти нужный файл в песочнице",
                "Прочитать его содержимое",
                "Сформировать вывод по содержимому",
            ]
        return [
            "Определить цель",
            "Разбить задачу на подцели",
            "Реализовать каждую подцель",
            "Собрать итоговый результат",
        ]

    def _llm_plan(
        self, goal: str, brain: Brain | None, model_config: ModelConfig | None
    ) -> list[str] | None:
        if brain is None:
            return None
        try:
            messages = [
                LLMMessage(role="system", content=PLANNER_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        f"Составь 3-6 шагов плана для задачи: {goal}. Ответ в виде списка шагов."
                    ),
                ),
            ]
            result = brain.generate(messages, model_config)
            text = result.text
            steps = self._parse_plan_text(text)
            if steps:
                return steps
            self.tracer.log("planning_validation_failed", "LLM план не прошёл валидацию")
            return None
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("planning_llm_error", str(exc))
            return None

    def _execute_with_tools(self, step: PlanStep, plan: TaskPlan, gateway: ToolGateway) -> str:
        text = step.description.lower()
        if any(keyword in text for keyword in ("web", "search", "поиск")):
            req = ToolRequest(name="web", args={"query": plan.goal})
            result = gateway.call(req)
        elif "файл" in text or "file" in text:
            req = ToolRequest(name="fs", args={"op": "list"})
            result = gateway.call(req)
        elif "shell" in text or "команда" in text:
            req = ToolRequest(name="shell", args={"command": "echo step"})
            result = gateway.call(req)
        else:
            return f"Выполнен: {step.description}"

        if result.ok:
            return str(result.data.get("output") or result.data)
        return f"Ошибка: {result.error}"

    def _parse_plan_text(self, text: str) -> list[str] | None:
        """Парсит и валидирует список шагов в диапазоне MIN_STEPS..MAX_STEPS."""
        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned: list[str] = []
        for line in raw_lines:
            stripped = line.lstrip("-•0123456789. ").strip()
            if stripped:
                cleaned.append(stripped)
        unique_steps = []
        for step in cleaned:
            if step not in unique_steps:
                unique_steps.append(step)
        if len(unique_steps) < MIN_STEPS or len(unique_steps) > MAX_STEPS:
            return None
        return unique_steps

    def _is_plan_valid(self, plan: TaskPlan) -> bool:
        if len(plan.steps) < MIN_STEPS or len(plan.steps) > MAX_STEPS:
            return False
        schema = TaskPlanSchema(
            goal=plan.goal,
            steps=[
                PlanStepSchema(
                    description=step.description,
                    status=step.status.value,
                    operation=step.operation,
                    result=step.result,
                )
                for step in plan.steps
            ],
        )
        return schema.is_valid(MIN_STEPS, MAX_STEPS)

    def _build_task_plan(self, goal: str, steps: list[str]) -> TaskPlan:
        plan_steps = [PlanStep(description=step) for step in steps]
        return TaskPlan(goal=goal, steps=plan_steps)

    def _assign_operations(self, plan: TaskPlan) -> TaskPlan:
        for step in plan.steps:
            if step.operation is None:
                step.operation = self._map_operation(step.description)
        return plan

    def _map_operation(self, description: str) -> str | None:
        text = description.lower()
        if any(keyword in text for keyword in PATCH_HINTS):
            return "workspace_patch"
        if any(keyword in text for keyword in WRITE_HINTS):
            return "workspace_write"
        if any(keyword in text for keyword in READ_HINTS) and any(
            hint in text for hint in FILE_HINTS
        ):
            return "workspace_read"
        if any(keyword in text for keyword in ("web", "search", "поиск")):
            return "web"
        if "файл" in text or "file" in text or "прочитать" in text:
            return "fs"
        if "shell" in text or "команда" in text:
            return "shell"
        if "project" in text or "индекс" in text:
            return "project"
        if "tts" in text or "озвуч" in text:
            return "tts"
        if "stt" in text or "распозна" in text:
            return "stt"
        return None

    def _is_workspace_read_goal(self, goal: str) -> bool:
        return self._has_any(goal, FILE_HINTS) and self._has_any(goal, READ_HINTS)

    def _is_workspace_write_goal(self, goal: str) -> bool:
        return self._has_any(goal, FILE_HINTS) and self._has_any(goal, WRITE_HINTS)

    def _is_workspace_patch_goal(self, goal: str) -> bool:
        return self._has_any(goal, FILE_HINTS) and self._has_any(goal, PATCH_HINTS)

    def _has_any(self, text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in text for keyword in keywords)
