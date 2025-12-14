from __future__ import annotations

from collections.abc import Callable

from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult

CriticCallback = Callable[[PlanStep], tuple[bool, str | None]]
AgentCallback = Callable[[PlanStep], str]


class Executor:
    """Выполняет шаги плана с трассировкой и опциональной критикой."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def run(
        self,
        plan: TaskPlan,
        tool_gateway: ToolGateway | None = None,
        agent_callback: AgentCallback | None = None,
        critic_callback: CriticCallback | None = None,
    ) -> TaskPlan:
        self.tracer.log("execution_start", f"Начато выполнение плана ({len(plan.steps)} шагов)")
        for index, step in enumerate(plan.steps, start=1):
            self.tracer.log("step_started", f"{index}. {step.description}")
            step.status = PlanStepStatus.IN_PROGRESS

            if critic_callback:
                ok, note = critic_callback(step)
                if not ok:
                    step.status = PlanStepStatus.ERROR
                    step.result = f"Отклонено критиком: {note or 'нет причины'}"
                    self.tracer.log("step_failed", step.result or "")
                    break

            try:
                if agent_callback:
                    result_text = agent_callback(step)
                elif tool_gateway:
                    result_text = self._execute_with_tools(step, plan, tool_gateway)
                else:
                    result_text = f"Выполнен: {step.description}"
                step.result = result_text
                step.status = PlanStepStatus.DONE
                self.tracer.log(
                    "step_finished", f"{index}. {step.description}", {"result": result_text}
                )
            except Exception as exc:  # noqa: BLE001
                step.status = PlanStepStatus.ERROR
                step.result = str(exc)
                self.tracer.log("step_failed", f"{index}. {step.description}", {"error": str(exc)})
                break

        self.tracer.log("execution_end", "План выполнен.")
        return plan

    def _execute_with_tools(self, step: PlanStep, plan: TaskPlan, gateway: ToolGateway) -> str:
        request: ToolRequest | None = None
        if step.operation is None:
            text = step.description.lower()
            if "web" in text or "search" in text or "поиск" in text:
                request = ToolRequest(name="web", args={"query": plan.goal})
            elif "shell" in text or "команда" in text:
                request = ToolRequest(name="shell", args={"command": "echo step"})
            else:
                request = ToolRequest(name=step.description or "unknown", args={"query": plan.goal})
        elif step.operation == "web":
            request = ToolRequest(name="web", args={"query": plan.goal})
        elif step.operation == "fs":
            request = ToolRequest(name="fs", args={"op": "list"})
        elif step.operation == "shell":
            request = ToolRequest(name="shell", args={"command": "echo step"})
        elif step.operation == "project":
            request = ToolRequest(name="project", args={"cmd": "find", "args": [plan.goal]})
        elif step.operation == "tts":
            request = ToolRequest(name="tts", args={"text": plan.goal})
        elif step.operation == "stt":
            request = ToolRequest(name="stt", args={"file_path": ""})
        else:
            return f"Выполнен: {step.description}"

        result = gateway.call(request)
        if isinstance(result, ToolResult) and result.ok:
            return str(result.data.get("output") or result.data)
        error = result.error if isinstance(result, ToolResult) else "Ошибка инструмента"
        raise RuntimeError(f"Ошибка: {error}")
