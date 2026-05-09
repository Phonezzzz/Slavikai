from __future__ import annotations

from collections.abc import Callable

from core.approval_policy import ApprovalRequired
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult

AgentCallback = Callable[[PlanStep], str]


class Executor:
    """Executes explicit plan steps without deriving tool arguments from prose."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def run(
        self,
        plan: TaskPlan,
        tool_gateway: ToolGateway | None = None,
        agent_callback: AgentCallback | None = None,
    ) -> TaskPlan:
        self.tracer.log("execution_start", f"Начато выполнение плана ({len(plan.steps)} шагов)")
        for index, step in enumerate(plan.steps, start=1):
            self.tracer.log("step_started", f"{index}. {step.description}")
            step.status = PlanStepStatus.IN_PROGRESS

            try:
                if agent_callback:
                    result_text = agent_callback(step)
                elif tool_gateway and step.operation:
                    result_text = self._execute_with_tools(step, tool_gateway)
                else:
                    result_text = f"Выполнен: {step.description}"
                step.result = result_text
                step.status = PlanStepStatus.DONE
                self.tracer.log(
                    "step_finished",
                    f"{index}. {step.description}",
                    {"result": result_text},
                )
            except ApprovalRequired:
                raise
            except Exception as exc:  # noqa: BLE001
                step.status = PlanStepStatus.ERROR
                step.result = str(exc)
                self.tracer.log("step_failed", f"{index}. {step.description}", {"error": str(exc)})
                break

        self.tracer.log("execution_end", "План выполнен.")
        return plan

    def _execute_with_tools(self, step: PlanStep, gateway: ToolGateway) -> str:
        if not step.operation:
            return f"Выполнен: {step.description}"

        result = gateway.call(ToolRequest(name=step.operation, args=dict(step.tool_args)))
        if isinstance(result, ToolResult) and result.ok:
            return str(result.data.get("output") or result.data)
        error = result.error if isinstance(result, ToolResult) else "Ошибка инструмента"
        raise RuntimeError(f"Ошибка: {error}")
