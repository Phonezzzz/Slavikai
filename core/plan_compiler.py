from __future__ import annotations

from dataclasses import dataclass

from llm.brain_base import Brain
from llm.types import ModelConfig, ToolSpec
from shared.models import JSONValue, LLMMessage

_MAX_PLAN_STEPS = 12


@dataclass(frozen=True)
class PlanCompilationError(RuntimeError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


def compile_structured_plan_steps(
    *,
    brain: Brain,
    config: ModelConfig | None,
    goal: str,
    audit_log: list[dict[str, JSONValue]],
    available_tools: list[ToolSpec],
) -> list[dict[str, JSONValue]]:
    if not brain.supports_native_tools:
        raise PlanCompilationError(
            code="native_tools_required",
            message="Plan требует provider с native tool calls. Выбери DeepSeek или Local model.",
        )
    tool_names = [tool.name for tool in available_tools]
    if not tool_names:
        raise PlanCompilationError(
            code="plan_tools_unavailable",
            message="Для Plan нет доступных runtime tools.",
        )
    submit_plan = ToolSpec(
        name="submit_plan",
        description=(
            "Submit an executable plan. Every step must select exactly one available operation "
            "and provide its complete tool arguments. Do not execute the plan."
        ),
        parameters_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": _MAX_PLAN_STEPS,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "operation": {"type": "string", "enum": tool_names},
                            "tool_args": {"type": "object"},
                            "expected_outputs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "acceptance_checks": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "title",
                            "description",
                            "operation",
                            "tool_args",
                            "expected_outputs",
                            "acceptance_checks",
                        ],
                    },
                }
            },
            "required": ["steps"],
        },
    )
    result = brain.generate(
        [
            LLMMessage(
                role="system",
                content=(
                    "You are the Plan runtime. Return exactly one submit_plan tool call. "
                    "Plan only: do not claim that any operation was executed. Each step must be "
                    "directly executable through one runtime tool with complete JSON arguments."
                ),
            ),
            LLMMessage(
                role="user",
                content=f"Goal:\n{goal}\n\nRead-only audit:\n{audit_log}",
            ),
        ],
        config=config,
        tools=[submit_plan],
    )
    if len(result.tool_calls) != 1 or result.tool_calls[0].name != "submit_plan":
        raise PlanCompilationError(
            code="structured_plan_required",
            message="Provider не вернул обязательный structured submit_plan tool call.",
        )
    steps_raw = result.tool_calls[0].arguments.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw or len(steps_raw) > _MAX_PLAN_STEPS:
        raise PlanCompilationError(
            code="invalid_structured_plan",
            message="submit_plan.steps должен содержать от 1 до 12 шагов.",
        )
    allowed = set(tool_names)
    steps: list[dict[str, JSONValue]] = []
    for index, raw_step in enumerate(steps_raw, start=1):
        if not isinstance(raw_step, dict):
            raise PlanCompilationError(
                code="invalid_structured_plan",
                message=f"submit_plan.steps[{index - 1}] должен быть объектом.",
            )
        title = _required_text(raw_step, "title", index)
        description = _required_text(raw_step, "description", index)
        operation = _required_text(raw_step, "operation", index)
        if operation not in allowed:
            raise PlanCompilationError(
                code="plan_operation_unavailable",
                message=f"Шаг {index}: operation '{operation}' недоступен.",
            )
        tool_args_raw = raw_step.get("tool_args")
        if not isinstance(tool_args_raw, dict):
            raise PlanCompilationError(
                code="invalid_structured_plan",
                message=f"Шаг {index}: tool_args должен быть объектом.",
            )
        expected_outputs = _required_text_list(raw_step, "expected_outputs", index)
        acceptance_checks = _required_text_list(raw_step, "acceptance_checks", index)
        steps.append(
            {
                "step_id": f"step-{index}",
                "title": title,
                "description": description,
                "allowed_tool_kinds": [operation],
                "inputs": {
                    "operation": operation,
                    "tool_args": {str(key): value for key, value in tool_args_raw.items()},
                },
                "expected_outputs": expected_outputs,
                "acceptance_checks": acceptance_checks,
                "status": "todo",
                "evidence": None,
            }
        )
    return steps


def _required_text(raw: dict[object, object], key: str, index: int) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanCompilationError(
            code="invalid_structured_plan",
            message=f"Шаг {index}: {key} должен быть непустой строкой.",
        )
    return value.strip()


def _required_text_list(raw: dict[object, object], key: str, index: int) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise PlanCompilationError(
            code="invalid_structured_plan",
            message=f"Шаг {index}: {key} должен быть непустым массивом строк.",
        )
    normalized = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if not normalized:
        raise PlanCompilationError(
            code="invalid_structured_plan",
            message=f"Шаг {index}: {key} должен быть непустым массивом строк.",
        )
    return normalized
