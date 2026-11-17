"""Каноническое выполнение утверждённого плана через AgentToolLoop (E01-T06).

План — это artifact: drafting/version/approval живут в UI-слое
(``server/http/handlers/plan.py`` + ``server/http/common/workflow_runtime.py``),
а выполнение утверждённого плана идёт тем же canonical loop, что и прямой
execute (auto/desktop): один ``AgentToolLoop`` на шаг плана, gateway из
RunSpec с ``intent=execute``, plan guard привязывает каждый вызов к
текущему шагу (plan_id/plan_hash/current_step_id/allowed_tool_kinds).

Правила этого модуля:

- Нет адаптера вокруг MWV и нет fallback на него: свежий запуск идёт
  только через ``AgentToolLoop`` + ``ToolGateway``. Resume старых
  MWV-чекпоинтов остаётся в ``run_plan_runner`` и для запусков после
  этого PR недостижим — его удаление принадлежит E01-T08.
- Шаг считается done только при чистом завершении loop и ``is_full_done``
  его outcome. Fake-DONE уровня шага из ``_execute_mwv_plan_step``
  (DONE из одного ``result.ok``) здесь не воспроизводится.
- State-changing шаг без подтверждённого postcondition даёт outcome
  UNVERIFIED, а не DONE (E01-T05 / ADR-0002). Проверка postcondition —
  это E05; до неё план с write-шагами завершается failed с честной
  причиной ``verification_unknown`` вместо ложного completed.
- Подмена plan hash/revision блокируется в ``validate_approved_plan``:
  хеш пересчитывается из содержимого плана, id/hash/revision
  сверяются с task-привязкой.
- Approval внутри шага не проглатывается: loop поднимает
  ``ApprovalRequired`` с continuation, executor сохраняет чекпоинт
  и поднимает ``PlanLoopApprovalPending`` — runner публикует UI decision,
  а resume идёт через тот же loop с сохранённой continuation
  (как ``resume_chat_approval``).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from core.approval_policy import ApprovalRequest, ApprovalRequired
from core.run_contract import RunBudget
from core.run_outcome import (
    Disposition,
    PostconditionCheck,
    RunOutcome,
    RunPhase,
    TerminalReason,
    Verification,
    VerificationOutcome,
    outcome_from_loop_result,
)
from core.tool_gateway import ToolGateway
from core.tool_loop import (
    AgentToolLoop,
    AgentToolLoopResult,
    ToolLoopContinuation,
    ToolLoopExecutionError,
)
from llm.brain_base import Brain
from llm.types import ModelConfig
from shared.models import JSONValue, LLMMessage
from tools.tool_registry import ToolRegistry
from tools.workspace_tools import workspace_root_context

StepStatus = Literal["done", "failed", "waiting_approval"]


class PlanExecutionBindingError(ValueError):
    """Привязка плана нарушена: статус, id/hash/revision или содержимое."""


class PlanLoopContinuationUnavailable(ValueError):
    """Сохранённая пауза loop-выполнения плана недоступна или не совпадает."""


class PlanExecuteAgentProtocol(Protocol):
    """То, что executor требует от агента (структурно — реальный Agent)."""

    tool_registry: ToolRegistry
    main_config: ModelConfig | None
    runtime_workspace_root: str | None

    def set_runtime_state(
        self,
        *,
        mode: str,
        active_plan: dict[str, JSONValue] | None,
        active_task: dict[str, JSONValue] | None,
        auto_state: dict[str, JSONValue] | None = None,
        enforce_plan_guard: bool,
    ) -> None: ...

    def _build_tool_gateway(self, *, budget: RunBudget | None = None) -> ToolGateway: ...

    def _get_main_brain(self) -> Brain: ...


@dataclass(frozen=True)
class PlanStepExecution:
    """Терминальное состояние одного шага плана после loop-выполнения."""

    step_id: str
    description: str
    status: StepStatus
    operation: str | None
    result: str
    tool_calls_used: int
    outcome: RunOutcome

    def snapshot(self) -> dict[str, JSONValue]:
        """Форма, которую понимает ``_apply_step_results_to_plan``."""
        return {
            "step_id": self.step_id,
            "description": self.description,
            "status": self.status,
            "operation": self.operation,
            "result": self.result,
            "tool_calls_used": self.tool_calls_used,
            "changes": [],
        }


@dataclass(frozen=True)
class PlanExecuteResult:
    """Итог выполнения утверждённого плана через canonical loop."""

    plan_id: str
    plan_hash: str
    plan_revision: int
    task_id: str
    steps: tuple[PlanStepExecution, ...]
    outcome: RunOutcome
    tool_calls_used: int

    @property
    def cancelled(self) -> bool:
        return self.outcome.terminal_reason is TerminalReason.CANCELLED


@dataclass(frozen=True)
class PlanLoopCheckpoint:
    """Пауза loop-выполнения плана на approval (аналог MWVWorkerCheckpoint)."""

    checkpoint_id: str
    plan_id: str
    plan_hash: str
    task_id: str
    session_id: str
    step_id: str
    step_index: int
    completed_steps: tuple[PlanStepExecution, ...]
    continuation: ToolLoopContinuation
    main_config: ModelConfig | None
    plan_revision: int
    brain: object = field(compare=False, repr=False)


class PlanLoopCheckpoints:
    """In-memory хранилище пауз loop-выполнения плана (как _mwv_checkpoints)."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, PlanLoopCheckpoint] = {}

    def create(self, checkpoint: PlanLoopCheckpoint) -> None:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint

    def take(self, checkpoint_id: str) -> PlanLoopCheckpoint | None:
        return self._checkpoints.pop(checkpoint_id, None)

    def peek(self, checkpoint_id: str) -> PlanLoopCheckpoint | None:
        return self._checkpoints.get(checkpoint_id)

    def __contains__(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self._checkpoints

    def __len__(self) -> int:
        return len(self._checkpoints)

    def clear_session(self, session_id: str) -> None:
        """Сбрасывает паузы сессии (паритет reset_runtime_transient_state)."""
        stale = [
            checkpoint_id
            for checkpoint_id, checkpoint in self._checkpoints.items()
            if checkpoint.session_id == session_id
        ]
        for checkpoint_id in stale:
            del self._checkpoints[checkpoint_id]


@dataclass(frozen=True)
class ValidatedPlan:
    """План после проверки статуса, привязки, хеша и структуры шагов."""

    plan_id: str
    plan_hash: str
    plan_revision: int
    steps: tuple[dict[str, JSONValue], ...]


class PlanLoopApprovalPending(Exception):
    """Loop запросил approval на шаге плана: выполнение приостановлено.

    Несёт чекпоинт для resume через ``run_plan_runner`` (как
    ``TaskPacketApprovalPending`` у MWV-пути).
    """

    def __init__(self, *, checkpoint: PlanLoopCheckpoint, approval: ApprovalRequest) -> None:
        super().__init__("plan_loop_approval_pending")
        self.checkpoint = checkpoint
        self.approval = approval


def _step_tool(step: dict[str, JSONValue]) -> str:
    allowed = step.get("allowed_tool_kinds")
    if isinstance(allowed, list) and allowed and isinstance(allowed[0], str):
        return allowed[0]
    raise PlanExecutionBindingError("plan_step_tool_missing")


def _step_tool_args(step: dict[str, JSONValue]) -> dict[str, JSONValue]:
    inputs = step.get("inputs")
    if not isinstance(inputs, dict):
        raise PlanExecutionBindingError("plan_step_inputs_missing")
    tool_args = inputs.get("tool_args")
    if not isinstance(tool_args, dict):
        raise PlanExecutionBindingError("plan_step_tool_args_missing")
    return tool_args


def validate_approved_plan(
    *,
    plan: dict[str, JSONValue],
    task: dict[str, JSONValue],
    plan_hash_fn: Callable[[dict[str, JSONValue]], str],
) -> ValidatedPlan:
    """Проверяет статус, привязку и хеш плана и структуру его шагов.

    Подмена plan hash/revision блокируется здесь: хеш пересчитывается из
    содержимого плана, id/hash сверяются с task-привязкой. Ревизия —
    CAS-счётчик, а не привязка контента: она растёт при каждом обновлении
    статуса шага (plan_mark_step), поэтому пакет, скомпилированный при
    запуске task, всегда отстаёт от живого плана. Для ревизии пакета
    проверяется только монотонность: пакет из ревизии новее живой
    невозможен и отклоняется. Возвращает шаги плана (та же валидация
    структуры, что и ``compile_plan_to_task_packet``: ровно один
    инструмент на шаг, operation совпадает с allowed_tool_kinds[0]).
    """
    status = plan.get("status")
    if status not in {"approved", "running"}:
        raise PlanExecutionBindingError(
            f"plan_not_approved: выполнить можно только approved план, статус {status!r}."
        )
    plan_id = plan.get("plan_id")
    plan_hash = plan.get("plan_hash")
    plan_revision = plan.get("plan_revision")
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise PlanExecutionBindingError("plan_id отсутствует или не строка.")
    if not isinstance(plan_hash, str) or not plan_hash.strip():
        raise PlanExecutionBindingError("plan_hash отсутствует или не строка.")
    if type(plan_revision) is not int or plan_revision <= 0:
        raise PlanExecutionBindingError("plan_revision должен быть положительным int.")
    for key in ("plan_id", "plan_hash"):
        if task.get(key) != plan.get(key):
            raise PlanExecutionBindingError(
                f"plan_binding_mismatch: task.{key} не совпадает с планом."
            )
    # Ревизия живёт не в task (normalize_task_payload её не несёт), а в
    # контексте task_packet, скомпилированного из плана при execute.
    # Ревизия живого плана только растёт (статусы шагов), поэтому
    # равенство здесь невозможно: пакет из ревизии новее живой —
    # подмена, пакет из более старой ревизии — норма.
    packet = task.get("task_packet")
    if isinstance(packet, dict):
        context = packet.get("context")
        if isinstance(context, dict) and "plan_revision" in context:
            packet_revision = context.get("plan_revision")
            if type(packet_revision) is not int or packet_revision <= 0:
                raise PlanExecutionBindingError(
                    "plan_binding_mismatch: task_packet.plan_revision не положительный int."
                )
            if packet_revision > plan_revision:
                raise PlanExecutionBindingError(
                    "plan_binding_mismatch: task_packet.plan_revision новее плана."
                )
    if plan_hash_fn(plan) != plan_hash:
        raise PlanExecutionBindingError(
            "plan_binding_mismatch: plan_hash не совпадает с содержимым плана."
        )
    steps_raw = plan.get("steps")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise PlanExecutionBindingError("plan.steps должен содержать хотя бы один шаг.")
    steps: list[dict[str, JSONValue]] = []
    for index, item in enumerate(steps_raw):
        if not isinstance(item, dict):
            raise PlanExecutionBindingError(f"plan.steps[{index}] должен быть объектом.")
        step_id = item.get("step_id")
        title = item.get("title")
        description = item.get("description")
        if not isinstance(step_id, str) or not step_id.strip():
            raise PlanExecutionBindingError(f"plan.steps[{index}].step_id обязателен.")
        if not isinstance(title, str) or not title.strip():
            raise PlanExecutionBindingError(f"plan.steps[{index}].title обязателен.")
        if not isinstance(description, str):
            raise PlanExecutionBindingError(f"plan.steps[{index}].description должен быть строкой.")
        allowed_raw = item.get("allowed_tool_kinds")
        allowed = (
            [tool.strip() for tool in allowed_raw if isinstance(tool, str) and tool.strip()]
            if isinstance(allowed_raw, list)
            else []
        )
        if len(allowed) != 1:
            raise PlanExecutionBindingError(
                f"plan.steps[{index}].allowed_tool_kinds должен содержать ровно один инструмент."
            )
        inputs_raw = item.get("inputs")
        if not isinstance(inputs_raw, dict):
            raise PlanExecutionBindingError(f"plan.steps[{index}].inputs должен быть объектом.")
        if inputs_raw.get("operation") != allowed[0]:
            raise PlanExecutionBindingError(
                f"plan.steps[{index}].inputs.operation должен совпадать с allowed_tool_kinds[0]."
            )
        if not isinstance(inputs_raw.get("tool_args"), dict):
            raise PlanExecutionBindingError(
                f"plan.steps[{index}].inputs.tool_args должен быть объектом."
            )
        steps.append(item)
    return ValidatedPlan(
        plan_id=plan_id,
        plan_hash=plan_hash,
        plan_revision=plan_revision,
        steps=tuple(steps),
    )


def _step_messages(*, plan: dict[str, JSONValue], step: dict[str, JSONValue]) -> list[LLMMessage]:
    plan_id = str(plan.get("plan_id"))
    revision = plan.get("plan_revision")
    acceptance_raw = step.get("acceptance_checks")
    acceptance = (
        [item.strip() for item in acceptance_raw if isinstance(item, str) and item.strip()]
        if isinstance(acceptance_raw, list)
        else []
    )
    system = (
        "Ты выполняешь утверждённый план. Работай только предложенным инструментом "
        "и только в рамках текущего шага. Не вызывай другие инструменты. "
        "Когда шаг выполнен, ответь коротко, что сделано."
    )
    user = (
        f"План {plan_id} (ревизия {revision}). Цель: {plan.get('goal')}\n"
        f"Шаг {step.get('step_id')}: {step.get('title')}\n"
        f"Описание: {step.get('description')}\n"
        f"Инструмент шага: {_step_tool(step)}\n"
        f"Аргументы инструмента: {json.dumps(_step_tool_args(step), ensure_ascii=False)}\n"
        f"Критерии приёмки: {'; '.join(acceptance) if acceptance else 'не заданы'}"
    )
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=user),
    ]


def _run_step_loop(
    *,
    agent: PlanExecuteAgentProtocol,
    plan: dict[str, JSONValue],
    step: dict[str, JSONValue],
    budget: RunBudget,
    cancellation_token: asyncio.Event | None,
    continuation: ToolLoopContinuation | None = None,
) -> AgentToolLoopResult:
    """Один AgentToolLoop на шаг плана с gateway из RunSpec intent=execute.

    Перед построением gateway runtime state агента ставится на этот шаг
    (как это делает MWV worker): mode=act, active_plan — утверждённый
    план, active_task — snapshot с current_step_id шага, plan guard on.
    Gateway строит замороженный RunSpec; listing при включённом guard
    возвращает ровно allowed tools шага, а loop не может вызвать
    инструмент вне listing.
    """
    step_id = str(step.get("step_id"))
    task_snapshot: dict[str, JSONValue] = {
        "plan_id": plan.get("plan_id"),
        "plan_hash": plan.get("plan_hash"),
        "current_step_id": step_id,
    }
    agent.set_runtime_state(
        mode="act",
        active_plan=plan,
        active_task=task_snapshot,
        auto_state=getattr(agent, "runtime_auto_state", None),
        enforce_plan_guard=True,
    )
    gateway = agent._build_tool_gateway(budget=budget)
    policy = gateway.run_policy
    if policy is None or policy.mode != "act" or not policy.enforce_plan_guard:
        raise PlanExecutionBindingError(
            "plan_execute_requires_execute_spec: gateway построен без act + plan guard."
        )
    specs = agent.tool_registry.list_tool_specs(policy=policy)
    tool_name = _step_tool(step)
    if tool_name not in {spec.name for spec in specs}:
        raise PlanExecutionBindingError(
            f"plan_step_tool_unavailable: инструмент шага {tool_name!r} недоступен в loop."
        )
    # Выполнять loop в контексте проверенного workspace сессии/пакета:
    # workspace tools и ToolGateway._workspace_binding читают root из
    # ContextVar, а apply_agent_runtime_state ставит только атрибут
    # агента (как было у MWV worker через workspace_root_context).
    with workspace_root_context(agent.runtime_workspace_root):
        return AgentToolLoop(**budget.loop_kwargs()).run(
            brain=agent._get_main_brain(),
            gateway=gateway,
            messages=_step_messages(plan=plan, step=step),
            tools=specs,
            config=agent.main_config,
            continuation=continuation,
            cancellation_token=cancellation_token,
        )


def _step_execution(
    *, step: dict[str, JSONValue], loop_result: AgentToolLoopResult, state_changing: bool
) -> PlanStepExecution:
    """Выводит терминальное состояние шага из результата loop.

    Шаг done — только при чистом завершении loop и ``is_full_done`` его
    outcome. Любой упавший вызов, ноль вызовов или loop-ошибка — failed.
    Отмена (cancelled) здесь не отображается: её обрабатывает
    ``_cancelled_step_execution`` в executor — частичный шаг сохраняется
    как failed с фактическим числом вызовов, а общий исход — CANCELLED.
    """
    step_id = str(step.get("step_id"))
    description = str(step.get("description"))
    tool_name = _step_tool(step)
    tool_calls_used = len(loop_result.tool_calls)
    if loop_result.error is not None:
        outcome = outcome_from_loop_result(loop_result, state_changing=state_changing)
        return PlanStepExecution(
            step_id=step_id,
            description=description,
            status="failed",
            operation=tool_name,
            result=f"Ошибка loop: {loop_result.error}",
            tool_calls_used=tool_calls_used,
            outcome=outcome,
        )
    failed_calls = [call for call in loop_result.tool_calls if not call.result.ok]
    if failed_calls:
        first = failed_calls[0]
        outcome = RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.FAILED,
            disposition=Disposition.NONE,
            verification=Verification(
                outcome=VerificationOutcome.UNKNOWN,
                checks=(
                    PostconditionCheck(
                        name="tool_call",
                        passed=False,
                        detail=first.result.error or "tool call failed",
                    ),
                ),
                source="plan_execute",
            ),
            state_changing=state_changing,
        )
        return PlanStepExecution(
            step_id=step_id,
            description=description,
            status="failed",
            operation=tool_name,
            result=f"Вызов {first.call.name} завершился ошибкой: {first.result.error}",
            tool_calls_used=tool_calls_used,
            outcome=outcome,
        )
    if not loop_result.tool_calls:
        outcome = RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.FAILED,
            disposition=Disposition.NONE,
            verification=Verification(
                outcome=VerificationOutcome.UNKNOWN,
                checks=(
                    PostconditionCheck(
                        name="tool_calls",
                        passed=False,
                        detail="шаг не выполнил ни одного вызова инструмента",
                    ),
                ),
                source="plan_execute",
            ),
            state_changing=state_changing,
        )
        return PlanStepExecution(
            step_id=step_id,
            description=description,
            status="failed",
            operation=tool_name,
            result="Шаг не выполнил ни одного вызова инструмента.",
            tool_calls_used=0,
            outcome=outcome,
        )
    outcome = outcome_from_loop_result(loop_result, state_changing=state_changing)
    if outcome.is_full_done:
        return PlanStepExecution(
            step_id=step_id,
            description=description,
            status="done",
            operation=tool_name,
            result=loop_result.text.strip() or "Шаг выполнен.",
            tool_calls_used=tool_calls_used,
            outcome=outcome,
        )
    return PlanStepExecution(
        step_id=step_id,
        description=description,
        status="done",
        operation=tool_name,
        result=f"Шаг выполнен, но результат не подтверждён ({outcome.ui_label}).",
        tool_calls_used=tool_calls_used,
        outcome=outcome,
    )


def _cancelled_step_execution(
    *, step: dict[str, JSONValue], loop_result: AgentToolLoopResult, state_changing: bool
) -> PlanStepExecution:
    """Частичный шаг, отменённый после уже выполненных вызовов.

    Отмена не откатывает произведённые эффекты: выполненные вызовы
    сохраняются с фактическим числом, а шаг НЕ объявляется done —
    он failed, и план завершается CANCELLED.
    """
    calls = len(loop_result.tool_calls)
    return PlanStepExecution(
        step_id=str(step.get("step_id")),
        description=str(step.get("description")),
        status="failed",
        operation=_step_tool(step),
        result=(
            f"Шаг отменён после {calls} выполненных вызовов; произведённые эффекты не откатываются."
        ),
        tool_calls_used=calls,
        outcome=RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.CANCELLED,
            disposition=Disposition.NONE,
            verification=Verification.unknown(source="plan_execute"),
            state_changing=state_changing,
        ),
    )


def _binding_failed_step_execution(
    *, step: dict[str, JSONValue], error: str, state_changing: bool
) -> PlanStepExecution:
    """Шаг, чей инструмент стал недоступен после уже выполненных шагов.

    Привязка проверяется перед dispatch шага, но отказ может прийти уже
    после эффектов предыдущих шагов (инструмент отключили после
    утверждения плана). Терять выполненный prefix нельзя: текущий шаг
    сохраняется как failed с причиной отказа, а план завершается FAILED.
    """
    return PlanStepExecution(
        step_id=str(step.get("step_id")),
        description=str(step.get("description")),
        status="failed",
        operation=_step_tool(step),
        result=f"Инструмент шага недоступен: {error}",
        tool_calls_used=0,
        outcome=RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.FAILED,
            disposition=Disposition.NONE,
            verification=Verification(
                outcome=VerificationOutcome.UNKNOWN,
                checks=(
                    PostconditionCheck(
                        name="step_tool_available",
                        passed=False,
                        detail=error,
                    ),
                ),
                source="plan_execute",
            ),
            state_changing=state_changing,
        ),
    )


def _overall_outcome(
    *,
    steps: tuple[PlanStepExecution, ...],
    any_state_changing: bool,
    cancelled: bool,
) -> RunOutcome:
    """Сводит per-step outcomes в общий исход плана."""
    if cancelled:
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.CANCELLED,
            disposition=Disposition.NONE,
            verification=Verification.unknown(source="plan_execute"),
            state_changing=any_state_changing,
        )
    for step in steps:
        reason = step.outcome.terminal_reason
        if reason is TerminalReason.ABORTED:
            return RunOutcome(
                phase=RunPhase.TERMINAL,
                terminal_reason=TerminalReason.ABORTED,
                disposition=Disposition.NONE,
                verification=Verification(
                    outcome=VerificationOutcome.UNKNOWN,
                    checks=(
                        PostconditionCheck(
                            name="step_aborted",
                            passed=False,
                            detail=f"шаг {step.step_id} остановлен политикой",
                        ),
                    ),
                    source="plan_execute",
                ),
                state_changing=any_state_changing,
            )
        if reason is TerminalReason.FAILED:
            return RunOutcome(
                phase=RunPhase.TERMINAL,
                terminal_reason=TerminalReason.FAILED,
                disposition=Disposition.NONE,
                verification=Verification(
                    outcome=VerificationOutcome.UNKNOWN,
                    checks=(
                        PostconditionCheck(
                            name="step_failed",
                            passed=False,
                            detail=step.result,
                        ),
                    ),
                    source="plan_execute",
                ),
                state_changing=any_state_changing,
            )
    if all(step.outcome.is_full_done for step in steps):
        verification = (
            Verification.not_applicable(source="plan_execute")
            if not any_state_changing
            else Verification.unknown(source="plan_execute")
        )
        return RunOutcome(
            phase=RunPhase.TERMINAL,
            terminal_reason=TerminalReason.COMPLETED,
            disposition=Disposition.FULL,
            verification=verification,
            state_changing=any_state_changing,
        )
    unverified = [step for step in steps if not step.outcome.is_full_done]
    return RunOutcome(
        phase=RunPhase.TERMINAL,
        terminal_reason=TerminalReason.COMPLETED,
        disposition=Disposition.FULL,
        verification=Verification(
            outcome=VerificationOutcome.UNKNOWN,
            checks=tuple(
                PostconditionCheck(
                    name="step_verification",
                    passed=False,
                    detail=f"шаг {step.step_id}: {step.outcome.ui_label}",
                )
                for step in unverified
            ),
            source="plan_execute",
        ),
        state_changing=any_state_changing,
    )


def _finish_result(
    *,
    plan_id: str,
    plan_hash: str,
    plan_revision: int,
    task_id: str,
    steps: tuple[PlanStepExecution, ...],
    any_state_changing: bool,
    cancelled: bool,
) -> PlanExecuteResult:
    return PlanExecuteResult(
        plan_id=plan_id,
        plan_hash=plan_hash,
        plan_revision=plan_revision,
        task_id=task_id,
        steps=steps,
        outcome=_overall_outcome(
            steps=steps, any_state_changing=any_state_changing, cancelled=cancelled
        ),
        tool_calls_used=sum(step.tool_calls_used for step in steps),
    )


def _execute_remaining_steps(
    *,
    agent: PlanExecuteAgentProtocol,
    plan: dict[str, JSONValue],
    validated: ValidatedPlan,
    task_id: str,
    session_id: str,
    start_index: int,
    completed: list[PlanStepExecution],
    any_state_changing: bool,
    budget: RunBudget,
    checkpoints: PlanLoopCheckpoints,
    cancellation_token: asyncio.Event | None,
    first_continuation: ToolLoopContinuation | None = None,
) -> PlanExecuteResult:
    """Выполняет шаги плана с ``start_index`` через canonical loop.

    Последовательный план: первый же failed/aborted шаг останавливает
    выполнение, остальные шаги не запускаются. Approval на шаге сохраняет
    чекпоинт и поднимает ``PlanLoopApprovalPending``.
    """
    plan_id = validated.plan_id
    plan_hash = validated.plan_hash
    plan_revision = validated.plan_revision
    steps = validated.steps
    done = list(completed)
    state_changing = any_state_changing
    for index in range(start_index, len(steps)):
        step = steps[index]
        if cancellation_token is not None and cancellation_token.is_set():
            return _finish_result(
                plan_id=plan_id,
                plan_hash=plan_hash,
                plan_revision=plan_revision,
                task_id=task_id,
                steps=tuple(done),
                any_state_changing=state_changing,
                cancelled=True,
            )
        tool_name = _step_tool(step)
        descriptor = agent.tool_registry.get_descriptor(tool_name)
        step_changing = descriptor is None or descriptor.capability != "read"
        state_changing = state_changing or step_changing
        continuation = first_continuation if index == start_index else None
        try:
            loop_result = _run_step_loop(
                agent=agent,
                plan=plan,
                step=step,
                budget=budget,
                cancellation_token=cancellation_token,
                continuation=continuation,
            )
        except ToolLoopExecutionError as exc:
            # Loop начался и упал (следующий запрос к модели после
            # выполненных вызовов). Отказ валидации до dispatch
            # (PlanExecutionBindingError / PlanLoopContinuationUnavailable)
            # ловится отдельным handler ниже: без выполненных шагов
            # чекпоинт resume сохраняется. А здесь continuation уже
            # недействительна как повтор: выполненный
            # prefix сохраняется как failed-шаг, план завершается FAILED,
            # и runner потребляет чекпоинт — повторное подтверждение не
            # воспроизведёт выполненные вызовы.
            executed = _step_execution(
                step=step, loop_result=exc.result, state_changing=step_changing
            )
            done.append(executed)
            return _finish_result(
                plan_id=plan_id,
                plan_hash=plan_hash,
                plan_revision=plan_revision,
                task_id=task_id,
                steps=tuple(done),
                any_state_changing=state_changing,
                cancelled=False,
            )
        except PlanExecutionBindingError as exc:
            # Привязка шага сломалась после уже выполненных в этом запуске
            # шагов (инструмент отключили после утверждения плана): терять
            # выполненный prefix нельзя — текущий шаг сохраняется как failed
            # с причиной, план завершается FAILED, runner потребляет
            # чекпоинт (тот же класс, что отмена и ошибка loop после
            # выполненных вызовов). Если в этом запуске ещё ничего не
            # выполнено (done == completed), отказ — до любого эффекта:
            # пробросить наружу, чтобы resume сохранил чекпоинт, а свежий
            # запуск упал общей ошибкой без потерянных эффектов.
            if len(done) == len(completed):
                raise
            done.append(
                _binding_failed_step_execution(
                    step=step, error=str(exc), state_changing=step_changing
                )
            )
            return _finish_result(
                plan_id=plan_id,
                plan_hash=plan_hash,
                plan_revision=plan_revision,
                task_id=task_id,
                steps=tuple(done),
                any_state_changing=state_changing,
                cancelled=False,
            )
        except ApprovalRequired as exc:
            if exc.continuation is None:
                raise RuntimeError("plan_loop_approval_without_continuation") from exc
            checkpoint = PlanLoopCheckpoint(
                checkpoint_id=f"plan-loop-{uuid.uuid4().hex}",
                plan_id=plan_id,
                plan_hash=plan_hash,
                task_id=task_id,
                session_id=session_id,
                step_id=str(step.get("step_id")),
                step_index=index,
                completed_steps=tuple(done),
                continuation=exc.continuation,
                main_config=agent.main_config,
                plan_revision=plan_revision,
                brain=agent._get_main_brain(),
            )
            checkpoints.create(checkpoint)
            raise PlanLoopApprovalPending(checkpoint=checkpoint, approval=exc.request) from exc
        if loop_result.cancelled:
            if loop_result.tool_calls:
                # Отмена не откатывает эффекты: сохранить частичный шаг
                # с фактическим числом вызовов, не объявляя его done.
                done.append(
                    _cancelled_step_execution(
                        step=step, loop_result=loop_result, state_changing=step_changing
                    )
                )
            return _finish_result(
                plan_id=plan_id,
                plan_hash=plan_hash,
                plan_revision=plan_revision,
                task_id=task_id,
                steps=tuple(done),
                any_state_changing=state_changing,
                cancelled=True,
            )
        executed = _step_execution(step=step, loop_result=loop_result, state_changing=step_changing)
        done.append(executed)
        if executed.outcome.terminal_reason in {
            TerminalReason.FAILED,
            TerminalReason.ABORTED,
        }:
            return _finish_result(
                plan_id=plan_id,
                plan_hash=plan_hash,
                plan_revision=plan_revision,
                task_id=task_id,
                steps=tuple(done),
                any_state_changing=state_changing,
                cancelled=False,
            )
    return _finish_result(
        plan_id=plan_id,
        plan_hash=plan_hash,
        plan_revision=plan_revision,
        task_id=task_id,
        steps=tuple(done),
        any_state_changing=state_changing,
        cancelled=False,
    )


def execute_approved_plan(
    *,
    agent: PlanExecuteAgentProtocol,
    plan: dict[str, JSONValue],
    task: dict[str, JSONValue],
    session_id: str,
    budget: RunBudget | None = None,
    plan_hash_fn: Callable[[dict[str, JSONValue]], str],
    checkpoints: PlanLoopCheckpoints,
    cancellation_token: asyncio.Event | None = None,
) -> PlanExecuteResult:
    """Выполняет утверждённый план как ``intent=execute`` через canonical loop."""
    validated = validate_approved_plan(plan=plan, task=task, plan_hash_fn=plan_hash_fn)
    task_id = task.get("task_id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise PlanExecutionBindingError("task_id отсутствует или не строка.")
    return _execute_remaining_steps(
        agent=agent,
        plan=plan,
        validated=validated,
        task_id=task_id,
        session_id=session_id,
        start_index=0,
        completed=[],
        any_state_changing=False,
        budget=budget if budget is not None else RunBudget(),
        checkpoints=checkpoints,
        cancellation_token=cancellation_token,
    )


def resume_approved_plan(
    *,
    agent: PlanExecuteAgentProtocol,
    checkpoint: PlanLoopCheckpoint,
    plan: dict[str, JSONValue],
    task: dict[str, JSONValue],
    budget: RunBudget | None = None,
    plan_hash_fn: Callable[[dict[str, JSONValue]], str],
    checkpoints: PlanLoopCheckpoints,
    cancellation_token: asyncio.Event | None = None,
) -> PlanExecuteResult:
    """Продолжает loop-выполнение плана с сохранённого чекпоинта.

    Проверяет scope паузы (план/контент/шаг/модель) до первого вызова:
    несовпадение — ``PlanLoopContinuationUnavailable``, а не молчаливый
    перезапуск с другим планом. Ревизия в scope не сравнивается на
    равенство: она растёт при каждом обновлении статуса шага, поэтому
    живой план всегда новее чекпоинта; контент привязан хешем.
    """
    validated = validate_approved_plan(plan=plan, task=task, plan_hash_fn=plan_hash_fn)
    if (
        validated.plan_id != checkpoint.plan_id
        or validated.plan_hash != checkpoint.plan_hash
        or checkpoint.plan_revision > validated.plan_revision
        or str(task.get("task_id")) != checkpoint.task_id
    ):
        raise PlanLoopContinuationUnavailable("plan_loop_continuation_scope_mismatch")
    if (
        checkpoint.step_index >= len(validated.steps)
        or str(validated.steps[checkpoint.step_index].get("step_id")) != checkpoint.step_id
    ):
        raise PlanLoopContinuationUnavailable("plan_loop_continuation_scope_mismatch")
    if (
        agent._get_main_brain() is not checkpoint.brain
        or agent.main_config != checkpoint.main_config
    ):
        raise PlanLoopContinuationUnavailable("plan_loop_continuation_unavailable")
    completed = list(checkpoint.completed_steps)
    any_state_changing = False
    for done_step in completed:
        descriptor = agent.tool_registry.get_descriptor(done_step.operation or "")
        if descriptor is None or descriptor.capability != "read":
            any_state_changing = True
    return _execute_remaining_steps(
        agent=agent,
        plan=plan,
        validated=validated,
        task_id=checkpoint.task_id,
        session_id=checkpoint.session_id,
        start_index=checkpoint.step_index,
        completed=completed,
        any_state_changing=any_state_changing,
        budget=budget if budget is not None else RunBudget(),
        checkpoints=checkpoints,
        cancellation_token=cancellation_token,
        first_continuation=checkpoint.continuation,
    )
