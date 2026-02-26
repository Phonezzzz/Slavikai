from __future__ import annotations

# ruff: noqa: F401
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from core.approval_policy import ApprovalCategory, ApprovalRequired
from core.decision.verifier_fail import build_verifier_fail_packet
from core.mwv.manager import ManagerRuntime, MWVRunResult
from core.mwv.models import (
    ChangeType,
    MWVMessage,
    RunContext,
    StopReasonCode,
    TaskPacket,
    TaskStepContract,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
    is_task_packet_hash_valid,
    with_task_packet_hash,
)
from core.mwv.routing import RouteDecision
from core.mwv.verifier_runtime import VerifierRuntime
from core.mwv.worker import WorkerRuntime
from llm.brain_base import Brain
from shared.models import (
    JSONValue,
    LLMMessage,
    PlanStep,
    PlanStepStatus,
    TaskPlan,
    WorkspaceDiffEntry,
)
from tools.workspace_tools import WORKSPACE_ROOT, workspace_root_context

MAX_MWV_ATTEMPTS = 3

if TYPE_CHECKING:
    import logging

    from config.memory_config import MemoryConfig
    from core.decision.models import DecisionPacket
    from core.executor import Executor
    from core.planner import Planner
    from core.skills.index import SkillMatch
    from core.tool_gateway import ToolGateway
    from core.tracer import Tracer
    from llm.types import ModelConfig
    from shared.models import ToolRequest, ToolResult


def _manager_runtime_cls() -> type[ManagerRuntime]:
    import core.agent as agent_module

    value = getattr(agent_module, "ManagerRuntime", ManagerRuntime)
    return value


def _verifier_runtime_cls() -> type[VerifierRuntime]:
    import core.agent as agent_module

    value = getattr(agent_module, "VerifierRuntime", VerifierRuntime)
    return value


def _workspace_root() -> str:
    import core.agent as agent_module

    value = getattr(agent_module, "WORKSPACE_ROOT", WORKSPACE_ROOT)
    return str(value)


class AgentMWVMixin:
    if TYPE_CHECKING:
        tracer: Tracer
        logger: logging.Logger
        planner: Planner
        executor: Executor
        tools_enabled: dict[str, bool]
        memory_config: MemoryConfig
        main_config: ModelConfig | None
        session_id: str | None
        approved_categories: set[ApprovalCategory]
        runtime_mode: str
        runtime_active_plan: dict[str, JSONValue] | None
        runtime_active_task: dict[str, JSONValue] | None
        runtime_auto_state: dict[str, JSONValue] | None
        runtime_plan_guard_enabled: bool
        _last_skill_match: SkillMatch | None

        def _inc_metric(self, metric_key: str) -> None: ...
        def _handle_decision_packet(
            self,
            packet: DecisionPacket,
            *,
            raw_input: str,
            record_in_history: bool,
        ) -> str: ...
        def save_to_memory(self, prompt: str, answer: str) -> None: ...
        def _log_chat_interaction(
            self,
            raw_input: str,
            response_text: str,
            *,
            retrieved_memory_ids: list[str] | None = None,
            applied_policy_ids: list[str] | None = None,
        ) -> str: ...
        def _append_short_term(
            self,
            messages: list[LLMMessage],
            *,
            history: list[LLMMessage] | None = None,
        ) -> None: ...
        def _reset_workspace_diffs(self) -> None: ...
        def set_runtime_state(
            self,
            *,
            mode: str,
            active_plan: dict[str, JSONValue] | None,
            active_task: dict[str, JSONValue] | None,
            auto_state: dict[str, JSONValue] | None = None,
            enforce_plan_guard: bool,
        ) -> None: ...
        def _build_tool_gateway(
            self,
            *,
            pre_call: Callable[[ToolRequest], object | None] | None = None,
            post_call: (Callable[[ToolRequest, ToolResult, object | None], None] | None) = None,
            safe_mode_override: bool | None = None,
        ) -> ToolGateway: ...
        def _workspace_diff_pre_call(self, request: ToolRequest) -> str | None: ...
        def _workspace_diff_post_call(
            self,
            request: ToolRequest,
            result: ToolResult,
            context: object | None,
        ) -> None: ...
        def consume_workspace_diffs(self) -> list[WorkspaceDiffEntry]: ...
        def _format_plan(self, plan: TaskPlan) -> str: ...
        def _format_stop_response(
            self,
            *,
            what: str,
            why: str,
            next_steps: list[str],
            stop_reason_code: StopReasonCode,
            route: str,
            trace_id: str | None = None,
            attempts: tuple[int, int] | None = None,
            verifier: VerificationResult | None = None,
            plan_summary: str | None = None,
            execution_summary: str | None = None,
        ) -> str: ...
        def _append_report_block(
            self,
            text: str,
            *,
            route: str,
            trace_id: str | None,
            attempts: tuple[int, int] | None,
            verifier: VerificationResult | None,
            next_steps: list[str] | None,
            stop_reason_code: StopReasonCode | None,
            plan_summary: str | None = None,
            execution_summary: str | None = None,
        ) -> str: ...

    last_plan: TaskPlan | None
    last_plan_original: TaskPlan | None

    def _run_mwv_flow(
        self,
        messages: list[LLMMessage],
        raw_input: str,
        decision: RouteDecision,
        record_in_history: bool,
    ) -> str:
        trace_id = str(uuid.uuid4())
        mwv_messages = self._to_mwv_messages(messages)
        context = self._build_mwv_context(trace_id=trace_id)
        manager = _manager_runtime_cls()(task_builder=self._mwv_task_builder(decision))
        worker = WorkerRuntime(runner=self._mwv_worker_runner)
        verifier_runtime = _verifier_runtime_cls()()

        def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
            self.tracer.log(
                "mwv_worker_start",
                f"Attempt {run_context.attempt}",
                {"goal": task.goal},
            )
            result = worker.run(task, run_context)
            self.tracer.log(
                "mwv_worker_done",
                f"Attempt {run_context.attempt}",
                {"status": result.status.value},
            )
            return result

        def _verifier(task: TaskPacket, run_context: RunContext) -> VerificationResult:
            try:
                result = verifier_runtime.run(task, run_context)
            except TypeError:
                # Backward compatibility for legacy verifier runtime stubs:
                # run(context) -> VerificationResult
                legacy_run = cast(Callable[[RunContext], VerificationResult], verifier_runtime.run)
                result = legacy_run(run_context)
            self.tracer.log(
                "mwv_verifier_done",
                result.status.value,
                {
                    "exit_code": result.exit_code,
                    "attempt": run_context.attempt,
                },
            )
            return result

        try:
            run_result = manager.run_flow(
                mwv_messages,
                context,
                worker=_worker,
                verifier=_verifier,
            )
        except ApprovalRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._handle_mwv_error(
                exc,
                raw_input=raw_input,
                record_in_history=record_in_history,
                trace_id=trace_id,
            )
        if run_result.verification_result.status != VerificationStatus.PASSED:
            self._inc_metric("verifier_fail_count")
            decision_packet = build_verifier_fail_packet(
                run_result.verification_result,
                task_id=run_result.task.task_id,
                trace_id=run_result.task.trace_id,
                attempt=run_result.attempt,
                max_attempts=run_result.max_attempts,
                retry_allowed=bool(
                    run_result.retry_decision and run_result.retry_decision.allow_retry
                ),
            )
            return self._handle_decision_packet(
                decision_packet,
                raw_input=raw_input,
                record_in_history=record_in_history,
            )
        response = self._format_mwv_response(run_result)
        if self.memory_config.auto_save_dialogue:
            self.save_to_memory(raw_input, response)
        self._log_chat_interaction(raw_input=raw_input, response_text=response)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=response)])
        return response

    def _build_mwv_context(self, *, trace_id: str | None = None) -> RunContext:
        session_id = self.session_id or "local"
        approved = sorted(self.approved_categories)
        resolved_trace_id = trace_id or str(uuid.uuid4())
        return RunContext(
            session_id=session_id,
            trace_id=resolved_trace_id,
            workspace_root=_workspace_root(),
            safe_mode=bool(self.tools_enabled.get("safe_mode", False)),
            approved_categories=[str(item) for item in approved],
            max_retries=max(0, MAX_MWV_ATTEMPTS - 1),
            attempt=1,
        )

    def _mwv_task_builder(
        self, decision: RouteDecision
    ) -> Callable[[Sequence[MWVMessage], RunContext], TaskPacket]:
        def _build(messages: Sequence[MWVMessage], context: RunContext) -> TaskPacket:
            goal = messages[-1].content if messages else ""
            skill_context: dict[str, JSONValue] = {}
            if self._last_skill_match is not None:
                skill_context = {
                    "skill_id": self._last_skill_match.entry.id,
                    "skill_pattern": self._last_skill_match.pattern,
                }
            build_capsule = getattr(self, "build_memory_capsule", None)
            if callable(build_capsule):
                try:
                    memory_capsule = build_capsule(goal, for_mwv=True)
                    if isinstance(memory_capsule, dict):
                        skill_context["memory_capsule"] = memory_capsule
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("MWV memory capsule build failed: %s", exc)
            plan_goal = goal
            plan = self.planner.build_plan(plan_goal)
            task_steps = self._plan_to_task_steps(plan)
            packet = TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=goal,
                messages=list(messages),
                steps=task_steps,
                constraints=[],
                policy={
                    "safe_mode": context.safe_mode,
                },
                scope={
                    "workspace_root": context.workspace_root,
                },
                budgets={
                    "max_attempts": max(1, context.max_retries + 1),
                },
                approvals={
                    "approved_categories": list(context.approved_categories),
                },
                verifier={
                    "command": "scripts/check.sh",
                },
                context={
                    "route_reason": decision.reason,
                    "risk_flags": decision.risk_flags,
                    **skill_context,
                },
            )
            return with_task_packet_hash(packet)

        return _build

    def _plan_to_task_steps(self, plan: TaskPlan) -> list[TaskStepContract]:
        steps: list[TaskStepContract] = []
        for index, step in enumerate(plan.steps, start=1):
            operation = step.operation.strip() if isinstance(step.operation, str) else ""
            allowed = [operation] if operation else []
            steps.append(
                TaskStepContract(
                    step_id=f"step-{index}",
                    title=f"Step {index}",
                    description=step.description,
                    allowed_tool_kinds=allowed,
                    inputs={
                        "operation": operation if operation else None,
                        "description": step.description,
                    },
                    expected_outputs=[],
                    acceptance_checks=[],
                )
            )
        return steps

    def _mwv_worker_runner(self, task: TaskPacket, context: RunContext) -> WorkResult:
        self._reset_workspace_diffs()
        if not is_task_packet_hash_valid(task):
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary="TaskPacket hash mismatch",
                changes=[],
                tool_summaries=[],
                diagnostics={
                    "steps_total": len(task.steps),
                    "step_errors": [
                        {
                            "description": "TaskPacket hash",
                            "result": "packet_hash mismatch",
                        }
                    ],
                    "stop_reason_code": StopReasonCode.REPLAN_REQUIRED.value,
                },
            )
        if not task.steps:
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary="TaskPacket is missing steps",
                changes=[],
                tool_summaries=[],
                diagnostics={
                    "steps_total": 0,
                    "step_errors": [
                        {
                            "description": "TaskPacket.steps",
                            "result": "empty",
                        }
                    ],
                    "stop_reason_code": StopReasonCode.REPLAN_REQUIRED.value,
                },
            )

        workspace_root, scope_error = self._resolve_packet_workspace_root(task, context)
        if scope_error is not None:
            return WorkResult(
                task_id=task.task_id,
                status=WorkStatus.FAILURE,
                summary=scope_error,
                changes=[],
                tool_summaries=[],
                diagnostics={
                    "steps_total": len(task.steps),
                    "step_errors": [
                        {
                            "description": "TaskPacket.scope.workspace_root",
                            "result": scope_error,
                        }
                    ],
                    "stop_reason_code": StopReasonCode.REPLAN_REQUIRED.value,
                },
            )

        runtime_snapshot = self._capture_runtime_execution_state()
        plan_snapshot = self._build_packet_plan_snapshot(task)
        step_states: list[PlanStep] = []
        stop_reason = StopReasonCode.WORKER_FAILED

        try:
            with workspace_root_context(workspace_root):
                for step in task.steps:
                    operation, operation_error = self._resolve_packet_step_operation(step)
                    plan_step = PlanStep(description=step.description, operation=operation)
                    task_snapshot: dict[str, JSONValue] = {
                        "plan_id": task.task_id,
                        "plan_hash": task.packet_hash,
                        "current_step_id": step.step_id,
                    }
                    self.set_runtime_state(
                        mode="act",
                        active_plan=plan_snapshot,
                        active_task=task_snapshot,
                        auto_state=self.runtime_auto_state,
                        enforce_plan_guard=True,
                    )
                    if operation_error is not None:
                        plan_step.status = PlanStepStatus.ERROR
                        plan_step.result = operation_error
                        step_states.append(plan_step)
                        stop_reason = StopReasonCode.REPLAN_REQUIRED
                        break
                    step_plan = TaskPlan(goal=task.goal, steps=[plan_step])
                    executed = self.executor.run(
                        step_plan,
                        tool_gateway=self._build_tool_gateway(
                            pre_call=self._workspace_diff_pre_call,
                            post_call=self._workspace_diff_post_call,
                        ),
                    )
                    executed_step = executed.steps[0]
                    step_states.append(executed_step)
                    if executed_step.status == PlanStepStatus.ERROR:
                        error_text = executed_step.result or ""
                        stop_reason = self._stop_reason_from_execution_error(error_text)
                        break
        finally:
            self._restore_runtime_execution_state(runtime_snapshot)

        executed_plan = TaskPlan(goal=task.goal, steps=step_states)
        self.last_plan_original = executed_plan
        self.last_plan = executed_plan
        status = (
            WorkStatus.FAILURE
            if any(step.status == PlanStepStatus.ERROR for step in step_states)
            else WorkStatus.SUCCESS
        )
        changes = self._mwv_changes_from_diffs(self.consume_workspace_diffs())
        diagnostics = self._build_mwv_diagnostics(executed_plan)
        if status == WorkStatus.FAILURE:
            diagnostics["stop_reason_code"] = stop_reason.value
        summary = self._format_plan(executed_plan)
        return WorkResult(
            task_id=task.task_id,
            status=status,
            summary=summary,
            changes=changes,
            tool_summaries=[],
            diagnostics=diagnostics,
        )

    def _to_mwv_messages(self, messages: list[LLMMessage]) -> list[MWVMessage]:
        mwv_messages: list[MWVMessage] = []
        for message in messages:
            if message.role not in {"system", "user", "assistant", "tool"}:
                continue
            mwv_messages.append(MWVMessage(role=message.role, content=message.content))
        return mwv_messages

    def _build_mwv_goal(self, task: TaskPacket) -> str:
        if not task.constraints:
            return task.goal
        constraints = "\n".join(f"- {item}" for item in task.constraints)
        return f"{task.goal}\nОграничения:\n{constraints}"

    def _resolve_packet_workspace_root(
        self,
        task: TaskPacket,
        context: RunContext,
    ) -> tuple[str, str | None]:
        context_root = Path(context.workspace_root).resolve()
        scope_root_raw = task.scope.get("workspace_root")
        if not isinstance(scope_root_raw, str) or not scope_root_raw.strip():
            return str(context_root), None
        candidate = Path(scope_root_raw).expanduser().resolve()
        try:
            candidate.relative_to(context_root)
        except ValueError:
            return "", f"scope workspace_root вне контекста: {candidate}"
        if not candidate.exists() or not candidate.is_dir():
            return "", f"scope workspace_root недоступен: {candidate}"
        return str(candidate), None

    def _build_packet_plan_snapshot(self, task: TaskPacket) -> dict[str, JSONValue]:
        steps: list[dict[str, JSONValue]] = []
        for step in task.steps:
            steps.append(
                {
                    "step_id": step.step_id,
                    "allowed_tool_kinds": list(step.allowed_tool_kinds),
                }
            )
        return {
            "plan_id": task.task_id,
            "plan_hash": task.packet_hash,
            "steps": steps,
            "policy": dict(task.policy),
            "scope": dict(task.scope),
            "budgets": dict(task.budgets),
            "verifier": dict(task.verifier),
        }

    def _capture_runtime_execution_state(self) -> dict[str, JSONValue]:
        return {
            "mode": self.runtime_mode,
            "active_plan": dict(self.runtime_active_plan) if self.runtime_active_plan else None,
            "active_task": dict(self.runtime_active_task) if self.runtime_active_task else None,
            "auto_state": dict(self.runtime_auto_state) if self.runtime_auto_state else None,
            "enforce_plan_guard": self.runtime_plan_guard_enabled,
        }

    def _restore_runtime_execution_state(self, snapshot: dict[str, JSONValue]) -> None:
        mode_raw = snapshot.get("mode")
        mode = mode_raw if isinstance(mode_raw, str) else "ask"
        active_plan_raw = snapshot.get("active_plan")
        active_task_raw = snapshot.get("active_task")
        auto_state_raw = snapshot.get("auto_state")
        enforce = bool(snapshot.get("enforce_plan_guard"))
        self.set_runtime_state(
            mode=mode,
            active_plan=active_plan_raw if isinstance(active_plan_raw, dict) else None,
            active_task=active_task_raw if isinstance(active_task_raw, dict) else None,
            auto_state=auto_state_raw if isinstance(auto_state_raw, dict) else None,
            enforce_plan_guard=enforce,
        )

    def _resolve_packet_step_operation(
        self,
        step: TaskStepContract,
    ) -> tuple[str | None, str | None]:
        operation_raw = step.inputs.get("operation")
        operation = operation_raw.strip() if isinstance(operation_raw, str) else ""
        if not operation:
            if len(step.allowed_tool_kinds) == 1:
                operation = step.allowed_tool_kinds[0].strip()
        if not operation:
            if step.allowed_tool_kinds:
                return None, "step operation не определён при непустом allowed_tool_kinds."
            return None, None
        if operation not in step.allowed_tool_kinds:
            return None, f"operation '{operation}' не входит в allowed_tool_kinds."
        return operation, None

    def _stop_reason_from_execution_error(self, error_text: str) -> StopReasonCode:
        normalized = error_text.lower()
        if "blocked_outside_plan" in normalized:
            return StopReasonCode.REPLAN_REQUIRED
        if "plan_read_only_block" in normalized:
            return StopReasonCode.REPLAN_REQUIRED
        if "operation '" in normalized and "allowed_tool_kinds" in normalized:
            return StopReasonCode.REPLAN_REQUIRED
        if "packet_hash mismatch" in normalized:
            return StopReasonCode.REPLAN_REQUIRED
        return StopReasonCode.WORKER_FAILED

    def _mwv_changes_from_diffs(self, diffs: list[WorkspaceDiffEntry]) -> list[WorkChange]:
        changes: list[WorkChange] = []
        for diff in diffs:
            if diff.added > 0 and diff.removed == 0:
                change_type = ChangeType.CREATE
            elif diff.removed > 0 and diff.added == 0:
                change_type = ChangeType.DELETE
            else:
                change_type = ChangeType.UPDATE
            summary = f"+{diff.added}/-{diff.removed}"
            changes.append(WorkChange(path=diff.path, change_type=change_type, summary=summary))
        return changes

    def _build_mwv_diagnostics(self, plan: TaskPlan) -> dict[str, JSONValue]:
        errors: list[dict[str, JSONValue]] = []
        for step in plan.steps:
            status_value = step.status.value if hasattr(step.status, "value") else str(step.status)
            if status_value == "error":
                errors.append(
                    {
                        "description": step.description,
                        "result": step.result or "",
                    }
                )
        return {
            "steps_total": len(plan.steps),
            "step_errors": errors,
        }

    def _format_mwv_response(self, result: MWVRunResult) -> str:
        trace_id = result.task.trace_id
        report_plan_summary = self._mwv_plan_summary_for_report(result)
        report_execution_summary = self._mwv_execution_summary_for_report(result)
        if result.work_result.status == WorkStatus.FAILURE:
            summary = self._summarize_work_failure(result.work_result)
            stop_reason = self._mwv_stop_reason_from_work_result(result.work_result)
            next_steps = [
                "Проверь шаги выполнения и уточни запрос.",
                "Запусти задачу снова с более узким фокусом.",
            ]
            if stop_reason == StopReasonCode.REPLAN_REQUIRED:
                next_steps = [
                    "Вернись в Plan и выпусти новый TaskPacket revision.",
                    "Сохрани policy/scope без изменений и уточни недостающие входы.",
                    "После approve повторно запусти Act.",
                ]
            return self._format_stop_response(
                what="MWV остановлен: ошибка выполнения",
                why=summary,
                next_steps=next_steps,
                stop_reason_code=stop_reason,
                route="mwv",
                trace_id=trace_id,
                attempts=(result.attempt, result.max_attempts),
                verifier=result.verification_result,
                plan_summary=report_plan_summary,
                execution_summary=report_execution_summary,
            )
        if result.verification_result.status != VerificationStatus.PASSED:
            note = self._mwv_verifier_note(result.verification_result)
            status_label = (
                "Ошибка проверки"
                if result.verification_result.status == VerificationStatus.ERROR
                else "Проверки не прошли"
            )
            return self._format_stop_response(
                what=status_label,
                why=note,
                next_steps=[
                    "Открой trace по trace_id и посмотри детали проверки.",
                    "Запусти scripts/check.sh вручную для диагностики.",
                    "Исправь проблему и повтори запрос.",
                ],
                stop_reason_code=StopReasonCode.VERIFIER_FAILED,
                route="mwv",
                trace_id=trace_id,
                attempts=(result.attempt, result.max_attempts),
                verifier=result.verification_result,
                plan_summary=report_plan_summary,
                execution_summary=report_execution_summary,
            )
        outcome = self._mwv_outcome_label(result)
        lines = [
            f"Итог: {outcome}",
            f"Попытка: {result.attempt}/{result.max_attempts}",
            f"Verifier: {self._mwv_verifier_label(result.verification_result)}",
            "Изменения:",
            *self._format_mwv_changes(result.work_result.changes),
        ]

        if self._mwv_needs_next_steps(result):
            lines.append("Что дальше:")
            lines.extend(self._mwv_next_steps(result))

        details = self._mwv_details(result)
        if details:
            lines.append("Детали:")
            lines.extend(details)

        next_steps = self._mwv_next_steps(result) if self._mwv_needs_next_steps(result) else []
        return self._append_report_block(
            "\n".join(lines).strip(),
            route="mwv",
            trace_id=trace_id,
            attempts=(result.attempt, result.max_attempts),
            verifier=result.verification_result,
            next_steps=next_steps,
            stop_reason_code=None,
            plan_summary=report_plan_summary,
            execution_summary=report_execution_summary,
        )

    def _summarize_work_failure(self, work_result: WorkResult) -> str:
        errors = work_result.diagnostics.get("step_errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                description = str(first.get("description", "")).strip()
                result = str(first.get("result", "")).strip()
                if description and result:
                    return f"{description}: {result[:200]}"
                if description:
                    return description
        return "Не удалось выполнить шаги."

    def _mwv_stop_reason_from_work_result(self, work_result: WorkResult) -> StopReasonCode:
        reason_raw = work_result.diagnostics.get("stop_reason_code")
        if isinstance(reason_raw, str):
            try:
                return StopReasonCode(reason_raw)
            except ValueError:
                return StopReasonCode.WORKER_FAILED
        return StopReasonCode.WORKER_FAILED

    def _mwv_outcome_label(self, result: MWVRunResult) -> str:
        if result.work_result.status == WorkStatus.FAILURE:
            return "ошибка выполнения"
        if result.verification_result.status == VerificationStatus.ERROR:
            return "ошибка проверки"
        if result.verification_result.status == VerificationStatus.FAILED:
            if result.retry_decision and result.retry_decision.reason == "retry_limit_reached":
                return "проверки не прошли (лимит попыток исчерпан)"
            return "проверки не прошли"
        return "проверки пройдены"

    def _mwv_verifier_label(self, verification: VerificationResult) -> str:
        if verification.status == VerificationStatus.PASSED:
            base = "PASS"
        elif verification.status == VerificationStatus.FAILED:
            base = "FAIL"
        else:
            base = "ERROR"
        if verification.exit_code is None:
            return base
        return f"{base} (exit_code={verification.exit_code})"

    def _mwv_plan_summary_for_report(self, result: MWVRunResult) -> str:
        summary = result.work_result.summary.strip()
        if summary:
            first_line = summary.splitlines()[0].strip()
            if first_line:
                return first_line
        if result.work_result.changes:
            return f"Изменено файлов: {len(result.work_result.changes)}"
        return "План выполнен без изменений."

    def _mwv_execution_summary_for_report(self, result: MWVRunResult) -> str:
        return (
            f"worker={result.work_result.status.value}; "
            f"verifier={self._mwv_verifier_label(result.verification_result)}; "
            f"changes={len(result.work_result.changes)}"
        )

    def _format_mwv_changes(self, changes: list[WorkChange]) -> list[str]:
        if not changes:
            return ["- нет изменений"]
        labels = {
            ChangeType.CREATE: "создан",
            ChangeType.UPDATE: "изменен",
            ChangeType.DELETE: "удален",
            ChangeType.RENAME: "переименован",
        }
        lines: list[str] = []
        for change in changes:
            label = labels.get(change.change_type, change.change_type.value)
            suffix = f", {change.summary}" if change.summary else ""
            lines.append(f"- {change.path} ({label}{suffix})")
        return lines

    def _mwv_needs_next_steps(self, result: MWVRunResult) -> bool:
        return (
            result.work_result.status == WorkStatus.FAILURE
            or result.verification_result.status != VerificationStatus.PASSED
        )

    def _mwv_next_steps(self, result: MWVRunResult) -> list[str]:
        if result.work_result.status == WorkStatus.FAILURE:
            return [
                "- Уточни, какой шаг/файл важен.",
                "- Разреши повтор с более узким фокусом.",
            ]
        if result.verification_result.status == VerificationStatus.ERROR:
            return [
                "- Проверь окружение проверки и доступность скрипта.",
                "- Запусти scripts/check.sh вручную для деталей.",
            ]
        if result.retry_decision and result.retry_decision.reason == "retry_limit_reached":
            return [
                "- Сузь задачу или уточни требования.",
                "- Запусти scripts/check.sh вручную, чтобы понять, что падает.",
                "- Разреши дополнительную попытку, если нужно.",
            ]
        steps = [
            "- Посмотри краткие детали ниже и уточни требования.",
            "- Разреши дополнительную попытку, если нужно.",
        ]
        skill_note = self._mwv_skill_failure_note(result)
        if skill_note:
            steps.append(skill_note)
        return steps

    def _mwv_details(self, result: MWVRunResult) -> list[str]:
        details: list[str] = []
        plan_summary = result.work_result.summary.strip()
        if plan_summary:
            details.append("План:")
            details.extend(self._truncate_lines(plan_summary, max_lines=6, max_chars=160))
        verifier_note = self._mwv_verifier_note(result.verification_result)
        if verifier_note:
            details.append(f"Проверка: {verifier_note}")
        return details

    def _mwv_verifier_note(self, verification: VerificationResult) -> str:
        if verification.status == VerificationStatus.PASSED:
            return "ok"
        if verification.error:
            return verification.error
        text = (verification.stderr or verification.stdout or "").strip()
        if not text:
            if verification.exit_code is None:
                return "нет деталей"
            return f"exit_code={verification.exit_code}"
        return text.splitlines()[0][:160]

    def _handle_mwv_error(
        self,
        exc: Exception,
        *,
        raw_input: str,
        record_in_history: bool,
        trace_id: str,
    ) -> str:
        self.logger.exception("MWV flow error: %s", exc)
        self.tracer.log("mwv_error", str(exc), {"trace_id": trace_id})
        response = self._format_stop_response(
            what="MWV internal error",
            why=str(exc),
            next_steps=[
                "Проверь логи и trace по trace_id.",
                "Повтори запрос или уточни входные данные.",
            ],
            stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
            route="mwv",
            trace_id=trace_id,
        )
        if self.memory_config.auto_save_dialogue:
            self.save_to_memory(raw_input, response)
        self._log_chat_interaction(raw_input=raw_input, response_text=response)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=response)])
        return response

    def _truncate_lines(self, text: str, max_lines: int, max_chars: int) -> list[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return []
        trimmed = [line[:max_chars] for line in lines[:max_lines]]
        if len(lines) > max_lines:
            trimmed.append("...")
        return trimmed

    def _mwv_skill_failure_note(self, result: MWVRunResult) -> str | None:
        if result.verification_result.status == VerificationStatus.PASSED:
            return None
        raw_skill = result.task.context.get("skill_id")
        if not isinstance(raw_skill, str) or not raw_skill:
            return None
        return f"- Навык {raw_skill} не прошел проверку. Нужна доработка skill."
