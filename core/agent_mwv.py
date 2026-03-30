from __future__ import annotations

# ruff: noqa: F401
import shlex
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

from core.approval_policy import ApprovalCategory, ApprovalRequest, ApprovalRequired
from core.decision.handler import DecisionEvent
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
from core.mwv.verifier_runtime import (
    NON_REPO_VERIFIER_REQUIRED_ERROR,
    VerifierRuntime,
    canonical_check_command,
    has_canonical_repo_verifier,
)
from core.mwv.verifier_summary import extract_verifier_excerpt
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
_ACTION_ORIENTED_RISK_FLAGS = frozenset({"code_change", "filesystem", "git", "install", "sudo"})
_NO_OBSERVABLE_ACTION_ERROR = (
    "MWV route требует наблюдаемого действия, но tool calls/diffs отсутствуют."
)


class TaskPacketApprovalPending(Exception):
    def __init__(
        self,
        *,
        request: ApprovalRequest,
        blocked_step_id: str,
        step_results: list[dict[str, JSONValue]],
        changes: list[dict[str, JSONValue]],
        tool_calls_used: int,
        diff_size: int,
    ) -> None:
        super().__init__("TaskPacket execution requires approval.")
        self.request = request
        self.blocked_step_id = blocked_step_id
        self.step_results = step_results
        self.changes = changes
        self.tool_calls_used = tool_calls_used
        self.diff_size = diff_size


if TYPE_CHECKING:
    import logging

    from config.memory_config import MemoryConfig
    from core.decision.handler import DecisionHandler
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
        decision_handler: DecisionHandler
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
        runtime_workspace_root: str | None
        _last_skill_match: SkillMatch | None
        _workspace_diffs: dict[str, WorkspaceDiffEntry]

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
        def apply_runtime_workspace_root(self, workspace_root: str | None) -> None: ...
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
            decision_packet = self.decision_handler.evaluate(
                event=DecisionEvent.verifier_fail(
                    verification_result=run_result.verification_result,
                    task_id=run_result.task.task_id,
                    trace_id=run_result.task.trace_id,
                    attempt=run_result.attempt,
                    max_attempts=run_result.max_attempts,
                    retry_allowed=bool(
                        run_result.retry_decision and run_result.retry_decision.allow_retry
                    ),
                )
            )
            if decision_packet is None:
                raise RuntimeError("DecisionHandler did not build verifier_fail packet.")
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
        workspace_root = self.runtime_workspace_root or _workspace_root()
        return RunContext(
            session_id=session_id,
            trace_id=resolved_trace_id,
            workspace_root=workspace_root,
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
                verifier=self._build_packet_verifier_config(context.workspace_root),
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
        started = time.monotonic()
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
                elapsed_ms=int((time.monotonic() - started) * 1000),
                files_touched=0,
                tool_calls_used=0,
                diff_size=0,
                root_cause_tag="packet_hash_mismatch",
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
                elapsed_ms=int((time.monotonic() - started) * 1000),
                files_touched=0,
                tool_calls_used=0,
                diff_size=0,
                root_cause_tag="missing_steps",
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
                elapsed_ms=int((time.monotonic() - started) * 1000),
                files_touched=0,
                tool_calls_used=0,
                diff_size=0,
                root_cause_tag="scope_error",
            )

        resume_state = self._mwv_resume_state(task)
        prior_step_results_raw = resume_state.get("step_results")
        prior_step_results = (
            [dict(item) for item in prior_step_results_raw if isinstance(item, dict)]
            if isinstance(prior_step_results_raw, list)
            else []
        )
        prior_changes_raw = resume_state.get("changes")
        prior_changes = (
            [
                change
                for change in (self._work_change_from_json(item) for item in prior_changes_raw)
                if change is not None
            ]
            if isinstance(prior_changes_raw, list)
            else []
        )
        prior_tool_calls_raw = resume_state.get("tool_calls_used")
        prior_tool_calls = (
            prior_tool_calls_raw
            if isinstance(prior_tool_calls_raw, int) and prior_tool_calls_raw >= 0
            else 0
        )
        prior_diff_size_raw = resume_state.get("diff_size")
        prior_diff_size = (
            prior_diff_size_raw
            if isinstance(prior_diff_size_raw, int) and prior_diff_size_raw >= 0
            else 0
        )

        runtime_snapshot = self._capture_runtime_execution_state()
        plan_snapshot = self._build_packet_plan_snapshot(task)
        step_states = [
            plan_step
            for plan_step in (self._plan_step_from_snapshot(item) for item in prior_step_results)
            if plan_step is not None
        ]
        step_results = list(prior_step_results)
        completed_step_ids = {
            str(item.get("step_id"))
            for item in prior_step_results
            if item.get("status") == "done" and isinstance(item.get("step_id"), str)
        }
        stop_reason = StopReasonCode.WORKER_FAILED
        action_oriented = self._mwv_is_action_oriented_task(task)
        successful_tool_calls = prior_tool_calls

        try:
            with workspace_root_context(workspace_root):

                def _mwv_post_call(
                    request: ToolRequest,
                    result: ToolResult,
                    call_context: object | None,
                ) -> None:
                    nonlocal successful_tool_calls
                    self._workspace_diff_post_call(request, result, call_context)
                    if result.ok:
                        successful_tool_calls += 1

                for step in task.steps:
                    if step.step_id in completed_step_ids:
                        continue
                    require_operation = self._mwv_step_requires_operation(
                        step,
                        action_oriented=action_oriented,
                    )
                    tool_calls_before = successful_tool_calls
                    diff_totals_before = self._workspace_diff_totals()
                    operation, operation_error = self._resolve_packet_step_operation(
                        step,
                        require_operation=require_operation,
                    )
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
                        step_results.append(
                            self._step_result_snapshot(
                                step_id=step.step_id,
                                description=step.description,
                                status="failed",
                                operation=operation,
                                result=operation_error,
                                tool_calls_used=0,
                                changes=[],
                            )
                        )
                        stop_reason = StopReasonCode.REPLAN_REQUIRED
                        break
                    step_plan = TaskPlan(goal=task.goal, steps=[plan_step])
                    try:
                        executed = self.executor.run(
                            step_plan,
                            tool_gateway=self._build_tool_gateway(
                                pre_call=self._workspace_diff_pre_call,
                                post_call=_mwv_post_call,
                            ),
                        )
                    except ApprovalRequired as exc:
                        step_results.append(
                            self._step_result_snapshot(
                                step_id=step.step_id,
                                description=step.description,
                                status="waiting_approval",
                                operation=operation,
                                result="Требуется подтверждение",
                                tool_calls_used=0,
                                changes=[],
                            )
                        )
                        current_diff_entries = self._workspace_diff_entries_delta(
                            {},
                            self._workspace_diff_totals(),
                        )
                        current_changes = self._mwv_changes_from_diffs(current_diff_entries)
                        current_diff_size = sum(
                            max(0, diff.added) + max(0, diff.removed)
                            for diff in current_diff_entries
                        )
                        raise TaskPacketApprovalPending(
                            request=exc.request,
                            blocked_step_id=step.step_id,
                            step_results=[
                                item for item in step_results if item.get("status") == "done"
                            ],
                            changes=[
                                self._work_change_to_json(change)
                                for change in self._merge_work_changes(
                                    prior_changes,
                                    current_changes,
                                )
                            ],
                            tool_calls_used=successful_tool_calls,
                            diff_size=prior_diff_size + current_diff_size,
                        ) from exc
                    executed_step = executed.steps[0]
                    step_states.append(executed_step)
                    step_diff_entries = self._workspace_diff_entries_delta(
                        diff_totals_before,
                        self._workspace_diff_totals(),
                    )
                    step_changes = self._mwv_changes_from_diffs(step_diff_entries)
                    step_results.append(
                        self._step_result_snapshot(
                            step_id=step.step_id,
                            description=step.description,
                            status="failed"
                            if executed_step.status == PlanStepStatus.ERROR
                            else "done",
                            operation=operation,
                            result=executed_step.result or "",
                            tool_calls_used=successful_tool_calls - tool_calls_before,
                            changes=step_changes,
                        )
                    )
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
        diff_entries = self.consume_workspace_diffs()
        changes = self._merge_work_changes(
            prior_changes,
            self._mwv_changes_from_diffs(diff_entries),
        )
        diagnostics = self._build_mwv_diagnostics(executed_plan)
        diagnostics["step_results"] = step_results
        diagnostics["tool_calls_ok"] = successful_tool_calls
        diagnostics["changes_total"] = len(changes)
        diagnostics["risk_flags"] = sorted(self._mwv_risk_flags(task))
        diff_size = prior_diff_size + sum(
            max(0, diff.added) + max(0, diff.removed) for diff in diff_entries
        )
        root_cause_tag = "success"
        if (
            status == WorkStatus.SUCCESS
            and action_oriented
            and successful_tool_calls == 0
            and not changes
        ):
            status = WorkStatus.FAILURE
            stop_reason = StopReasonCode.REPLAN_REQUIRED
            step_errors = diagnostics.get("step_errors")
            if isinstance(step_errors, list):
                step_errors.append(
                    {
                        "description": "MWV execution",
                        "result": _NO_OBSERVABLE_ACTION_ERROR,
                    }
                )
            root_cause_tag = "no_observable_action"
        if status == WorkStatus.FAILURE:
            diagnostics["stop_reason_code"] = stop_reason.value
            if root_cause_tag == "success":
                root_cause_tag = stop_reason.value.lower()
        summary = self._format_plan(executed_plan)
        return WorkResult(
            task_id=task.task_id,
            status=status,
            summary=summary,
            changes=changes,
            tool_summaries=[],
            diagnostics=diagnostics,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            files_touched=len({change.path for change in changes}),
            tool_calls_used=successful_tool_calls,
            diff_size=diff_size,
            root_cause_tag=root_cause_tag,
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

    def run_task_packet(self, packet: TaskPacket, context: RunContext) -> MWVRunResult:
        verifier_runtime = _verifier_runtime_cls()()
        worker = WorkerRuntime(runner=self._mwv_worker_runner)
        manager = _manager_runtime_cls()(task_builder=lambda _messages, _context: packet)

        def _worker(task: TaskPacket, run_context: RunContext) -> WorkResult:
            return worker.run(task, run_context)

        def _verifier(task: TaskPacket, run_context: RunContext) -> VerificationResult:
            try:
                return verifier_runtime.run(task, run_context)
            except TypeError:
                legacy_run = cast(Callable[[RunContext], VerificationResult], verifier_runtime.run)
                return legacy_run(run_context)

        return manager.run_flow(packet.messages, context, worker=_worker, verifier=_verifier)

    def _mwv_resume_state(self, task: TaskPacket) -> dict[str, JSONValue]:
        raw = task.context.get("plan_runner_resume")
        return dict(raw) if isinstance(raw, dict) else {}

    def _work_change_to_json(self, change: WorkChange) -> dict[str, JSONValue]:
        return {
            "path": change.path,
            "change_type": change.change_type.value,
            "summary": change.summary,
        }

    def _work_change_from_json(self, raw: object) -> WorkChange | None:
        if not isinstance(raw, dict):
            return None
        path_raw = raw.get("path")
        change_type_raw = raw.get("change_type")
        summary_raw = raw.get("summary")
        if not isinstance(path_raw, str) or not path_raw.strip():
            return None
        if not isinstance(change_type_raw, str):
            return None
        try:
            change_type = ChangeType(change_type_raw)
        except ValueError:
            return None
        summary = summary_raw if isinstance(summary_raw, str) else ""
        return WorkChange(path=path_raw.strip(), change_type=change_type, summary=summary)

    def _merge_work_changes(
        self,
        prior_changes: list[WorkChange],
        current_changes: list[WorkChange],
    ) -> list[WorkChange]:
        merged: dict[str, WorkChange] = {change.path: change for change in prior_changes}
        for change in current_changes:
            merged[change.path] = change
        ordered_paths = [change.path for change in prior_changes]
        existing = set(ordered_paths)
        for change in current_changes:
            if change.path not in existing:
                ordered_paths.append(change.path)
                existing.add(change.path)
        seen: set[str] = set()
        ordered: list[WorkChange] = []
        for path in [*ordered_paths, *merged.keys()]:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(merged[path])
        return ordered

    def _step_result_snapshot(
        self,
        *,
        step_id: str,
        description: str,
        status: str,
        operation: str | None,
        result: str,
        tool_calls_used: int,
        changes: list[WorkChange],
    ) -> dict[str, JSONValue]:
        return {
            "step_id": step_id,
            "description": description,
            "status": status,
            "operation": operation,
            "result": result,
            "tool_calls_used": tool_calls_used,
            "changes": [self._work_change_to_json(change) for change in changes],
        }

    def _plan_step_from_snapshot(self, snapshot: dict[str, JSONValue]) -> PlanStep | None:
        status_raw = snapshot.get("status")
        description_raw = snapshot.get("result")
        if not isinstance(status_raw, str):
            return None
        if status_raw == "done":
            status = PlanStepStatus.DONE
        elif status_raw in {"failed", "waiting_approval"}:
            status = PlanStepStatus.ERROR
        else:
            return None
        operation_raw = snapshot.get("operation")
        return PlanStep(
            description=str(snapshot.get("description") or snapshot.get("step_id") or ""),
            operation=operation_raw if isinstance(operation_raw, str) else None,
            status=status,
            result=description_raw if isinstance(description_raw, str) else "",
        )

    def _workspace_diff_totals(self) -> dict[str, tuple[int, int]]:
        return {path: (entry.added, entry.removed) for path, entry in self._workspace_diffs.items()}

    def _workspace_diff_entries_delta(
        self,
        before: dict[str, tuple[int, int]],
        after: dict[str, tuple[int, int]],
    ) -> list[WorkspaceDiffEntry]:
        deltas: list[WorkspaceDiffEntry] = []
        for path, (added_after, removed_after) in after.items():
            added_before, removed_before = before.get(path, (0, 0))
            added_delta = max(0, added_after - added_before)
            removed_delta = max(0, removed_after - removed_before)
            if added_delta == 0 and removed_delta == 0:
                continue
            deltas.append(
                WorkspaceDiffEntry(path=path, added=added_delta, removed=removed_delta, diff="")
            )
        return deltas

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
            "workspace_root": self.runtime_workspace_root,
        }

    def _restore_runtime_execution_state(self, snapshot: dict[str, JSONValue]) -> None:
        mode_raw = snapshot.get("mode")
        mode = mode_raw if isinstance(mode_raw, str) else "ask"
        active_plan_raw = snapshot.get("active_plan")
        active_task_raw = snapshot.get("active_task")
        auto_state_raw = snapshot.get("auto_state")
        enforce = bool(snapshot.get("enforce_plan_guard"))
        workspace_root_raw = snapshot.get("workspace_root")
        self.apply_runtime_workspace_root(
            workspace_root_raw if isinstance(workspace_root_raw, str) else None
        )
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
        *,
        require_operation: bool,
    ) -> tuple[str | None, str | None]:
        allowed_tool_kinds = [item.strip() for item in step.allowed_tool_kinds if item.strip()]
        operation_raw = step.inputs.get("operation")
        operation = operation_raw.strip() if isinstance(operation_raw, str) else ""
        if not operation:
            if len(allowed_tool_kinds) == 1:
                operation = allowed_tool_kinds[0]
            elif len(allowed_tool_kinds) > 1:
                return None, "step operation неоднозначен: нужно явно выбрать operation."
        if not operation:
            if require_operation:
                return None, "step operation не определён для исполнимого MWV-шага."
            if allowed_tool_kinds:
                return None, "step operation не определён при непустом allowed_tool_kinds."
            return None, None
        if not allowed_tool_kinds:
            if require_operation:
                return None, "allowed_tool_kinds пуст для исполнимого MWV-шага."
            return operation, None
        if operation not in allowed_tool_kinds:
            return None, f"operation '{operation}' не входит в allowed_tool_kinds."
        return operation, None

    def _mwv_risk_flags(self, task: TaskPacket) -> set[str]:
        raw = task.context.get("risk_flags")
        if not isinstance(raw, list):
            return set()
        return {item.strip() for item in raw if isinstance(item, str) and item.strip()}

    def _mwv_is_action_oriented_task(self, task: TaskPacket) -> bool:
        return bool(self._mwv_risk_flags(task) & _ACTION_ORIENTED_RISK_FLAGS)

    def _mwv_step_requires_operation(
        self,
        step: TaskStepContract,
        *,
        action_oriented: bool,
    ) -> bool:
        if action_oriented:
            return True
        operation_raw = step.inputs.get("operation")
        if isinstance(operation_raw, str) and operation_raw.strip():
            return True
        return bool([item for item in step.allowed_tool_kinds if item.strip()])

    def _build_packet_verifier_config(self, workspace_root: str) -> dict[str, JSONValue]:
        root = Path(workspace_root).resolve()
        if has_canonical_repo_verifier(root):
            return {
                "command": canonical_check_command(),
                "cwd": ".",
            }
        return {}

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
                next_steps=self._mwv_verifier_stop_next_steps(result.verification_result),
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
            return self._mwv_bulletize(
                self._mwv_verifier_stop_next_steps(result.verification_result)[:2]
            )
        if result.retry_decision and result.retry_decision.reason == "retry_limit_reached":
            steps = ["Сузь задачу или уточни требования."]
            command = self._mwv_verifier_manual_command(result.verification_result)
            if command:
                steps.append(f"Запусти вручную: {command}.")
            else:
                steps.append("Проверь verifier.command и рабочую директорию проверки.")
            steps.append("Разреши дополнительную попытку, если нужно.")
            return self._mwv_bulletize(steps)
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
        if verification.error == NON_REPO_VERIFIER_REQUIRED_ERROR:
            return (
                "Для non-repo workspace нужен явный verifier.command или repo-like workspace_root."
            )
        return extract_verifier_excerpt(verification, max_lines=3, max_chars=300)

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

    def _mwv_verifier_manual_command(self, verification: VerificationResult) -> str | None:
        if not verification.command:
            return None
        return shlex.join(verification.command)

    def _mwv_verifier_stop_next_steps(self, verification: VerificationResult) -> list[str]:
        command = self._mwv_verifier_manual_command(verification)
        if verification.error == NON_REPO_VERIFIER_REQUIRED_ERROR:
            return [
                "Укажи verifier.command для этого workspace или выбери repo-like workspace_root.",
                "Повтори запрос после настройки проверки.",
            ]
        steps = ["Открой trace по trace_id и посмотри детали проверки."]
        if command:
            steps.append(f"Запусти вручную: {command}.")
        else:
            steps.append("Проверь verifier.command и рабочую директорию проверки.")
        steps.append("Исправь проблему и повтори запрос.")
        return steps

    def _mwv_bulletize(self, steps: list[str]) -> list[str]:
        return [f"- {item}" for item in steps]
