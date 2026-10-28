from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core.approval_policy import ApprovalRequired
from core.mwv.models import (
    RunContext,
    StopReasonCode,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
)
from core.mwv.verifier_runtime import (
    VerifierRuntime,
    canonical_check_command,
    has_canonical_repo_verifier,
    is_repo_workspace,
)
from core.mwv.verifier_summary import extract_verifier_excerpt
from core.response_preferences import compose_response_preferences
from core.skills.index import SkillResolution
from core.skills.runtime import skill_run_metadata, skill_run_report
from core.tool_loop import (
    AgentToolLoop,
    AgentToolLoopResult,
    ExecutedToolCall,
    ModelUsageSummary,
    ToolLoopContinuation,
    ToolLoopExecutionError,
)
from llm.brain_base import Brain
from shared.auto_models import (
    AutoPlan,
    AutoRunStatus,
    AutoShard,
    normalize_auto_state,
    utc_now_iso,
)
from shared.models import JSONValue, LLMMessage
from tools.workspace_tools import WORKSPACE_ROOT, get_workspace_root

if TYPE_CHECKING:
    from core.agent import Agent


AUTO_MAX_RUNTIME_SECONDS_ENV = "AUTO_MAX_RUNTIME_SECONDS"
AUTO_MAX_TOOL_CALLS_ENV = "AUTO_MAX_TOOL_CALLS"
AUTO_MAX_FILES_TOUCHED_ENV = "AUTO_MAX_FILES_TOUCHED"
AUTO_MAX_LLM_TOKENS_ENV = "AUTO_MAX_LLM_TOKENS"
AUTO_MAX_RETRIES_ENV = "AUTO_MAX_RETRIES"
AUTO_DEFAULT_MAX_RUNTIME_SECONDS = 900
AUTO_DEFAULT_MAX_TOOL_CALLS = 80
AUTO_DEFAULT_MAX_FILES_TOUCHED = 120
AUTO_DEFAULT_MAX_LLM_TOKENS = 120_000
AUTO_DEFAULT_MAX_RETRIES = 0
AUTO_V1_SYSTEM_PROMPT = (
    "You are running Auto v1. Decide from the user's request whether workspace action is "
    "actually required. For conversation or information that needs no action, answer directly "
    "without tool calls. For every workspace action, use native tool calls and never invent "
    "filesystem results. Stop with a concise final answer after any tools finish."
)


@dataclass(frozen=True)
class AutoRunOutcome:
    text: str
    status: AutoRunStatus
    stop_reason_code: StopReasonCode | None
    verifier: VerificationResult | None
    next_steps: list[str]


@dataclass(frozen=True)
class AutoBudgets:
    max_runtime_seconds: int
    max_tool_calls: int
    max_files_touched: int
    max_llm_tokens: int
    max_retries: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_tool_calls": self.max_tool_calls,
            "max_files_touched": self.max_files_touched,
            "max_llm_tokens": self.max_llm_tokens,
            "max_retries": self.max_retries,
        }


@dataclass
class _RunFrame:
    run_id: str
    goal: str
    started_at: str
    started_monotonic: float
    workspace_root: Path
    budgets: AutoBudgets
    state: dict[str, JSONValue]
    skill_resolution: SkillResolution | None
    brain: Brain
    principal_id: str
    session_id: str | None
    runtime_root: str | None


@dataclass
class _PausedRun:
    frame: _RunFrame
    continuation: ToolLoopContinuation


class AutoContinuationUnavailable(ValueError):
    pass


class AutoOrchestrator:
    def __init__(
        self,
        parent_agent: Agent,
        *,
        workspace_root: Path | None = None,
        progress_callback: Callable[[dict[str, JSONValue]], None] | None = None,
    ) -> None:
        self.parent = parent_agent
        self.workspace_root = (workspace_root or WORKSPACE_ROOT).resolve()
        self.progress_callback = progress_callback
        self._paused_runs: dict[str, _PausedRun] = {}

    def run_v1(
        self,
        goal: str,
        *,
        skill_resolution: SkillResolution | None = None,
        run_root_override: Path | None = None,
    ) -> AutoRunOutcome:
        run_id_value = f"auto-{uuid.uuid4().hex}"
        started = utc_now_iso()
        started_monotonic = time.monotonic()
        budgets = _resolve_auto_budgets()
        runtime_root = get_workspace_root().resolve()
        if run_root_override is not None:
            run_root = run_root_override.resolve()
        elif runtime_root != WORKSPACE_ROOT:
            run_root = runtime_root
        else:
            run_root = self.workspace_root
        state: dict[str, JSONValue] = {
            "run_id": run_id_value,
            "status": AutoRunStatus.IDLE.value,
            "goal": goal,
            "root_path": str(run_root),
            "pool_size": 1,
            "started_at": started,
            "updated_at": started,
            "planner": {"status": "idle", "runtime": "auto_v1_tool_loop"},
            "plan": None,
            "coders": [],
            "merge": {"status": "idle", "changed_paths": []},
            "execution_metrics": None,
            "budgets": budgets.to_dict(),
            "verifier": None,
            "approval": None,
            "error": None,
            "error_code": None,
            "missing_paths": [],
            "skill": (
                skill_run_metadata(skill_resolution)
                if skill_resolution is not None
                else skill_run_report(None, status="skipped", reason="no_match")
            ),
        }
        self._set_state(state)
        frame = _RunFrame(
            run_id_value,
            goal,
            started,
            started_monotonic,
            run_root,
            budgets,
            state,
            skill_resolution,
            self.parent._get_main_brain(),
            self.parent.user_id,
            self.parent.session_id,
            self.parent.runtime_workspace_root,
        )
        return self._execute_frame(frame)

    def _execute_frame(
        self,
        frame: _RunFrame,
        *,
        continuation: ToolLoopContinuation | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> AutoRunOutcome:
        goal, run_id_value = frame.goal, frame.run_id
        run_root, budgets, state = frame.workspace_root, frame.budgets, frame.state
        started_monotonic, skill_resolution = frame.started_monotonic, frame.skill_resolution
        try:
            brain = frame.brain
            capability = brain.native_tool_capability()
            if not capability.allowed:
                reason = capability.reason
                state["error"] = reason
                state["error_code"] = "native_tools_required"
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: native tools недоступны",
                        why=reason,
                        next_steps=["Выбери модель с доступными native tools и повтори запуск."],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto требует native tool-calling provider.",
                        execution_summary=reason,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=["Выбери модель с доступными native tools и повтори запуск."],
                )
            self._set_status(state, AutoRunStatus.PLANNING)
            budget_stop = _budget_runtime_stop(
                budgets=budgets,
                started_monotonic=started_monotonic,
            )
            if budget_stop is not None:
                state["error"] = budget_stop
                state["error_code"] = "budget_runtime"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return _budget_stop_outcome(
                    self.parent,
                    budget_stop,
                    skill=_skill_state(state),
                )

            if continuation is None:
                plan = AutoPlan(
                    plan_id=f"plan-{uuid.uuid4().hex}",
                    goal=goal,
                    shards=[
                        AutoShard(
                            shard_id="tool-loop",
                            goal=goal,
                            path_scope=["."],
                            acceptance_checks=[
                                "Native tool loop completed",
                                "Canonical verifier completed",
                            ],
                        )
                    ],
                )
                state["planner"] = {
                    "status": "completed",
                    "runtime": "auto_v1_tool_loop",
                    "shards_total": 1,
                }
                state["plan"] = plan.to_dict()
                self._set_state(state)

            self._set_status(state, AutoRunStatus.CODING)
            gateway = self.parent._build_tool_gateway()
            tool_specs = self.parent.tool_registry.list_tool_specs()
            system_prompt = AUTO_V1_SYSTEM_PROMPT
            if skill_resolution is not None:
                system_prompt = f"{system_prompt}\n\n{skill_resolution.system_instruction()}"
            loop_result = AgentToolLoop(
                max_iterations=budgets.max_tool_calls,
                max_tool_calls=budgets.max_tool_calls,
                max_model_tokens=budgets.max_llm_tokens,
                expires_at_monotonic=started_monotonic + budgets.max_runtime_seconds,
            ).run(
                brain=brain,
                gateway=gateway,
                messages=compose_response_preferences(
                    [
                        LLMMessage(role="system", content=system_prompt),
                        LLMMessage(role="user", content=goal),
                    ],
                    self.parent.response_preferences,
                ),
                tools=tool_specs,
                config=self.parent.main_config,
                continuation=continuation,
                cancellation_token=cancellation_token,
                record_model_usage=True,
            )
            tool_call_states = [_auto_v1_tool_call_state(item) for item in loop_result.tool_calls]
            state["coders"] = tool_call_states
            state["merge"] = {
                "status": "completed",
                "changed_paths": [],
                "runtime": "auto_v1_tool_loop",
            }
            state["execution_metrics"] = {
                "tool_calls_used": len(loop_result.tool_calls),
                "iterations": loop_result.iterations,
                "files_touched": 0,
            }
            _record_model_usage(state, loop_result.model_usage)
            self._set_state(state)
            if loop_result.cancelled or (
                cancellation_token is not None and cancellation_token.is_set()
            ):
                state["error"] = "cancelled_by_user"
                self._set_status(state, AutoRunStatus.CANCELLED)
                return AutoRunOutcome(
                    text="Auto-run отменён; дальнейшие tools не выполняются.",
                    status=AutoRunStatus.CANCELLED,
                    stop_reason_code=None,
                    verifier=None,
                    next_steps=[],
                )
            if getattr(loop_result, "error", None) == "runtime_budget_exhausted":
                reason = "Auto run expired; no further operation was started."
                state["error"] = reason
                state["error_code"] = "budget_runtime"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return _budget_stop_outcome(self.parent, reason, skill=_skill_state(state))
            if getattr(loop_result, "error", None) == "tool_call_budget_exhausted":
                reason = (
                    f"Budget exhausted before next native tool call: "
                    f"completed_calls={len(loop_result.tool_calls)}, "
                    f"max_tool_calls={budgets.max_tool_calls}"
                )
                state["error"] = reason
                state["error_code"] = "budget_tool_calls"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return _budget_stop_outcome(
                    self.parent,
                    reason,
                    skill=_skill_state(state),
                )
            if loop_result.error in {"model_usage_unknown", "model_token_budget_exhausted"}:
                unknown = loop_result.error == "model_usage_unknown"
                reason = (
                    "Reported usage предыдущего вызова модели неизвестен; "
                    "следующий model request с token limit не начат."
                    if unknown
                    else "Reported token threshold достигнут; следующий model request не начат."
                )
                state["error"] = reason
                state["error_code"] = "budget_tokens_unknown" if unknown else "budget_tokens"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return _budget_stop_outcome(
                    self.parent,
                    reason,
                    skill=_skill_state(state),
                    usage_unknown=unknown,
                )
            loop_error = getattr(loop_result, "error", None)
            if loop_error is not None:
                state["error"] = loop_error
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: tool loop failed",
                        why=loop_error,
                        next_steps=[
                            "Проверь auto_state.coders.",
                            "Исправь причину ошибки tool call и перезапусти auto.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto v1 выполнил задачу через native tool loop.",
                        execution_summary=loop_error,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Проверь auto_state.coders.",
                        "Исправь причину ошибки tool call и перезапусти auto.",
                    ],
                )

            if _has_failed_tool_call(loop_result.tool_calls):
                reason = (
                    "Auto-run остановлен: tool call завершился ошибкой, а verifier "
                    "evidence выполнения исходного действия отсутствует."
                )
                state["error"] = reason
                state["error_code"] = "tool_failure_unrecovered"
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: неустранённый сбой tool call",
                        why=reason,
                        next_steps=[
                            "Проверь auto_state.coders на failed outcome.",
                            "Докажи выполнение действия verifier-ом или перезапусти auto.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto v1 обнаружил неустранённый сбой tool call.",
                        execution_summary=reason,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Проверь auto_state.coders на failed outcome.",
                        "Докажи выполнение действия verifier-ом или перезапусти auto.",
                    ],
                )

            self._set_status(state, AutoRunStatus.VERIFYING)
            if cancellation_token is not None and cancellation_token.is_set():
                state["error"] = "cancelled_by_user"
                self._set_status(state, AutoRunStatus.CANCELLED)
                return AutoRunOutcome(
                    text="Auto-run отменён; verifier не запускался.",
                    status=AutoRunStatus.CANCELLED,
                    stop_reason_code=None,
                    verifier=None,
                    next_steps=[],
                )
            budget_stop = _budget_runtime_stop(budgets=budgets, started_monotonic=started_monotonic)
            if budget_stop is not None:
                state["error"] = budget_stop
                state["error_code"] = "budget_runtime"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return _budget_stop_outcome(self.parent, budget_stop, skill=_skill_state(state))
            verification = self._run_verifier(
                run_id=run_id_value,
                goal=goal,
                run_root=run_root,
                budgets=budgets,
                loop_result=loop_result,
                cancellation_token=cancellation_token,
            )
            state["verifier"] = _verification_state(verification)
            self._set_state(state)
            if cancellation_token is not None and cancellation_token.is_set():
                state["error"] = "cancelled_by_user"
                self._set_status(state, AutoRunStatus.CANCELLED)
                return AutoRunOutcome(
                    text="Auto-run отменён.",
                    status=AutoRunStatus.CANCELLED,
                    stop_reason_code=None,
                    verifier=verification,
                    next_steps=[],
                )
            if verification.status != VerificationStatus.PASSED:
                self._set_status(state, AutoRunStatus.FAILED_VERIFIER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: verifier не прошёл",
                        why=_verifier_reason(verification),
                        next_steps=[
                            "Открой verifier stdout/stderr в отчёте.",
                            "Исправь проблему и перезапусти auto.",
                        ],
                        stop_reason_code=StopReasonCode.VERIFIER_FAILED,
                        route="auto",
                        verifier=verification,
                        attempts=(1, 1),
                        plan_summary="Auto v1 выполнил native tool loop и дошёл до verifier.",
                        execution_summary="Tool loop завершён, но verifier вернул fail/error.",
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_VERIFIER,
                    stop_reason_code=StopReasonCode.VERIFIER_FAILED,
                    verifier=verification,
                    next_steps=[
                        "Открой verifier stdout/stderr в отчёте.",
                        "Исправь проблему и перезапусти auto.",
                    ],
                )

            self._set_status(state, AutoRunStatus.COMPLETED)
            response_only = not loop_result.tool_calls
            text = self.parent._append_report_block(
                loop_result.text,
                route="auto",
                trace_id=None,
                attempts=(1, 1),
                verifier=verification,
                next_steps=[],
                stop_reason_code=None,
                plan_summary=(
                    "Auto v1 определил, что workspace-действия не требуются."
                    if response_only
                    else "Auto v1 использовал AgentToolLoop и ToolGateway."
                ),
                execution_summary=(
                    "Модель вернула final response без tool calls."
                    if response_only
                    else (
                        f"Native tool loop iterations={loop_result.iterations}, "
                        f"tool_calls={len(loop_result.tool_calls)}."
                    )
                ),
                skill=_skill_state(state),
            )
            return AutoRunOutcome(
                text=text,
                status=AutoRunStatus.COMPLETED,
                stop_reason_code=None,
                verifier=verification,
                next_steps=[],
            )
        except ApprovalRequired as exc:
            if exc.continuation is None:
                raise AutoContinuationUnavailable("auto_continuation_unavailable") from exc
            _record_model_usage(state, exc.continuation.model_usage)
            if cancellation_token is not None and cancellation_token.is_set():
                state["coders"] = [
                    _auto_v1_tool_call_state(item) for item in exc.continuation.executed
                ]
                state["error"] = "cancelled_by_user"
                self._set_status(state, AutoRunStatus.CANCELLED)
                return AutoRunOutcome(
                    text="Auto-run отменён.",
                    status=AutoRunStatus.CANCELLED,
                    stop_reason_code=None,
                    verifier=None,
                    next_steps=[],
                )
            self._paused_runs[run_id_value] = _PausedRun(frame, exc.continuation)
            state["coders"] = [_auto_v1_tool_call_state(item) for item in exc.continuation.executed]
            state["approval"] = {
                "status": "required",
                "required_categories": list(exc.request.required_categories),
                "tool": exc.request.tool,
                "details": dict(exc.request.details),
                "resume_token": run_id_value,
            }
            self._set_status(state, AutoRunStatus.WAITING_APPROVAL)
            raise
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, ToolLoopExecutionError):
                _record_model_usage(state, exc.result.model_usage)
                state["coders"] = [_auto_v1_tool_call_state(item) for item in exc.result.tool_calls]
            state["error"] = str(exc)
            self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
            return AutoRunOutcome(
                text=self.parent._format_stop_response(
                    what="Auto-run остановлен: внутренняя ошибка",
                    why=str(exc),
                    next_steps=[
                        "Проверь логи и trace.",
                        "Повтори запуск auto после исправления.",
                    ],
                    stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                    route="auto",
                    plan_summary="Auto v1 tool loop завершился внутренней ошибкой.",
                    execution_summary=str(exc),
                    skill=_skill_state(state),
                ),
                status=AutoRunStatus.FAILED_INTERNAL,
                stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                verifier=None,
                next_steps=[
                    "Проверь логи и trace.",
                    "Повтори запуск auto после исправления.",
                ],
            )

    def pending_run_id(self, continuation: ToolLoopContinuation) -> str:
        for run_id, paused in self._paused_runs.items():
            if paused.continuation is continuation:
                return run_id
        raise AutoContinuationUnavailable("auto_continuation_unavailable")

    def validate_resume(
        self, run_id: str, continuation: ToolLoopContinuation | None = None
    ) -> None:
        paused = self._paused_runs.get(run_id)
        if paused is None:
            raise AutoContinuationUnavailable("auto_continuation_unavailable")
        frame = paused.frame
        if (
            (continuation is not None and paused.continuation is not continuation)
            or self.parent.session_id != frame.session_id
            or self.parent.user_id != frame.principal_id
            or self.parent.runtime_mode != "auto"
            or self.parent.runtime_workspace_root != frame.runtime_root
            or self.parent._get_main_brain() is not frame.brain
            or self.parent.main_config != paused.continuation.config
        ):
            raise AutoContinuationUnavailable("auto_continuation_unavailable")

    def resume(
        self,
        run_id: str,
        *,
        cancellation_token: asyncio.Event | None = None,
    ) -> AutoRunOutcome:
        self.validate_resume(run_id)
        if cancellation_token is not None and cancellation_token.is_set():
            raise AutoContinuationUnavailable("auto_continuation_not_started")
        paused = self._paused_runs.pop(run_id)
        paused.frame.state["approval"] = None
        return self._execute_frame(
            paused.frame,
            continuation=paused.continuation,
            cancellation_token=cancellation_token,
        )

    def cancel(
        self,
        run_id: str,
        *,
        reason: str = "cancelled_by_user",
    ) -> dict[str, JSONValue] | None:
        paused = self._paused_runs.pop(run_id, None)
        if paused is None:
            return None
        state = dict(paused.frame.state)
        state["status"] = AutoRunStatus.CANCELLED.value
        state["error"] = reason
        state["approval"] = {"status": "rejected"}
        return self._set_status(state, AutoRunStatus.CANCELLED)

    def _run_verifier(
        self,
        *,
        run_id: str,
        goal: str,
        run_root: Path,
        budgets: AutoBudgets,
        loop_result: AgentToolLoopResult,
        cancellation_token: asyncio.Event | None = None,
    ) -> VerificationResult:
        if not loop_result.tool_calls:
            response = loop_result.text.strip()
            if response:
                return VerificationResult(
                    status=VerificationStatus.PASSED,
                    command=[],
                    exit_code=0,
                    stdout="Verified non-empty Auto response without workspace actions.",
                    stderr="",
                    duration_seconds=0.0,
                    error=None,
                    fail_type=None,
                    excerpt="Auto completed without requesting workspace tools.",
                    verifier_profile="response_only",
                )
            reason = "auto_no_observable_result"
            return VerificationResult(
                status=VerificationStatus.FAILED,
                command=[],
                exit_code=1,
                stdout="",
                stderr=reason,
                duration_seconds=0.0,
                error=reason,
                fail_type="no_observable_result",
                excerpt=reason,
                verifier_profile="response_only",
            )
        if not has_canonical_repo_verifier(run_root):
            if is_repo_workspace(run_root):
                reason = "canonical_repo_verifier_unavailable"
                return VerificationResult(
                    status=VerificationStatus.ERROR,
                    command=[],
                    exit_code=None,
                    stdout="",
                    stderr=reason,
                    duration_seconds=0.0,
                    error=reason,
                    fail_type="verifier_unavailable",
                    excerpt=reason,
                    verifier_profile="repository",
                )
            failed_calls = [item for item in loop_result.tool_calls if not item.result.ok]
            if failed_calls:
                first_error = failed_calls[0].result.error or "tool call failed"
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    command=[],
                    exit_code=1,
                    stdout="",
                    stderr=first_error,
                    duration_seconds=0.0,
                    error=first_error,
                    fail_type="tool_outcome_failed",
                    excerpt=first_error,
                    verifier_profile="tool_outcomes",
                )
            return VerificationResult(
                status=VerificationStatus.PASSED,
                command=[],
                exit_code=0,
                stdout=f"Verified {len(loop_result.tool_calls)} successful tool call(s).",
                stderr="",
                duration_seconds=0.0,
                error=None,
                fail_type=None,
                excerpt="All native tool calls completed successfully.",
                verifier_profile="tool_outcomes",
            )
        verifier = VerifierRuntime(project_root=run_root)
        context = RunContext(
            session_id=self.parent.session_id or "local",
            trace_id=str(uuid.uuid4()),
            workspace_root=str(run_root),
            safe_mode=bool(self.parent.tools_enabled.get("safe_mode", False)),
            approved_categories=sorted(self.parent.approved_categories),
            max_retries=budgets.max_retries,
            attempt=1,
        )
        verifier_task = TaskPacket(
            task_id=run_id,
            session_id=context.session_id,
            trace_id=context.trace_id,
            goal=goal,
            scope={"workspace_root": str(run_root)},
            verifier={"command": canonical_check_command(), "cwd": str(run_root)},
        )
        if cancellation_token is not None:
            return verifier.run(verifier_task, context, cancelled=cancellation_token.is_set)
        return verifier.run(verifier_task, context)

    def _set_status(
        self,
        state: dict[str, JSONValue],
        status: AutoRunStatus,
    ) -> dict[str, JSONValue]:
        state["status"] = status.value
        skill = state.get("skill")
        if isinstance(skill, dict) and isinstance(skill.get("skill_id"), str):
            if status == AutoRunStatus.COMPLETED:
                skill["status"] = "completed"
                skill.pop("reason", None)
            elif status in {
                AutoRunStatus.FAILED_CONFLICT,
                AutoRunStatus.FAILED_VERIFIER,
                AutoRunStatus.FAILED_WORKER,
                AutoRunStatus.FAILED_INTERNAL,
                AutoRunStatus.CANCELLED,
            }:
                skill["status"] = "failed"
                reason = state.get("error")
                if isinstance(reason, str) and reason:
                    skill["reason"] = reason
        return self._set_state(state)

    def _set_state(self, state: dict[str, JSONValue]) -> dict[str, JSONValue]:
        state["updated_at"] = utc_now_iso()
        normalized = normalize_auto_state(state)
        if normalized is None:
            raise RuntimeError("auto_state_normalization_failed")
        self.parent.last_auto_state = normalized
        if self.progress_callback is not None:
            self.progress_callback(dict(normalized))
        return normalized


def _auto_v1_tool_call_state(item: ExecutedToolCall) -> dict[str, JSONValue]:
    result = item.result
    diagnostics: list[str] = []
    if result.error:
        diagnostics.append(result.error)
    return {
        "coder_id": item.call.id,
        "shard_id": "tool-loop",
        "status": "completed" if result.ok else "failed",
        "changed_paths": [],
        "diagnostics": diagnostics,
        "tool": item.call.name,
    }


def _has_failed_tool_call(tool_calls: list[ExecutedToolCall]) -> bool:
    """Fail closed on any failed tool call.

    Tool-call order is not evidence: a later, unrelated success does not prove
    the failed action was performed. The canonical verifier checks the repository
    state, not a specific tool outcome, so it cannot provide that evidence.
    """
    return any(not item.result.ok for item in tool_calls)


def _verification_state(verification: VerificationResult) -> dict[str, JSONValue]:
    return {
        "status": verification.status.value,
        "command": list(verification.command),
        "exit_code": verification.exit_code,
        "error": verification.error,
        "duration_ms": verification.duration_ms,
        "fail_type": verification.fail_type,
        "excerpt": verification.excerpt,
        "verifier_profile": verification.verifier_profile,
    }


def _env_int(name: str, default: int, *, min_value: int = 0) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    if value < min_value:
        return default
    return value


def _resolve_auto_budgets() -> AutoBudgets:
    return AutoBudgets(
        max_runtime_seconds=_env_int(
            AUTO_MAX_RUNTIME_SECONDS_ENV,
            AUTO_DEFAULT_MAX_RUNTIME_SECONDS,
            min_value=1,
        ),
        max_tool_calls=_env_int(
            AUTO_MAX_TOOL_CALLS_ENV,
            AUTO_DEFAULT_MAX_TOOL_CALLS,
            min_value=1,
        ),
        max_files_touched=_env_int(
            AUTO_MAX_FILES_TOUCHED_ENV,
            AUTO_DEFAULT_MAX_FILES_TOUCHED,
            min_value=1,
        ),
        max_llm_tokens=_env_int(
            AUTO_MAX_LLM_TOKENS_ENV,
            AUTO_DEFAULT_MAX_LLM_TOKENS,
            min_value=1,
        ),
        max_retries=_env_int(
            AUTO_MAX_RETRIES_ENV,
            AUTO_DEFAULT_MAX_RETRIES,
            min_value=0,
        ),
    )


def _budget_runtime_stop(
    *,
    budgets: AutoBudgets,
    started_monotonic: float,
) -> str | None:
    now = time.monotonic()
    elapsed = max(0.0, now - started_monotonic)
    if now < started_monotonic + budgets.max_runtime_seconds:
        return None
    return (
        f"Budget exhausted: runtime_seconds={elapsed} "
        f">= max_runtime_seconds={budgets.max_runtime_seconds}"
    )


def _budget_stop_outcome(
    parent_agent: Agent,
    reason: str,
    *,
    skill: dict[str, JSONValue] | None = None,
    usage_unknown: bool = False,
) -> AutoRunOutcome:
    next_steps = [
        "Сузь задачу и перезапусти auto.",
        "Увеличь budgets для auto-run при необходимости.",
    ]
    if usage_unknown:
        next_steps = [
            "Проверь поддержку полного reported usage у выбранного provider.",
            "Проверь выполненные действия перед явным новым запуском.",
        ]
    return AutoRunOutcome(
        text=parent_agent._format_stop_response(
            what=(
                "Auto-run остановлен: token usage неизвестен"
                if usage_unknown
                else "Auto-run остановлен: budget exhausted"
            ),
            why=reason,
            next_steps=next_steps,
            stop_reason_code=StopReasonCode.BUDGET_EXHAUSTED,
            route="auto",
            plan_summary=(
                "Token admission невозможен без reported usage."
                if usage_unknown
                else "Auto FSM остановлен бюджетным лимитом."
            ),
            execution_summary=reason,
            skill=skill,
        ),
        status=AutoRunStatus.FAILED_INTERNAL,
        stop_reason_code=StopReasonCode.BUDGET_EXHAUSTED,
        verifier=None,
        next_steps=next_steps,
    )


def _skill_state(state: dict[str, JSONValue]) -> dict[str, JSONValue] | None:
    raw = state.get("skill")
    return dict(raw) if isinstance(raw, dict) else None


def _verifier_reason(verification: VerificationResult) -> str:
    return extract_verifier_excerpt(verification, max_lines=3, max_chars=300)


def _record_model_usage(state: dict[str, JSONValue], usage: ModelUsageSummary | None) -> None:
    if usage is None:
        return
    metrics = state.get("execution_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
        state["execution_metrics"] = metrics
    metrics["tool_loop_model_usage"] = {
        "started_calls": usage.started_calls,
        "reported_calls": usage.reported_calls,
        "unknown_calls": usage.unknown_calls,
        "reported_sum": {
            "prompt_tokens": usage.reported_sum.prompt_tokens,
            "completion_tokens": usage.reported_sum.completion_tokens,
            "total_tokens": usage.reported_sum.total_tokens,
        },
        "total_tokens": usage.total_tokens,
    }
