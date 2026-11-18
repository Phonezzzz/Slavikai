"""Auto as a policy preset over the canonical tool loop (E01-T07).

Auto is no longer an orchestration engine. There is no planner, no coder
pool, no merge phase and no shard plan: a run in mode ``"auto"`` is a
canonical :class:`core.tool_loop.AgentToolLoop` execution with the Auto
budget+policy preset (900s / 80 tool calls / 120 touched files / 120k
model tokens, native tool calling required, fail-closed on any failed
tool call, approval pause/resume with the same scope checks as before).

What changed relative to the old ``AutoOrchestrator``:

- The fabricated ``AutoPlan`` (a single ``"tool-loop"`` shard with prose
  acceptance checks) and the ``planner`` / ``plan`` / ``coders`` /
  ``merge`` / ``pool_size`` progress theatre are gone. Progress state
  carries only what the run really knows: run id, honest phase, goal,
  budgets, verifier, approval, error and execution metrics. The UI
  normalizer still accepts the state shape (missing theatre keys
  normalize to empty values).
- The fake ``PASSED`` verifier profiles are gone. A response-only run
  (no tool calls) and a run in a non-repository workspace make no
  verification claim at all (``verifier is None``); the canonical
  outcome derivation says ``NOT_APPLICABLE`` for the former and
  ``UNKNOWN`` for the latter. Only a repository workspace with a
  canonical verifier runs a real verifier (``VerifierRuntime``), and
  a repository workspace without one still fails honestly with
  ``canonical_repo_verifier_unavailable``.
- Completion is gated on the canonical outcome derivation
  (:func:`core.run_outcome.outcome_from_loop_result`): the run reports
  ``COMPLETED`` only when the canonical derivation says the loop
  terminated cleanly. ``COMPLETED`` means the run terminated normally;
  it does not claim full done — that needs a confirmed postcondition
  (``RunOutcome.is_full_done``), which a preset run without a repo
  verifier does not have.
- Budgets, approval pause/resume, capture scope check, skill
  injection/reporting, usage accounting and the file-effects cap are
  preserved verbatim from the old engine: they were real mechanisms,
  not theatre.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from core.approval_policy import ApprovalRequired
from core.managed_workspace import WorkspaceAccess
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
    has_canonical_check_definition,
    is_repo_workspace,
)
from core.mwv.verifier_summary import extract_verifier_excerpt
from core.response_preferences import compose_response_preferences
from core.run_contract import RunBudget
from core.run_outcome import TerminalReason, outcome_from_loop_result
from core.skills.index import SkillResolution
from core.skills.runtime import skill_run_metadata, skill_run_report
from core.tool_gateway import AutoFileEffects
from core.tool_loop import (
    AgentToolLoop,
    AgentToolLoopResult,
    ExecutedToolCall,
    ModelUsageSummary,
    ToolLoopContinuation,
    ToolLoopExecutionError,
)
from core.workspace_python_execution import (
    RootlessCommandExecutor,
    RootlessPythonExecutor,
)
from llm.brain_base import Brain
from shared.auto_models import (
    AutoRunStatus,
    normalize_auto_state,
    utc_now_iso,
)
from shared.models import JSONValue, LLMMessage
from shared.tool_capture import CaptureBinding, ToolCaptureFault
from shared.workspace_binding import WorkspaceBinding
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

    def to_run_budget(self) -> RunBudget:
        """The preset as a canonical loop budget."""
        return RunBudget(
            max_iterations=self.max_tool_calls,
            max_tool_calls=self.max_tool_calls,
            max_model_tokens=self.max_llm_tokens,
            max_runtime_seconds=float(self.max_runtime_seconds),
        )


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
    file_effects: AutoFileEffects
    capture_binding: CaptureBinding | None = None
    # Хэш Makefile, зафиксированный доверенным кодом до начала loop (условие 2
    # владельца): критерий проверки задаёт доверенная сторона, а не агент.
    verifier_criterion_sha256: str | None = None


@dataclass
class _PausedRun:
    frame: _RunFrame
    continuation: ToolLoopContinuation


class AutoContinuationUnavailable(ValueError):
    pass


def _criterion_sha256(run_root: Path) -> str | None:
    """Хэш доверенного критерия проверки (Makefile) до запуска агента.

    None — Makefile отсутствует или нечитаем: критерий не задан, verifier
    такой run не проверяет.
    """
    try:
        return hashlib.sha256((run_root / "Makefile").read_bytes()).hexdigest()
    except OSError:
        return None


def _unavailable_verification() -> VerificationResult:
    """Fail-closed: backend недоступен → ERROR, никогда PASSED и никогда None."""
    reason = "sandbox_execution_unavailable"
    return VerificationResult(
        status=VerificationStatus.ERROR,
        command=[],
        exit_code=None,
        stdout="",
        stderr=reason,
        duration_seconds=0.0,
        error=reason,
        fail_type="execution_unavailable",
        excerpt=reason,
        verifier_profile="unavailable",
    )


class AutoPresetRunner:
    """Runs mode ``"auto"`` as a canonical loop with the Auto preset.

    The only orchestration left is the loop itself plus budget/approval
    bookkeeping around it. There is no plan step, no shard and no merge:
    the model drives the single canonical loop until it stops, the
    budget stops it, approval pauses it, or a tool call fails.
    """

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

    def run(
        self,
        goal: str,
        *,
        skill_resolution: SkillResolution | None = None,
        run_root_override: Path | None = None,
        capture_binding: CaptureBinding | None = None,
        cancellation_token: asyncio.Event | None = None,
    ) -> AutoRunOutcome:
        if capture_binding is not None and (
            capture_binding.run.scope.principal_id != self.parent.user_id
            or capture_binding.run.scope.session_id != self.parent.session_id
        ):
            raise ToolCaptureFault("workspace_capture_scope_mismatch")
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
        # Критерий проверки фиксируется доверенным кодом до запуска агента
        # (условие 2 владельца); resume его не перезахватывает — frame
        # персистится в _PausedRun.
        criterion_sha256 = _criterion_sha256(run_root)
        state: dict[str, JSONValue] = {
            "run_id": run_id_value,
            "status": AutoRunStatus.RUNNING.value,
            "goal": goal,
            "root_path": str(run_root),
            "started_at": started,
            "updated_at": started,
            # No fabricated plan: the preset never had planner/coders/merge.
            # The keys stay present (empty) so paused/cancelled states carry
            # the same honest shape as finished ones.
            "planner": {},
            "plan": None,
            "coders": [],
            "merge": {},
            # The run has not established a trustworthy file-effect count yet.
            # Keep unknown visible through approvals, failures and cancellation.
            "execution_metrics": {"files_touched": None},
            "budgets": budgets.to_dict(),
            "verifier": None,
            "verifier_criterion_sha256": criterion_sha256,
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
            AutoFileEffects(budgets.max_files_touched),
            capture_binding,
            verifier_criterion_sha256=criterion_sha256,
        )
        return self._execute_frame(frame, cancellation_token=cancellation_token)

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
                        plan_summary="Auto preset требует native tool-calling provider.",
                        execution_summary=reason,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=["Выбери модель с доступными native tools и повтори запуск."],
                )
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

            run_budget = budgets.to_run_budget()
            gateway = replace(
                self.parent._build_tool_gateway(budget=run_budget),
                auto_file_effects=frame.file_effects,
            )
            tool_specs = self.parent.tool_registry.list_tool_specs(policy=gateway.run_policy)
            system_prompt = AUTO_V1_SYSTEM_PROMPT
            if skill_resolution is not None:
                system_prompt = f"{system_prompt}\n\n{skill_resolution.system_instruction()}"
            loop_result = AgentToolLoop(
                **run_budget.loop_kwargs(started_monotonic=started_monotonic)
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
                capture_binding=frame.capture_binding,
            )
            state["execution_metrics"] = {
                "tool_calls_used": len(loop_result.tool_calls),
                "iterations": loop_result.iterations,
                "files_touched": frame.file_effects.count,
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
                review_required = loop_error == "workspace_execution_review_required"
                next_steps = (
                    [
                        "Проверь сохранённый результат и состояние среды "
                        "перед решением о новом запуске."
                    ]
                    if review_required
                    else [
                        "Проверь результат loop и исправь причину ошибки tool call, "
                        "затем перезапусти auto.",
                    ]
                )
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: tool loop failed",
                        why=(
                            "Результат изолированного исполнения не согласован; "
                            "дальнейшие вызовы остановлены."
                            if review_required
                            else loop_error
                        ),
                        next_steps=next_steps,
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto preset выполнил задачу через canonical tool loop.",
                        execution_summary=loop_error,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=next_steps,
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
                            "Проверь выполненные tool calls на failed outcome.",
                            "Докажи выполнение действия verifier-ом или перезапусти auto.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto preset обнаружил неустранённый сбой tool call.",
                        execution_summary=reason,
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Проверь выполненные tool calls на failed outcome.",
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
            verification = self._verify_after_loop(
                run_id=run_id_value,
                goal=goal,
                run_root=run_root,
                budgets=budgets,
                loop_result=loop_result,
                cancellation_token=cancellation_token,
                criterion_sha256=frame.verifier_criterion_sha256,
                principal_id=frame.principal_id,
                session_id=frame.session_id,
            )
            state["verifier"] = (
                _verification_state(verification) if verification is not None else None
            )
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
            if verification is not None and verification.status != VerificationStatus.PASSED:
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
                        plan_summary="Auto preset выполнил canonical loop и дошёл до verifier.",
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

            # The canonical outcome derivation is the completion authority:
            # a clean loop terminates COMPLETED, and the verification claim
            # is exactly what the derivation says — NOT_APPLICABLE when the
            # run changed nothing, UNKNOWN when it may have changed state
            # without a confirmed postcondition. COMPLETED here means the
            # run terminated normally, not that full done is proven.
            canonical = outcome_from_loop_result(
                loop_result, state_changing=bool(loop_result.tool_calls)
            )
            if canonical.terminal_reason is not TerminalReason.COMPLETED:
                state["error"] = "canonical_outcome_not_completed"
                state["error_code"] = "outcome_not_completed"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: canonical outcome не completed",
                        why="Tool loop завершился, но канонический outcome — не COMPLETED.",
                        next_steps=["Проверь логи и trace.", "Повтори запуск auto."],
                        stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                        route="auto",
                        plan_summary="Auto preset выполнил canonical tool loop.",
                        execution_summary="canonical_outcome_not_completed",
                        skill=_skill_state(state),
                    ),
                    status=AutoRunStatus.FAILED_INTERNAL,
                    stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                    verifier=verification,
                    next_steps=["Проверь логи и trace.", "Повтори запуск auto после исправления."],
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
                    "Auto preset определил, что workspace-действия не требуются."
                    if response_only
                    else "Auto preset использовал canonical AgentToolLoop и ToolGateway."
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
            metrics = state.get("execution_metrics")
            if isinstance(metrics, dict):
                metrics["files_touched"] = frame.file_effects.count
                metrics["tool_calls_used"] = len(exc.continuation.executed)
            _record_model_usage(state, exc.continuation.model_usage)
            if cancellation_token is not None and cancellation_token.is_set():
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
                metrics = state.get("execution_metrics")
                if isinstance(metrics, dict):
                    metrics["files_touched"] = None
                    metrics["tool_calls_used"] = len(exc.result.tool_calls)
                    metrics["iterations"] = exc.result.iterations
                _record_model_usage(state, exc.result.model_usage)
            capture_cause = (
                exc.cause
                if isinstance(exc, ToolLoopExecutionError)
                and isinstance(exc.cause, ToolCaptureFault)
                else exc
                if isinstance(exc, ToolCaptureFault)
                else None
            )
            capture_fault = capture_cause is not None
            reason = str(capture_cause) if capture_fault else str(exc)
            steps = (
                [
                    "Проверь сохранённый intent/output и фактические effects.",
                    "Не повторяй операцию до reconciliation неизвестного результата.",
                ]
                if capture_fault
                else ["Проверь логи и trace.", "Повтори запуск auto после исправления."]
            )
            state["error"] = reason
            if capture_fault:
                state["error_code"] = "tool_capture_failed"
            self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
            return AutoRunOutcome(
                text=self.parent._format_stop_response(
                    what="Auto-run остановлен: внутренняя ошибка",
                    why=reason,
                    next_steps=steps,
                    stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                    route="auto",
                    plan_summary="Auto preset tool loop завершился внутренней ошибкой.",
                    execution_summary=reason,
                    skill=_skill_state(state),
                ),
                status=AutoRunStatus.FAILED_INTERNAL,
                stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                verifier=None,
                next_steps=steps,
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
        # The resumed frame still carries waiting_approval: publish
        # RUNNING before the loop starts so last_auto_state and the UI
        # progress show execution, not a pending approval, while the
        # confirmed tool runs and the next model request happens.
        self._set_status(paused.frame.state, AutoRunStatus.RUNNING)
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

    def _verify_after_loop(
        self,
        *,
        run_id: str,
        goal: str,
        run_root: Path,
        budgets: AutoBudgets,
        loop_result: AgentToolLoopResult,
        cancellation_token: asyncio.Event | None = None,
        criterion_sha256: str | None = None,
        principal_id: str | None = None,
        session_id: str | None = None,
    ) -> VerificationResult | None:
        """Honest verification after a clean loop.

        Returns ``None`` when there is nothing real to verify — a
        non-empty response-only run, or a run whose trusted criterion was
        never fixed before the loop. No ``PASSED`` is ever claimed from the
        bare fact that tool calls succeeded or the model answered. An
        empty response-only run is rejected with
        ``auto_no_observable_result`` instead of completing: the user
        got neither an answer nor an executed action.

        The criterion decision is stored with the criterion itself: once
        ``criterion_sha256`` was fixed before the loop, verification is
        mandatory until the end of the run. A Makefile deleted afterwards
        (or changed, or replaced on the staged snapshot) fails with
        ``verifier_criterion_changed`` — never ``None``. Without a
        pre-loop criterion there is no mandatory verification: a Makefile
        planted by the agent mid-run is never executed. The only
        exception is a repository workspace without a Makefile, which
        fails honestly with ``canonical_repo_verifier_unavailable``.
        A Makefile workspace whose backend is unavailable fails closed
        with ``sandbox_execution_unavailable`` (ERROR, never ``None``).
        The verifier runs the explicit ``make -f Makefile check`` on the
        staged snapshot; the executor re-checks the criterion hash on
        those exact bytes before any process starts.
        """
        if not loop_result.tool_calls:
            if not (loop_result.text or "").strip():
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
            return None
        if criterion_sha256 is None:
            # Доверенный критерий до loop не зафиксирован: обязательной
            # проверки нет, и Makefile, появившийся во время loop
            # (его мог подложить агент), никогда не исполняется.
            if is_repo_workspace(run_root) and not has_canonical_check_definition(run_root):
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
            return None
        if not has_canonical_check_definition(run_root):
            # Критерий был зафиксирован до loop, а Makefile к концу run
            # исчез: обязательная проверка потеряна — ERROR, никогда None.
            reason = "verifier_criterion_changed"
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=[],
                exit_code=None,
                stdout="",
                stderr=reason,
                duration_seconds=0.0,
                error=reason,
                fail_type="criterion_changed",
                excerpt=reason,
                verifier_profile="canonical",
            )
        executor, binding = self._verifier_command_executor(
            run_root, principal_id=principal_id, session_id=session_id
        )
        if executor is None or binding is None or not executor.is_available():
            return _unavailable_verification()
        current = _criterion_sha256(run_root)
        if current != criterion_sha256:
            reason = "verifier_criterion_changed"
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=[],
                exit_code=None,
                stdout="",
                stderr=reason,
                duration_seconds=0.0,
                error=reason,
                fail_type="criterion_changed",
                excerpt=reason,
                verifier_profile="canonical",
            )
        verifier = VerifierRuntime(
            project_root=run_root,
            executor=executor,
            binding=binding,
            criterion_sha256=criterion_sha256,
        )
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

    def _verifier_command_executor(
        self,
        run_root: Path,
        *,
        principal_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[RootlessCommandExecutor | None, WorkspaceBinding | None]:
        """Command executor + binding для штатного verifier, или (None, None).

        Источник явного разрешения на запуск verifier для этого run —
        вызов auto run владельцем (``handle_auto_command`` после routing
        decision gate); запись о нём — ``_RunFrame`` этого run (run_id,
        principal, session). Здесь разрешение проверяется, а не
        предполагается: binding обязан принадлежать principal/session
        этого run, указывать ровно на run_root и нести environment
        revision квалифицированного executor. Parent держит
        ``_workspace_executor``/``_workspace_access`` как приватные
        атрибуты; stub-родители в тестах их не имеют — getattr.
        """
        python_executor = getattr(self.parent, "_workspace_executor", None)
        if not isinstance(python_executor, RootlessPythonExecutor):
            return None, None
        access = getattr(self.parent, "_workspace_access", None)
        if not isinstance(access, WorkspaceAccess):
            return None, None
        try:
            workspace_root = access.workspace.root.resolve()
            run_root_resolved = run_root.resolve()
        except OSError:
            return None, None
        if workspace_root != run_root_resolved:
            return None, None
        executor = RootlessCommandExecutor.from_python_executor(python_executor)
        if executor is None:
            return None, None
        binding = access.bind()
        if principal_id is not None and binding.principal_id != principal_id:
            return None, None
        if session_id is not None and binding.session_id != session_id:
            return None, None
        try:
            environment_revision = executor.environment_revision
        except (OSError, ValueError):
            return None, None
        if (
            binding.environment_revision is None
            or binding.environment_revision != environment_revision
        ):
            return None, None
        return executor, binding

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
                else "Auto preset остановлен бюджетным лимитом."
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
