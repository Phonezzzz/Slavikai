from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
from core.tool_loop import AgentToolLoop, AgentToolLoopResult, ExecutedToolCall
from shared.auto_models import (
    AUTO_CODER_POOL_DEFAULT,
    AUTO_CODER_POOL_MAX,
    AUTO_CODER_POOL_MIN,
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


AUTO_CODER_POOL_ENV = "AUTO_CODER_POOL_SIZE"
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
AUTO_DEFAULT_ACCEPTANCE_CHECKS = [
    "Изменения применены без конфликтов",
    "Проверки завершены успешно",
]
AUTO_V1_SYSTEM_PROMPT = (
    "You are running Auto v1. Decide from the user's request whether workspace action is "
    "actually required. For conversation or information that needs no action, answer directly "
    "without tool calls. For every workspace action, use native tool calls and never invent "
    "filesystem results. Stop with a concise final answer after any tools finish."
)


@dataclass(frozen=True)
class PatchOperation:
    op: str
    path: str
    content: bytes | None = None


@dataclass(frozen=True)
class PatchBundle:
    operations: list[PatchOperation] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    status: str = "ok"


@dataclass(frozen=True)
class CoderResult:
    coder_id: str
    shard_id: str
    status: str
    bundle: PatchBundle
    error: str | None = None


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
class _PausedRun:
    run_id: str
    goal: str
    pool_size: int
    plan: AutoPlan | None
    started_at: str
    workspace_root: Path
    runtime: str = "legacy"


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
        run_id: str | None = None,
        started_at: str | None = None,
        run_root_override: Path | None = None,
    ) -> AutoRunOutcome:
        run_id_value = run_id or f"auto-{uuid.uuid4().hex}"
        started = started_at or utc_now_iso()
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
        }
        self._set_state(state)

        try:
            brain = self.parent._get_main_brain()
            if not brain.supports_native_tools:
                reason = "Выбранный provider не поддерживает native tool calls для Auto."
                state["error"] = reason
                state["error_code"] = "native_tools_required"
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: provider несовместим с Auto",
                        why=reason,
                        next_steps=["Выбери DeepSeek или Local model и повтори запуск."],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto требует native tool-calling provider.",
                        execution_summary=reason,
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=["Выбери DeepSeek или Local model и повтори запуск."],
                )
            self._set_status(state, AutoRunStatus.PLANNING)
            budget_stop = _budget_runtime_stop(
                budgets=budgets,
                started_monotonic=started_monotonic,
            )
            if budget_stop is not None:
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                state["error"] = budget_stop
                state["error_code"] = "budget_runtime"
                self._set_state(state)
                return _budget_stop_outcome(self.parent, budget_stop)

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
            loop_result = AgentToolLoop(max_iterations=budgets.max_tool_calls).run(
                brain=brain,
                gateway=gateway,
                messages=[
                    LLMMessage(role="system", content=AUTO_V1_SYSTEM_PROMPT),
                    LLMMessage(role="user", content=goal),
                ],
                tools=tool_specs,
                config=self.parent.main_config,
            )
            tool_call_states = [_auto_v1_tool_call_state(item) for item in loop_result.tool_calls]
            state["coders"] = tool_call_states
            failed_calls = [
                item
                for item in tool_call_states
                if isinstance(item.get("status"), str) and item.get("status") != "completed"
            ]
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
            self._set_state(state)
            if len(loop_result.tool_calls) >= budgets.max_tool_calls:
                reason = (
                    f"Budget exhausted: tool_calls={len(loop_result.tool_calls)} "
                    f">= max_tool_calls={budgets.max_tool_calls}"
                )
                state["error"] = reason
                state["error_code"] = "budget_tool_calls"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                self._set_state(state)
                return _budget_stop_outcome(self.parent, reason)
            if failed_calls:
                diagnostics_raw = failed_calls[0].get("diagnostics")
                if isinstance(diagnostics_raw, list) and diagnostics_raw:
                    first_error = str(diagnostics_raw[0])
                else:
                    first_error = "tool call failed"
                state["error"] = first_error
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: tool loop failed",
                        why=first_error,
                        next_steps=[
                            "Проверь auto_state.coders.",
                            "Исправь причину ошибки tool call и перезапусти auto.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary="Auto v1 выполнил задачу через native tool loop.",
                        execution_summary=first_error,
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Проверь auto_state.coders.",
                        "Исправь причину ошибки tool call и перезапусти auto.",
                    ],
                )

            self._set_status(state, AutoRunStatus.VERIFYING)
            verification = self._run_verifier(
                run_id=run_id_value,
                goal=goal,
                run_root=run_root,
                budgets=budgets,
                loop_result=loop_result,
            )
            state["verifier"] = _verification_state(verification)
            self._set_state(state)
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
            visible_result = loop_result.text
            if not response_only:
                visible_result = (
                    "Auto-run v1 завершён успешно.\n"
                    f"Tool calls: {len(loop_result.tool_calls)}\n"
                    f"Verifier: {verification.status.value}\n"
                    f"Result: {loop_result.text}"
                )
            text = self.parent._append_report_block(
                visible_result,
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
            )
            return AutoRunOutcome(
                text=text,
                status=AutoRunStatus.COMPLETED,
                stop_reason_code=None,
                verifier=verification,
                next_steps=[],
            )
        except ApprovalRequired as exc:
            self._paused_runs[run_id_value] = _PausedRun(
                run_id=run_id_value,
                goal=goal,
                pool_size=1,
                plan=None,
                started_at=started,
                workspace_root=run_root,
                runtime="tool_loop_v1",
            )
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
                ),
                status=AutoRunStatus.FAILED_INTERNAL,
                stop_reason_code=StopReasonCode.MWV_INTERNAL_ERROR,
                verifier=None,
                next_steps=[
                    "Проверь логи и trace.",
                    "Повтори запуск auto после исправления.",
                ],
            )

    def resume(self, run_id: str) -> AutoRunOutcome | None:
        paused = self._paused_runs.pop(run_id, None)
        if paused is None:
            return None
        if paused.runtime != "tool_loop_v1":
            return None
        return self.run_v1(
            paused.goal,
            run_id=paused.run_id,
            started_at=paused.started_at,
            run_root_override=paused.workspace_root,
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
        state: dict[str, JSONValue] = {
            "run_id": run_id,
            "status": AutoRunStatus.CANCELLED.value,
            "goal": paused.goal,
            "root_path": str(paused.workspace_root),
            "pool_size": paused.pool_size,
            "started_at": paused.started_at,
            "updated_at": utc_now_iso(),
            "planner": {"status": "completed", "runtime": "auto_v1_tool_loop"},
            "plan": paused.plan.to_dict() if paused.plan is not None else None,
            "coders": [],
            "merge": {"status": "cancelled", "runtime": "auto_v1_tool_loop"},
            "verifier": None,
            "approval": {"status": "rejected"},
            "error": reason,
            "error_code": None,
            "missing_paths": [],
        }
        return self._set_state(state)

    def _run_verifier(
        self,
        *,
        run_id: str,
        goal: str,
        run_root: Path,
        budgets: AutoBudgets,
        loop_result: AgentToolLoopResult,
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
        verifier_run: Any = verifier.run
        try:
            return cast(VerificationResult, verifier_run(verifier_task, context))
        except TypeError:
            # Backward compatibility for legacy verifier stubs.
            return cast(VerificationResult, verifier_run(context))

    def _set_status(
        self,
        state: dict[str, JSONValue],
        status: AutoRunStatus,
    ) -> dict[str, JSONValue]:
        state["status"] = status.value
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


def _resolve_pool_size() -> int:
    raw = os.getenv(AUTO_CODER_POOL_ENV, "").strip()
    if not raw:
        return AUTO_CODER_POOL_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return AUTO_CODER_POOL_DEFAULT
    return max(AUTO_CODER_POOL_MIN, min(AUTO_CODER_POOL_MAX, value))


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
    elapsed = int(max(0.0, time.monotonic() - started_monotonic))
    if elapsed <= budgets.max_runtime_seconds:
        return None
    return (
        f"Budget exhausted: runtime_seconds={elapsed} "
        f"> max_runtime_seconds={budgets.max_runtime_seconds}"
    )


def _budget_stop_outcome(parent_agent: Agent, reason: str) -> AutoRunOutcome:
    next_steps = [
        "Сузь задачу и перезапусти auto.",
        "Увеличь budgets для auto-run при необходимости.",
    ]
    return AutoRunOutcome(
        text=parent_agent._format_stop_response(
            what="Auto-run остановлен: budget exhausted",
            why=reason,
            next_steps=next_steps,
            stop_reason_code=StopReasonCode.BUDGET_EXHAUSTED,
            route="auto",
            plan_summary="Auto FSM остановлен бюджетным лимитом.",
            execution_summary=reason,
        ),
        status=AutoRunStatus.FAILED_INTERNAL,
        stop_reason_code=StopReasonCode.BUDGET_EXHAUSTED,
        verifier=None,
        next_steps=next_steps,
    )


def _fallback_shard(goal: str) -> AutoShard:
    return AutoShard(
        shard_id="shard-1",
        goal=goal,
        path_scope=["."],
        depends_on=[],
        acceptance_checks=list(AUTO_DEFAULT_ACCEPTANCE_CHECKS),
    )


def _parse_auto_plan_payload(payload: object, *, goal: str) -> AutoPlan | None:
    if not isinstance(payload, dict):
        return None
    plan_id_raw = payload.get("plan_id")
    shards_raw = payload.get("shards")
    plan_id = plan_id_raw.strip() if isinstance(plan_id_raw, str) and plan_id_raw.strip() else ""
    if not plan_id:
        return None
    if not isinstance(shards_raw, list):
        return None
    shards: list[AutoShard] = []
    for index, item in enumerate(shards_raw, start=1):
        if not isinstance(item, dict):
            continue
        shard_id_raw = item.get("shard_id")
        shard_goal_raw = item.get("goal")
        shard_id = (
            shard_id_raw.strip()
            if isinstance(shard_id_raw, str) and shard_id_raw.strip()
            else f"shard-{index}"
        )
        shard_goal = shard_goal_raw.strip() if isinstance(shard_goal_raw, str) else ""
        if not shard_goal:
            shard_goal = goal
        path_scope = _string_list(item.get("path_scope")) or ["."]
        depends_on = _string_list(item.get("depends_on"))
        acceptance = _string_list(item.get("acceptance_checks")) or list(
            AUTO_DEFAULT_ACCEPTANCE_CHECKS
        )
        shards.append(
            AutoShard(
                shard_id=shard_id,
                goal=shard_goal,
                path_scope=path_scope,
                depends_on=depends_on,
                acceptance_checks=acceptance,
            )
        )
    if not shards:
        shards = [_fallback_shard(goal)]
    plan_goal_raw = payload.get("goal")
    plan_goal = (
        plan_goal_raw.strip() if isinstance(plan_goal_raw, str) and plan_goal_raw.strip() else goal
    )
    return AutoPlan(plan_id=plan_id, goal=plan_goal, shards=shards)


def _plan_from_payload(payload: dict[str, JSONValue], *, goal: str) -> AutoPlan:
    parsed = _parse_auto_plan_payload(payload, goal=goal)
    if parsed is not None:
        return parsed
    return AutoPlan(
        plan_id=f"plan-{uuid.uuid4().hex}",
        goal=goal,
        shards=[_fallback_shard(goal)],
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    output: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
            if cleaned:
                output.append(cleaned)
    return output


def _snapshot_workspace(root: Path) -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if not parts:
            continue
        if parts[0] == ".auto" or ".git" in parts:
            continue
        snapshot[relative.as_posix()] = path.read_bytes()
    return snapshot


def _build_patch_bundle(before: dict[str, bytes], after: dict[str, bytes]) -> PatchBundle:
    paths = sorted(set(before.keys()) | set(after.keys()))
    operations: list[PatchOperation] = []
    changed_paths: list[str] = []
    for path in paths:
        before_value = before.get(path)
        after_value = after.get(path)
        if before_value is None and after_value is not None:
            operations.append(PatchOperation(op="create", path=path, content=after_value))
            changed_paths.append(path)
            continue
        if before_value is not None and after_value is None:
            operations.append(PatchOperation(op="delete", path=path, content=None))
            changed_paths.append(path)
            continue
        if before_value != after_value and after_value is not None:
            operations.append(PatchOperation(op="update", path=path, content=after_value))
            changed_paths.append(path)
    return PatchBundle(
        operations=operations,
        changed_paths=changed_paths,
        diagnostics=[],
        status="ok",
    )


def _topological_order(plan: AutoPlan) -> list[AutoShard]:
    shards = {item.shard_id: item for item in plan.shards}
    indegree: dict[str, int] = {item.shard_id: 0 for item in plan.shards}
    graph: dict[str, list[str]] = {item.shard_id: [] for item in plan.shards}
    for item in plan.shards:
        for dep in item.depends_on:
            if dep not in shards:
                continue
            graph[dep].append(item.shard_id)
            indegree[item.shard_id] += 1

    queue: list[str] = [shard_id for shard_id, value in indegree.items() if value == 0]
    ordered: list[AutoShard] = []
    while queue:
        current = queue.pop(0)
        ordered.append(shards[current])
        for neighbor in graph[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(plan.shards):
        raise ValueError("auto_plan_cycle_detected")
    return ordered


def _depends_on(
    *,
    left: str,
    right: str,
    depends_map: dict[str, set[str]],
) -> bool:
    pending = [left]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        deps = depends_map.get(current, set())
        if right in deps:
            return True
        pending.extend(deps)
    return False


def _detect_conflict(
    coder_results: list[CoderResult],
    plan: AutoPlan,
) -> tuple[CoderResult, CoderResult, list[str]] | None:
    depends_map = {item.shard_id: set(item.depends_on) for item in plan.shards}
    for left_index, left in enumerate(coder_results):
        left_paths = set(left.bundle.changed_paths)
        if not left_paths:
            continue
        for right in coder_results[left_index + 1 :]:
            right_paths = set(right.bundle.changed_paths)
            if not right_paths:
                continue
            overlap = sorted(left_paths & right_paths)
            if not overlap:
                continue
            if _depends_on(left=left.shard_id, right=right.shard_id, depends_map=depends_map):
                continue
            if _depends_on(left=right.shard_id, right=left.shard_id, depends_map=depends_map):
                continue
            return left, right, overlap
    return None


def _update_coder_state(
    state: dict[str, JSONValue],
    coder_id: str,
    *,
    status: str,
    changed_paths: list[str] | None = None,
    diagnostics: list[str] | None = None,
) -> None:
    coders_raw = state.get("coders")
    if not isinstance(coders_raw, list):
        return
    for item in coders_raw:
        if not isinstance(item, dict):
            continue
        item_coder_id = item.get("coder_id")
        if item_coder_id != coder_id:
            continue
        item["status"] = status
        if changed_paths is not None:
            item["changed_paths"] = list(changed_paths)
        if diagnostics is not None:
            item["diagnostics"] = list(diagnostics)
        return


_MISSING_FILE_PATTERN = re.compile(r"(?:Файл не найден|File not found):\s*(.+)")
_MISSING_TARGET_PATH_PATTERNS = (
    re.compile(r"не\s+указан\s+путь\s+к\s+файлу\s+workspace\s+для\s+записи", re.IGNORECASE),
    re.compile(r"workspace[_\s-]?write.*(path|путь)", re.IGNORECASE),
    re.compile(r"missing[_\s-]?target[_\s-]?path", re.IGNORECASE),
    re.compile(r"(target[_\s-]?path|путь).*(required|не\s+указан)", re.IGNORECASE),
)


def _extract_missing_paths(results: list[CoderResult]) -> list[str]:
    paths: set[str] = set()
    for item in results:
        diagnostics = item.bundle.diagnostics
        for diag in diagnostics:
            if not isinstance(diag, str):
                continue
            match = _MISSING_FILE_PATTERN.search(diag)
            if not match:
                continue
            raw_path = match.group(1).strip()
            if raw_path:
                paths.add(raw_path)
    return sorted(paths)


def _is_missing_target_path_error(error: str) -> bool:
    if not error.strip():
        return False
    return any(pattern.search(error) for pattern in _MISSING_TARGET_PATH_PATTERNS)


def _auto_plan_summary(plan: AutoPlan) -> str:
    shard_count = len(plan.shards)
    return f"План разбит на {shard_count} shard(ов)."


def _verifier_reason(verification: VerificationResult) -> str:
    return extract_verifier_excerpt(verification, max_lines=3, max_chars=300)
