from __future__ import annotations

import concurrent.futures
import json
import os
import re
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.approval_policy import ApprovalRequired
from core.executor import Executor
from core.mwv.models import (
    RunContext,
    StopReasonCode,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
)
from core.mwv.verifier_runtime import VerifierRuntime
from core.planner import Planner
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
from shared.models import JSONValue, LLMMessage, PlanStepStatus
from tools.workspace_tools import WORKSPACE_ROOT, get_workspace_root, workspace_root_context

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
    plan: AutoPlan
    started_at: str
    workspace_root: Path


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

    def run(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        plan_override: AutoPlan | None = None,
        started_at: str | None = None,
        run_root_override: Path | None = None,
    ) -> AutoRunOutcome:
        run_id_value = run_id or f"auto-{uuid.uuid4().hex}"
        pool_size = _resolve_pool_size()
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
            "pool_size": pool_size,
            "started_at": started,
            "updated_at": started,
            "planner": {"status": "idle"},
            "plan": None,
            "coders": [],
            "merge": {"status": "idle", "changed_paths": []},
            "budgets": budgets.to_dict(),
            "verifier": None,
            "approval": None,
            "error": None,
            "error_code": None,
            "missing_paths": [],
        }
        self._set_state(state)

        try:
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
            plan = plan_override or self._build_plan(goal)
            state["planner"] = {
                "status": "completed",
                "shards_total": len(plan.shards),
            }
            state["plan"] = plan.to_dict()
            self._set_state(state)

            self._set_status(state, AutoRunStatus.CODING)
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
            baseline_snapshot = _snapshot_workspace(run_root)
            coder_results = self._run_coder_pool(
                run_id=run_id_value,
                plan=plan,
                pool_size=pool_size,
                baseline_snapshot=baseline_snapshot,
                state=state,
                workspace_root=run_root,
            )

            worker_fail = [item for item in coder_results if item.status != "completed"]
            if worker_fail:
                first_error = worker_fail[0].error or "Worker failed"
                state["error"] = first_error
                missing_paths = _extract_missing_paths(worker_fail)
                if missing_paths:
                    state["error_code"] = "missing_file"
                    state["missing_paths"] = missing_paths
                elif _is_missing_target_path_error(first_error):
                    state["error_code"] = "missing_target_path"
                self._set_status(state, AutoRunStatus.FAILED_WORKER)
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: ошибка coder-воркера",
                        why=first_error,
                        next_steps=[
                            "Проверь diagnostics в auto_state.coders.",
                            "Уточни задачу и запусти auto повторно.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary=(
                            "План шардирован, но часть coder-воркеров завершилась ошибкой."
                        ),
                        execution_summary=first_error,
                    ),
                    status=AutoRunStatus.FAILED_WORKER,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Проверь diagnostics в auto_state.coders.",
                        "Уточни задачу и запусти auto повторно.",
                    ],
                )

            tool_calls_used = len(coder_results)
            if tool_calls_used > budgets.max_tool_calls:
                reason = (
                    f"Budget exhausted: tool_calls={tool_calls_used} "
                    f"> max_tool_calls={budgets.max_tool_calls}"
                )
                state["error"] = reason
                state["error_code"] = "budget_tool_calls"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                self._set_state(state)
                return _budget_stop_outcome(self.parent, reason)

            self._set_status(state, AutoRunStatus.MERGING)
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
            topo = _topological_order(plan)
            topo_ids = [item.shard_id for item in topo]
            ordered_results = sorted(
                coder_results,
                key=lambda item: topo_ids.index(item.shard_id)
                if item.shard_id in topo_ids
                else len(topo_ids),
            )

            conflict = _detect_conflict(ordered_results, plan)
            if conflict is not None:
                left, right, paths = conflict
                changed_paths_union = sorted(
                    {
                        *left.bundle.changed_paths,
                        *right.bundle.changed_paths,
                    }
                )
                state["merge"] = {
                    "status": "failed_conflict",
                    "changed_paths": changed_paths_union,
                    "details": {
                        "left_shard": left.shard_id,
                        "right_shard": right.shard_id,
                        "paths": paths,
                    },
                }
                state["error"] = "merge_conflict"
                self._set_status(state, AutoRunStatus.FAILED_CONFLICT)
                conflict_why = (
                    f"Shard '{left.shard_id}' и '{right.shard_id}' "
                    f"изменяют одни и те же пути: {', '.join(paths)}"
                )
                return AutoRunOutcome(
                    text=self.parent._format_stop_response(
                        what="Auto-run остановлен: конфликт merge",
                        why=conflict_why,
                        next_steps=[
                            "Сузь path_scope шарда или добавь depends_on.",
                            "Перезапусти auto после уточнения зависимостей.",
                        ],
                        stop_reason_code=StopReasonCode.WORKER_FAILED,
                        route="auto",
                        plan_summary=(
                            "План построен, merge остановлен fail-fast политикой конфликтов."
                        ),
                        execution_summary=(
                            "Обнаружено пересечение changed_paths между независимыми shard-ами."
                        ),
                    ),
                    status=AutoRunStatus.FAILED_CONFLICT,
                    stop_reason_code=StopReasonCode.WORKER_FAILED,
                    verifier=None,
                    next_steps=[
                        "Сузь path_scope шарда или добавь depends_on.",
                        "Перезапусти auto после уточнения зависимостей.",
                    ],
                )

            merged_paths = self._apply_patch_bundles(ordered_results, workspace_root=run_root)
            if len(merged_paths) > budgets.max_files_touched:
                reason = (
                    f"Budget exhausted: files_touched={len(merged_paths)} "
                    f"> max_files_touched={budgets.max_files_touched}"
                )
                state["error"] = reason
                state["error_code"] = "budget_files_touched"
                self._set_status(state, AutoRunStatus.FAILED_INTERNAL)
                self._set_state(state)
                return _budget_stop_outcome(self.parent, reason)
            state["merge"] = {
                "status": "completed",
                "changed_paths": sorted(merged_paths),
            }
            self._set_state(state)

            self._set_status(state, AutoRunStatus.VERIFYING)
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
                task_id=run_id_value,
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=goal,
                scope={"workspace_root": str(run_root)},
                verifier={"command": "scripts/check.sh", "cwd": str(run_root)},
            )
            verifier_run: Any = verifier.run
            try:
                verification = verifier_run(verifier_task, context)
            except TypeError:
                # Backward compatibility for legacy verifier stubs.
                verification = verifier_run(context)
            state["verifier"] = {
                "status": verification.status.value,
                "command": list(verification.command),
                "exit_code": verification.exit_code,
                "error": verification.error,
                "duration_ms": verification.duration_ms,
            }
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
                        plan_summary=_auto_plan_summary(plan),
                        execution_summary="Merge завершён, но verifier вернул fail/error.",
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
            text = self.parent._append_report_block(
                (
                    "Auto-run завершён успешно.\n"
                    f"Planner shards: {len(plan.shards)}\n"
                    f"Merged paths: {len(merged_paths)}\n"
                    f"Verifier: {verification.status.value}"
                ),
                route="auto",
                trace_id=context.trace_id,
                attempts=(1, 1),
                verifier=verification,
                next_steps=[],
                stop_reason_code=None,
                plan_summary=_auto_plan_summary(plan),
                execution_summary=(
                    f"Применено {len(merged_paths)} изменённых путей, verifier passed."
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
            plan_payload = state.get("plan")
            if not isinstance(plan_payload, dict):
                fallback_plan = plan_override or AutoPlan(
                    plan_id=f"plan-{uuid.uuid4().hex}",
                    goal=goal,
                    shards=[_fallback_shard(goal)],
                )
                plan_payload = fallback_plan.to_dict()
            plan = _plan_from_payload(plan_payload, goal=goal)
            self._paused_runs[run_id_value] = _PausedRun(
                run_id=run_id_value,
                goal=goal,
                pool_size=pool_size,
                plan=plan,
                started_at=started,
                workspace_root=run_root,
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
                    plan_summary="Auto pipeline завершился внутренней ошибкой.",
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
        return self.run(
            paused.goal,
            run_id=paused.run_id,
            plan_override=paused.plan,
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
            "planner": {"status": "completed"},
            "plan": paused.plan.to_dict(),
            "coders": [],
            "merge": {"status": "cancelled"},
            "verifier": None,
            "approval": {"status": "rejected"},
            "error": reason,
            "error_code": None,
            "missing_paths": [],
        }
        return self._set_state(state)

    def _build_plan(self, goal: str) -> AutoPlan:
        prompt = (
            "Верни JSON объект AutoPlan с полями plan_id, goal, shards. "
            "Каждый shard: shard_id, goal, path_scope, depends_on, acceptance_checks. "
            "Без markdown и без пояснений."
        )
        content = f"goal={goal}\nmax_shards=6"
        try:
            result = self.parent._get_main_brain().generate(
                [
                    LLMMessage(role="system", content=prompt),
                    LLMMessage(role="user", content=content),
                ],
                self.parent.main_config,
            )
            parsed = json.loads(result.text)
            plan = _parse_auto_plan_payload(parsed, goal=goal)
            if plan is not None:
                return plan
        except Exception:  # noqa: BLE001
            self.parent.tracer.log("auto_planner_fallback", "planner json parse failed")
        return AutoPlan(
            plan_id=f"plan-{uuid.uuid4().hex}",
            goal=goal,
            shards=[_fallback_shard(goal)],
        )

    def _run_coder_pool(
        self,
        *,
        run_id: str,
        plan: AutoPlan,
        pool_size: int,
        baseline_snapshot: dict[str, bytes],
        state: dict[str, JSONValue],
        workspace_root: Path,
    ) -> list[CoderResult]:
        shards = list(plan.shards)
        coders: list[dict[str, JSONValue]] = []
        for index, shard in enumerate(shards, start=1):
            coders.append(
                {
                    "coder_id": f"coder-{index}",
                    "shard_id": shard.shard_id,
                    "status": "queued",
                    "changed_paths": [],
                    "diagnostics": [],
                }
            )
        state["coders"] = coders
        self._set_state(state)

        run_root = workspace_root / ".auto" / run_id
        run_root.mkdir(parents=True, exist_ok=True)

        results: list[CoderResult] = []
        max_workers = min(pool_size, max(1, len(shards)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map: dict[concurrent.futures.Future[CoderResult], tuple[str, str]] = {}
            for index, shard in enumerate(shards, start=1):
                coder_id = f"coder-{index}"
                future = pool.submit(
                    self._run_single_coder,
                    run_root,
                    coder_id,
                    shard,
                    baseline_snapshot,
                    workspace_root,
                )
                future_map[future] = (coder_id, shard.shard_id)
                _update_coder_state(state, coder_id, status="running")
                self._set_state(state)

            try:
                for future in concurrent.futures.as_completed(future_map):
                    coder_id, shard_id = future_map[future]
                    result = future.result()
                    results.append(result)
                    _update_coder_state(
                        state,
                        coder_id,
                        status=result.status,
                        changed_paths=result.bundle.changed_paths,
                        diagnostics=result.bundle.diagnostics,
                    )
                    if result.error:
                        state["error"] = result.error
                    self._set_state(state)
            except ApprovalRequired:
                for future in future_map:
                    future.cancel()
                raise

        return results

    def _run_single_coder(
        self,
        run_root: Path,
        coder_id: str,
        shard: AutoShard,
        baseline_snapshot: dict[str, bytes],
        workspace_root: Path,
    ) -> CoderResult:
        workspace_copy = run_root / coder_id
        if workspace_copy.exists():
            shutil.rmtree(workspace_copy)
        shutil.copytree(
            workspace_root,
            workspace_copy,
            dirs_exist_ok=False,
            ignore=shutil.ignore_patterns(".auto"),
        )

        planner = Planner()
        executor = Executor(self.parent.tracer)
        try:
            plan = planner.build_plan(
                shard.goal,
                brain=self.parent._get_main_brain(),
                model_config=self.parent.main_config,
            )
            with workspace_root_context(workspace_copy):
                executed = executor.run(
                    plan,
                    tool_gateway=self.parent._build_tool_gateway(),
                )
            worker_errors = [
                step.result or "step failed"
                for step in executed.steps
                if step.status == PlanStepStatus.ERROR
            ]
            if worker_errors:
                return CoderResult(
                    coder_id=coder_id,
                    shard_id=shard.shard_id,
                    status="failed",
                    bundle=PatchBundle(status="failed", diagnostics=worker_errors),
                    error=worker_errors[0],
                )
            after_snapshot = _snapshot_workspace(workspace_copy)
            bundle = _build_patch_bundle(baseline_snapshot, after_snapshot)
            return CoderResult(
                coder_id=coder_id,
                shard_id=shard.shard_id,
                status="completed",
                bundle=bundle,
                error=None,
            )
        except ApprovalRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            return CoderResult(
                coder_id=coder_id,
                shard_id=shard.shard_id,
                status="failed",
                bundle=PatchBundle(status="failed", diagnostics=[str(exc)]),
                error=str(exc),
            )

    def _apply_patch_bundles(
        self,
        results: list[CoderResult],
        *,
        workspace_root: Path,
    ) -> list[str]:
        changed_paths: set[str] = set()
        for item in results:
            for operation in item.bundle.operations:
                target = (workspace_root / operation.path).resolve()
                try:
                    target.relative_to(workspace_root)
                except ValueError as exc:
                    raise ValueError(f"path outside workspace: {operation.path}") from exc
                if operation.op == "delete":
                    if target.exists() and target.is_file():
                        target.unlink()
                elif operation.op in {"create", "update"}:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(operation.content or b"")
                else:
                    raise ValueError(f"unsupported patch operation: {operation.op}")
                changed_paths.add(operation.path)
        return sorted(changed_paths)

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
    if verification.error:
        return verification.error
    if verification.stderr.strip():
        return verification.stderr.splitlines()[0][:300]
    if verification.stdout.strip():
        return verification.stdout.splitlines()[0][:300]
    if verification.exit_code is None:
        return "Verifier завершился с ошибкой без exit code."
    return f"Verifier завершился с кодом {verification.exit_code}."
