from __future__ import annotations

# ruff: noqa: F401
# mypy: ignore-errors
import uuid
from collections.abc import Callable, Sequence

from core.approval_policy import ApprovalRequired
from core.decision.verifier_fail import build_verifier_fail_packet
from core.mwv.manager import ManagerRuntime, MWVRunResult
from core.mwv.models import (
    ChangeType,
    MWVMessage,
    RunContext,
    StopReasonCode,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from core.mwv.routing import RouteDecision
from core.mwv.verifier_runtime import VerifierRuntime
from core.mwv.worker import WorkerRuntime
from llm.brain_base import Brain
from shared.models import JSONValue, LLMMessage, TaskPlan, WorkspaceDiffEntry
from tools.workspace_tools import WORKSPACE_ROOT

MAX_MWV_ATTEMPTS = 3


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

        def _verifier(run_context: RunContext) -> VerificationResult:
            result = verifier_runtime.run(run_context)
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
            return TaskPacket(
                task_id=str(uuid.uuid4()),
                session_id=context.session_id,
                trace_id=context.trace_id,
                goal=goal,
                messages=list(messages),
                constraints=[],
                context={
                    "route_reason": decision.reason,
                    "risk_flags": decision.risk_flags,
                    **skill_context,
                },
            )

        return _build

    def _mwv_worker_runner(self, task: TaskPacket, context: RunContext) -> WorkResult:
        self._reset_workspace_diffs()
        plan_goal = self._build_mwv_goal(task)
        plan_brain = self._get_main_brain()
        plan = self.planner.build_plan(plan_goal, brain=plan_brain, model_config=self.main_config)
        self.last_plan_original = plan
        executed = self.executor.run(
            plan,
            tool_gateway=self._build_tool_gateway(
                pre_call=self._workspace_diff_pre_call,
                post_call=self._workspace_diff_post_call,
            ),
        )
        self.last_plan = executed
        status = (
            WorkStatus.FAILURE
            if any(step.status.value == "error" for step in executed.steps)
            else WorkStatus.SUCCESS
        )
        changes = self._mwv_changes_from_diffs(self.consume_workspace_diffs())
        diagnostics = self._build_mwv_diagnostics(executed)
        summary = self._format_plan(executed)
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
            return self._format_stop_response(
                what="MWV остановлен: ошибка выполнения",
                why=summary,
                next_steps=[
                    "Проверь шаги выполнения и уточни запрос.",
                    "Запусти задачу снова с более узким фокусом.",
                ],
                stop_reason_code=StopReasonCode.WORKER_FAILED,
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
