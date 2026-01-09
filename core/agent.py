from __future__ import annotations

import difflib
import logging
import re
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from config.memory_config import MemoryConfig, load_memory_config
from config.mode_config import load_mode, save_mode
from config.model_store import load_model_configs, save_model_configs
from config.shell_config import DEFAULT_SHELL_CONFIG_PATH
from config.system_prompts import CRITIC_PROMPT
from config.tools_config import ToolsConfig, load_tools_config, save_tools_config
from core.approval_policy import (
    ApprovalCategory,
    ApprovalContext,
    ApprovalRequest,
    ApprovalRequired,
)
from core.auto_agent import AutoAgent
from core.batch_review import BatchReviewer
from core.critic_policy import (
    CriticDecision,
    CriticFailure,
    CriticMode,
    classify_critic_status,
)
from core.executor import Executor
from core.mwv.manager import ManagerRuntime, MWVRunResult, summarize_verifier_failure
from core.mwv.models import (
    ChangeType,
    MWVMessage,
    RunContext,
    TaskPacket,
    VerificationResult,
    VerificationStatus,
    WorkChange,
    WorkResult,
    WorkStatus,
)
from core.mwv.routing import RouteDecision, classify_request
from core.mwv.verifier_runtime import VerifierRuntime
from core.mwv.worker import WorkerRuntime
from core.planner import Planner
from core.rule_engine import PolicyApplication, RuleEngine
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.brain_manager import BrainManager
from llm.dual_brain import DualBrain
from llm.types import ModelConfig
from memory.memory_companion_store import MemoryCompanionStore
from memory.memory_manager import MemoryManager
from memory.vector_index import VectorIndex
from shared.batch_review_models import (
    BatchReviewRun,
    CandidateStatus,
    PolicyRuleCandidate,
)
from shared.memory_companion_models import (
    BlockedReason,
    ChatInteractionLog,
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionLog,
    InteractionMode,
    ToolInteractionLog,
    ToolStatus,
)
from shared.models import (
    JSONValue,
    LLMMessage,
    MemoryKind,
    MemoryRecord,
    PlanStep,
    TaskPlan,
    ToolCallRecord,
    ToolRequest,
    ToolResult,
    WorkspaceDiffEntry,
)
from shared.policy_models import (
    PolicyRule,
    PolicyScope,
    policy_action_from_json,
    policy_trigger_from_json,
)
from shared.sandbox import normalize_sandbox_path
from tools.filesystem_tool import FilesystemTool
from tools.http_client import HttpClient
from tools.image_analyze_tool import ImageAnalyzeTool
from tools.image_generate_tool import ImageGenerateTool
from tools.project_tool import ProjectTool
from tools.shell_tool import ShellTool
from tools.stt_tool import SttTool
from tools.tool_registry import ToolRegistry
from tools.tts_tool import TtsTool
from tools.web_search_tool import WebSearchTool
from tools.workspace_tools import (
    MAX_FILE_BYTES,
    WORKSPACE_ROOT,
    ApplyPatchTool,
    ListFilesTool,
    ReadFileTool,
    RunCodeTool,
    WriteFileTool,
)

DEFAULT_TOOLS = {
    "fs": True,
    "shell": False,
    "web": False,
    "project": True,
    "image_analyze": False,
    "image_generate": False,
    "tts": False,
    "stt": False,
    "workspace_run": True,
    "safe_mode": True,
}
SAFE_MODE_TOOLS_OFF = {
    "web",
    "shell",
    "project",
    "tts",
    "stt",
    "image_analyze",
    "image_generate",
    "workspace_run",
}
MAX_MWV_ATTEMPTS = 3
_DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS = 30
MAX_SHORT_TERM_MESSAGES = 20
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_MIN_BASE64_LEN = 64
_FEEDBACK_LABEL_HINTS: dict[FeedbackLabel, tuple[str, str]] = {
    FeedbackLabel.OFF_TOPIC: ("fatal", "Держись темы вопроса."),
    FeedbackLabel.HALLUCINATION: ("major", "Проверь факты и избегай галлюцинаций."),
    FeedbackLabel.INCORRECT: ("major", "Проверяй корректность ответа."),
    FeedbackLabel.NO_SOURCES: ("major", "Добавляй источники при необходимости."),
    FeedbackLabel.TOO_LONG: ("minor", "Делай ответ короче."),
    FeedbackLabel.TOO_COMPLEX: ("minor", "Упрощай объяснение."),
    FeedbackLabel.OTHER: ("minor", "Улучшай качество ответа."),
}


def _looks_like_base64(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < _MIN_BASE64_LEN or len(stripped) % 4 != 0:
        return False
    return bool(_BASE64_RE.fullmatch(stripped))


class Agent:
    """SlavikAI Core v1.0 — Распределённый рассуждающий агент."""

    def __init__(
        self,
        brain: Brain | None = None,
        critic: Brain | None = None,
        enable_tools: dict[str, bool] | None = None,
        main_config: ModelConfig | None = None,
        critic_config: ModelConfig | None = None,
        main_api_key: str | None = None,
        critic_api_key: str | None = None,
        brain_manager: BrainManager | None = None,
        user_id: str = "local",
        memory_companion_db_path: str | None = None,
    ) -> None:
        saved_main, saved_critic = load_model_configs()
        self.main_config = main_config or saved_main
        self.critic_config = critic_config or saved_critic
        self.main_api_key = main_api_key
        self.critic_api_key = critic_api_key
        self.shell_config_path = str(DEFAULT_SHELL_CONFIG_PATH)
        self._external_brain = brain
        self._external_critic = critic
        self._brain_manager = brain_manager
        self.user_id = user_id
        self.memory_config: MemoryConfig = load_memory_config()
        self._interaction_store = (
            MemoryCompanionStore(memory_companion_db_path)
            if memory_companion_db_path
            else MemoryCompanionStore()
        )
        self._rule_engine = RuleEngine()
        self.last_chat_interaction_id: str | None = None

        self.brain = self._build_brain()
        self.logger = logging.getLogger("SlavikAI.Agent")
        self.tracer = Tracer()
        self.planner = Planner()
        self.executor = Executor(self.tracer)
        self.auto_agent = AutoAgent(self)
        self.tools_enabled = enable_tools or self._load_tools()
        self.tool_registry = ToolRegistry(safe_block=SAFE_MODE_TOOLS_OFF)
        self.web_tool = WebSearchTool()
        self._register_tools()
        if self.tools_enabled.get("safe_mode", False):
            self._apply_safe_mode(True)
        self.memory = MemoryManager("memory/memory.db")
        self.vectors = VectorIndex("memory/vectors.db")
        self.short_term: list[LLMMessage] = []
        self.critic_short_term: list[LLMMessage] = []
        self.conversation_id = str(uuid.uuid4())
        self.session_id: str | None = None
        self.approved_categories: set[ApprovalCategory] = set()
        self.last_plan: TaskPlan | None = None
        self.last_plan_original: TaskPlan | None = None
        self.last_hints_used: list[str] = []
        self.last_hints_meta: list[dict[str, str]] = []
        self.last_context_text: str | None = None
        self.last_critic_response: str | None = None
        self.last_critic_status: str = "disabled"
        self.last_critic_reasons: list[str] = []
        self._critic_step_rejected = False
        self.last_approval_request: ApprovalRequest | None = None
        self.last_reasoning: str | None = None
        self.workspace_file_path: str | None = None
        self.workspace_file_content: str | None = None
        self.workspace_selection: str | None = None
        self._workspace_diff_baselines: dict[str, str] = {}
        self._workspace_diffs: dict[str, WorkspaceDiffEntry] = {}
        self._init_mode_from_config()

    def respond(self, messages: list[LLMMessage]) -> str:
        if not messages:
            return "[Пустое сообщение]"

        last_content = messages[-1].content.strip()
        record_in_history = self._should_record_in_history(last_content)
        try:
            if record_in_history:
                self._append_short_term(messages)
                self._append_short_term(messages, history=self.critic_short_term)
            self.tracer.log("user_input", last_content)
            self._reset_critic_state()
            self._reset_approval_state()
            self.last_reasoning = None
            self._reset_workspace_diffs()

            if last_content.lower().startswith("авто") or last_content.startswith("/auto"):
                auto_goal = last_content.replace("/auto", "").strip()
                result = self.handle_auto_command(auto_goal)
                self._log_chat_interaction(raw_input=last_content, response_text=result)
                if record_in_history:
                    self._append_short_term([LLMMessage(role="assistant", content=result)])
                return result

            if last_content.startswith("/"):
                return self.handle_tool_command(last_content)

            decision = classify_request(
                messages,
                last_content,
                context={"safe_mode": bool(self.tools_enabled.get("safe_mode", False))},
            )
            self.tracer.log(
                "routing_decision",
                decision.route,
                {"reason": decision.reason, "flags": decision.risk_flags},
            )
            if decision.route == "mwv":
                return self._run_mwv_flow(messages, last_content, decision, record_in_history)
            return self._run_chat_response(messages, last_content, record_in_history)
        except ApprovalRequired as exc:
            return self._handle_approval_required(
                exc.request,
                raw_input=last_content,
                record_in_history=record_in_history,
            )
        except Exception as exc:
            self.logger.exception("Agent.respond error: %s", exc)
            self.tracer.log("error", f"Ошибка Agent.respond: {exc}")
            error_text = f"[Ошибка ответа: {exc}]"
            try:
                self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            except Exception as log_exc:  # noqa: BLE001
                self.logger.error("Ошибка записи InteractionLog: %s", log_exc)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return error_text

    def _run_chat_response(
        self,
        messages: list[LLMMessage],
        last_content: str,
        record_in_history: bool,
    ) -> str:
        try:
            self.tracer.log("reasoning_start", "Генерация ответа моделью")
            policy_application = self._apply_policies(last_content)
            messages_with_context = self._build_context_messages(self.short_term, last_content)
            messages_with_context = self._append_policy_instructions(
                messages_with_context,
                policy_application,
            )
            reply = self._get_main_brain().generate(messages_with_context)
            reviewed = self._review_answer(reply.text)
            if self.main_config and self.main_config.thinking_enabled:
                self.last_reasoning = reply.reasoning
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            if self.memory_config.auto_save_dialogue:
                self.save_to_memory(last_content, reviewed)
            self._log_chat_interaction(
                raw_input=last_content,
                response_text=reviewed,
                applied_policy_ids=policy_application.applied_policy_ids,
            )
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=reviewed)])
            return reviewed
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM error: %s", exc)
            self.tracer.log("error", f"Ошибка модели: {exc}")
            error_text = f"[Ошибка модели: {exc}]"
            self._log_chat_interaction(raw_input=last_content, response_text=error_text)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
            return error_text

    def _run_mwv_flow(
        self,
        messages: list[LLMMessage],
        raw_input: str,
        decision: RouteDecision,
        record_in_history: bool,
    ) -> str:
        mwv_messages = self._to_mwv_messages(messages)
        context = self._build_mwv_context()
        manager = ManagerRuntime(task_builder=self._mwv_task_builder(decision))
        worker = WorkerRuntime(runner=self._mwv_worker_runner)
        verifier_runtime = VerifierRuntime()

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

        run_result = manager.run_flow(
            mwv_messages,
            context,
            worker=_worker,
            verifier=_verifier,
        )
        response = self._format_mwv_response(run_result)
        if self.memory_config.auto_save_dialogue:
            self.save_to_memory(raw_input, response)
        self._log_chat_interaction(raw_input=raw_input, response_text=response)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=response)])
        return response

    def _build_mwv_context(self) -> RunContext:
        session_id = self.session_id or "local"
        approved = sorted(self.approved_categories)
        return RunContext(
            session_id=session_id,
            trace_id=str(uuid.uuid4()),
            workspace_root=str(WORKSPACE_ROOT),
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
        if (
            result.work_result.status == WorkStatus.SUCCESS
            and result.verification_result.status == VerificationStatus.PASSED
        ):
            return f"Готово. Проверки пройдены.\n{result.work_result.summary}".strip()
        if result.work_result.status == WorkStatus.FAILURE:
            detail = self._summarize_work_failure(result.work_result)
            return f"Ошибка выполнения: {detail}".strip()
        if result.verification_result.status == VerificationStatus.ERROR:
            detail = summarize_verifier_failure(result.verification_result)
            return f"Ошибка проверки: {detail}".strip()
        detail = summarize_verifier_failure(result.verification_result)
        extra = ""
        if result.retry_decision and result.retry_decision.reason == "retry_limit_reached":
            extra = "Лимит попыток исчерпан."
        return "\n".join(part for part in [f"Проверки не прошли. {detail}", extra] if part).strip()

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

    def _get_main_brain(self) -> Brain:
        if isinstance(self.brain, DualBrain):
            if self.brain.mode == "critic-only":
                self.tracer.log("critic_mode_ignored", "Critic-only mode ignored in MWV routing")
            return self.brain.main
        return self.brain

    def handle_tool_command(self, command: str) -> str:
        parts = command.split()
        cmd = parts[0][1:].lower()
        args = parts[1:]
        self.tracer.log("tool_invoked", cmd, {"args": args})

        try:
            if cmd == "auto":
                goal = " ".join(args)
                result = self.handle_auto_command(goal)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "plan":
                goal = " ".join(args)
                plan = self.planner.build_plan(goal)
                self.last_plan_original = plan
                self.last_plan = plan
                executed: TaskPlan = self.executor.run(
                    plan,
                    tool_gateway=ToolGateway(
                        self.tool_registry,
                        pre_call=self._workspace_diff_pre_call,
                        post_call=self._workspace_diff_post_call,
                    ),
                )
                result = self._format_plan(executed)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "fs":
                operation = args[0] if args else "list"
                path_arg = args[1] if len(args) > 1 else ""
                req = ToolRequest(name="fs", args={"op": operation, "path": path_arg})
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "web":
                query = " ".join(args)
                req = ToolRequest(name="web", args={"query": query})
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "sh":
                req = ToolRequest(
                    name="shell",
                    args={
                        "command": " ".join(args),
                        "config_path": str(getattr(self, "shell_config_path", "")) or None,
                    },
                )
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "project":
                if not args:
                    result = "[Нужно указать подкоманду: index|find]"
                    self._log_chat_interaction(raw_input=command, response_text=result)
                    return result
                req = ToolRequest(name="project", args={"cmd": args[0], "args": args[1:]})
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd in {"imggen", "img_generate"}:
                prompt = " ".join(args) or "image"
                req = ToolRequest(name="image_generate", args={"prompt": prompt})
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd in {"imganalyze", "img_analyze"}:
                if not args:
                    result = "[Нужно указать base64 или путь]"
                    self._log_chat_interaction(raw_input=command, response_text=result)
                    return result
                raw_value = args[0].strip()
                if raw_value.startswith("base64:"):
                    payload = raw_value.removeprefix("base64:").strip()
                    req = ToolRequest(name="image_analyze", args={"base64": payload})
                elif _looks_like_base64(raw_value):
                    req = ToolRequest(name="image_analyze", args={"base64": raw_value})
                else:
                    req = ToolRequest(name="image_analyze", args={"path": raw_value})
                tool_result = self._call_tool_logged(command, req)
                result = self._format_tool_result(tool_result)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            if cmd == "trace":
                logs = self.tracer.read_recent(40)
                lines: list[str] = []
                for log in logs:
                    timestamp = log.get("timestamp", "?")
                    event = log.get("event", "?")
                    message = log.get("message", "")
                    lines.append(f"[{timestamp}] {event}: {message}")
                result = "\n".join(lines)
                self._log_chat_interaction(raw_input=command, response_text=result)
                return result

            unknown = f"[Инструмент '{cmd}' неактивен или неизвестен]"
            self._log_tool_interaction(
                raw_input=command,
                request=ToolRequest(name=cmd, args={"args": args}),
                result=ToolResult.failure(f"Инструмент {cmd} не зарегистрирован"),
            )
            self._log_chat_interaction(raw_input=command, response_text=unknown)
            return unknown
        except ApprovalRequired as exc:
            return self._handle_approval_required(
                exc.request,
                raw_input=command,
                record_in_history=False,
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("error", f"Ошибка при вызове инструмента: {exc}")
            error_text = f"[Ошибка при вызове инструмента: {exc}]"
            self._log_chat_interaction(raw_input=command, response_text=error_text)
            return error_text

    def _should_record_in_history(self, content: str) -> bool:
        if content.startswith("/"):
            return False
        if content.lower().startswith("авто"):
            return False
        return True

    def _append_short_term(
        self,
        messages: list[LLMMessage],
        *,
        history: list[LLMMessage] | None = None,
    ) -> None:
        target = history if history is not None else self.short_term
        for message in messages:
            if message.role not in {"user", "assistant"}:
                continue
            target.append(message)
        self._trim_short_term(target)

    def _trim_short_term(self, history: list[LLMMessage]) -> None:
        if len(history) <= MAX_SHORT_TERM_MESSAGES:
            return
        overflow = len(history) - MAX_SHORT_TERM_MESSAGES
        del history[:overflow]

    def _reset_workspace_diffs(self) -> None:
        self._workspace_diff_baselines.clear()
        self._workspace_diffs.clear()

    def _reset_critic_state(self) -> None:
        self.last_critic_response = None
        self.last_critic_status = "disabled"
        self.last_critic_reasons = []
        self._critic_step_rejected = False

    def _reset_approval_state(self) -> None:
        self.last_approval_request = None

    def _current_mode(self) -> CriticMode:
        if isinstance(self.brain, DualBrain):
            mode = self.brain.mode
            if mode == "single":
                return "single"
            if mode == "dual":
                return "dual"
            if mode == "critic-only":
                return "critic-only"
        return "single"

    def set_session_context(
        self,
        session_id: str | None,
        approved_categories: set[ApprovalCategory],
    ) -> None:
        self.session_id = session_id
        self.approved_categories = set(approved_categories)

    def _approval_context(self) -> ApprovalContext:
        safe_mode = bool(self.tools_enabled.get("safe_mode", False))
        normalized: set[ApprovalCategory] = set(self.approved_categories)
        return ApprovalContext(
            safe_mode=safe_mode,
            session_id=self.session_id,
            approved_categories=normalized,
        )

    def _build_tool_gateway(
        self,
        *,
        pre_call: Callable[[ToolRequest], object | None] | None = None,
        post_call: (Callable[[ToolRequest, ToolResult, object | None], None] | None) = None,
    ) -> ToolGateway:
        return ToolGateway(
            self.tool_registry,
            pre_call=pre_call,
            post_call=post_call,
            approval_context=self._approval_context(),
            log_event=self.tracer.log,
        )

    def consume_workspace_diffs(self) -> list[WorkspaceDiffEntry]:
        diffs = list(self._workspace_diffs.values())
        self._workspace_diff_baselines.clear()
        self._workspace_diffs.clear()
        return diffs

    def _normalize_workspace_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        try:
            return normalize_sandbox_path(raw_path, WORKSPACE_ROOT)
        except Exception:  # noqa: BLE001
            return None

    def _read_workspace_text(self, path: Path) -> str | None:
        try:
            if not path.exists() or not path.is_file():
                return ""
            if path.stat().st_size > MAX_FILE_BYTES:
                return None
            return path.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            return None

    def _workspace_diff_pre_call(self, request: ToolRequest) -> str | None:
        if request.name not in {"workspace_write", "workspace_patch"}:
            return None
        if request.name == "workspace_patch" and bool(request.args.get("dry_run", False)):
            return None
        raw_path = request.args.get("path")
        if not isinstance(raw_path, str):
            return None
        path = self._normalize_workspace_path(raw_path)
        if path is None:
            return None
        before = self._read_workspace_text(path)
        if before is None:
            return None
        rel_path = str(path.relative_to(WORKSPACE_ROOT))
        self._workspace_diff_baselines.setdefault(rel_path, before)
        return rel_path

    def _workspace_diff_post_call(
        self,
        request: ToolRequest,
        result: ToolResult,
        context: object | None,
    ) -> None:
        if not isinstance(context, str) or not context:
            return
        if not result.ok:
            return
        if request.name == "workspace_patch" and bool(result.data.get("dry_run", False)):
            return
        raw_path = request.args.get("path")
        if not isinstance(raw_path, str):
            return
        path = self._normalize_workspace_path(raw_path)
        if path is None:
            return
        after = self._read_workspace_text(path)
        if after is None:
            return
        baseline = self._workspace_diff_baselines.get(context)
        if baseline is None:
            baseline = ""
        diff_text = self._build_workspace_diff(baseline, after, context)
        if not diff_text.strip():
            self._workspace_diffs.pop(context, None)
            return
        added, removed = self._count_diff_lines(diff_text)
        self._workspace_diffs[context] = WorkspaceDiffEntry(
            path=context,
            added=added,
            removed=removed,
            diff=diff_text,
        )

    def _build_workspace_diff(self, before: str, after: str, label: str) -> str:
        diff_lines = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{label}",
            tofile=f"b/{label}",
            lineterm="",
        )
        return "\n".join(diff_lines)

    def _count_diff_lines(self, diff_text: str) -> tuple[int, int]:
        added = 0
        removed = 0
        for line in diff_text.splitlines():
            if line.startswith(("+++ ", "--- ", "@@")):
                continue
            if line.startswith("+"):
                added += 1
            elif line.startswith("-"):
                removed += 1
        return added, removed

    def call_tool(
        self,
        name: str,
        args: dict[str, JSONValue] | None = None,
        raw_input: str | None = None,
    ) -> ToolResult:
        request = ToolRequest(name=name, args=args or {})
        return self._call_tool_logged(raw_input or f"tool:{name}", request)

    def _call_tool_logged(self, raw_input: str, request: ToolRequest) -> ToolResult:
        pre_call = None
        post_call = None
        if not raw_input.startswith("ui:"):
            pre_call = self._workspace_diff_pre_call
            post_call = self._workspace_diff_post_call
        gateway = self._build_tool_gateway(pre_call=pre_call, post_call=post_call)
        try:
            result = gateway.call(request)
        except ApprovalRequired:
            result = ToolResult.failure("Требуется подтверждение")
            self._log_tool_interaction(raw_input=raw_input, request=request, result=result)
            raise
        self._log_tool_interaction(raw_input=raw_input, request=request, result=result)
        return result

    def _log_chat_interaction(
        self,
        raw_input: str,
        response_text: str,
        *,
        retrieved_memory_ids: list[str] | None = None,
        applied_policy_ids: list[str] | None = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        log = ChatInteractionLog(
            interaction_id=interaction_id,
            user_id=self.user_id,
            interaction_kind=InteractionKind.CHAT,
            raw_input=raw_input,
            mode=InteractionMode.STANDARD,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            response_text=response_text,
            retrieved_memory_ids=retrieved_memory_ids or [],
            applied_policy_ids=applied_policy_ids or [],
        )
        self._interaction_store.log_interaction(log)
        self.last_chat_interaction_id = interaction_id
        self.tracer.log(
            "interaction_logged",
            "Chat interaction stored",
            {"interaction_id": interaction_id},
        )
        return interaction_id

    def _log_tool_interaction(
        self,
        raw_input: str,
        request: ToolRequest,
        result: ToolResult,
    ) -> None:
        status, blocked_reason = self._classify_tool_result(result)
        output_preview = None
        if result.ok:
            if "output" in result.data:
                output_preview = str(result.data.get("output") or "")
            else:
                output_preview = str(result.data)
        log = ToolInteractionLog(
            interaction_id=str(uuid.uuid4()),
            user_id=self.user_id,
            interaction_kind=InteractionKind.TOOL,
            raw_input=raw_input,
            mode=InteractionMode.STANDARD,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            tool_name=request.name,
            tool_args=request.args,
            tool_status=status,
            blocked_reason=blocked_reason,
            tool_output_preview=output_preview,
            tool_error=None if result.ok else (result.error or "unknown error"),
            tool_meta=result.meta,
        )
        self._interaction_store.log_interaction(log)

    def _classify_tool_result(self, result: ToolResult) -> tuple[ToolStatus, BlockedReason | None]:
        if result.ok:
            return ToolStatus.OK, None
        error = (result.error or "").strip()
        error_lower = error.lower()

        approval_markers = ("требуется подтверждение", "approval required")
        if any(marker in error_lower for marker in approval_markers):
            return ToolStatus.BLOCKED, BlockedReason.APPROVAL_REQUIRED

        if error == "Safe mode: инструмент отключён":
            return ToolStatus.BLOCKED, BlockedReason.SAFE_MODE_BLOCKED
        if "не зарегистрирован" in error_lower:
            return ToolStatus.BLOCKED, BlockedReason.TOOL_NOT_REGISTERED
        if error.startswith("Инструмент ") and error.endswith(" отключён"):
            return ToolStatus.BLOCKED, BlockedReason.TOOL_DISABLED

        sandbox_markers = (
            "sandbox violation",
            "путь вне",
            "песочниц",
            "sandbox_root",
            "выход за пределы песоч",
        )
        if any(marker in error_lower for marker in sandbox_markers):
            return ToolStatus.BLOCKED, BlockedReason.SANDBOX_VIOLATION

        validation_markers = (
            "не указан",
            "нужны ",
            "должен быть",
            "некоррект",
            "неизвестн",
            "запрещ",
            "опасн",
            "цепоч",
            "команда пуста",
        )
        if any(marker in error_lower for marker in validation_markers):
            return ToolStatus.BLOCKED, BlockedReason.VALIDATION_ERROR

        return ToolStatus.ERROR, None

    def _apply_policies(self, user_message: str) -> PolicyApplication:
        rules = self._interaction_store.list_policy_rules(self.user_id)
        return self._rule_engine.apply(user_message=user_message, rules=rules)

    def _append_policy_instructions(
        self,
        messages: list[LLMMessage],
        policy_application: PolicyApplication,
    ) -> list[LLMMessage]:
        if not policy_application.instructions:
            return messages
        lines = [
            "Политики (approved):",
            *[f"- {t}" for t in policy_application.instructions],
        ]
        return [*messages, LLMMessage(role="system", content="\n".join(lines))]

    def record_feedback_event(
        self,
        *,
        interaction_id: str,
        rating: FeedbackRating,
        labels: list[FeedbackLabel] | None = None,
        free_text: str | None = None,
    ) -> None:
        cleaned_free_text = free_text.strip() if free_text else ""
        normalized_free_text = cleaned_free_text if cleaned_free_text else None

        unique_labels: list[FeedbackLabel] = []
        seen: set[FeedbackLabel] = set()
        for label in labels or []:
            if label in seen:
                continue
            unique_labels.append(label)
            seen.add(label)

        event = FeedbackEvent(
            feedback_id=str(uuid.uuid4()),
            interaction_id=interaction_id,
            user_id=self.user_id,
            rating=rating,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            labels=unique_labels,
            free_text=normalized_free_text,
        )
        self._interaction_store.add_feedback_event(event)
        self.tracer.log(
            "feedback_event_saved",
            rating.value,
            {
                "labels": [label.value for label in unique_labels],
                "interaction_id": interaction_id,
            },
        )

    def handle_auto_command(self, goal: str) -> str:
        """Создаёт подагентов и выполняет задачу параллельно."""
        self.tracer.log("auto_invoke", f"Создание автоагентов для: {goal}")
        return self.auto_agent.auto_execute(goal)

    def save_to_memory(self, prompt: str, answer: str) -> None:
        item = MemoryRecord(
            id=str(uuid.uuid4()),
            content=f"Q: {prompt}\nA: {answer}",
            kind=MemoryKind.NOTE,
            tags=["dialogue"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            meta={},
        )
        self.memory.save(item)
        self.tracer.log("memory_saved", prompt[:100])

    def set_mode(self, mode: str) -> None:
        if isinstance(self.brain, DualBrain):
            self.brain.set_mode(mode)
            self.tracer.log("mode_set", mode)
        elif mode != "single":
            raise ValueError("Режимы dual/critic-only недоступны без критика.")
        save_mode(mode)

    def reconfigure_models(
        self,
        main_config: ModelConfig,
        critic_config: ModelConfig | None = None,
        main_api_key: str | None = None,
        critic_api_key: str | None = None,
    ) -> None:
        """Переинициализирует мозги с новыми настройками."""
        self.main_config = main_config
        self.critic_config = critic_config
        self.main_api_key = main_api_key
        self.critic_api_key = critic_api_key
        self.brain = self._build_brain()
        save_model_configs(self.main_config, self.critic_config)
        self.tracer.log("brain_reconfigured", "Мозг переинициализирован")

    def _format_plan(self, plan: TaskPlan) -> str:
        lines: list[str] = []
        for index, step in enumerate(plan.steps, start=1):
            status_key = step.status.value if hasattr(step.status, "value") else str(step.status)
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "done": "✅",
                "error": "❌",
            }.get(status_key, "•")
            result_preview = f" — {step.result}" if step.result else ""
            lines.append(f"{index}. {status_icon} {step.description}{result_preview}")
        return "\n".join(lines)

    def _format_tool_result(self, result: ToolResult) -> str:
        if result.ok:
            if "output" in result.data:
                return str(result.data["output"])
            return str(result.data)
        error = result.error or "Неизвестная ошибка"
        return f"[Ошибка инструмента: {error}]"

    def _record_critic_decision(self, decision: CriticDecision, mode: str) -> None:
        self.last_critic_reasons = list(decision.reasons)
        self.tracer.log(
            "critic_decision",
            "Critic decision",
            {
                "mode": mode,
                "should_run": decision.should_run_critic,
                "reasons": decision.reasons,
            },
        )

    def _finalize_critic_status(self, decision: CriticDecision, critic_text: str | None) -> None:
        if self._critic_step_rejected:
            self.last_critic_status = "risky"
        else:
            self.last_critic_status = classify_critic_status(
                decision=decision, critic_text=critic_text
            )
        self.last_critic_response = critic_text
        meta: dict[str, JSONValue] = {"status": self.last_critic_status}
        if critic_text is not None:
            meta["text"] = critic_text
        preview = (critic_text or "")[:120]
        self.tracer.log("critic_response", preview or "Critic response", meta)

    def _handle_critic_failure(
        self,
        exc: Exception,
        decision: CriticDecision,
        *,
        raw_input: str | None = None,
        record_in_history: bool = False,
    ) -> str:
        self.last_critic_status = "internal_error"
        self.last_critic_response = None
        self.tracer.log(
            "critic_error",
            f"Ошибка критика: {exc}",
            {"reasons": decision.reasons},
        )
        error_text = f"[Ошибка критика: {exc}]"
        if raw_input is not None:
            self._log_chat_interaction(raw_input=raw_input, response_text=error_text)
            if record_in_history:
                self._append_short_term([LLMMessage(role="assistant", content=error_text)])
        return error_text

    def _handle_approval_required(
        self,
        request: ApprovalRequest,
        *,
        raw_input: str,
        record_in_history: bool = False,
    ) -> str:
        self.last_approval_request = request
        error_text = "[Требуется подтверждение действия]"
        self._log_chat_interaction(raw_input=raw_input, response_text=error_text)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=error_text)])
        return error_text

    def _run_answer_critic(self, *, user_message: str, main_reply: str) -> str:
        critic = self._get_critic_brain()
        if critic is None:
            raise CriticFailure("Критик недоступен.")
        critic_messages = self._build_critic_messages(
            user_message=user_message,
            main_reply=main_reply,
        )
        try:
            critic_reply = critic.generate(critic_messages)
        except Exception as exc:  # noqa: BLE001
            raise CriticFailure(str(exc)) from exc
        critic_text = critic_reply.text
        self.tracer.log(
            "critic_review",
            "Ответ проверен критиком",
            {"stage": "answer"},
        )
        return critic_text

    def _run_critic_only(self, *, user_message: str) -> str:
        critic = self._get_critic_brain()
        if critic is None:
            raise CriticFailure("Критик недоступен.")
        critic_messages = self._build_critic_only_messages(user_message=user_message)
        try:
            critic_reply = critic.generate(critic_messages)
        except Exception as exc:  # noqa: BLE001
            raise CriticFailure(str(exc)) from exc
        critic_text = critic_reply.text
        self.tracer.log(
            "critic_review",
            "Ответ критика получен",
            {"stage": "critic-only"},
        )
        return critic_text

    def _review_plan(
        self,
        plan: TaskPlan,
        *,
        store_critic: bool = False,
        enforce_critic: bool = False,
    ) -> tuple[str, str | None]:
        plan_text = self._format_plan(plan)
        if isinstance(self.brain, DualBrain) and self.brain.mode != "single":
            try:
                critic_messages = [
                    LLMMessage(role="system", content=CRITIC_PROMPT),
                    LLMMessage(
                        role="user",
                        content=f"Проверь план и предложи улучшения:\n{plan_text}",
                    ),
                ]
                critic = self._get_critic_brain()
                if critic is None:
                    raise CriticFailure("Критик недоступен.")
                review = critic.generate(critic_messages)
                review_text = review.text
                self.tracer.log(
                    "critic_review",
                    "План проверен критиком",
                    {"stage": "plan", "text": review_text},
                )
                if store_critic:
                    return plan_text, review_text
                return f"{plan_text}\n\n🧠 Критик плана:\n{review_text}", None
            except CriticFailure as exc:
                self.tracer.log("critic_error", f"Критик не доступен: {exc}")
                if enforce_critic:
                    raise
            except Exception as exc:  # noqa: BLE001
                self.tracer.log("critic_error", f"Критик не доступен: {exc}")
                if enforce_critic:
                    raise CriticFailure(str(exc)) from exc
                self.logger.warning("Не удалось получить критику плана: %s", exc)
        return plan_text, None

    def _critic_plan(self, plan: TaskPlan, *, enforce_critic: bool = False) -> TaskPlan:
        if not isinstance(self.brain, DualBrain):
            return plan
        plan_text = self._format_plan(plan)
        try:
            critic_messages = [
                LLMMessage(role="system", content=CRITIC_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        "Проверь план и предложи улучшения. Верни только список шагов:\n"
                        f"{plan_text}"
                    ),
                ),
            ]
            critic = self._get_critic_brain()
            if critic is None:
                raise CriticFailure("Критик недоступен.")
            review = critic.generate(critic_messages)
            review_text = review.text
            self.tracer.log(
                "critic_review",
                "План переписан критиком",
                {"stage": "plan_rewrite", "text": review_text},
            )
            new_steps = self.planner.parse_plan_text(review_text) or []
            if len(new_steps) >= 2:
                rewritten = TaskPlan(
                    goal=plan.goal,
                    steps=[PlanStep(description=s) for s in new_steps],
                )
                return self.planner.assign_operations(rewritten)
        except CriticFailure as exc:
            self.tracer.log("critic_error", f"Критик не доступен: {exc}")
            if enforce_critic:
                raise
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("critic_error", f"Критик не доступен: {exc}")
            if enforce_critic:
                raise CriticFailure(str(exc)) from exc
            self.logger.warning("Не удалось получить критику плана: %s", exc)
        return plan

    def _critic_step(
        self, step: PlanStep, *, enforce_critic: bool = False
    ) -> tuple[bool, str | None]:
        if not isinstance(self.brain, DualBrain):
            return True, None
        try:
            critic_messages = [
                LLMMessage(role="system", content=CRITIC_PROMPT),
                LLMMessage(
                    role="user",
                    content=(
                        f"Оцени шаг плана: '{step.description}'. "
                        "Ответь 'approve' если выполнять безопасно, иначе укажи краткую причину."
                    ),
                ),
            ]
            critic = self._get_critic_brain()
            if critic is None:
                raise CriticFailure("Критик недоступен.")
            review = critic.generate(critic_messages)
            review_text = review.text
            if "approve" in review_text.lower():
                self.tracer.log(
                    "critic_step_approved",
                    step.description,
                )
                return True, None
            self._critic_step_rejected = True
            self.tracer.log(
                "critic_step_rejected",
                step.description,
                {"note": review_text.strip()},
            )
            return False, review_text.strip()
        except CriticFailure as exc:
            if enforce_critic:
                raise
            self.tracer.log("critic_error", f"Критик не доступен: {exc}")
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("critic_error", f"Критик не доступен: {exc}")
            if enforce_critic:
                raise CriticFailure(str(exc)) from exc
            return False, str(exc)

    def _should_use_step_critic(self) -> bool:
        return isinstance(self.brain, DualBrain) and self.brain.mode == "dual"

    def _should_use_dual_response(self) -> bool:
        return isinstance(self.brain, DualBrain) and self.brain.mode == "dual"

    def get_primary_brain(self) -> Brain:
        if isinstance(self.brain, DualBrain):
            if self.brain.mode == "critic-only":
                return self.brain.critic
            return self.brain.main
        return self.brain

    def _get_primary_brain(self) -> Brain:
        return self.get_primary_brain()

    def _get_critic_brain(self) -> Brain | None:
        if isinstance(self.brain, DualBrain):
            return self.brain.critic
        return None

    def _get_plan_brain_config(self) -> tuple[Brain | None, ModelConfig | None]:
        if isinstance(self.brain, DualBrain):
            if self.brain.mode == "critic-only":
                return self.brain.critic, self.critic_config or self.main_config
            return self.brain.main, self.main_config
        return self.brain, self.main_config

    def _history_without_latest_user(self, history: list[LLMMessage]) -> list[LLMMessage]:
        if history and history[-1].role == "user":
            return history[:-1]
        return history

    def _append_context_message(
        self,
        messages: list[LLMMessage],
        context_text: str | None,
    ) -> list[LLMMessage]:
        if not context_text:
            return messages
        return [*messages, LLMMessage(role="system", content=context_text)]

    def _build_critic_messages(
        self,
        *,
        user_message: str,
        main_reply: str,
    ) -> list[LLMMessage]:
        history = self._history_without_latest_user(self.critic_short_term)
        messages = self._append_context_message(history, self.last_context_text)
        critic_prompt = (
            "Пользователь:\n"
            f"{user_message}\n\n"
            "Ответ основной модели:\n"
            f"{main_reply}\n\n"
            "Дай краткую критику/улучшения."
        )
        return [
            *messages,
            LLMMessage(role="system", content=CRITIC_PROMPT),
            LLMMessage(role="user", content=critic_prompt),
        ]

    def _build_critic_only_messages(
        self,
        *,
        user_message: str,
        plan_text: str | None = None,
    ) -> list[LLMMessage]:
        history = self._history_without_latest_user(self.critic_short_term)
        messages = self._append_context_message(history, self.last_context_text)
        if plan_text:
            critic_prompt = (
                "Пользователь:\n"
                f"{user_message}\n\n"
                "План:\n"
                f"{plan_text}\n\n"
                "Дай краткую критику и возможные риски."
            )
        else:
            critic_prompt = (
                f"Пользователь:\n{user_message}\n\nДай краткую критику и возможные риски."
            )
        return [
            *messages,
            LLMMessage(role="system", content=CRITIC_PROMPT),
            LLMMessage(role="user", content=critic_prompt),
        ]

    def _review_answer(self, answer: str) -> str:
        return answer

    def _build_brain(self) -> Brain:
        if self._brain_manager:
            return self._brain_manager.build()
        if self._external_brain:
            if self._external_critic:
                return DualBrain(self._external_brain, self._external_critic)
            return self._external_brain
        if self.main_config is None:
            raise RuntimeError("Не выбрана модель. Укажите model id в настройках.")
        main_brain = create_brain(self.main_config, api_key=self.main_api_key)
        critic_brain = (
            create_brain(self.critic_config, api_key=self.critic_api_key)
            if self.critic_config
            else None
        )
        return DualBrain(main_brain, critic_brain) if critic_brain else main_brain

    def _register_tools(self) -> None:
        self.tool_registry.register(
            "fs",
            FilesystemTool(),
            enabled=self.tools_enabled.get("fs", False),
        )
        self.tool_registry.register(
            "web",
            self.web_tool.handle,
            enabled=self.tools_enabled.get("web", False),
        )
        self.tool_registry.register(
            "shell",
            ShellTool(),
            enabled=self.tools_enabled.get("shell", False),
        )
        self.tool_registry.register(
            "project",
            ProjectTool(),
            enabled=self.tools_enabled.get("project", False),
        )
        self.tool_registry.register(
            "image_analyze",
            ImageAnalyzeTool(),
            enabled=self.tools_enabled.get("image_analyze", False),
        )
        self.tool_registry.register(
            "image_generate",
            ImageGenerateTool(),
            enabled=self.tools_enabled.get("image_generate", False),
        )
        http_client = HttpClient()
        self.tool_registry.register(
            "tts",
            TtsTool(http_client),
            enabled=self.tools_enabled.get("tts", False),
        )
        self.tool_registry.register(
            "stt",
            SttTool(http_client),
            enabled=self.tools_enabled.get("stt", False),
        )
        self.tool_registry.register("workspace_list", ListFilesTool(), enabled=True)
        self.tool_registry.register("workspace_read", ReadFileTool(), enabled=True)
        self.tool_registry.register("workspace_write", WriteFileTool(), enabled=True)
        self.tool_registry.register("workspace_patch", ApplyPatchTool(), enabled=True)
        self.tool_registry.register(
            "workspace_run",
            RunCodeTool(),
            enabled=self.tools_enabled.get("workspace_run", True),
        )

    def synthesize_speech(
        self,
        text: str,
        voice_id: str | None = None,
        fmt: str | None = None,
    ) -> ToolResult:
        args: dict[str, JSONValue] = {"text": text}
        if voice_id:
            args["voice_id"] = voice_id
        if fmt:
            args["format"] = fmt
        return self.call_tool("tts", args=args, raw_input="api:tts")

    def transcribe_audio(self, file_path: str, language: str | None = None) -> ToolResult:
        args: dict[str, JSONValue] = {"file_path": file_path}
        if language:
            args["language"] = language
        return self.call_tool("stt", args=args, raw_input="api:stt")

    def set_workspace_context(
        self,
        path: str | None,
        content: str | None,
        selection: str | None = None,
    ) -> None:
        """Сохраняет текущий контекст файла для LLM."""
        self.workspace_file_path = path
        self.workspace_file_content = content
        self.workspace_selection = selection

    def _init_mode_from_config(self) -> None:
        try:
            mode = load_mode()
            if isinstance(self.brain, DualBrain):
                self.brain.set_mode(mode)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Не удалось загрузить режим: %s", exc)

    def _load_tools(self) -> dict[str, bool]:
        try:
            return load_tools_config().to_dict()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Не удалось загрузить инструменты, используем значения по умолчанию: %s",
                exc,
            )
            return DEFAULT_TOOLS.copy()

    def get_available_tool_keys(self) -> list[str]:
        return [key for key in self.tools_enabled.keys() if key != "safe_mode"]

    def update_tools_enabled(self, state: dict[str, bool]) -> None:
        self.tools_enabled.update(state)
        save_tools_config(ToolsConfig(**self.tools_enabled))
        self.tracer.log("tools_updated", "Инструменты обновлены", {"tools": self.tools_enabled})
        for name, enabled in state.items():
            if name in self.tool_registry.list_tools():
                self.tool_registry.set_enabled(name, enabled)
        if state.get("safe_mode") is not None:
            self._apply_safe_mode(state["safe_mode"])

    def _apply_safe_mode(self, enabled: bool) -> None:
        self.tool_registry.apply_safe_mode(enabled)
        if enabled:
            self.tracer.log("safe_mode", "Safe mode enabled, unsafe tools disabled")
        else:
            self.tracer.log("safe_mode", "Safe mode disabled")

    def get_recent_tool_calls(self, limit: int = 50) -> list[ToolCallRecord]:
        return self.tool_registry.read_recent_calls(limit)

    def get_recent_feedback_events(self, limit: int = 50) -> list[FeedbackEvent]:
        return self._interaction_store.get_recent_feedback(user_id=self.user_id, limit=limit)

    def get_feedback_stats(self) -> dict[FeedbackRating, int]:
        return self._interaction_store.get_feedback_stats(user_id=self.user_id)

    def get_interaction_log(self, interaction_id: str) -> InteractionLog | None:
        return self._interaction_store.get_interaction(interaction_id)

    def run_batch_review(self, *, period_days: int) -> BatchReviewRun:
        reviewer = BatchReviewer(self._interaction_store)
        result = reviewer.run(user_id=self.user_id, period_days=period_days)
        self.tracer.log(
            "batch_review_completed",
            f"candidates={result.run.candidate_count}",
            {"run_id": result.run.batch_review_run_id, "period_days": period_days},
        )
        return result.run

    def get_recent_batch_review_runs(self, limit: int = 20) -> list[BatchReviewRun]:
        return self._interaction_store.get_recent_batch_review_runs(
            user_id=self.user_id,
            limit=limit,
        )

    def list_policy_rule_candidates(
        self,
        *,
        run_id: str | None = None,
        status: CandidateStatus | None = None,
        limit: int = 200,
    ) -> list[PolicyRuleCandidate]:
        return self._interaction_store.list_policy_rule_candidates(
            user_id=self.user_id,
            run_id=run_id,
            status=status,
            limit=limit,
        )

    def approve_policy_rule_candidate(
        self,
        *,
        candidate_id: str,
        scope: PolicyScope = PolicyScope.USER,
        decay_half_life_days: int = _DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS,
        override_trigger_json: str | None = None,
        override_action_json: str | None = None,
        override_priority: int | None = None,
        override_confidence: float | None = None,
    ) -> PolicyRule:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")
        if decay_half_life_days <= 0:
            raise ValueError("decay_half_life_days должен быть > 0.")

        trigger = (
            policy_trigger_from_json(override_trigger_json)
            if override_trigger_json is not None
            else candidate.proposed_trigger
        )
        action = (
            policy_action_from_json(override_action_json)
            if override_action_json is not None
            else candidate.proposed_action
        )
        priority = (
            override_priority if override_priority is not None else candidate.priority_suggestion
        )
        confidence = (
            float(override_confidence)
            if override_confidence is not None
            else candidate.confidence_suggestion
        )
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("confidence должен быть в диапазоне 0..1.")

        feedback_ids = sorted({e.feedback_id for e in candidate.evidence if e.feedback_id})
        provenance = (
            f"batch_review_run_id:{candidate.batch_review_run_id};"
            f"candidate_id:{candidate.candidate_id}"
        )
        if len(feedback_ids) == 1:
            provenance += f";feedback_id:{feedback_ids[0]}"
        elif feedback_ids:
            provenance += f";feedback_ids:{','.join(feedback_ids)}"

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        rule = PolicyRule(
            rule_id=str(uuid.uuid4()),
            user_id=self.user_id,
            scope=scope,
            trigger=trigger,
            action=action,
            priority=priority,
            confidence=confidence,
            decay_half_life_days=decay_half_life_days,
            provenance=provenance,
            created_at=now,
            updated_at=now,
        )

        self._interaction_store.approve_policy_rule_candidate(
            candidate_id=candidate_id,
            user_id=self.user_id,
            approved_rule=rule,
            final_trigger=trigger,
            final_action=action,
            final_priority=priority,
            final_confidence=confidence,
            updated_at=now,
        )
        self.tracer.log(
            "policy_rule_approved",
            rule.rule_id,
            {"candidate_id": candidate_id, "run_id": candidate.batch_review_run_id},
        )
        return rule

    def update_policy_rule_candidate_suggestion(
        self,
        *,
        candidate_id: str,
        proposed_trigger_json: str,
        proposed_action_json: str,
        priority_suggestion: int,
        confidence_suggestion: float,
    ) -> PolicyRuleCandidate:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")

        trigger = policy_trigger_from_json(proposed_trigger_json)
        action = policy_action_from_json(proposed_action_json)
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self._interaction_store.update_policy_rule_candidate_suggestion(
            candidate_id=candidate_id,
            user_id=self.user_id,
            proposed_trigger=trigger,
            proposed_action=action,
            priority_suggestion=priority_suggestion,
            confidence_suggestion=confidence_suggestion,
            updated_at=now,
        )
        self.tracer.log("policy_candidate_updated", candidate_id)
        updated = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if updated is None:
            raise RuntimeError("Candidate missing after update (unexpected).")
        return updated

    def reject_policy_rule_candidate(self, *, candidate_id: str) -> None:
        candidate = self._interaction_store.get_policy_rule_candidate(candidate_id=candidate_id)
        if candidate is None:
            raise ValueError(f"Candidate not found: {candidate_id!r}")
        if candidate.user_id != self.user_id:
            raise ValueError("Candidate принадлежит другому user_id.")
        if candidate.status is not CandidateStatus.PROPOSED:
            raise ValueError(f"Candidate status must be proposed, got: {candidate.status.value!r}")

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self._interaction_store.reject_policy_rule_candidate(
            candidate_id=candidate_id,
            user_id=self.user_id,
            updated_at=now,
        )
        self.tracer.log("policy_candidate_rejected", candidate_id)

    def _build_context_messages(self, messages: list[LLMMessage], query: str) -> list[LLMMessage]:
        context_parts: list[str] = []

        recent_notes = self.memory.get_recent(3, kind=MemoryKind.NOTE)
        if recent_notes:
            context_parts.append("Недавняя память:")
            for note in recent_notes:
                context_parts.append(f"- {note.content[:200]}")

        hints_meta = self._collect_feedback_hints(2, severity_filter=["major", "fatal"])
        if hints_meta:
            context_parts.append("Подсказки от пользователя:")
            for hint_meta in hints_meta:
                context_parts.append(f"- ({hint_meta.get('severity')}) {hint_meta.get('hint')}")
            self.last_hints_used = [h["hint"] for h in hints_meta]
            self.last_hints_meta = hints_meta
            self.tracer.log("auto_hint_applied", "Использованы подсказки", {"hints": hints_meta})
        else:
            self.last_hints_used = []
            self.last_hints_meta = []

        prefs = self.memory.get_user_prefs()
        if prefs:
            context_parts.append("Предпочтения пользователя:")
            for pref in prefs:
                meta = pref.meta or {}
                context_parts.append(f"- {meta.get('key')}: {meta.get('value')}")

        # Векторный поиск по проектному индексу (code + docs)
        try:
            vec_results_code = self.vectors.search(query, namespace="code", top_k=3)
            vec_results_docs = self.vectors.search(query, namespace="docs", top_k=3)
            vec_results = [*vec_results_code, *vec_results_docs]
            if vec_results:
                context_parts.append("Контекст проекта (code/docs):")
                for res in vec_results:
                    context_parts.append(f"- {res.path}: {res.snippet}")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Vector search failed: %s", exc)

        if self.workspace_file_path and self.workspace_file_content is not None:
            context_parts.append("Текущий файл:")
            context_parts.append(f"- path: {self.workspace_file_path}")
            if self.workspace_selection:
                selection_snippet = self.workspace_selection[:1200]
                context_parts.append(f"- выделение:\n{selection_snippet}")
            content_snippet = self.workspace_file_content
            if len(content_snippet) > 6000:
                content_snippet = content_snippet[:6000]
            context_parts.append(f"- содержимое:\n{content_snippet}")

        if context_parts:
            context_msg = "\n".join(context_parts)
            self.last_context_text = context_msg
            return [*messages, LLMMessage(role="system", content=context_msg)]
        self.last_context_text = None
        return messages

    def _collect_feedback_hints(
        self,
        limit: int,
        severity_filter: list[str] | None = None,
    ) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        scan_limit = max(limit * 5, limit)
        events = self._interaction_store.get_recent_feedback(user_id=self.user_id, limit=scan_limit)
        hints: list[dict[str, str]] = []
        for event in events:
            meta = self._feedback_event_to_hint(event)
            if not meta:
                continue
            severity = meta.get("severity")
            if severity_filter and severity not in severity_filter:
                continue
            hints.append(meta)
            if len(hints) >= limit:
                break
        return hints

    def _feedback_event_to_hint(self, event: FeedbackEvent) -> dict[str, str] | None:
        if event.rating is FeedbackRating.GOOD:
            return None
        free_text = event.free_text.strip() if event.free_text else ""
        label_hint = self._best_label_hint(event.labels)
        severity = label_hint[0] if label_hint else "minor"
        hint = label_hint[1] if label_hint else ""
        if event.rating is FeedbackRating.BAD:
            severity = self._max_severity(severity, "major")
            if not hint:
                hint = "Проверь факты и избегай галлюцинаций."
        if free_text:
            hint = free_text
        if not hint:
            return None
        return {
            "severity": severity,
            "hint": hint,
            "timestamp": event.created_at,
            "feedback_id": event.feedback_id,
            "interaction_id": event.interaction_id,
            "rating": event.rating.value,
        }

    def _best_label_hint(self, labels: list[FeedbackLabel]) -> tuple[str, str] | None:
        best: tuple[str, str] | None = None
        best_rank = -1
        for label in labels:
            severity, hint = _FEEDBACK_LABEL_HINTS.get(label, ("minor", "Улучшай качество ответа."))
            rank = self._severity_rank(severity)
            if rank > best_rank:
                best = (severity, hint)
                best_rank = rank
        return best

    def _max_severity(self, current: str, incoming: str) -> str:
        if self._severity_rank(incoming) > self._severity_rank(current):
            return incoming
        return current

    def _severity_rank(self, severity: str) -> int:
        ranks = {"minor": 0, "major": 1, "fatal": 2}
        return ranks.get(severity, 0)
