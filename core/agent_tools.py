from __future__ import annotations

# ruff: noqa: F401
import difflib
import json
import re
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from config.model_store import save_model_configs
from core.approval_policy import (
    ApprovalCategory,
    ApprovalContext,
    ApprovalRequest,
    ApprovalRequired,
)
from core.computer_activity_log import ComputerActivityLog
from core.decision.handler import DecisionContext
from core.decision.models import DecisionPacket
from core.desktop_policy import DesktopApprovalRule, DesktopPolicyRuntime
from core.mwv.models import (
    MWV_REPORT_PREFIX,
    StopReasonCode,
    VerificationResult,
    VerificationStatus,
)
from core.mwv.routing import RouteDecision
from core.rule_engine import PolicyApplication
from core.skills.candidates import CandidateDraft, sanitize_text, suggest_patterns
from core.skills.index import SkillMatch, SkillResolution
from core.skills.models import SkillRisk
from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.types import ModelConfig
from shared.memory_companion_models import (
    BlockedReason,
    ChatInteractionLog,
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionKind,
    InteractionMode,
    ToolInteractionLog,
    ToolStatus,
)
from shared.models import (
    JSONValue,
    LLMMessage,
    PlanStepStatus,
    TaskPlan,
    ToolCallRecord,
    ToolRequest,
    ToolResult,
    UXContractSummary,
    WorkspaceDiffEntry,
)
from shared.policy_models import PolicyAction, PolicyRule, PolicyScope
from shared.sandbox import normalize_sandbox_path
from tools.workspace_tools import MAX_FILE_BYTES, WORKSPACE_ROOT

SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD = 3

MAX_SHORT_TERM_MESSAGES = 20
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_MIN_BASE64_LEN = 64
_MAX_UX_SUMMARY_CHARS = 220

if TYPE_CHECKING:
    from config.memory_config import MemoryConfig
    from core.auto_agent import AutoAgent
    from core.decision.handler import DecisionHandler
    from core.desktop_policy import DesktopPolicyStore
    from core.desktop_security import DesktopPathSecurity
    from core.rule_engine import RuleEngine
    from core.skills.candidates import SkillCandidateWriter
    from core.tracer import Tracer
    from memory.memory_companion_store import MemoryCompanionStore
    from memory.memory_inbox_writer import MemoryInboxWriter
    from memory.memory_manager import MemoryManager
    from tools.tool_registry import ToolRegistry


def _workspace_root() -> Path:
    import core.agent as agent_module

    value = getattr(agent_module, "WORKSPACE_ROOT", WORKSPACE_ROOT)
    return Path(value)


def _max_file_bytes() -> int:
    import core.agent as agent_module

    value = getattr(agent_module, "MAX_FILE_BYTES", MAX_FILE_BYTES)
    return int(value)


def _looks_like_base64(value: str) -> bool:
    stripped = value.strip()
    if len(stripped) < _MIN_BASE64_LEN or len(stripped) % 4 != 0:
        return False
    return bool(_BASE64_RE.fullmatch(stripped))


class AgentToolsMixin:
    if TYPE_CHECKING:
        brain: Brain
        tracer: Tracer
        decision_handler: DecisionHandler
        memory: MemoryManager
        memory_config: MemoryConfig
        auto_agent: AutoAgent
        desktop_policy_store: DesktopPolicyStore
        desktop_policy_runtime: DesktopPolicyRuntime
        desktop_security: DesktopPathSecurity
        tool_registry: ToolRegistry
        tools_enabled: dict[str, bool]
        short_term: list[LLMMessage]
        user_id: str
        shell_config_path: str
        _interaction_store: MemoryCompanionStore
        _rule_engine: RuleEngine
        _skill_candidate_writer: SkillCandidateWriter
        _memory_inbox_writer: MemoryInboxWriter
        _skill_metrics: dict[str, int]
        _tool_error_counts: dict[str, int]
        _workspace_diff_baselines: dict[str, str]
        _workspace_diffs: dict[str, WorkspaceDiffEntry]
        _last_user_input: str | None
        _computer_log: ComputerActivityLog

        def _build_brain(self) -> Brain: ...
        def build_memory_save_preview(
            self,
            text: str,
            *,
            source_kind: str,
            source_id: str | None = None,
            lang_hint: str | None = None,
        ) -> dict[str, JSONValue]: ...
        def summarize_current_session(self) -> dict[str, JSONValue] | None: ...

    main_config: ModelConfig | None
    main_api_key: str | None
    last_chat_interaction_id: str | None
    last_approval_request: ApprovalRequest | None
    last_decision_packet: DecisionPacket | None
    last_plan_summary: str | None
    last_execution_summary: str | None
    last_reasoning: str | None
    last_stream_response_raw: str | None
    last_plan: TaskPlan | None
    last_plan_original: TaskPlan | None
    runtime_mode: str
    runtime_active_plan: dict[str, JSONValue] | None
    runtime_active_task: dict[str, JSONValue] | None
    runtime_auto_state: dict[str, JSONValue] | None
    runtime_plan_guard_enabled: bool
    runtime_workspace_root: str | None
    last_auto_state: dict[str, JSONValue] | None
    last_approval_source_endpoint: str | None
    last_approval_resume_payload: dict[str, JSONValue] | None

    def _inc_metric(self, name: str) -> None:
        current = self._skill_metrics.get(name, 0) + 1
        self._skill_metrics[name] = current
        self.tracer.log(name, str(current))

    def _get_main_brain(self) -> Brain:
        return self.brain

    def handle_tool_command(self, command: str) -> str:
        parts = command.split()
        cmd = parts[0][1:].lower()
        args = parts[1:]
        self.tracer.log("tool_invoked", cmd, {"args": args})

        def _wrap(response: str) -> str:
            return self._format_command_lane_response(response)

        try:
            if cmd in {"end-session", "end_session"}:
                summary = self.summarize_current_session()
                if summary is None:
                    result = "Сессия не содержит сообщений для резюме."
                else:
                    stable_key = summary.get("stable_key")
                    self.short_term.clear()
                    result = f"Сессия закрыта, резюме сохранено: {stable_key}"
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "trace":
                logs = self.tracer.read_recent(40)
                lines: list[str] = []
                for log in logs:
                    timestamp = log.get("timestamp", "?")
                    event = log.get("event", "?")
                    message = log.get("message", "")
                    lines.append(f"[{timestamp}] {event}: {message}")
                result = "\n".join(lines)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            unknown = (
                f"[Команда '/{cmd}' неактивна и отключена. Command lane поддерживает только "
                "/trace и /end-session.]"
            )
            self._log_tool_interaction(
                raw_input=command,
                request=ToolRequest(name=cmd, args={"args": args}),
                result=ToolResult.failure(f"Инструмент {cmd} не зарегистрирован"),
            )
            response = _wrap(unknown)
            self._log_chat_interaction(raw_input=command, response_text=response)
            return response
        except ApprovalRequired as exc:
            return self._handle_approval_required(
                exc.request,
                raw_input=command,
                record_in_history=False,
                command_lane=True,
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("error", f"Ошибка при вызове инструмента: {exc}")
            error_text = f"[Ошибка при вызове инструмента: {exc}]"
            response = _wrap(error_text)
            self._log_chat_interaction(raw_input=command, response_text=response)
            return response

    def _should_record_in_history(self, content: str) -> bool:
        if content.startswith("/"):
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

    def _reset_approval_state(self) -> None:
        self.last_approval_request = None
        self.last_approval_source_endpoint = None
        self.last_approval_resume_payload = None

    def reset_runtime_transient_state(self) -> None:
        self._reset_workspace_diffs()
        self._reset_approval_state()
        self.last_decision_packet = None
        self.last_reasoning = None
        self.last_stream_response_raw = None
        self.last_plan = None
        self.last_plan_original = None
        self.last_plan_summary = None
        self.last_execution_summary = None
        self.last_auto_state = None
        self.runtime_mode = "ask"
        self.runtime_active_plan = None
        self.runtime_active_task = None
        self.runtime_auto_state = None
        self.runtime_plan_guard_enabled = False
        self.runtime_workspace_root = None
        short_term = getattr(self, "short_term", None)
        if isinstance(short_term, list):
            short_term.clear()
        auto_events = getattr(self, "_auto_progress_events", None)
        if isinstance(auto_events, list):
            auto_events.clear()
        tool_registry = getattr(self, "tool_registry", None)
        set_execution_policy = getattr(tool_registry, "set_execution_policy", None)
        if callable(set_execution_policy):
            set_execution_policy(
                mode="ask",
                active_plan=None,
                active_task=None,
                enforce_plan_guard=False,
            )
        self.tracer.log("runtime_reset", "Transient runtime state reset")

    def _record_decision_packet(self, packet: DecisionPacket) -> None:
        self.last_decision_packet = packet

    def _handle_decision_packet(
        self,
        packet: DecisionPacket,
        *,
        raw_input: str,
        record_in_history: bool,
    ) -> str:
        self._record_decision_packet(packet)
        response = packet.to_json()
        self.tracer.log(
            "decision_packet",
            packet.summary,
            {"id": packet.id, "reason": packet.reason.value},
        )
        if packet.decision_type != "memory_save":
            self._log_chat_interaction(raw_input=raw_input, response_text=response)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=response)])
        return response

    def set_session_context(
        self,
        session_id: str | None,
        approved_categories: set[ApprovalCategory],
    ) -> None:
        self.session_id = session_id
        self.approved_categories = set(approved_categories)

    def set_desktop_policy_context(
        self,
        rules: list[DesktopApprovalRule],
        principal_id: str = "legacy",
    ) -> None:
        persistent = self.desktop_policy_store.list_rules(subject_principal_id=principal_id)
        self.desktop_policy_runtime = DesktopPolicyRuntime([*persistent, *rules])

    def clear_desktop_policy_context(self) -> None:
        self.desktop_policy_runtime = DesktopPolicyRuntime()

    def drain_consumed_desktop_rule_ids(self) -> list[str]:
        return self.desktop_policy_runtime.drain_consumed_rule_ids()

    def set_runtime_state(
        self,
        *,
        mode: str,
        active_plan: dict[str, JSONValue] | None,
        active_task: dict[str, JSONValue] | None,
        auto_state: dict[str, JSONValue] | None = None,
        enforce_plan_guard: bool,
    ) -> None:
        self.runtime_mode = mode.strip().lower() if isinstance(mode, str) else "ask"
        self.runtime_active_plan = dict(active_plan) if isinstance(active_plan, dict) else None
        self.runtime_active_task = dict(active_task) if isinstance(active_task, dict) else None
        self.runtime_auto_state = dict(auto_state) if isinstance(auto_state, dict) else None
        self.runtime_plan_guard_enabled = bool(enforce_plan_guard)

    def apply_runtime_workspace_root(self, workspace_root: str | None) -> None:
        if not isinstance(workspace_root, str) or not workspace_root.strip():
            self.runtime_workspace_root = None
            return
        candidate = Path(workspace_root.strip()).expanduser().resolve()
        if not candidate.exists() or not candidate.is_dir():
            self.runtime_workspace_root = None
            return
        self.runtime_workspace_root = str(candidate)

    def _approval_context(self, *, safe_mode_override: bool | None = None) -> ApprovalContext:
        safe_mode = bool(self.tools_enabled.get("safe_mode", False))
        if safe_mode_override is not None:
            safe_mode = safe_mode_override
        normalized: set[ApprovalCategory] = set(self.approved_categories)
        return ApprovalContext(
            safe_mode=safe_mode,
            session_id=self.session_id,
            approved_categories=normalized,
            execution_target=(
                "desktop" if getattr(self, "runtime_mode", "ask") == "desktop" else "sandbox"
            ),
            desktop_policy=getattr(self, "desktop_policy_runtime", None),
            desktop_security=getattr(self, "desktop_security", None),
        )

    def _build_tool_gateway(
        self,
        *,
        pre_call: Callable[[ToolRequest], object | None] | None = None,
        post_call: (Callable[[ToolRequest, ToolResult, object | None], None] | None) = None,
        safe_mode_override: bool | None = None,
        confirmed_decision: bool = False,
    ) -> ToolGateway:
        self.tool_registry.set_execution_policy(
            mode=getattr(self, "runtime_mode", "act"),
            active_plan=getattr(self, "runtime_active_plan", None),
            active_task=getattr(self, "runtime_active_task", None),
            enforce_plan_guard=bool(getattr(self, "runtime_plan_guard_enabled", False)),
        )

        computer_log: ComputerActivityLog | None = getattr(self, "_computer_log", None)

        def _combined_pre_call(request: ToolRequest) -> object | None:
            main_ctx = pre_call(request) if pre_call else None
            if computer_log is not None:
                computer_log.pre_call(request)
            return main_ctx

        def _post_call(request: ToolRequest, result: ToolResult, context: object | None) -> None:
            if post_call:
                post_call(request, result, context)
            if computer_log is not None:
                computer_log.post_call(request, result, context)
            self._track_tool_error(request, result)

        combined_pre: Callable[[ToolRequest], object | None] | None
        has_pre = pre_call is not None or computer_log is not None
        combined_pre = _combined_pre_call if has_pre else None

        return ToolGateway(
            self.tool_registry,
            pre_call=combined_pre,
            post_call=_post_call,
            approval_context=self._approval_context(safe_mode_override=safe_mode_override),
            log_event=self.tracer.log,
            confirmed_decision=confirmed_decision,
        )

    def _track_tool_error(self, request: ToolRequest, result: ToolResult) -> None:
        if result.ok:
            self._tool_error_counts.pop(request.name, None)
            return
        if not self._should_track_tool_error(result):
            return
        count = self._tool_error_counts.get(request.name, 0) + 1
        self._tool_error_counts[request.name] = count
        if count < SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD:
            return
        self._tool_error_counts[request.name] = 0
        self._record_tool_error_inbox(request, result, count)
        self._record_tool_error_candidate(request, result, count)

    def _should_track_tool_error(self, result: ToolResult) -> bool:
        error_text = (result.error or "").lower()
        if not error_text:
            return True
        ignore_markers = (
            "safe mode",
            "отключ",
            "не зарегистрирован",
            "требуется подтверждение",
            "confirmed_decision_required",
            "approval",
        )
        return not any(marker in error_text for marker in ignore_markers)

    def _record_unknown_skill_candidate(self, user_input: str, decision: RouteDecision) -> None:
        patterns = suggest_patterns(user_input)
        if not patterns:
            patterns = ["unknown"]
        draft = CandidateDraft(
            title=f"Unknown request: {patterns[0]}",
            reason="unknown_request",
            requests=[sanitize_text(user_input)],
            patterns=patterns,
            entrypoints=["unknown"],
            expected_behavior=[
                "Handle the request safely using tools and code changes.",
            ],
            risk=self._risk_from_flags(decision.risk_flags),
            notes=[f"route_reason={decision.reason}"],
        )
        key = f"unknown:{patterns[0]}"
        try:
            path = self._skill_candidate_writer.write_once(key, draft)
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("skill_candidate_error", str(exc))
            return
        if path is not None:
            self._inc_metric("candidate_written_count")
            self.tracer.log(
                "skill_candidate_created",
                path.name,
                {"reason": draft.reason, "key": key},
            )

    def _record_unknown_inbox(self, user_input: str, decision: RouteDecision) -> None:
        summary = f"Неизвестный запрос: {sanitize_text(user_input)}"
        meta: dict[str, JSONValue] = {
            "reason": "unknown_request",
            "route": decision.route,
            "risk_flags": list(decision.risk_flags),
            "skill_status": decision.skill_decision.status if decision.skill_decision else "none",
        }
        try:
            item = self._memory_inbox_writer.write_once(
                summary,
                source="agent",
                meta=meta,
                title="Unknown request",
                tags=["unknown_request"],
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("memory_inbox_error", str(exc), {"reason": "unknown_request"})
            return
        if item is not None:
            self.tracer.log("memory_inbox_written", item.id, {"reason": "unknown_request"})

    def _record_tool_error_candidate(
        self,
        request: ToolRequest,
        result: ToolResult,
        count: int,
    ) -> None:
        error_text = sanitize_text(result.error or "unknown error")
        request_text = sanitize_text(self._last_user_input or "")
        draft = CandidateDraft(
            title=f"Tool error: {request.name}",
            reason="tool_error",
            requests=[request_text] if request_text else ["unknown"],
            patterns=[request.name],
            entrypoints=[request.name],
            expected_behavior=[
                "Provide a stable tool workflow and recover from failures.",
            ],
            risk="medium",
            notes=[f"error={error_text}", f"count={count}"],
        )
        key = f"tool_error:{request.name}"
        try:
            path = self._skill_candidate_writer.write_once(key, draft)
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("skill_candidate_error", str(exc))
            return
        if path is not None:
            self._inc_metric("candidate_written_count")
            self.tracer.log(
                "skill_candidate_created",
                path.name,
                {"reason": draft.reason, "tool": request.name},
            )

    def _record_tool_error_inbox(
        self,
        request: ToolRequest,
        result: ToolResult,
        count: int,
    ) -> None:
        summary = f"Tool error threshold reached: {request.name}"
        meta: dict[str, JSONValue] = {
            "reason": "tool_error",
            "tool": request.name,
            "error": sanitize_text(result.error or "unknown error"),
            "count": count,
            "threshold": SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD,
        }
        try:
            item = self._memory_inbox_writer.write_once(
                summary,
                source="agent",
                meta=meta,
                title="Tool error threshold",
                tags=["tool_error"],
            )
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("memory_inbox_error", str(exc), {"reason": "tool_error"})
            return
        if item is not None:
            self.tracer.log("memory_inbox_written", item.id, {"reason": "tool_error"})

    def _risk_from_flags(self, flags: list[str]) -> SkillRisk:
        high = {"sudo", "system", "install", "git"}
        if any(flag in high for flag in flags):
            return "high"
        if "tools" in flags or "filesystem" in flags:
            return "medium"
        return "low"

    def drain_computer_events(self) -> list[dict[str, JSONValue]]:
        computer_log: ComputerActivityLog | None = getattr(self, "_computer_log", None)
        if computer_log is None:
            return []
        return computer_log.drain()

    def consume_workspace_diffs(self) -> list[WorkspaceDiffEntry]:
        diffs = list(self._workspace_diffs.values())
        self._workspace_diff_baselines.clear()
        self._workspace_diffs.clear()
        return diffs

    def _normalize_workspace_path(self, raw_path: str) -> Path | None:
        if not raw_path:
            return None
        try:
            return normalize_sandbox_path(raw_path, _workspace_root())
        except Exception:  # noqa: BLE001
            return None

    def _read_workspace_text(self, path: Path) -> str | None:
        try:
            if not path.exists() or not path.is_file():
                return ""
            if path.stat().st_size > _max_file_bytes():
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
        rel_path = str(path.relative_to(_workspace_root()))
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

    def _call_tool_logged(
        self,
        raw_input: str,
        request: ToolRequest,
        *,
        safe_mode_override: bool | None = None,
        confirmed_decision: bool = False,
    ) -> ToolResult:
        pre_call = None
        post_call = None
        if not raw_input.startswith("ui:"):
            pre_call = self._workspace_diff_pre_call
            post_call = self._workspace_diff_post_call
        gateway = self._build_tool_gateway(
            pre_call=pre_call,
            post_call=post_call,
            safe_mode_override=safe_mode_override,
            confirmed_decision=confirmed_decision,
        )
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

    def handle_auto_command(
        self,
        goal: str,
        *,
        command_lane: bool = False,
        skill_resolution: SkillResolution | None = None,
    ) -> str:
        goal_clean = goal.strip() or "auto run"
        self.tracer.log("auto_invoke", f"Auto v1 tool loop: {goal_clean}")
        outcome = self.auto_agent.run_outcome(
            goal_clean,
            skill_resolution=skill_resolution,
        )
        if command_lane:
            return self._strip_report_block(outcome.text)
        return outcome.text

    def resume_auto_run(self, run_id: str) -> str | None:
        run_id_clean = run_id.strip()
        if not run_id_clean:
            return None
        outcome = self.auto_agent.resume_outcome(run_id_clean)
        if outcome is None:
            return None
        return outcome.text

    def cancel_auto_run(
        self,
        run_id: str,
        *,
        reason: str = "cancelled_by_user",
    ) -> dict[str, JSONValue] | None:
        run_id_clean = run_id.strip()
        if not run_id_clean:
            return None
        return self.auto_agent.cancel_run(run_id_clean, reason=reason)

    def reconfigure_models(
        self,
        main_config: ModelConfig,
        main_api_key: str | None = None,
        *,
        persist: bool = True,
    ) -> None:
        """Переинициализирует мозг с новыми настройками."""
        self.main_config = main_config
        self.main_api_key = main_api_key
        self.brain = self._build_brain()
        if persist:
            save_model_configs(self.main_config)
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

    def _format_command_lane_response(self, response: str) -> str:
        prefix = "Командный режим (без MWV)"
        base = prefix if not response else f"{prefix}\n{response}".strip()
        return self._append_report_block(
            base,
            route="command",
            trace_id=None,
            attempts=None,
            verifier=None,
            next_steps=[],
            stop_reason_code=StopReasonCode.COMMAND_LANE_NOTICE,
            plan_summary="Ручной command lane без MWV-плана.",
            execution_summary=response or "Командный ответ сформирован.",
        )

    def _strip_report_block(self, text: str) -> str:
        lines = [line for line in text.splitlines() if not line.startswith(MWV_REPORT_PREFIX)]
        return "\n".join(lines).strip()

    def _format_report_block(
        self,
        *,
        route: str,
        trace_id: str | None,
        attempts: tuple[int, int] | None,
        verifier: VerificationResult | None,
        next_steps: list[str] | None,
        stop_reason_code: StopReasonCode | None,
        ux_contract: UXContractSummary,
        skill: dict[str, JSONValue] | None = None,
    ) -> str:
        payload: dict[str, JSONValue] = {"route": route, "trace_id": trace_id}
        payload["plan_summary"] = ux_contract.plan_summary
        payload["execution_summary"] = ux_contract.execution_summary
        if skill is not None:
            payload["skill"] = dict(skill)
        is_stop = stop_reason_code is not None
        if attempts is not None:
            payload["attempts"] = {"current": attempts[0], "max": attempts[1]}
        elif is_stop:
            payload["attempts"] = {"current": 1, "max": 1}
        if verifier is not None:
            payload["verifier"] = {
                "status": "ok" if verifier.status == VerificationStatus.PASSED else "fail",
                "duration_ms": verifier.duration_ms,
            }
        elif is_stop:
            payload["verifier"] = {"status": "unknown", "duration_ms": None}
        if next_steps is not None:
            payload["next_steps"] = self._normalize_report_steps(next_steps)
        elif is_stop:
            payload["next_steps"] = []
        if stop_reason_code is not None:
            payload["stop_reason_code"] = stop_reason_code.value
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return f"{MWV_REPORT_PREFIX}{encoded}"

    def _normalize_report_steps(self, steps: list[str]) -> list[str]:
        normalized: list[str] = []
        for step in steps:
            cleaned = step.strip()
            if cleaned.startswith("- "):
                cleaned = cleaned[2:].strip()
            normalized.append(cleaned)
        return normalized[:3]

    def _compact_summary_text(self, text: str, *, fallback: str) -> str:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            cleaned = fallback
        if len(cleaned) <= _MAX_UX_SUMMARY_CHARS:
            return cleaned
        return f"{cleaned[: _MAX_UX_SUMMARY_CHARS - 3].rstrip()}..."

    def _build_ux_contract(
        self,
        *,
        route: str,
        text: str,
        plan_summary: str | None,
        execution_summary: str | None,
    ) -> UXContractSummary:
        plan_fallbacks = {
            "chat": "План не требуется для chat-маршрута.",
            "command": "Ручной command lane без MWV-плана.",
        }
        execution_fallbacks = {
            "chat": "Ответ сформирован моделью.",
            "command": "Командный ответ сформирован.",
        }
        plan = self._compact_summary_text(
            plan_summary or "",
            fallback=plan_fallbacks.get(route, "План не указан."),
        )
        execution = self._compact_summary_text(
            execution_summary or "",
            fallback=execution_fallbacks.get(
                route,
                self._compact_summary_text(text, fallback="Выполнение завершено."),
            ),
        )
        self.last_plan_summary = plan
        self.last_execution_summary = execution
        return UXContractSummary(plan_summary=plan, execution_summary=execution)

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
        skill: dict[str, JSONValue] | None = None,
    ) -> str:
        ux_contract = self._build_ux_contract(
            route=route,
            text=text,
            plan_summary=plan_summary,
            execution_summary=execution_summary,
        )
        report = self._format_report_block(
            route=route,
            trace_id=trace_id,
            attempts=attempts,
            verifier=verifier,
            next_steps=next_steps,
            stop_reason_code=stop_reason_code,
            ux_contract=ux_contract,
            skill=skill,
        )
        if not text:
            return report
        return f"{text}\n{report}"

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
        skill: dict[str, JSONValue] | None = None,
    ) -> str:
        steps = next_steps or ["Уточни запрос или попробуй снова."]
        lines = [
            f"Что случилось: {what}",
            f"Почему: {why}",
            "Что делать дальше:",
            *[f"- {step}" for step in steps[:3]],
        ]
        if trace_id:
            lines.append(f"trace_id={trace_id}")
        self.tracer.log(
            "stop_response",
            what,
            {
                "stop_reason_code": stop_reason_code.value,
                "route": route,
                "trace_id": trace_id or "",
            },
        )
        return self._append_report_block(
            "\n".join(lines).strip(),
            route=route,
            trace_id=trace_id,
            attempts=attempts,
            verifier=verifier,
            next_steps=steps,
            stop_reason_code=stop_reason_code,
            plan_summary=plan_summary or what,
            execution_summary=execution_summary or why,
            skill=skill,
        )

    def _handle_approval_required(
        self,
        request: ApprovalRequest,
        *,
        raw_input: str,
        record_in_history: bool = False,
        command_lane: bool = False,
        source_endpoint: str | None = None,
        resume_payload: dict[str, JSONValue] | None = None,
    ) -> str:
        self.last_approval_request = request
        required = ", ".join(request.required_categories) if request.required_categories else "n/a"
        why_parts = [f"category={request.category}", f"required={required}"]
        if command_lane:
            why_parts.append("mode=command_lane (без MWV)")
        route = "command" if command_lane else "mwv"
        next_source = source_endpoint.strip() if isinstance(source_endpoint, str) else ""
        next_resume = dict(resume_payload) if isinstance(resume_payload, dict) else {}
        if not command_lane and self.runtime_mode == "auto":
            route = "auto"
            if not next_source:
                next_source = "auto.run"
            if not next_resume and isinstance(self.last_auto_state, dict):
                run_id_raw = self.last_auto_state.get("run_id")
                if isinstance(run_id_raw, str) and run_id_raw.strip():
                    next_resume = {"run_id": run_id_raw.strip()}
        self.last_approval_source_endpoint = next_source or None
        self.last_approval_resume_payload = next_resume or None
        error_text = self._format_stop_response(
            what="Требуется подтверждение действия",
            why="; ".join(why_parts),
            next_steps=[
                "Подтверди действие или отмени его.",
                "При необходимости уточни команду.",
            ],
            stop_reason_code=StopReasonCode.APPROVAL_REQUIRED,
            route=route,
        )
        self._log_chat_interaction(raw_input=raw_input, response_text=error_text)
        if record_in_history:
            self._append_short_term([LLMMessage(role="assistant", content=error_text)])
        return error_text
