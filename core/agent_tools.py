from __future__ import annotations

# ruff: noqa: F401
# mypy: ignore-errors
import difflib
import json
import re
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from config.model_store import save_model_configs
from core.approval_policy import (
    ApprovalCategory,
    ApprovalContext,
    ApprovalRequest,
    ApprovalRequired,
)
from core.decision.handler import DecisionContext, DecisionRequired
from core.decision.models import DecisionPacket
from core.decision.tool_fail import build_tool_fail_packet
from core.mwv.models import (
    MWV_REPORT_PREFIX,
    StopReasonCode,
    VerificationResult,
    VerificationStatus,
)
from core.mwv.routing import RouteDecision
from core.rule_engine import PolicyApplication
from core.skills.candidates import CandidateDraft, sanitize_text, suggest_patterns
from core.skills.index import SkillMatch
from core.skills.models import SkillRisk
from core.tool_gateway import ToolGateway
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.types import ModelConfig
from memory.memory_manager import MemoryRecord
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
    MemoryKind,
    PlanStepStatus,
    TaskPlan,
    ToolCallRecord,
    ToolRequest,
    ToolResult,
    WorkspaceDiffEntry,
)
from shared.policy_models import PolicyAction, PolicyRule, PolicyScope
from shared.sandbox import normalize_sandbox_path
from tools.workspace_tools import MAX_FILE_BYTES, WORKSPACE_ROOT

SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD = 3

MAX_SHORT_TERM_MESSAGES = 20
_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_MIN_BASE64_LEN = 64


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
    main_config: ModelConfig | None
    main_api_key: str | None
    last_chat_interaction_id: str | None
    last_approval_request: ApprovalRequest | None
    last_decision_packet: DecisionPacket | None
    _pending_decision_packet: DecisionPacket | None

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
            if cmd == "auto":
                goal = " ".join(args)
                result = self.handle_auto_command(goal)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "plan":
                goal = " ".join(args)
                plan = self.planner.build_plan(goal)
                self.last_plan_original = plan
                self.last_plan = plan
                executed: TaskPlan = self.executor.run(
                    plan,
                    tool_gateway=self._build_tool_gateway(
                        pre_call=self._workspace_diff_pre_call,
                        post_call=self._workspace_diff_post_call,
                        safe_mode_override=True,
                    ),
                )
                result = self._format_plan(executed)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "fs":
                operation = args[0] if args else "list"
                path_arg = args[1] if len(args) > 1 else ""
                req = ToolRequest(name="fs", args={"op": operation, "path": path_arg})
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "web":
                query = " ".join(args)
                req = ToolRequest(name="web", args={"query": query})
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "sh":
                req = ToolRequest(
                    name="shell",
                    args={
                        "command": " ".join(args),
                        "config_path": str(getattr(self, "shell_config_path", "")) or None,
                    },
                )
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd == "project":
                if not args:
                    result = "[Нужно указать подкоманду: index|find]"
                    response = _wrap(result)
                    self._log_chat_interaction(raw_input=command, response_text=response)
                    return response
                req = ToolRequest(name="project", args={"cmd": args[0], "args": args[1:]})
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd in {"imggen", "img_generate"}:
                prompt = " ".join(args) or "image"
                req = ToolRequest(name="image_generate", args={"prompt": prompt})
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
                response = _wrap(result)
                self._log_chat_interaction(raw_input=command, response_text=response)
                return response

            if cmd in {"imganalyze", "img_analyze"}:
                if not args:
                    result = "[Нужно указать base64 или путь]"
                    response = _wrap(result)
                    self._log_chat_interaction(raw_input=command, response_text=response)
                    return response
                raw_value = args[0].strip()
                if raw_value.startswith("base64:"):
                    payload = raw_value.removeprefix("base64:").strip()
                    req = ToolRequest(name="image_analyze", args={"base64": payload})
                elif _looks_like_base64(raw_value):
                    req = ToolRequest(name="image_analyze", args={"base64": raw_value})
                else:
                    req = ToolRequest(name="image_analyze", args={"path": raw_value})
                tool_result = self._call_tool_logged(command, req, safe_mode_override=True)
                result = self._format_tool_result(tool_result)
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

            unknown = f"[Инструмент '{cmd}' неактивен или неизвестен]"
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
        except DecisionRequired as exc:
            return self._handle_decision_packet(
                exc.packet,
                raw_input=command,
                record_in_history=False,
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

    def _reset_approval_state(self) -> None:
        self.last_approval_request = None

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
        if self.memory_config.auto_save_dialogue:
            self.save_to_memory(raw_input, response)
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

    def _approval_context(self, *, safe_mode_override: bool | None = None) -> ApprovalContext:
        safe_mode = bool(self.tools_enabled.get("safe_mode", False))
        if safe_mode_override is not None:
            safe_mode = safe_mode_override
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
        safe_mode_override: bool | None = None,
    ) -> ToolGateway:
        def _post_call(request: ToolRequest, result: ToolResult, context: object | None) -> None:
            if post_call:
                post_call(request, result, context)
            self._track_tool_error(request, result)

        return ToolGateway(
            self.tool_registry,
            pre_call=pre_call,
            post_call=_post_call,
            approval_context=self._approval_context(safe_mode_override=safe_mode_override),
            log_event=self.tracer.log,
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
        error_text = sanitize_text(result.error or "unknown error")
        self._pending_decision_packet = build_tool_fail_packet(
            tool_name=request.name,
            error_text=error_text,
            count=count,
            threshold=SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD,
            user_input=self._last_user_input,
        )
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
        )
        try:
            result = gateway.call(request)
        except ApprovalRequired:
            result = ToolResult.failure("Требуется подтверждение")
            self._log_tool_interaction(raw_input=raw_input, request=request, result=result)
            raise
        self._log_tool_interaction(raw_input=raw_input, request=request, result=result)
        if self._pending_decision_packet is not None:
            packet = self._pending_decision_packet
            self._pending_decision_packet = None
            raise DecisionRequired(packet)
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
        )

    def _format_report_block(
        self,
        *,
        route: str,
        trace_id: str | None,
        attempts: tuple[int, int] | None,
        verifier: VerificationResult | None,
        next_steps: list[str] | None,
        stop_reason_code: StopReasonCode | None,
    ) -> str:
        payload: dict[str, JSONValue] = {"route": route, "trace_id": trace_id}
        if attempts is not None:
            payload["attempts"] = {"current": attempts[0], "max": attempts[1]}
        if verifier is not None:
            payload["verifier"] = {
                "status": "ok" if verifier.status == VerificationStatus.PASSED else "fail",
                "duration_ms": verifier.duration_ms,
            }
        if next_steps is not None:
            payload["next_steps"] = self._normalize_report_steps(next_steps)
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
    ) -> str:
        report = self._format_report_block(
            route=route,
            trace_id=trace_id,
            attempts=attempts,
            verifier=verifier,
            next_steps=next_steps,
            stop_reason_code=stop_reason_code,
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
        )

    def _handle_approval_required(
        self,
        request: ApprovalRequest,
        *,
        raw_input: str,
        record_in_history: bool = False,
        command_lane: bool = False,
    ) -> str:
        self.last_approval_request = request
        required = ", ".join(request.required_categories) if request.required_categories else "n/a"
        why_parts = [f"category={request.category}", f"required={required}"]
        if command_lane:
            why_parts.append("mode=command_lane (без MWV)")
        route = "command" if command_lane else "mwv"
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
