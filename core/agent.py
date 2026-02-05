from __future__ import annotations

import logging
import time
import uuid

from config.memory_config import MemoryConfig, load_memory_config
from config.model_store import load_model_configs
from config.model_whitelist import ensure_model_allowed
from config.shell_config import DEFAULT_SHELL_CONFIG_PATH
from config.tools_config import ToolsConfig, load_tools_config, save_tools_config
from core.agent_mwv import AgentMWVMixin
from core.agent_routing import AgentRoutingMixin
from core.agent_tools import AgentToolsMixin
from core.approval_policy import ApprovalCategory, ApprovalRequest
from core.auto_agent import AutoAgent
from core.batch_review import BatchReviewer
from core.decision.handler import DecisionHandler
from core.decision.models import DecisionPacket
from core.executor import Executor
from core.mwv.manager import ManagerRuntime
from core.mwv.verifier_runtime import VerifierRuntime
from core.planner import Planner
from core.rule_engine import RuleEngine
from core.skills.candidates import SkillCandidateWriter
from core.skills.index import SkillIndex, SkillMatch
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.brain_manager import BrainManager
from llm.types import ModelConfig
from memory.categorized_memory_store import CategorizedMemoryStore
from memory.memory_companion_store import MemoryCompanionStore
from memory.memory_inbox_writer import MemoryInboxWriter
from memory.memory_manager import MemoryManager
from memory.vector_index import VectorIndex
from shared.batch_review_models import (
    BatchReviewRun,
    CandidateStatus,
    PolicyRuleCandidate,
)
from shared.memory_companion_models import (
    FeedbackEvent,
    FeedbackLabel,
    FeedbackRating,
    InteractionLog,
)
from shared.models import (
    JSONValue,
    LLMMessage,
    MemoryKind,
    TaskPlan,
    ToolCallRecord,
    ToolResult,
    WorkspaceDiffEntry,
)
from shared.policy_models import (
    PolicyRule,
    PolicyScope,
    policy_action_from_json,
    policy_trigger_from_json,
)
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
    "web_search",
    "shell",
    "project",
    "tts",
    "stt",
    "http_client",
    "image_analyze",
    "image_generate",
    "workspace_run",
}
MAX_MWV_ATTEMPTS = 3
SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD = 3
_DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS = 30
_FEEDBACK_LABEL_HINTS: dict[FeedbackLabel, tuple[str, str]] = {
    FeedbackLabel.OFF_TOPIC: ("fatal", "Держись темы вопроса."),
    FeedbackLabel.HALLUCINATION: ("major", "Проверь факты и избегай галлюцинаций."),
    FeedbackLabel.INCORRECT: ("major", "Проверяй корректность ответа."),
    FeedbackLabel.NO_SOURCES: ("major", "Добавляй источники при необходимости."),
    FeedbackLabel.TOO_LONG: ("minor", "Делай ответ короче."),
    FeedbackLabel.TOO_COMPLEX: ("minor", "Упрощай объяснение."),
    FeedbackLabel.OTHER: ("minor", "Улучшай качество ответа."),
}

_COMPAT_EXPORTS = (ManagerRuntime, VerifierRuntime, WORKSPACE_ROOT, MAX_FILE_BYTES)


class Agent(AgentRoutingMixin, AgentMWVMixin, AgentToolsMixin):
    """SlavikAI Core v1.0 — Распределённый рассуждающий агент."""

    def __init__(
        self,
        brain: Brain | None = None,
        enable_tools: dict[str, bool] | None = None,
        main_config: ModelConfig | None = None,
        main_api_key: str | None = None,
        brain_manager: BrainManager | None = None,
        user_id: str = "local",
        memory_companion_db_path: str | None = None,
        memory_inbox_db_path: str | None = None,
    ) -> None:
        saved_main = load_model_configs()
        self.main_config = main_config or saved_main
        if self.main_config is not None:
            ensure_model_allowed(self.main_config.model)
        self.main_api_key = main_api_key
        self.shell_config_path = str(DEFAULT_SHELL_CONFIG_PATH)
        self._external_brain = brain
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
        if self.memory_config.auto_save_dialogue:
            self.logger.warning(
                "auto_save_dialogue включен явно через config/memory.json "
                "(policies-first override)."
            )
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
        self._memory_inbox_store = (
            CategorizedMemoryStore(memory_inbox_db_path)
            if memory_inbox_db_path
            else CategorizedMemoryStore()
        )
        self._memory_inbox_writer = MemoryInboxWriter(self._memory_inbox_store, self.memory_config)
        self.vectors = VectorIndex("memory/vectors.db")
        self.skill_index = SkillIndex.load_default()
        self._skill_candidate_writer = SkillCandidateWriter()
        self.short_term: list[LLMMessage] = []
        self.conversation_id = str(uuid.uuid4())
        self.session_id: str | None = None
        self.approved_categories: set[ApprovalCategory] = set()
        self.last_plan: TaskPlan | None = None
        self.last_plan_original: TaskPlan | None = None
        self.last_hints_used: list[str] = []
        self.last_hints_meta: list[dict[str, str]] = []
        self.last_context_text: str | None = None
        self._last_skill_match: SkillMatch | None = None
        self._last_user_input: str | None = None
        self._tool_error_counts: dict[str, int] = {}
        self._skill_metrics: dict[str, int] = {
            "skill_match_hit": 0,
            "skill_match_miss": 0,
            "ambiguous_count": 0,
            "deprecated_count": 0,
            "verifier_fail_count": 0,
            "candidate_written_count": 0,
        }
        self.decision_handler = DecisionHandler()
        self.last_approval_request: ApprovalRequest | None = None
        self.last_decision_packet: DecisionPacket | None = None
        self.last_reasoning: str | None = None
        self._pending_decision_packet: DecisionPacket | None = None
        self.workspace_file_path: str | None = None
        self.workspace_file_content: str | None = None
        self.workspace_selection: str | None = None
        self._workspace_diff_baselines: dict[str, str] = {}
        self._workspace_diffs: dict[str, WorkspaceDiffEntry] = {}

    def _review_answer(self, answer: str) -> str:
        return answer

    def _build_brain(self) -> Brain:
        if self._brain_manager:
            return self._brain_manager.build()
        if self._external_brain:
            return self._external_brain
        if self.main_config is None:
            raise RuntimeError("Не выбрана модель. Укажите model id в настройках.")
        main_brain = create_brain(self.main_config, api_key=self.main_api_key)
        return main_brain

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
