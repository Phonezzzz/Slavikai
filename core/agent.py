from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import replace
from typing import Literal, cast

from config.memory_config import MemoryConfig, load_memory_config
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
from memory.atom_embedding_index import AtomEmbeddingIndex
from memory.canonical_aggregator import CanonicalAggregator
from memory.canonical_atom_store import CanonicalAtomStore
from memory.categorized_memory_store import CategorizedMemoryStore
from memory.claim_extractor import ClaimExtractor, ExtractorConfig
from memory.memory_companion_store import MemoryCompanionStore
from memory.memory_inbox_writer import MemoryInboxWriter
from memory.memory_manager import MemoryManager
from memory.memory_retrieval import (
    RetrievalConfig,
)
from memory.memory_retrieval import (
    build_memory_capsule as build_memory_capsule_payload,
)
from memory.session_summarizer import SessionSummarizer
from memory.vector_index import VectorIndex
from shared.batch_review_models import (
    BatchReviewRun,
    CandidateStatus,
    PolicyRuleCandidate,
)
from shared.canonical_atom_models import (
    AtomStatus,
    CanonicalAtom,
    Claim,
    ClaimExtractionInput,
    ClaimType,
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
from tools.protocols import Tool
from tools.shell_tool import ShellTool
from tools.stt_tool import SttTool
from tools.tool_descriptors import get_tool_metadata
from tools.tool_registry import ToolCapability, ToolHandler, ToolRegistry
from tools.tts_tool import TtsTool
from tools.web_search_tool import WebSearchTool
from tools.workspace_tools import (
    MAX_FILE_BYTES,
    WORKSPACE_ROOT,
    ApplyPatchTool,
    CreateFileTool,
    DeleteFileTool,
    ListFilesTool,
    MoveFileTool,
    ReadFileTool,
    RenameFileTool,
    RunCodeTool,
    WorkspaceTerminalRunTool,
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
    "workspace_terminal_run",
}
MAX_MWV_ATTEMPTS = 3
SKILL_CANDIDATE_TOOL_ERROR_THRESHOLD = 3
_DEFAULT_POLICY_DECAY_HALF_LIFE_DAYS = 30
_EXPLICIT_MEMORY_PREFIX = re.compile(r"^\s*(?:запомни|remember)\b[:\-\s]*", re.IGNORECASE)
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
        canonical_atoms_db_path: str | None = None,
    ) -> None:
        self.main_config = main_config
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
        self.auto_agent.set_progress_callback(self._record_auto_progress)
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
        self._canonical_store = (
            CanonicalAtomStore(canonical_atoms_db_path)
            if canonical_atoms_db_path
            else CanonicalAtomStore()
        )
        self._claim_extractor = ClaimExtractor(
            config=ExtractorConfig(enable_llm_enrichment=True),
            brain=self.brain,
        )
        self._canonical_aggregator = CanonicalAggregator(self._canonical_store)
        self._atom_embedding_index = AtomEmbeddingIndex(self.vectors)
        self._session_summarizer = SessionSummarizer(self.brain)
        self._retrieval_config = RetrievalConfig()
        self.skill_index = SkillIndex.load_default()
        self._skill_candidate_writer = SkillCandidateWriter()
        self.short_term: list[LLMMessage] = []
        self.conversation_id = str(uuid.uuid4())
        self.session_id: str | None = None
        self.approved_categories: set[ApprovalCategory] = set()
        self.runtime_mode = "act"
        self.runtime_active_plan: dict[str, JSONValue] | None = None
        self.runtime_active_task: dict[str, JSONValue] | None = None
        self.runtime_auto_state: dict[str, JSONValue] | None = None
        self.runtime_plan_guard_enabled = False
        self.runtime_workspace_root: str | None = None
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
        self.last_approval_source_endpoint: str | None = None
        self.last_approval_resume_payload: dict[str, JSONValue] | None = None
        self.last_decision_packet: DecisionPacket | None = None
        self.last_reasoning: str | None = None
        self.last_stream_response_raw: str | None = None
        self.last_plan_summary: str | None = None
        self.last_execution_summary: str | None = None
        self._pending_decision_packet: DecisionPacket | None = None
        self.last_auto_state: dict[str, JSONValue] | None = None
        self._auto_progress_events: list[dict[str, JSONValue]] = []
        self.workspace_file_path: str | None = None
        self.workspace_file_content: str | None = None
        self.workspace_selection: str | None = None
        self._workspace_diff_baselines: dict[str, str] = {}
        self._workspace_diffs: dict[str, WorkspaceDiffEntry] = {}

    def _review_answer(self, answer: str) -> str:
        return answer

    def _record_auto_progress(self, state: dict[str, JSONValue]) -> None:
        self._auto_progress_events.append(dict(state))

    def drain_auto_progress_events(self) -> list[dict[str, JSONValue]]:
        events = [dict(item) for item in self._auto_progress_events]
        self._auto_progress_events.clear()
        return events

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
        def register_tool(
            name: str,
            handler: Tool | ToolHandler,
            *,
            enabled: bool,
            capability: ToolCapability,
            risk_classes: list[str] | None = None,
        ) -> None:
            metadata = get_tool_metadata(name)
            self.tool_registry.register(
                name,
                handler,
                enabled=enabled,
                capability=capability,
                risk_classes=risk_classes,
                description=metadata.description,
                parameters_schema=metadata.parameters_schema,
            )

        register_tool(
            "fs",
            FilesystemTool(),
            enabled=self.tools_enabled.get("fs", False),
            capability="exec",
        )
        register_tool(
            "web",
            self.web_tool.handle,
            enabled=self.tools_enabled.get("web", False),
            capability="read",
            risk_classes=["read", "network", "external_side_effect"],
        )
        register_tool(
            "shell",
            ShellTool(),
            enabled=self.tools_enabled.get("shell", False),
            capability="exec",
            risk_classes=["execute"],
        )
        register_tool(
            "project",
            ProjectTool(),
            enabled=self.tools_enabled.get("project", False),
            capability="exec",
            risk_classes=["execute"],
        )
        register_tool(
            "image_analyze",
            ImageAnalyzeTool(),
            enabled=self.tools_enabled.get("image_analyze", False),
            capability="read",
        )
        register_tool(
            "image_generate",
            ImageGenerateTool(),
            enabled=self.tools_enabled.get("image_generate", False),
            capability="exec",
            risk_classes=["execute", "network", "external_side_effect"],
        )
        http_client = HttpClient()
        register_tool(
            "tts",
            TtsTool(http_client),
            enabled=self.tools_enabled.get("tts", False),
            capability="exec",
            risk_classes=["execute", "network", "external_side_effect"],
        )
        register_tool(
            "stt",
            SttTool(http_client),
            enabled=self.tools_enabled.get("stt", False),
            capability="exec",
            risk_classes=["execute", "network", "external_side_effect"],
        )
        register_tool(
            "workspace_list",
            ListFilesTool(),
            enabled=True,
            capability="read",
        )
        register_tool(
            "workspace_read",
            ReadFileTool(),
            enabled=True,
            capability="read",
        )
        register_tool(
            "workspace_write",
            WriteFileTool(),
            enabled=True,
            capability="write",
        )
        register_tool(
            "workspace_create",
            CreateFileTool(),
            enabled=True,
            capability="write",
        )
        register_tool(
            "workspace_rename",
            RenameFileTool(),
            enabled=True,
            capability="write",
        )
        register_tool(
            "workspace_move",
            MoveFileTool(),
            enabled=True,
            capability="write",
        )
        register_tool(
            "workspace_delete",
            DeleteFileTool(),
            enabled=True,
            capability="write",
            risk_classes=["write", "destructive"],
        )
        register_tool(
            "workspace_patch",
            ApplyPatchTool(),
            enabled=True,
            capability="write",
        )
        register_tool(
            "workspace_run",
            RunCodeTool(),
            enabled=self.tools_enabled.get("workspace_run", True),
            capability="exec",
            risk_classes=["execute"],
        )
        register_tool(
            "workspace_terminal_run",
            WorkspaceTerminalRunTool(),
            enabled=self.tools_enabled.get("workspace_run", True),
            capability="exec",
            risk_classes=["execute"],
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

    def update_tools_enabled(self, state: dict[str, bool], *, persist: bool = True) -> None:
        self.tools_enabled.update(state)
        if persist:
            save_tools_config(ToolsConfig(**self.tools_enabled))
            self.tracer.log("tools_updated", "Инструменты обновлены", {"tools": self.tools_enabled})
        else:
            self.tracer.log(
                "tools_runtime_updated",
                "Runtime-инструменты обновлены",
                {"tools": self.tools_enabled},
            )
        for name, enabled in state.items():
            if name in self.tool_registry.list_tools():
                self.tool_registry.set_enabled(name, enabled)
        if state.get("safe_mode") is not None:
            self._apply_safe_mode(state["safe_mode"])

    def apply_runtime_tools_enabled(self, state: dict[str, bool]) -> None:
        self.update_tools_enabled(state, persist=False)

    def get_embeddings_model(self) -> str:
        return self.vectors.model_name

    def set_embeddings_model(self, model_name: str) -> None:
        normalized = model_name.strip()
        if not normalized:
            raise ValueError("embeddings model не должен быть пустым")
        self.set_embeddings_config(
            provider="local",
            local_model=normalized,
            openai_model=self.vectors.openai_model,
            openai_api_key=self.vectors.openai_api_key,
        )

    def set_embeddings_config(
        self,
        *,
        provider: Literal["local", "openai"],
        local_model: str,
        openai_model: str,
        openai_api_key: str | None,
    ) -> None:
        normalized_provider = provider.strip().lower()
        normalized_local_model = local_model.strip()
        normalized_openai_model = openai_model.strip()
        if normalized_provider not in {"local", "openai"}:
            raise ValueError("embeddings provider должен быть local|openai")
        if not normalized_local_model:
            raise ValueError("local embeddings model не должен быть пустым")
        if not normalized_openai_model:
            raise ValueError("openai embeddings model не должен быть пустым")
        if (
            self.vectors.provider == normalized_provider
            and self.vectors.local_model == normalized_local_model
            and self.vectors.openai_model == normalized_openai_model
            and self.vectors.openai_api_key == openai_api_key
        ):
            return

        self.vectors.close()
        self.vectors = VectorIndex(
            "memory/vectors.db",
            provider=cast(Literal["local", "openai"], normalized_provider),
            local_model=normalized_local_model,
            openai_model=normalized_openai_model,
            openai_api_key=openai_api_key,
        )
        self._atom_embedding_index = AtomEmbeddingIndex(self.vectors)
        self.tracer.log(
            "embeddings_model_updated",
            "Embeddings model updated",
            {
                "provider": normalized_provider,
                "local_model": normalized_local_model,
                "openai_model": normalized_openai_model,
            },
        )

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

    def capture_memory_claims_from_text(
        self,
        text: str,
        *,
        source_kind: str,
        source_id: str,
        lang_hint: str | None = None,
    ) -> list[dict[str, JSONValue]]:
        payload = ClaimExtractionInput(
            text=text,
            source_kind=source_kind,
            source_id=source_id,
            lang_hint=lang_hint,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        )
        claims = self._claim_extractor.extract(payload)
        applied: list[dict[str, JSONValue]] = []
        for claim in claims:
            if not self._should_promote_claim(claim):
                continue
            atom = self._canonical_aggregator.upsert_claim(claim)
            self._atom_embedding_index.sync_atom(atom)
            applied.append(
                {
                    "stable_key": atom.stable_key,
                    "status": atom.status.value,
                    "claim_type": atom.claim_type.value,
                    "confidence": atom.confidence,
                }
            )
        if applied:
            self.tracer.log(
                "memory_claims_applied",
                f"Applied {len(applied)} claims",
                {"claims": applied, "source_kind": source_kind, "source_id": source_id},
            )
        return applied

    def is_explicit_memory_request(self, text: str) -> bool:
        return _EXPLICIT_MEMORY_PREFIX.match(text.strip()) is not None

    def remember_explicit_text(
        self,
        text: str,
        *,
        source_kind: str,
        source_id: str | None = None,
        lang_hint: str | None = None,
    ) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "Нечего запомнить."

        capture_text = (
            cleaned if self.is_explicit_memory_request(cleaned) else f"remember {cleaned}"
        )
        resolved_source_id = source_id or self.session_id or self.conversation_id
        applied = self.capture_memory_claims_from_text(
            capture_text,
            source_kind=source_kind,
            source_id=resolved_source_id,
            lang_hint=lang_hint,
        )
        if not applied:
            return "Не удалось выделить факт для памяти."

        stable_keys = [
            str(item["stable_key"])
            for item in applied
            if isinstance(item.get("stable_key"), str) and item["stable_key"]
        ]
        if not stable_keys:
            return f"Запомнил: {len(applied)}"
        return f"Запомнил: {', '.join(stable_keys)}"

    def build_memory_capsule(
        self,
        query: str,
        *,
        for_mwv: bool = False,
        allow_vector_runtime_init: bool = True,
    ) -> dict[str, JSONValue]:
        return build_memory_capsule_payload(
            query=query,
            store=self._canonical_store,
            vector_index=self.vectors,
            for_mwv=for_mwv,
            config=self._retrieval_config,
            allow_vector_runtime_init=allow_vector_runtime_init,
        )

    def list_memory_conflicts(self, limit: int = 50) -> list[dict[str, JSONValue]]:
        atoms = self._canonical_store.list_conflicts(limit=limit)
        return [self._atom_to_payload(atom) for atom in atoms]

    def list_pinned_memory_atoms(self, limit: int = 20) -> list[dict[str, JSONValue]]:
        atoms = self._canonical_store.list_pinned(limit=limit)
        return [self._atom_to_payload(atom) for atom in atoms]

    def pin_memory_atom(self, stable_key: str) -> dict[str, JSONValue] | None:
        if not self._canonical_store.set_pinned(stable_key, True):
            return None
        atom = self._canonical_store.get_by_stable_key(stable_key)
        if atom is None:
            return None
        self.tracer.log("memory_atom_pinned", stable_key, {"stable_key": stable_key})
        return self._atom_to_payload(atom)

    def unpin_memory_atom(self, stable_key: str) -> dict[str, JSONValue] | None:
        if not self._canonical_store.set_pinned(stable_key, False):
            return None
        atom = self._canonical_store.get_by_stable_key(stable_key)
        if atom is None:
            return None
        self.tracer.log("memory_atom_unpinned", stable_key, {"stable_key": stable_key})
        return self._atom_to_payload(atom)

    def summarize_current_session(self) -> dict[str, JSONValue] | None:
        summary = self._session_summarizer.summarize(self.short_term)
        if summary is None:
            self.tracer.log("session_summary_skipped", "No session messages to summarize")
            return None

        atom = self._canonical_aggregator.upsert_claim(summary.claim)
        self._atom_embedding_index.sync_atom(atom)
        self.tracer.log(
            "session_summary_saved",
            atom.stable_key,
            {"stable_key": atom.stable_key, "chars": len(summary.text)},
        )
        return self._atom_to_payload(atom)

    def resolve_memory_conflict(
        self,
        *,
        stable_key: str,
        action: str,
        value_json: JSONValue | None = None,
    ) -> dict[str, JSONValue] | None:
        atom = self._canonical_aggregator.resolve_conflict(
            stable_key=stable_key,
            action=action,
            value_json=value_json,
        )
        if atom is None:
            return None
        self._atom_embedding_index.sync_atom(atom)
        self.tracer.log(
            "memory_conflict_resolved",
            stable_key,
            {"action": action, "status": atom.status.value},
        )
        return self._atom_to_payload(atom)

    def _should_promote_claim(self, claim: Claim) -> bool:
        if claim.is_explicit:
            return True
        return claim.claim_type in {
            ClaimType.PREFERENCE,
            ClaimType.ENVIRONMENT,
            ClaimType.FACT,
        }

    def _atom_to_payload(self, atom: CanonicalAtom) -> dict[str, JSONValue]:
        return {
            "atom_id": atom.atom_id,
            "stable_key": atom.stable_key,
            "claim_type": atom.claim_type.value,
            "value_json": atom.value_json,
            "confidence": atom.confidence,
            "support_count": atom.support_count,
            "contradict_count": atom.contradict_count,
            "last_seen_at": atom.last_seen_at,
            "status": atom.status.value,
            "summary_text": atom.summary_text,
            "pinned": atom.pinned,
        }

    def _build_context_messages(self, messages: list[LLMMessage], query: str) -> list[LLMMessage]:
        budget = self.memory_config.context_budget
        remaining = budget.total_chars
        filled_slots: list[str] = []
        slot_sizes: dict[str, int] = {}

        def _append_slot(name: str, raw_text: str, max_chars: int) -> None:
            nonlocal remaining
            text = raw_text.strip()
            if not text or max_chars <= 0:
                slot_sizes[name] = 0
                return
            if remaining <= 0:
                slot_sizes[name] = 0
                self.tracer.log("context_budget_exhausted", name, {"slot": name})
                return

            text = text[:max_chars].rstrip()
            prefix = f"[[SLOT:{name}]]\n"
            suffix = "\n[[/SLOT]]"
            separator_len = 2 if filled_slots else 0
            overhead = separator_len + len(prefix) + len(suffix)
            if remaining <= overhead:
                slot_sizes[name] = 0
                self.tracer.log(
                    "context_budget_exhausted",
                    name,
                    {"slot": name, "remaining": remaining},
                )
                return

            allowed_payload = min(len(text), remaining - overhead)
            text = text[:allowed_payload].rstrip()
            if not text:
                slot_sizes[name] = 0
                return
            slot = f"{prefix}{text}{suffix}"
            filled_slots.append(slot)
            slot_sizes[name] = len(slot)
            remaining -= len(slot) + separator_len

        pinned_atoms = self._canonical_store.list_pinned(limit=20)
        if pinned_atoms:
            pinned_parts = ["Закрепленная память:"]
            for atom in pinned_atoms:
                pinned_parts.append(
                    f"- [{atom.claim_type.value}] {atom.stable_key}: {atom.summary_text}"
                )
            _append_slot("pinned_atoms", "\n".join(pinned_parts), budget.pinned_atoms_chars)

        session_atoms = self._canonical_store.list_atoms(
            statuses={AtomStatus.ACTIVE},
            claim_types={ClaimType.FACT},
            stable_key_prefix="session:",
            limit=3,
        )
        if session_atoms:
            session_parts = ["Резюме прошлых сессий:"]
            for atom in session_atoms:
                summary_text = atom.summary_text
                value = atom.value_json
                if isinstance(value, dict):
                    raw_text = value.get("text")
                    if isinstance(raw_text, str) and raw_text.strip():
                        summary_text = raw_text.strip()
                session_parts.append(f"- {atom.stable_key}: {summary_text[:500]}")
            _append_slot(
                "session_summary",
                "\n".join(session_parts),
                budget.session_summary_chars,
            )

        recent_notes = self.memory.get_recent(
            max(1, budget.legacy_notes_chars // 200),
            kind=MemoryKind.NOTE,
        )
        if recent_notes:
            note_parts = ["Недавняя память:"]
            for note in recent_notes:
                note_parts.append(f"- {note.content[:200]}")
            _append_slot("legacy_notes", "\n".join(note_parts), budget.legacy_notes_chars)

        hints_meta = self._collect_feedback_hints(
            budget.feedback_max_items,
            severity_filter=["major", "fatal"],
        )
        if hints_meta:
            hint_parts = ["Подсказки от пользователя:"]
            for hint_meta in hints_meta:
                hint_parts.append(f"- ({hint_meta.get('severity')}) {hint_meta.get('hint')}")
            _append_slot("feedback", "\n".join(hint_parts), budget.feedback_chars)
            self.last_hints_used = [h["hint"] for h in hints_meta]
            self.last_hints_meta = hints_meta
            self.tracer.log("auto_hint_applied", "Использованы подсказки", {"hints": hints_meta})
        else:
            self.last_hints_used = []
            self.last_hints_meta = []

        prefs_all = self.memory.get_user_prefs()
        prefs = prefs_all[: budget.prefs_max_items]
        if len(prefs_all) > len(prefs):
            self.tracer.log(
                "prefs_truncated",
                f"{len(prefs_all)} -> {len(prefs)}",
                {"total": len(prefs_all), "kept": len(prefs)},
            )
        if prefs:
            pref_parts = ["Предпочтения пользователя:"]
            for pref in prefs:
                meta = pref.meta or {}
                pref_parts.append(f"- {meta.get('key')}: {meta.get('value')}")
            _append_slot("prefs", "\n".join(pref_parts), budget.prefs_chars)

        try:
            retrieval_config = replace(
                self._retrieval_config,
                max_context_chars=budget.canonical_memory_chars,
            )
            memory_capsule = build_memory_capsule_payload(
                query=query,
                store=self._canonical_store,
                vector_index=self.vectors,
                for_mwv=False,
                config=retrieval_config,
                allow_vector_runtime_init=True,
            )
            capsule_text = memory_capsule.get("text")
            if isinstance(capsule_text, str) and capsule_text.strip():
                canonical_text = "\n".join(["Каноническая память:", *capsule_text.splitlines()])
                _append_slot(
                    "canonical_memory",
                    canonical_text,
                    budget.canonical_memory_chars,
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Canonical memory retrieval failed: %s", exc)

        # Векторный поиск по проектному индексу (code + docs)
        try:
            code_top_k = max(1, budget.vector_code_chars // 400)
            docs_top_k = max(1, budget.vector_docs_chars // 400)
            vec_results_code = self.vectors.search(
                query,
                namespace="code",
                top_k=code_top_k,
                allow_runtime_init=True,
            )
            vec_results_docs = self.vectors.search(
                query,
                namespace="docs",
                top_k=docs_top_k,
                allow_runtime_init=True,
            )
            if vec_results_code:
                code_parts = ["Контекст проекта (code):"]
                for res in vec_results_code:
                    code_parts.append(f"- {res.path}: {res.snippet}")
                _append_slot("vectors_code", "\n".join(code_parts), budget.vector_code_chars)
            if vec_results_docs:
                docs_parts = ["Контекст проекта (docs):"]
                for res in vec_results_docs:
                    docs_parts.append(f"- {res.path}: {res.snippet}")
                _append_slot("vectors_docs", "\n".join(docs_parts), budget.vector_docs_chars)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Vector search failed: %s", exc)

        if self.workspace_file_path and self.workspace_file_content is not None:
            workspace_parts = ["Текущий файл:", f"- path: {self.workspace_file_path}"]
            if self.workspace_selection:
                selection_snippet = self.workspace_selection[: budget.workspace_file_chars]
                workspace_parts.append(f"- выделение:\n{selection_snippet}")
            content_snippet = self.workspace_file_content
            if len(content_snippet) > budget.workspace_file_chars:
                content_snippet = content_snippet[: budget.workspace_file_chars]
            workspace_parts.append(f"- содержимое:\n{content_snippet}")
            _append_slot("workspace_file", "\n".join(workspace_parts), budget.workspace_file_chars)

        if filled_slots:
            context_msg = "\n\n".join(filled_slots)
            self.last_context_text = context_msg
            self.tracer.log(
                "context_built",
                f"total_chars={len(context_msg)}",
                {"total_chars": len(context_msg), "slots": slot_sizes},
            )
            return [LLMMessage(role="system", content=context_msg), *messages]
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
