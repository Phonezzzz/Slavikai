from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Final, Literal, cast

from config.computer_backend_config import resolve_computer_backend_config
from config.memory_config import MemoryConfig, load_memory_config
from config.shell_config import DEFAULT_SHELL_CONFIG_PATH
from config.tools_config import ToolsConfig, load_tools_config, save_tools_config
from core.agent_computer import AgentComputerRuntime
from core.agent_memory import AgentMemoryMixin
from core.agent_mwv import AgentMWVMixin
from core.agent_routing import AgentRoutingMixin
from core.agent_tools import AgentToolsMixin
from core.approval_policy import ApprovalCategory, ApprovalRequest
from core.auto_agent import AutoAgent
from core.computer_activity_log import ComputerActivityLog
from core.computer_backend import ComputerBackend, LocalComputerBackend
from core.container_computer_backend import ContainerComputerBackend
from core.decision.handler import DecisionHandler
from core.decision.models import DecisionPacket
from core.desktop_policy import DesktopPolicyRuntime, DesktopPolicyStore
from core.desktop_runtime import DesktopExecutionControl, DesktopRunCoordinator, DesktopRuntime
from core.desktop_security import DesktopPathSecurity
from core.mwv.manager import ManagerRuntime
from core.mwv.verifier_runtime import VerifierRuntime
from core.plan_compiler import compile_structured_plan_steps
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
from memory.session_summarizer import SessionSummarizer
from memory.vector_index import VectorIndex
from shared.models import (
    JSONValue,
    LLMMessage,
    TaskPlan,
    ToolCallRecord,
    ToolResult,
    WorkspaceDiffEntry,
)
from tools.desktop_browser_tool import DesktopBrowserTool
from tools.desktop_gui_tool import DesktopGuiTool
from tools.desktop_system_tools import (
    DesktopClipboardTool,
    DesktopPackageTool,
    DesktopProcessTool,
    DesktopSessionTool,
    DesktopSystemdTool,
    DesktopSystemInfoTool,
    DesktopUnverifiedLaunchCleanupTool,
)
from tools.desktop_tools import (
    DesktopArchiveExtractTool,
    DesktopFileDeleteTool,
    DesktopFileReadTool,
    DesktopFileSearchTool,
    DesktopFileTransferTool,
    DesktopFileWriteTool,
    DesktopLaunchTool,
    DesktopOpenTool,
    DesktopShellTool,
    DesktopVerifyTool,
)
from tools.filesystem_tool import SANDBOX_ROOT, FilesystemTool
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
CHAT_EXPOSED_READ_TOOLS: Final[frozenset[str]] = frozenset(
    {"web", "workspace_list", "workspace_read"}
)

_COMPAT_EXPORTS = (ManagerRuntime, VerifierRuntime, WORKSPACE_ROOT, MAX_FILE_BYTES)


class Agent(AgentRoutingMixin, AgentMWVMixin, AgentToolsMixin, AgentMemoryMixin):
    """SlavikAI Core v1.0 — Распределённый рассуждающий агент."""

    def compile_plan_steps(
        self,
        goal: str,
        audit_log: list[dict[str, JSONValue]],
    ) -> list[dict[str, JSONValue]]:
        return compile_structured_plan_steps(
            brain=self._get_main_brain(),
            config=self.main_config,
            goal=goal,
            audit_log=audit_log,
            available_tools=self.tool_registry.list_tool_specs(),
        )

    def __init__(
        self,
        brain: Brain | None = None,
        enable_tools: dict[str, bool] | None = None,
        main_config: ModelConfig | None = None,
        main_api_key: str | None = None,
        brain_manager: BrainManager | None = None,
        user_id: str = "local",
        memory_db_path: str | None = None,
        vectors_db_path: str | None = None,
        memory_companion_db_path: str | None = None,
        memory_inbox_db_path: str | None = None,
        canonical_atoms_db_path: str | None = None,
        embeddings_provider: Literal["local", "openai"] = "local",
        embeddings_local_model: str = "all-MiniLM-L6-v2",
        embeddings_openai_model: str = "text-embedding-3-small",
        embeddings_openai_api_key: str | None = None,
        desktop_policy_store: DesktopPolicyStore | None = None,
        desktop_home: Path | None = None,
        desktop_run_coordinator: DesktopRunCoordinator | None = None,
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
        self.tracer = Tracer()
        self.auto_agent = AutoAgent(self)
        self.auto_agent.set_progress_callback(self._record_auto_progress)
        self.desktop_execution_control = DesktopExecutionControl()
        self.desktop_runtime = DesktopRuntime(
            self,
            run_coordinator=desktop_run_coordinator,
        )
        self.tools_enabled = enable_tools or self._load_tools()
        self.desktop_policy_store = desktop_policy_store or DesktopPolicyStore()
        project_root = Path(__file__).resolve().parents[1]
        self.desktop_security = DesktopPathSecurity(
            home=desktop_home,
            policy_store_path=self.desktop_policy_store.path,
            protected_paths=(
                project_root / "config",
                project_root / "core" / "approval_policy.py",
                project_root / "core" / "agent.py",
                project_root / "core" / "desktop_policy.py",
                project_root / "core" / "desktop_runtime.py",
                project_root / "core" / "desktop_security.py",
                project_root / "core" / "tool_gateway.py",
                project_root / "core" / "tool_loop.py",
                project_root / "requirements.in",
                project_root / "requirements.txt",
                project_root / "constraints.txt",
                project_root / "tools" / "desktop_tools.py",
                project_root / "tools" / "desktop_browser_tool.py",
                project_root / "tools" / "desktop_gui_tool.py",
                project_root / "tools" / "desktop_system_tools.py",
                project_root / "tools" / "tool_descriptors.py",
                project_root / "tools" / "tool_registry.py",
                project_root / "server" / "http" / "handlers" / "decision.py",
                project_root / "server" / "http" / "handlers" / "desktop.py",
            ),
        )
        self.desktop_policy_runtime = DesktopPolicyRuntime(self.desktop_policy_store.list_rules())
        self.tool_registry = ToolRegistry(safe_block=SAFE_MODE_TOOLS_OFF)
        self.web_tool = WebSearchTool()
        self._register_tools()
        if self.tools_enabled.get("safe_mode", False):
            self._apply_safe_mode(True)
        self.memory = MemoryManager(memory_db_path or "memory/memory.db")
        self._memory_inbox_store = (
            CategorizedMemoryStore(memory_inbox_db_path)
            if memory_inbox_db_path
            else CategorizedMemoryStore()
        )
        self._memory_inbox_writer = MemoryInboxWriter(self._memory_inbox_store, self.memory_config)
        self._vectors_db_path = vectors_db_path or "memory/vectors.db"
        self.vectors = VectorIndex(
            self._vectors_db_path,
            provider=embeddings_provider,
            local_model=embeddings_local_model,
            openai_model=embeddings_openai_model,
            openai_api_key=embeddings_openai_api_key,
        )
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
        self.last_auto_state: dict[str, JSONValue] | None = None
        self._auto_progress_events: list[dict[str, JSONValue]] = []
        self.workspace_file_path: str | None = None
        self.workspace_file_content: str | None = None
        self.workspace_selection: str | None = None
        self._workspace_diff_baselines: dict[str, str] = {}
        self._workspace_diffs: dict[str, WorkspaceDiffEntry] = {}
        self._computer_log = ComputerActivityLog()

    def _review_answer(self, answer: str) -> str:
        return answer

    def _record_auto_progress(self, state: dict[str, JSONValue]) -> None:
        self._auto_progress_events.append(dict(state))

    def drain_auto_progress_events(self) -> list[dict[str, JSONValue]]:
        events = [dict(item) for item in self._auto_progress_events]
        self._auto_progress_events.clear()
        return events

    def make_computer_runtime(self) -> AgentComputerRuntime:
        cfg = resolve_computer_backend_config()
        backend: ComputerBackend
        if cfg.mode == "container":
            backend = ContainerComputerBackend(
                image=cfg.container_image,
                project_root=WORKSPACE_ROOT,
                container_workspace=cfg.container_workspace,
                runtime=cfg.container_engine,
            )
        else:
            backend = LocalComputerBackend(gateway=self._build_tool_gateway())
        return AgentComputerRuntime(backend=backend)

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
            execution_targets: set[str] | None = None,
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
                chat_exposed=name in CHAT_EXPOSED_READ_TOOLS,
                execution_targets=execution_targets,
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
            ImageAnalyzeTool(sandbox_root=SANDBOX_ROOT),
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
        desktop_open = DesktopOpenTool(self.desktop_security)
        desktop_launcher = DesktopLaunchTool(
            self.desktop_security,
            cancelled=self.desktop_execution_control.cancelled,
            on_launch=self.desktop_execution_control.register_launch,
        )
        self.desktop_browser_tool = DesktopBrowserTool(
            self.desktop_security,
            cancelled=self.desktop_execution_control.cancelled,
        )
        self.desktop_gui_tool = DesktopGuiTool()
        desktop_process = DesktopProcessTool(
            self.desktop_security,
            launcher=desktop_launcher,
            cancelled=self.desktop_execution_control.cancelled,
        )
        desktop_tools: list[tuple[str, Tool, ToolCapability, list[str]]] = [
            ("desktop_file_search", DesktopFileSearchTool(self.desktop_security), "read", ["read"]),
            ("desktop_file_read", DesktopFileReadTool(self.desktop_security), "read", ["read"]),
            ("desktop_file_write", DesktopFileWriteTool(self.desktop_security), "write", ["write"]),
            (
                "desktop_file_transfer",
                DesktopFileTransferTool(self.desktop_security),
                "write",
                ["write"],
            ),
            (
                "desktop_file_delete",
                DesktopFileDeleteTool(self.desktop_security),
                "write",
                ["write", "destructive"],
            ),
            (
                "desktop_archive_extract",
                DesktopArchiveExtractTool(self.desktop_security),
                "write",
                ["write"],
            ),
            (
                "desktop_shell",
                DesktopShellTool(
                    self.desktop_security,
                    cancelled=self.desktop_execution_control.cancelled,
                ),
                "exec",
                ["execute"],
            ),
            ("desktop_clipboard", DesktopClipboardTool(), "exec", ["external_side_effect"]),
            ("desktop_system_info", DesktopSystemInfoTool(), "read", ["read"]),
            (
                "desktop_process",
                desktop_process,
                "exec",
                ["execute"],
            ),
            ("desktop_systemd", DesktopSystemdTool(), "exec", ["execute", "privileged"]),
            ("desktop_package", DesktopPackageTool(), "exec", ["install", "privileged"]),
            ("desktop_session", DesktopSessionTool(), "exec", ["external_side_effect"]),
            ("desktop_open", desktop_open, "exec", ["execute", "external_side_effect"]),
            (
                "desktop_browser",
                self.desktop_browser_tool,
                "exec",
                ["network", "external_side_effect"],
            ),
            (
                "desktop_gui",
                self.desktop_gui_tool,
                "exec",
                ["execute", "external_side_effect"],
            ),
            ("desktop_verify", DesktopVerifyTool(self.desktop_security), "read", ["read"]),
        ]
        for name, handler, capability, risk_classes in desktop_tools:
            register_tool(
                name,
                handler,
                enabled=True,
                capability=capability,
                risk_classes=risk_classes,
                execution_targets={"desktop"},
            )
        self.tool_registry.register(
            "desktop_cleanup_unverified_launches",
            DesktopUnverifiedLaunchCleanupTool(
                self.desktop_execution_control,
                desktop_process,
            ),
            enabled=True,
            capability="exec",
            risk_classes=["execute"],
            description="Rollback processes launched by the current failed Desktop run.",
            parameters_schema={"type": "object", "properties": {}, "additionalProperties": False},
            model_exposed=False,
            execution_targets={"desktop"},
        )

    def close_desktop_resources(self) -> None:
        self.desktop_browser_tool.close()
        self.desktop_gui_tool.close()

    def close(self) -> None:
        self.close_desktop_resources()
        self.memory.close()
        self.vectors.close()
        self._interaction_store.close()
        self._memory_inbox_store.close()
        self._canonical_store.close()

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
            self._vectors_db_path,
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
