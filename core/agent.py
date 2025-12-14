from __future__ import annotations

import logging
import time
import uuid

from config.mode_config import load_mode, save_mode
from config.model_store import load_model_configs, save_model_configs
from config.shell_config import DEFAULT_SHELL_CONFIG_PATH
from config.system_prompts import CRITIC_PROMPT
from config.tools_config import ToolsConfig, load_tools_config, save_tools_config
from core.auto_agent import AutoAgent
from core.executor import Executor
from core.planner import Planner
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from llm.brain_base import Brain
from llm.brain_factory import create_brain
from llm.brain_manager import BrainManager
from llm.dual_brain import DualBrain
from llm.types import LLMResult, ModelConfig
from memory.feedback_manager import FeedbackManager
from memory.memory_manager import MemoryManager
from memory.vector_index import VectorIndex
from shared.models import (
    JSONValue,
    LLMMessage,
    MemoryKind,
    MemoryRecord,
    PlanStep,
    TaskComplexity,
    TaskPlan,
    ToolCallRecord,
    ToolRequest,
    ToolResult,
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
    "img": False,
    "tts": False,
    "stt": False,
    "safe_mode": True,
}
SAFE_MODE_TOOLS_OFF = {"web", "shell", "project", "tts", "stt"}


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
        self.feedback = FeedbackManager("memory/feedback.db")
        self.short_term: list[LLMMessage] = []
        self.last_plan: TaskPlan | None = None
        self.last_plan_original: TaskPlan | None = None
        self.last_hints_used: list[str] = []
        self.last_hints_meta: list[dict[str, str]] = []
        self.last_context_text: str | None = None
        self.workspace_file_path: str | None = None
        self.workspace_file_content: str | None = None
        self.workspace_selection: str | None = None
        self._init_mode_from_config()

    def respond(self, messages: list[LLMMessage]) -> str:
        if not messages:
            return "[Пустое сообщение]"

        last_content = messages[-1].content.strip()
        self.short_term.extend(messages)
        self.tracer.log("user_input", last_content)

        if last_content.lower().startswith("авто") or last_content.startswith("/auto"):
            return self.handle_auto_command(last_content.replace("/auto", "").strip())

        if last_content.startswith("/"):
            return self.handle_tool_command(last_content)

        complexity = self.planner.classify_complexity(last_content)

        if (
            last_content.lower().startswith(("план", "plan"))
            or complexity == TaskComplexity.COMPLEX
        ):
            plan = self.planner.build_plan(
                last_content, brain=self.brain, model_config=self.main_config
            )
            self.last_plan_original = plan
            dual_mode = "single"
            if isinstance(self.brain, DualBrain):
                dual_mode = getattr(self.brain, "mode", "single")
            if isinstance(self.brain, DualBrain) and dual_mode != "single":
                plan = self._critic_plan(plan)
            if dual_mode == "critic-only":
                self.last_plan = plan
                return self._review_plan(plan)
            executed = self.executor.run(
                plan,
                tool_gateway=ToolGateway(self.tool_registry),
                critic_callback=self._critic_step if self._should_use_step_critic() else None,
            )
            self.last_plan = executed
            reviewed = self._review_plan(executed)
            return reviewed

        try:
            self.tracer.log("reasoning_start", "Генерация ответа моделью")
            messages_with_context = self._build_context_messages(self.short_term, last_content)
            reply = self.brain.generate(messages_with_context)
            reply_text = reply.text if isinstance(reply, LLMResult) else str(reply)
            reviewed = self._review_answer(reply_text)
            self.tracer.log("reasoning_end", "Ответ получен", {"reply_preview": reviewed[:120]})
            self.save_to_memory(last_content, reviewed)
            return reviewed
        except Exception as exc:  # noqa: BLE001
            self.logger.error("LLM error: %s", exc)
            self.tracer.log("error", f"Ошибка модели: {exc}")
            return f"[Ошибка модели: {exc}]"

    def handle_tool_command(self, command: str) -> str:
        parts = command.split()
        cmd = parts[0][1:].lower()
        args = parts[1:]
        self.tracer.log("tool_invoked", cmd, {"args": args})

        try:
            if cmd == "auto":
                goal = " ".join(args)
                return self.handle_auto_command(goal)

            if cmd == "plan":
                goal = " ".join(args)
                plan = self.planner.build_plan(goal)
                executed = self.planner.execute_plan(plan)
                return self._format_plan(executed)

            if cmd == "fs":
                operation = args[0] if args else "list"
                path_arg = args[1] if len(args) > 1 else ""
                req = ToolRequest(name="fs", args={"op": operation, "path": path_arg})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd == "web":
                query = " ".join(args)
                req = ToolRequest(name="web", args={"query": query})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd == "sh":
                req = ToolRequest(name="shell", args={"command": " ".join(args)})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd == "project":
                if not args:
                    return "[Нужно указать подкоманду: index|find]"
                req = ToolRequest(name="project", args={"cmd": args[0], "args": args[1:]})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd in {"imggen", "img_generate"}:
                prompt = " ".join(args) or "image"
                req = ToolRequest(name="image_generate", args={"prompt": prompt})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd in {"imganalyze", "img_analyze"}:
                if not args:
                    return "[Нужно указать base64 или путь]"
                req = ToolRequest(name="image_analyze", args={"path": args[0]})
                result = self.tool_registry.call(req)
                return self._format_tool_result(result)

            if cmd == "trace":
                logs = self.tracer.read_recent(40)
                return "\n".join(
                    [f"[{log['timestamp']}] {log['event']}: {log['message']}" for log in logs]
                )

            return f"[Инструмент '{cmd}' неактивен или неизвестен]"
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("error", f"Ошибка при вызове инструмента: {exc}")
            return f"[Ошибка при вызове инструмента: {exc}]"

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

    def save_feedback(self, prompt: str, answer: str, rating: str, hint: str | None = None) -> None:
        severity = "minor"
        if rating in {"bad", "offtopic"}:
            severity = "major"
            hint = hint or "Перепроверь факты и избегай галлюцинаций."
        self.feedback.save_feedback(prompt, answer, rating, severity=severity, hint=hint)
        self.tracer.log("feedback_saved", rating, {"prompt": prompt[:80]})

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
        lines = []
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

    def _review_plan(self, plan: TaskPlan) -> str:
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
                critic = self.brain.critic if isinstance(self.brain, DualBrain) else None
                if critic is None:
                    return plan_text
                review = critic.generate(critic_messages)
                review_text = review.text if isinstance(review, LLMResult) else str(review)
                return f"{plan_text}\n\n🧠 Критик плана:\n{review_text}"
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Не удалось получить критику плана: %s", exc)
        return plan_text

    def _critic_plan(self, plan: TaskPlan) -> TaskPlan:
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
            critic = self.brain.critic if isinstance(self.brain, DualBrain) else None
            if critic is None:
                return plan
            review = critic.generate(critic_messages)
            review_text = review.text if isinstance(review, LLMResult) else str(review)
            new_steps = self.planner._parse_plan_text(review_text) or []  # noqa: SLF001
            if len(new_steps) >= 2:
                rewritten = TaskPlan(
                    goal=plan.goal,
                    steps=[PlanStep(description=s) for s in new_steps],
                )
                if hasattr(self.planner, "_assign_operations"):
                    return self.planner._assign_operations(rewritten)  # noqa: SLF001
                return rewritten
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Не удалось получить критику плана: %s", exc)
        return plan

    def _critic_step(self, step: PlanStep) -> tuple[bool, str | None]:
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
            critic = self.brain.critic if isinstance(self.brain, DualBrain) else None
            if critic is None:
                return True, None
            review = critic.generate(critic_messages)
            review_text = review.text if isinstance(review, LLMResult) else str(review)
            if "approve" in review_text.lower():
                return True, None
            return False, review_text.strip()
        except Exception as exc:  # noqa: BLE001
            self.tracer.log("critic_error", f"Критик не доступен: {exc}")
            return True, None

    def _should_use_step_critic(self) -> bool:
        return isinstance(self.brain, DualBrain) and self.brain.mode == "dual"

    def _review_answer(self, answer: str) -> str:
        if isinstance(self.brain, DualBrain) and self.brain.mode != "single":
            return answer  # уже включён критик в DualBrain
        if isinstance(self.brain, DualBrain):
            try:
                critic_messages = [
                    LLMMessage(role="system", content=CRITIC_PROMPT),
                    LLMMessage(
                        role="user",
                        content=f"Проверь ответ и кратко укажи ошибки/улучшения:\n{answer}",
                    ),
                ]
                critic = self.brain.critic if isinstance(self.brain, DualBrain) else None
                if critic is None:
                    return answer
                review = critic.generate(critic_messages)
                review_text = review.text if isinstance(review, LLMResult) else str(review)
                return f"{answer}\n\n🧠 Критик:\n{review_text}"
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Не удалось получить критику ответа: %s", exc)
                return answer
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
            "fs", FilesystemTool(), enabled=self.tools_enabled.get("fs", False)
        )
        self.tool_registry.register(
            "web", self.web_tool.handle, enabled=self.tools_enabled.get("web", False)
        )
        self.tool_registry.register(
            "shell", ShellTool(), enabled=self.tools_enabled.get("shell", False)
        )
        self.tool_registry.register(
            "project", ProjectTool(), enabled=self.tools_enabled.get("project", False)
        )
        self.tool_registry.register(
            "image_analyze",
            ImageAnalyzeTool(),
            enabled=self.tools_enabled.get("img", False),
        )
        self.tool_registry.register(
            "image_generate",
            ImageGenerateTool(),
            enabled=self.tools_enabled.get("img", False),
        )
        http_client = HttpClient()
        self.tool_registry.register(
            "tts", TtsTool(http_client), enabled=self.tools_enabled.get("tts", False)
        )
        self.tool_registry.register(
            "stt", SttTool(http_client), enabled=self.tools_enabled.get("stt", False)
        )
        self.tool_registry.register("workspace_list", ListFilesTool(), enabled=True)
        self.tool_registry.register("workspace_read", ReadFileTool(), enabled=True)
        self.tool_registry.register("workspace_write", WriteFileTool(), enabled=True)
        self.tool_registry.register("workspace_patch", ApplyPatchTool(), enabled=True)
        self.tool_registry.register("workspace_run", RunCodeTool(), enabled=True)

    def synthesize_speech(
        self, text: str, voice_id: str | None = None, fmt: str | None = None
    ) -> ToolResult:
        args: dict[str, JSONValue] = {"text": text}
        if voice_id:
            args["voice_id"] = voice_id
        if fmt:
            args["format"] = fmt
        return self.tool_registry.call(ToolRequest(name="tts", args=args))

    def transcribe_audio(self, file_path: str, language: str | None = None) -> ToolResult:
        args: dict[str, JSONValue] = {"file_path": file_path}
        if language:
            args["language"] = language
        return self.tool_registry.call(ToolRequest(name="stt", args=args))

    def set_workspace_context(
        self, path: str | None, content: str | None, selection: str | None = None
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
            self.tracer.log("safe_mode", "Safe mode enabled, web/shell disabled")
        else:
            self.tracer.log("safe_mode", "Safe mode disabled")

    def get_recent_tool_calls(self, limit: int = 50) -> list[ToolCallRecord]:
        return self.tool_registry.read_recent_calls(limit)

    def _build_context_messages(self, messages: list[LLMMessage], query: str) -> list[LLMMessage]:
        context_parts = []

        recent_notes = self.memory.get_recent(3, kind=MemoryKind.NOTE)
        if recent_notes:
            context_parts.append("Недавняя память:")
            for note in recent_notes:
                context_parts.append(f"- {note.content[:200]}")

        hints_meta = self.feedback.get_recent_hints_meta(2, severity_filter=["major", "fatal"])
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
