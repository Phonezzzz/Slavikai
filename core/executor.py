from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from core.approval_policy import ApprovalRequired
from core.tool_gateway import ToolGateway
from core.tracer import Tracer
from shared.models import PlanStep, PlanStepStatus, TaskPlan, ToolRequest, ToolResult

AgentCallback = Callable[[PlanStep], str]

_TEXT_EXTENSIONS = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def _extract_path(text: str, allowed_exts: set[str] | None = None) -> str | None:
    if not text:
        return None
    tokens = text.replace(",", " ").split()
    for raw in tokens:
        candidate = raw.strip("()[]{}<>\"'`.;:")
        if not candidate:
            continue
        if candidate.startswith(("path=", "path:")):
            candidate = candidate.split("=", 1)[-1].split(":", 1)[-1].strip()
            if not candidate:
                continue
        if "/" in candidate or "\\" in candidate:
            return candidate
        if allowed_exts:
            if Path(candidate).suffix.lower() in allowed_exts:
                return candidate
    return None


def _extract_labeled_value(text: str, key: str) -> str | None:
    if not text:
        return None
    normalized = text.lower()
    for marker in (f"{key.lower()}=", f"{key.lower()}:"):
        idx = normalized.find(marker)
        if idx < 0:
            continue
        value = text[idx + len(marker) :].strip()
        if not value:
            return None
        quote = value[:1]
        if quote in {'"', "'", "`"} and value.endswith(quote) and len(value) > 1:
            value = value[1:-1]
        return value.replace("\\n", "\n")
    return None


class Executor:
    """Выполняет шаги плана с трассировкой."""

    def __init__(self, tracer: Tracer | None = None) -> None:
        self.tracer = tracer or Tracer()

    def run(
        self,
        plan: TaskPlan,
        tool_gateway: ToolGateway | None = None,
        agent_callback: AgentCallback | None = None,
    ) -> TaskPlan:
        self.tracer.log("execution_start", f"Начато выполнение плана ({len(plan.steps)} шагов)")
        for index, step in enumerate(plan.steps, start=1):
            self.tracer.log("step_started", f"{index}. {step.description}")
            step.status = PlanStepStatus.IN_PROGRESS

            try:
                if agent_callback:
                    result_text = agent_callback(step)
                elif tool_gateway:
                    result_text = self._execute_with_tools(step, plan, tool_gateway)
                else:
                    result_text = f"Выполнен: {step.description}"
                step.result = result_text
                step.status = PlanStepStatus.DONE
                self.tracer.log(
                    "step_finished",
                    f"{index}. {step.description}",
                    {"result": result_text},
                )
            except ApprovalRequired:
                raise
            except Exception as exc:  # noqa: BLE001
                step.status = PlanStepStatus.ERROR
                step.result = str(exc)
                self.tracer.log("step_failed", f"{index}. {step.description}", {"error": str(exc)})
                break

        self.tracer.log("execution_end", "План выполнен.")
        return plan

    def _execute_with_tools(self, step: PlanStep, plan: TaskPlan, gateway: ToolGateway) -> str:
        if step.operation is None:
            return f"Выполнен: {step.description}"

        request: ToolRequest
        if step.operation == "web":
            request = ToolRequest(name="web", args={"query": plan.goal})
        elif step.operation == "fs":
            path = _extract_path(step.description, _TEXT_EXTENSIONS) or _extract_path(
                plan.goal,
                _TEXT_EXTENSIONS,
            )
            if path:
                request = ToolRequest(name="fs", args={"op": "read", "path": path})
            else:
                request = ToolRequest(name="fs", args={"op": "list"})
        elif step.operation == "shell":
            request = ToolRequest(name="shell", args={"command": "echo step"})
        elif step.operation == "project":
            request = ToolRequest(name="project", args={"cmd": "find", "args": [plan.goal]})
        elif step.operation == "tts":
            request = ToolRequest(name="tts", args={"text": plan.goal})
        elif step.operation == "stt":
            audio_path = _extract_path(step.description, _AUDIO_EXTENSIONS) or _extract_path(
                plan.goal,
                _AUDIO_EXTENSIONS,
            )
            if not audio_path:
                raise RuntimeError("Не указан аудиофайл для распознавания.")
            request = ToolRequest(name="stt", args={"file_path": audio_path})
        elif step.operation == "image_analyze":
            image_path = _extract_path(step.description, _IMAGE_EXTENSIONS) or _extract_path(
                plan.goal,
                _IMAGE_EXTENSIONS,
            )
            if not image_path:
                raise RuntimeError("Не указан путь к изображению для анализа.")
            request = ToolRequest(name="image_analyze", args={"path": image_path})
        elif step.operation == "image_generate":
            request = ToolRequest(name="image_generate", args={"prompt": plan.goal})
        elif step.operation == "workspace_read":
            file_path = _extract_path(step.description, _TEXT_EXTENSIONS) or _extract_path(
                plan.goal,
                _TEXT_EXTENSIONS,
            )
            if not file_path:
                raise RuntimeError("Не указан путь к файлу workspace для чтения.")
            request = ToolRequest(name="workspace_read", args={"path": file_path})
        elif step.operation == "workspace_write":
            file_path = _extract_path(step.description, _TEXT_EXTENSIONS) or _extract_path(
                plan.goal,
                _TEXT_EXTENSIONS,
            )
            if not file_path:
                raise RuntimeError("Не указан путь к файлу workspace для записи.")
            content = _extract_labeled_value(step.description, "content") or _extract_labeled_value(
                plan.goal,
                "content",
            )
            request = ToolRequest(
                name="workspace_write",
                args={"path": file_path, "content": content or ""},
            )
        elif step.operation == "workspace_patch":
            file_path = _extract_path(step.description, _TEXT_EXTENSIONS) or _extract_path(
                plan.goal,
                _TEXT_EXTENSIONS,
            )
            if not file_path:
                raise RuntimeError("Не указан путь к файлу workspace для патча.")
            patch_text = _extract_labeled_value(step.description, "patch")
            if not patch_text:
                patch_text = _extract_labeled_value(plan.goal, "patch")
            if not patch_text:
                raise RuntimeError("Не указан patch для workspace_patch.")
            request = ToolRequest(
                name="workspace_patch",
                args={"path": file_path, "patch": patch_text},
            )
        elif step.operation == "workspace_run":
            script_path = _extract_path(step.description, {".py"}) or _extract_path(
                plan.goal,
                {".py"},
            )
            if not script_path:
                raise RuntimeError("Не указан путь к .py для запуска.")
            request = ToolRequest(name="workspace_run", args={"path": script_path})
        else:
            return f"Выполнен: {step.description}"

        result = gateway.call(request)
        if isinstance(result, ToolResult) and result.ok:
            return str(result.data.get("output") or result.data)
        error = result.error if isinstance(result, ToolResult) else "Ошибка инструмента"
        raise RuntimeError(f"Ошибка: {error}")
