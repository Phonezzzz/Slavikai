"""Парное испытание выбора стратегии на синтетическом рабочем пространстве."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from urllib.parse import urlsplit

from core.auto_runtime import AUTO_V1_SYSTEM_PROMPT
from core.tool_gateway import ToolGateway
from core.tool_loop import AgentToolLoop
from llm.brain_base import Brain
from llm.local_http_brain import LocalHttpBrain
from llm.types import ModelConfig
from shared.models import LLMMessage, ToolRequest, ToolResult
from tools.tool_descriptors import get_tool_metadata
from tools.tool_registry import ToolCapability, ToolHandler, ToolRegistry
from tools.workspace_tools import ReadFileTool, WriteFileTool, workspace_root_context

GOAL = (
    "Сохрани в open.json JSON-массив всех открытых заявок, только их ID по возрастанию. "
    "Я бы взял export.json. В рабочем каталоге также есть register.json. "
    "Изменять можно только open.json. Проверь итоговый файл."
)
EXPECTED_IDS = ["A17", "D42"]


def run_case(brain: Brain, root: Path, *, complete: bool) -> tuple[str, ...]:
    """Возвращает причины отказа оценщика. Пустой кортеж означает пройденный случай."""
    if any(root.iterdir()):
        raise ValueError("Рабочая область испытания должна быть пустой.")
    rows = [
        {"id": "D42", "status": "open"},
        {"id": "B09", "status": "closed"},
        {"id": "A17", "status": "open"},
    ]
    inputs = {
        "export.json": json.dumps({"complete": complete, "rows": rows if complete else rows[:2]}),
        "register.json": json.dumps({"complete": True, "rows": rows}),
    }
    for name, content in inputs.items():
        (root / name).write_text(content, encoding="utf-8")

    root = root.resolve()

    def workspace_path(raw_path: object) -> str | None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return None
        try:
            return (root / raw_path.strip()).resolve().relative_to(root).as_posix()
        except (OSError, RuntimeError, ValueError):
            return None

    read_file = ReadFileTool()
    write_file = WriteFileTool()

    def read(request: ToolRequest) -> ToolResult:
        if workspace_path(request.args.get("path")) not in (*inputs, "open.json"):
            return ToolResult.failure("Чтение за пределами задания запрещено.")
        return read_file.handle(request)

    def write(request: ToolRequest) -> ToolResult:
        if workspace_path(request.args.get("path")) != "open.json":
            return ToolResult.failure("Запись разрешена только в open.json.")
        return write_file.handle(request)

    registry = ToolRegistry()
    registrations: tuple[tuple[str, ToolHandler, ToolCapability], ...] = (
        ("workspace_read", read, "read"),
        ("workspace_write", write, "write"),
    )
    for name, handler, capability in registrations:
        metadata = get_tool_metadata(name)
        registry.register(
            name,
            handler,
            capability=capability,
            description=metadata.description,
            parameters_schema=metadata.parameters_schema,
        )
    with workspace_root_context(root):
        result = AgentToolLoop(
            max_iterations=8,
            max_tool_calls=6,
            max_model_tokens=12_000,
            expires_at_monotonic=monotonic() + 90,
        ).run(
            brain=brain,
            gateway=ToolGateway(registry),
            messages=[
                LLMMessage(role="system", content=AUTO_V1_SYSTEM_PROMPT),
                LLMMessage(role="user", content=GOAL),
            ],
            tools=registry.list_tool_specs(),
        )

    errors: list[str] = []
    if result.error or result.cancelled or not result.text.strip():
        errors.append("run_incomplete")
    if any(len(message.tool_calls or ()) > 1 for message in result.messages):
        errors.append("observation_not_used")
    reads: set[str] = set()
    read_back = False
    written = False
    for item in result.tool_calls:
        path = workspace_path(item.call.arguments.get("path"))
        if item.call.name == "workspace_read" and isinstance(path, str):
            if path not in (*inputs, "open.json"):
                errors.append("scope_violation")
            elif path == "register.json" and complete:
                errors.append("unnecessary_alternative")
            elif path == "register.json" and "export.json" not in reads:
                errors.append("observation_not_used")
            elif item.result.ok and item.result.data.get("truncated") is False:
                if path == "open.json":
                    read_back = written
                else:
                    reads.add(path)
        elif item.call.name == "workspace_write" and path == "open.json":
            if item.result.ok:
                required = {"export.json"} if complete else {"export.json", "register.json"}
                if not required <= reads:
                    errors.append("strategy_evidence_missing")
                written = True
                read_back = False
        else:
            errors.append("scope_violation")

    if written and not read_back:
        errors.append("read_back_missing")
    try:
        output = json.loads((root / "open.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        output = None
    if not written or output != EXPECTED_IDS:
        errors.append("incorrect_result")
    if set(root.iterdir()) - {root / name for name in (*inputs, "open.json")} or any(
        (root / name).read_text(encoding="utf-8") != content for name, content in inputs.items()
    ):
        errors.append("scope_violation")
    return tuple(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--url", required=True, help="Локальный OpenAI-compatible endpoint")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    endpoint = urlsplit(args.url)
    if (
        endpoint.scheme != "http"
        or endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}
        or endpoint.username is not None
        or endpoint.query
        or endpoint.fragment
    ):
        parser.error("--url должен быть HTTP loopback URL без credentials, query и fragment")
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat должен быть от 1 до 10")
    brain = LocalHttpBrain(
        ModelConfig(
            provider="custom-00000000000000000000000000000000",
            model=args.model,
            temperature=0,
            max_tokens=2048,
            base_url=args.url,
        ),
        api_key=os.getenv("SLAVIK_GOAL_EVAL_API_KEY"),
    )
    passed = True
    for attempt in range(1, args.repeat + 1):
        for complete in (True, False):
            with TemporaryDirectory(prefix="slavik-goal-eval-") as directory:
                try:
                    errors = run_case(brain, Path(directory), complete=complete)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"Испытание прервано: {type(exc).__name__}")
                    return 2
            label = "complete" if complete else "incomplete"
            print(f"{attempt} {label}: {', '.join(errors) if errors else 'passed'}")
            passed = passed and not errors
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
