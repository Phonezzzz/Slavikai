from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from config.ui_embeddings_settings import load_ui_embeddings_settings, resolve_openai_api_key
from memory.vector_index import VectorIndex
from shared.models import JSONValue, ToolRequest, ToolResult
from shared.sandbox import SandboxViolationError, normalize_sandbox_path

ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (".py", ".md", ".txt")
IGNORED_DIRS: Final[set[str]] = {".git", "__pycache__", "venv", ".venv"}
MAX_FILE_BYTES: Final[int] = 1_000_000  # 1 MB
MAX_DEPTH: Final[int] = 5
SANDBOX_ROOT: Final[Path] = Path("sandbox/project").resolve()

SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)


def _normalize_path(raw: str) -> Path:
    try:
        return normalize_sandbox_path(raw or ".", SANDBOX_ROOT)
    except SandboxViolationError as exc:
        raise ValueError("Путь вне sandbox/project запрещён") from exc


def _resolve_in_sandbox(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SANDBOX_ROOT)
    except ValueError as exc:
        raise ValueError("Путь вне sandbox/project запрещён") from exc
    return resolved


def handle_project_request(request: ToolRequest) -> ToolResult:
    cmd = str(request.args.get("cmd") or "").strip()
    args_raw = request.args.get("args") or []
    args = [str(a) for a in args_raw] if isinstance(args_raw, list) else [str(args_raw)]
    embeddings = load_ui_embeddings_settings()
    with VectorIndex(
        "memory/vectors.db",
        provider=embeddings.provider,
        local_model=embeddings.local_model,
        openai_model=embeddings.openai_model,
        openai_api_key=resolve_openai_api_key(),
    ) as index:
        if cmd == "index":
            try:
                index.ensure_runtime_ready()
            except RuntimeError as exc:
                return ToolResult.failure(str(exc))
            path_str = args[0] if args else "."
            try:
                base = _normalize_path(path_str)
            except ValueError as exc:
                return ToolResult.failure(str(exc))
            if not base.exists() or not base.is_dir():
                return ToolResult.failure(f"Каталог не найден в sandbox/project: {path_str}")

            indexed_code = 0
            indexed_docs = 0
            skipped: list[str] = []

            for root, dirs, files in os.walk(base):
                try:
                    current_root = _resolve_in_sandbox(Path(root))
                    rel_depth = len(current_root.relative_to(base).parts)
                except Exception:
                    skipped.append(f"{root}: путь вне sandbox/project")
                    continue
                if rel_depth > MAX_DEPTH:
                    skipped.append(f"{root}: превышена глубина {MAX_DEPTH}")
                    dirs[:] = []
                    continue
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
                for filename in files:
                    if not filename.endswith(ALLOWED_EXTENSIONS):
                        continue
                    full_path = Path(root, filename)
                    try:
                        full_path = _resolve_in_sandbox(full_path)
                        if full_path.stat().st_size > MAX_FILE_BYTES:
                            skipped.append(f"{full_path}: файл больше {MAX_FILE_BYTES} байт")
                            continue
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                        namespace = "code" if full_path.suffix == ".py" else "docs"
                        index.index_text(str(full_path), content, namespace=namespace)
                        if namespace == "code":
                            indexed_code += 1
                        else:
                            indexed_docs += 1
                    except Exception as exc:  # noqa: BLE001
                        skipped.append(f"{full_path}: {exc}")

            return ToolResult.success(
                {
                    "output": f"📁 Code: {indexed_code}, Docs: {indexed_docs}",
                    "indexed_code": indexed_code,
                    "indexed_docs": indexed_docs,
                    "skipped": skipped,
                },
                meta={
                    "indexed_code": indexed_code,
                    "indexed_docs": indexed_docs,
                    "skipped": len(skipped),
                    "base": str(base),
                },
            )

        if cmd == "find":
            query = " ".join(args).strip()
            if not query:
                return ToolResult.failure("Пустой поисковый запрос.")
            try:
                index.ensure_runtime_ready()
            except RuntimeError as exc:
                return ToolResult.failure(str(exc))
            results_code = index.search(query, namespace="code")
            results_docs = index.search(query, namespace="docs")
            results = results_code + results_docs
            matches: list[dict[str, JSONValue]] = [
                {"path": item.path, "snippet": item.snippet, "score": round(item.score, 3)}
                for item in results
            ]

            def _safe_str(value: JSONValue) -> str:
                if isinstance(value, (bytes, bytearray)):
                    return value.decode("utf-8", errors="replace")
                return str(value)

            output = "\n".join(
                [
                    "🔍 {path} [{score}]\n→ {snippet}".format(
                        path=_safe_str(m["path"]),
                        score=_safe_str(m["score"]),
                        snippet=_safe_str(m["snippet"]),
                    )
                    for m in matches
                ]
            )
            return ToolResult.success(
                {"output": output, "matches": matches},
                meta={
                    "matches": len(matches),
                    "code_hits": len(results_code),
                    "doc_hits": len(results_docs),
                },
            )

    return ToolResult.failure("Неизвестная команда проекта")


class ProjectTool:
    """Класс-обёртка для проектного инструмента (Tool.handle совместимый)."""

    def handle(self, request: ToolRequest) -> ToolResult:  # noqa: D401 - очевидно
        return handle_project_request(request)
