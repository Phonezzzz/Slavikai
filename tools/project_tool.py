from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from memory.vector_index import VectorIndex
from shared.models import JSONValue, ToolRequest, ToolResult
from shared.sandbox import WORKSPACE_ROOT, SandboxViolationError, normalize_workspace_path

ALLOWED_EXTENSIONS: Final[tuple[str, ...]] = (".py", ".md", ".txt")
IGNORED_DIRS: Final[set[str]] = {".git", "__pycache__", "venv", ".venv"}
MAX_FILE_BYTES: Final[int] = 1_000_000  # 1 MB
MAX_DEPTH: Final[int] = 5
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)


def handle_project_request(request: ToolRequest) -> ToolResult:
    cmd = str(request.args.get("cmd") or "").strip()
    args_raw = request.args.get("args") or []
    args = [str(a) for a in args_raw] if isinstance(args_raw, list) else [str(args_raw)]
    index = VectorIndex("memory/vectors.db")
    if cmd == "index":
        path_str = args[0] if args else "."
        try:
            base = normalize_workspace_path(path_str)
        except SandboxViolationError as exc:
            return ToolResult.failure(
                "Путь вне sandbox/project запрещён",
                meta={"raw": exc.raw_path, "normalized": str(exc.normalized_path)},
            )
        if not base.exists() or not base.is_dir():
            return ToolResult.failure(f"Каталог не найден в sandbox/project: {path_str}")

        indexed_code = 0
        indexed_docs = 0
        skipped: list[str] = []

        for root, dirs, files in os.walk(base):
            try:
                rel_depth = len(Path(root).resolve().relative_to(WORKSPACE_ROOT).parts)
            except Exception:
                skipped.append(f"{root}: вне sandbox/project")
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
                f"🔍 {_safe_str(m['path'])} [{_safe_str(m['score'])}]\n→ {_safe_str(m['snippet'])}"
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
