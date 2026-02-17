from __future__ import annotations

import ast
import re
from pathlib import Path


def _iter_python_files() -> list[Path]:
    roots = [Path("server/http_api.py"), Path("server/http")]
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
            continue
        for path in root.rglob("*.py"):
            files.append(path)
    return files


def _collect_private_functions(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("_"):
            continue
        if node.name.startswith("__"):
            continue
        names.append(node.name)
    return names


def test_private_helpers_have_references_in_repo() -> None:
    source_roots = [Path("server"), Path("tests"), Path("ui/src")]
    all_sources: list[str] = []
    for root in source_roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".py", ".ts", ".tsx"}:
                continue
            all_sources.append(path.read_text(encoding="utf-8", errors="ignore"))
    repo_blob = "\n".join(all_sources)

    unused: list[str] = []
    for module_path in _iter_python_files():
        for func_name in _collect_private_functions(module_path):
            hits = len(re.findall(rf"\b{re.escape(func_name)}\b", repo_blob))
            if hits <= 1:
                unused.append(f"{module_path}:{func_name}")

    assert unused == [], f"Unused private helpers detected: {unused}"
