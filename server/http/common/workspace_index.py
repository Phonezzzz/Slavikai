from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

from config.ui_embeddings_settings import UIEmbeddingsSettings
from memory.vector_index import VectorIndex
from shared.models import JSONValue

DEFAULT_IGNORED_DIRS: Final[set[str]] = {
    "venv",
    "__pycache__",
    ".git",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".cache",
}
DEFAULT_ALLOWED_EXTENSIONS: Final[set[str]] = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".html",
    ".css",
    ".sql",
    ".sh",
}
DEFAULT_MAX_FILE_BYTES: Final[int] = 1_000_000


def env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def is_index_enabled(*, index_enabled_env: str) -> bool:
    return env_flag_enabled(index_enabled_env, default=True)


def index_workspace_root(
    *,
    root: Path,
    load_embeddings_settings: Callable[[], UIEmbeddingsSettings],
    resolve_provider_api_key: Callable[[str], str | None],
    index_enabled_env: str,
    ignored_dirs: set[str],
    allowed_extensions: set[str],
    max_file_bytes: int,
) -> dict[str, JSONValue]:
    if not is_index_enabled(index_enabled_env=index_enabled_env):
        return {
            "ok": False,
            "message": "INDEX disabled",
            "root_path": str(root),
            "indexed_code": 0,
            "indexed_docs": 0,
            "skipped": 0,
        }

    embeddings = load_embeddings_settings()
    vector_index = VectorIndex(
        "memory/vectors.db",
        provider=embeddings.provider,
        local_model=embeddings.local_model,
        openai_model=embeddings.openai_model,
        openai_api_key=resolve_provider_api_key("openai"),
    )
    vector_index.ensure_runtime_ready()
    indexed_code = 0
    indexed_docs = 0
    skipped = 0
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [name for name in dirs if name not in ignored_dirs]
        current = Path(current_root)
        for filename in files:
            full_path = current / filename
            if filename.endswith(".sqlite"):
                skipped += 1
                continue
            suffix = full_path.suffix.lower()
            if suffix not in allowed_extensions:
                skipped += 1
                continue
            try:
                if full_path.stat().st_size > max_file_bytes:
                    skipped += 1
                    continue
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            namespace = "code" if suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".sh"} else "docs"
            vector_index.upsert_text(str(full_path), content, namespace=namespace)
            if namespace == "code":
                indexed_code += 1
            else:
                indexed_docs += 1
    return {
        "ok": True,
        "message": None,
        "root_path": str(root),
        "indexed_code": indexed_code,
        "indexed_docs": indexed_docs,
        "skipped": skipped,
    }


def workspace_git_diff(root: Path) -> tuple[str, str | None]:
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--no-ext-diff", "--", "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "not a git repository" in stderr.lower():
            return "", None
        return "", stderr or "git diff failed"
    return result.stdout, None
