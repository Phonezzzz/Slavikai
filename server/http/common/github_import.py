from __future__ import annotations

import asyncio
import logging
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from core.approval_policy import ApprovalCategory, ApprovalPrompt, ApprovalRequest
from shared.models import ToolRequest
from tools.project_tool import handle_project_request

GITHUB_ROOT: Final[Path] = Path("sandbox/project/github").resolve()
SANDBOX_PROJECT_ROOT: Final[Path] = Path("sandbox/project").resolve()

logger = logging.getLogger("SlavikAI.HttpAPI")


def parse_github_import_args(args_raw: str) -> tuple[str, str | None]:
    try:
        parts = shlex.split(args_raw)
    except ValueError as exc:
        raise ValueError(f"Некорректные аргументы github_import: {exc}") from exc
    if not parts:
        raise ValueError("Укажи ссылку на GitHub репозиторий.")
    repo_url = parts[0].strip()
    if not repo_url:
        raise ValueError("Укажи ссылку на GitHub репозиторий.")
    branch = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
    return repo_url, branch


def _validate_github_segment(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("Некорректный сегмент пути GitHub.")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(char not in allowed for char in normalized):
        raise ValueError("Допустимы только owner/repo с символами [a-zA-Z0-9-_.].")
    return normalized


def resolve_github_target(repo_url: str) -> tuple[Path, str]:
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("Поддерживаются только http(s) GitHub URL.")
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Поддерживаются только github.com репозитории.")
    path_parts = [part for part in parsed.path.split("/") if part.strip()]
    if len(path_parts) < 2:
        raise ValueError("URL должен содержать owner/repo.")
    owner = _validate_github_segment(path_parts[0])
    repo_name_raw = path_parts[1].removesuffix(".git")
    repo_name = _validate_github_segment(repo_name_raw)
    GITHUB_ROOT.mkdir(parents=True, exist_ok=True)
    target_root = GITHUB_ROOT / owner
    target_root.mkdir(parents=True, exist_ok=True)
    candidate = target_root / repo_name
    suffix = 1
    while candidate.exists():
        candidate = target_root / f"{repo_name}-{suffix}"
        suffix += 1
    relative_target = candidate.resolve().relative_to(SANDBOX_PROJECT_ROOT)
    return candidate, str(relative_target)


def build_github_import_approval_request(
    *,
    session_id: str,
    repo_url: str,
    branch: str | None,
    required_categories: list[ApprovalCategory],
) -> ApprovalRequest:
    branch_text = f", branch={branch}" if branch else ""
    return ApprovalRequest(
        category=required_categories[0],
        required_categories=required_categories,
        prompt=ApprovalPrompt(
            what=f"GitHub import: {repo_url}{branch_text}",
            why="Нужно скачать внешний репозиторий и проиндексировать код.",
            risk="Сетевой доступ и выполнение git clone.",
            changes=[
                "Создание каталога в sandbox/project/github/*",
                "Скачивание внешнего репозитория",
                "Индексация файлов в memory/vectors.db",
            ],
        ),
        tool="project",
        details={
            "command": "github_import",
            "repo_url": repo_url,
            "branch": branch,
        },
        session_id=session_id,
    )


async def clone_github_repository(
    *,
    repo_url: str,
    branch: str | None,
    target_path: Path,
) -> tuple[bool, str]:
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    cmd.extend([repo_url, str(target_path)])
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"Git clone failed: {exc}"
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "unknown error"
        try:
            if target_path.exists():
                shutil.rmtree(target_path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to cleanup clone target after error", exc_info=True)
        return False, f"Git clone failed: {details}"
    return True, "ok"


def index_imported_project(relative_path: str) -> tuple[bool, str]:
    try:
        tool_result = handle_project_request(
            ToolRequest(
                name="project",
                args={"cmd": "index", "args": [relative_path]},
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"project index exception: {exc}"
    if not tool_result.ok:
        message = tool_result.error or "project index failed"
        return False, message
    data = tool_result.data if isinstance(tool_result.data, dict) else {}
    indexed_code = data.get("indexed_code")
    indexed_docs = data.get("indexed_docs")
    return True, f"Code={indexed_code}, Docs={indexed_docs}"
