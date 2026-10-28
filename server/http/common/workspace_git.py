from __future__ import annotations

from pathlib import Path

from shared.models import JSONValue
from shared.sandbox_execution import SANDBOX_EXECUTION_MESSAGE


def git_status(root: Path) -> dict[str, JSONValue]:
    return {
        "ok": False,
        "error": SANDBOX_EXECUTION_MESSAGE,
        "branch": None,
        "upstream": None,
        "ahead": 0,
        "behind": 0,
        "staged": [],
        "unstaged": [],
        "untracked": [],
        "conflicted": [],
    }


def git_stage(
    root: Path, paths: list[str] | None = None, *, all_files: bool = False
) -> tuple[bool, str]:
    if not all_files and not paths:
        return False, "paths or all_files required"
    return False, SANDBOX_EXECUTION_MESSAGE


def git_unstage(
    root: Path, paths: list[str] | None = None, *, all_files: bool = False
) -> tuple[bool, str]:
    if not all_files and not paths:
        return False, "paths or all_files required"
    return False, SANDBOX_EXECUTION_MESSAGE


def git_commit(root: Path, message: str) -> tuple[bool, str]:
    return False, SANDBOX_EXECUTION_MESSAGE


def parse_git_operation_paths(
    paths_raw: object,
    all_raw: object,
) -> tuple[list[str] | None, bool]:
    """Strict contract validation for stage/unstage payloads.

    Rules:
    - `all`, если передан, обязан быть boolean;
    - `paths`, если передан, обязан быть непустым list[str], каждый
      элемент — непустая строка (проверка через strip, но возвращается
      ОРИГИНАЛЬНЫЙ filename без модификации);
    - `all == True` требует отсутствие paths (paths must be None);
    - `all == False` (или отсутствует) требует валидные paths;
    - если нет ни all==True, ни paths — ValueError.

    Returns (paths, all_flag). Raises ValueError on malformed input.
    """
    all_flag: bool = False
    if all_raw is not None:
        if not isinstance(all_raw, bool):
            raise ValueError("all должен быть boolean.")
        all_flag = all_raw

    if all_flag:
        if paths_raw is not None:
            raise ValueError("Нельзя передавать одновременно all=true и paths.")
        return None, True

    if paths_raw is None:
        raise ValueError("Необходимо передать paths (или all=true).")
    if not isinstance(paths_raw, list) or len(paths_raw) == 0:
        raise ValueError("paths должен быть непустым списком строк.")
    normalized: list[str] = []
    for item in paths_raw:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("paths должен быть списком непустых строк.")
        normalized.append(item)
    return normalized, False
