from __future__ import annotations

import subprocess
from pathlib import Path

from shared.models import JSONValue


def _run_git(root: Path, *args: str) -> tuple[str, str, int]:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout, result.stderr, result.returncode


def _parse_branch_ab(value: str) -> tuple[int, int]:
    ahead = 0
    behind = 0
    for part in value.split():
        part = part.strip()
        if part.startswith("+"):
            try:
                ahead = int(part[1:])
            except ValueError:
                ahead = 0
        elif part.startswith("-"):
            try:
                behind = int(part[1:])
            except ValueError:
                behind = 0
    return ahead, behind


def _git_upstream(root: Path, branch: str | None) -> str | None:
    if not branch:
        return None
    stdout, _, rc = _run_git(
        root, "for-each-ref", "--format=%(upstream:short)", f"refs/heads/{branch}"
    )
    if rc != 0 or not stdout.strip():
        return None
    return stdout.strip()


def git_status(root: Path) -> dict[str, JSONValue]:
    stdout, stderr, rc = _run_git(root, "status", "--porcelain=v2", "-z", "--branch")
    if rc != 0:
        return {
            "ok": False,
            "error": stderr.strip() or "git status failed",
            "branch": None,
            "upstream": None,
            "ahead": 0,
            "behind": 0,
            "staged": [],
            "unstaged": [],
            "untracked": [],
            "conflicted": [],
        }

    branch: str | None = None
    upstream: str | None = None
    ahead = 0
    behind = 0
    staged: list[dict[str, str]] = []
    unstaged: list[dict[str, str]] = []
    untracked: list[dict[str, str]] = []
    conflicted: list[dict[str, str]] = []

    tokens = stdout.split("\0")
    for token in tokens:
        if not token:
            continue

        if token.startswith("# branch.head "):
            branch = token.removeprefix("# branch.head ").strip()
            continue
        if token.startswith("# branch.upstream "):
            upstream = token.removeprefix("# branch.upstream ").strip()
            continue
        if token.startswith("# branch.ab "):
            ahead, behind = _parse_branch_ab(token.removeprefix("# branch.ab "))
            continue
        if token.startswith("# "):
            continue

        if token.startswith("? "):
            untracked.append({"path": token[2:], "status": "??"})
            continue

        if token.startswith("1 "):
            fields = token.split(" ", 8)
            if len(fields) < 9:
                continue
            xy = fields[1]
            path = fields[8]
            x = xy[0]
            y = xy[1]
            if x == "U" or y == "U" or (x == "D" and y == "D") or (x == "A" and y == "A"):
                conflicted.append({"path": path, "status": xy})
                continue
            entry = {"path": path, "status": xy}
            if x != ".":
                staged.append(entry)
            if y != ".":
                unstaged.append(entry)
            continue

        if token.startswith("2 "):
            fields = token.split(" ", 9)
            if len(fields) < 10:
                continue
            xy = fields[1]
            path = fields[9]
            x = xy[0]
            y = xy[1]
            if x == "U" or y == "U":
                conflicted.append({"path": path, "status": xy})
                continue
            entry = {"path": path, "status": xy}
            if x != ".":
                staged.append(entry)
            if y != ".":
                unstaged.append(entry)
            continue

        if token.startswith("u "):
            fields = token.split(" ", 10)
            if len(fields) < 11:
                continue
            conflicted.append({"path": fields[10], "status": fields[1]})
            continue

    if upstream is None:
        upstream = _git_upstream(root, branch)

    return {
        "ok": True,
        "error": None,
        "branch": branch,
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "staged": staged,
        "unstaged": unstaged,
        "untracked": untracked,
        "conflicted": conflicted,
    }


def git_stage(
    root: Path, paths: list[str] | None = None, *, all_files: bool = False
) -> tuple[bool, str]:
    args = ["add"]
    if all_files:
        args.append("--all")
    elif paths:
        args.extend(["--", *paths])
    else:
        return False, "paths or all_files required"
    _, stderr, rc = _run_git(root, *args)
    if rc != 0:
        return False, stderr.strip() or "git add failed"
    return True, "ok"


def git_unstage(
    root: Path, paths: list[str] | None = None, *, all_files: bool = False
) -> tuple[bool, str]:
    if all_files:
        _, stderr, rc = _run_git(root, "reset", "HEAD", "--", ".")
    elif paths:
        _, stderr, rc = _run_git(root, "reset", "HEAD", "--", *paths)
    else:
        return False, "paths or all_files required"
    if rc != 0:
        return False, stderr.strip() or "git reset failed"
    return True, "ok"


def git_commit(root: Path, message: str) -> tuple[bool, str]:
    _, stderr, rc = _run_git(root, "commit", "-m", message)
    if rc != 0:
        return False, stderr.strip() or "git commit failed"
    return True, "ok"


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
