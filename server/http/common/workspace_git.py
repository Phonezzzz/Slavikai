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


def git_status(root: Path) -> dict[str, JSONValue]:
    stdout, stderr, rc = _run_git(root, "status", "--porcelain=v2", "--branch")
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

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# branch.oid "):
            continue
        if line.startswith("# branch.head "):
            branch = line.removeprefix("# branch.head ").strip()
        elif line.startswith("# branch.upstream "):
            upstream = line.removeprefix("# branch.upstream ").strip()
        elif line.startswith("# branch.ab "):
            parts = line.removeprefix("# branch.ab ").split()
            if len(parts) >= 1:
                ahead = int(parts[0].removeprefix("+"))
            if len(parts) >= 2:
                behind = int(parts[1].removeprefix("-"))
        elif line.startswith("1 ") or line.startswith("2 "):
            fields = line.split(" ")
            if len(fields) < 9:
                continue
            xy = fields[1]
            sub = fields[2]
            path = fields[8] if len(fields) > 8 else fields[-1]
            entry = {"path": path, "status": xy}
            if xy == "??":
                untracked.append(entry)
            elif "u" in xy.lower() or "U" in xy:
                conflicted.append(entry)
            elif sub in {"0", "1", "2", "3"}:
                staged.append(entry)
            elif sub in {"4", "5", "6", "7"}:
                unstaged.append(entry)
        elif line.startswith("? "):
            path = line.removeprefix("? ").strip()
            untracked.append({"path": path, "status": "??"})

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
        args.extend(paths)
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


def git_pull(root: Path) -> tuple[bool, str]:
    _, stderr, rc = _run_git(root, "pull", "--ff-only")
    if rc != 0:
        return False, stderr.strip() or "git pull failed"
    return True, "ok"


def git_push(root: Path) -> tuple[bool, str]:
    _, stderr, rc = _run_git(root, "push")
    if rc != 0:
        return False, stderr.strip() or "git push failed"
    return True, "ok"


def git_switch(root: Path, branch: str, *, create: bool = False) -> tuple[bool, str]:
    cmd = ["checkout", "-b" if create else "", branch]
    args = [a for a in cmd if a]
    _, stderr, rc = _run_git(root, *args)
    if rc != 0:
        return False, stderr.strip() or "git switch failed"
    return True, "ok"
