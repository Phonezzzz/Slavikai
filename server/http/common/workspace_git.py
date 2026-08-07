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
    stdout, stderr, rc = _run_git(root, "status", "--porcelain", "--branch")
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
        if not line:
            continue

        if line.startswith("## "):
            branch_info = line.removeprefix("## ").strip()
            if "..." in branch_info:
                head_part, rest = branch_info.split("...", 1)
                branch = head_part.strip()
                upstream_part = rest.strip()
                if "[" in upstream_part and "]" in upstream_part:
                    upstream = upstream_part.split("[")[0].strip()
                    bracket = upstream_part[
                        upstream_part.index("[") + 1 : upstream_part.rindex("]")
                    ]
                    for token in bracket.split(","):
                        token = token.strip()
                        if token.startswith("ahead "):
                            ahead = int(token.removeprefix("ahead ").strip())
                        elif token.startswith("behind "):
                            behind = int(token.removeprefix("behind ").strip())
                else:
                    upstream = upstream_part.strip()
            else:
                branch = branch_info
            continue

        # porcelain v1: XY PATH for ordinary, R  OLD -> NEW for renames
        raw = line.rstrip()
        if len(raw) < 2:
            continue

        x = raw[0]
        y = raw[1]

        entry: dict[str, str]

        if raw.startswith("??"):
            path = raw[3:]
            untracked.append({"path": path.lstrip(), "status": "??"})
            continue

        if raw.startswith("!!"):
            continue

        # rename: "R  old -> new" or "R  "old -> new""
        if x == "R":
            rest = raw[3:]
            if " -> " in rest:
                parts = rest.split(" -> ", 1)
                new_path = parts[1].strip()
                # remove surrounding quotes if present
                if new_path.startswith('"') and new_path.endswith('"'):
                    new_path = new_path[1:-1]
                if y != " ":
                    unstaged.append({"path": new_path, "status": raw[:2]})
                else:
                    staged.append({"path": new_path, "status": raw[:2]})
            continue

        # copy: "C  new"
        if x == "C":
            path = raw[3:]
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]
            staged.append({"path": path, "status": raw[:2]})
            continue

        # ordinary entry
        path = raw[3:]
        # remove surrounding quotes if present
        if path.startswith('"') and path.endswith('"'):
            path = path[1:-1]
        entry = {"path": path, "status": raw[:2]}

        x_staged = x in {"M", "A", "D"}
        y_unstaged = y in {"M", "D"}
        has_conflict = x in {"U", "D"} and y in {"U", "A", "D"} or "U" in {x, y}

        if has_conflict:
            conflicted.append(entry)
            continue

        if x_staged:
            staged.append(entry)
        if y_unstaged:
            unstaged.append(entry)

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
