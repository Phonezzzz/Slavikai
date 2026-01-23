#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: not inside a git repository" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
timestamp="$(date +%Y%m%d-%H%M)"
output_dir="${repo_root}/docs/status"
output_file="${output_dir}/status-${timestamp}.md"

mkdir -p "$output_dir"

{
    echo "## repo root"
    git -C "$repo_root" rev-parse --show-toplevel
    echo
    echo "## status"
    git -C "$repo_root" status -sb
    echo
    echo "## branches"
    git -C "$repo_root" branch -vv
    echo
    echo "## recent commits"
    git -C "$repo_root" log --oneline --decorate -n 40
    echo
    echo "## diff stat (working tree)"
    git -C "$repo_root" diff --stat
    echo
    echo "## diff (working tree)"
    git -C "$repo_root" diff
    echo
    echo "## diff stat (staged)"
    git -C "$repo_root" diff --cached --stat
    echo
    echo "## diff (staged)"
    git -C "$repo_root" diff --cached
    echo
    echo "## modified/untracked files"
    git -C "$repo_root" ls-files -m -o --exclude-standard
    echo
    echo "## quick grep refactor markers"
    (cd "$repo_root" && rg -n "_mwv_worker_runner|command_handler|/command|CommandHandler|core/agent.py" core -S || true)
} | tee "$output_file"
