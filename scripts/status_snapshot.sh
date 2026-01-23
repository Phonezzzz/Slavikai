#!/usr/bin/env bash
set -euo pipefail

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: not inside a git repository" >&2
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="${repo_root}/docs/status"
output_file="${output_dir}/status-${timestamp}.md"

mkdir -p "$output_dir"

if [ -e "$output_file" ]; then
    echo "error: output file exists: $output_file" >&2
    exit 1
fi

{
    echo "## repo root"
    git -C "$repo_root" --no-pager rev-parse --show-toplevel
    echo
    echo "## status"
    git -C "$repo_root" --no-pager status -sb --no-color
    echo
    echo "## branches"
    git -C "$repo_root" --no-pager branch -vv --no-color
    echo
    echo "## recent commits"
    git -C "$repo_root" --no-pager log --oneline --decorate -n 40 --no-color
    echo
    echo "## diff stat (working tree)"
    git -C "$repo_root" --no-pager diff --stat --no-color
    echo
    echo "## diff (working tree)"
    git -C "$repo_root" --no-pager diff --no-color
    echo
    echo "## diff stat (staged)"
    git -C "$repo_root" --no-pager diff --cached --stat --no-color
    echo
    echo "## diff (staged)"
    git -C "$repo_root" --no-pager diff --cached --no-color
    echo
    echo "## modified/untracked files"
    git -C "$repo_root" --no-pager ls-files -m -o --exclude-standard --exclude="docs/status/*"
    echo
    echo "## quick grep refactor markers"
    rg -n "_mwv_worker_runner|command_handler|/command|CommandHandler" "${repo_root}/core" -S || true
} | tee "$output_file"
