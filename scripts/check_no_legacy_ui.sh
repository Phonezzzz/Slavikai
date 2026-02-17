#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

forbidden_dirs=(
  "ui/src/legacy"
  "ui_archive/legacy-ui"
)

fail=0

for dir in "${forbidden_dirs[@]}"; do
  if [[ -d "$dir" ]]; then
    echo "Forbidden legacy directory exists: $dir"
    fail=1
  fi
done

tracked="$(git ls-files -- "${forbidden_dirs[@]}")"
if [[ -n "$tracked" ]]; then
  echo "Forbidden tracked legacy files found:"
  echo "$tracked"
  fail=1
fi

untracked="$(git ls-files --others --exclude-standard -- "${forbidden_dirs[@]}")"
if [[ -n "$untracked" ]]; then
  echo "Forbidden untracked legacy files found:"
  echo "$untracked"
  fail=1
fi

if [[ "$fail" -ne 0 ]]; then
  echo "Remove legacy UI paths before continuing."
  exit 1
fi

echo "OK: no forbidden legacy UI paths found."
