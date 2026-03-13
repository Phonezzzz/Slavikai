#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

forbidden_dirs=(
  "ui/src/legacy"
  "ui_archive/legacy-ui"
)

forbidden_files=(
  "ui/src/app/components/ChatArea.tsx"
  "ui/src/app/components/ChatCanvas.tsx"
  "ui/src/app/components/Sidebar.tsx"
)

fail=0

print_tracked_existing() {
  local found=1
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ -e "$path" ]]; then
      echo "$path"
      found=0
    fi
  done < <(git ls-files -- "$@")
  return "$found"
}

for dir in "${forbidden_dirs[@]}"; do
  if [[ -d "$dir" ]]; then
    echo "Forbidden legacy directory exists: $dir"
    fail=1
  fi
done

if tracked="$(print_tracked_existing "${forbidden_dirs[@]}")"; then
  echo "Forbidden tracked legacy files found:"
  echo "$tracked"
  fail=1
fi

if tracked_files="$(print_tracked_existing "${forbidden_files[@]}")"; then
  echo "Forbidden tracked legacy UI files found:"
  echo "$tracked_files"
  fail=1
fi

untracked="$(git ls-files --others --exclude-standard -- "${forbidden_dirs[@]}")"
if [[ -n "$untracked" ]]; then
  echo "Forbidden untracked legacy files found:"
  echo "$untracked"
  fail=1
fi

untracked_files="$(git ls-files --others --exclude-standard -- "${forbidden_files[@]}")"
if [[ -n "$untracked_files" ]]; then
  echo "Forbidden untracked legacy UI files found:"
  echo "$untracked_files"
  fail=1
fi

for file in "${forbidden_files[@]}"; do
  if [[ -e "$file" ]]; then
    echo "Forbidden legacy UI file exists: $file"
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "Remove legacy UI paths before continuing."
  exit 1
fi

echo "OK: no forbidden legacy UI paths found."
