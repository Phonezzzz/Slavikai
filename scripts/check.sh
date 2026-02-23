#!/usr/bin/env bash
set -euo pipefail

# Go to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3 || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found."
    exit 1
fi

echo "================================================="
echo "   🔍 Running full project quality check"
echo "================================================="
echo "Python: $PYTHON_BIN"
echo

run_step() {
    local name="$1"
    shift
    echo "▶ $name..."
    if "$@"; then
        echo "✓ $name: OK"
    else
        echo "✗ $name: FAILED"
        exit 1
    fi
    echo
}

# 1. Ruff lint
run_step "Ruff lint" \
    "$PYTHON_BIN" -m ruff check .

# 2. Ruff formatting
run_step "Ruff format check" \
    "$PYTHON_BIN" -m ruff format --check .

# 3. Skills lint
run_step "Skills lint" \
    "$PYTHON_BIN" skills/tools/lint_skills.py

# 4. Skills manifest check
run_step "Skills manifest check" \
    "$PYTHON_BIN" skills/tools/build_manifest.py --check

# 5. MyPy strict typing
run_step "MyPy strict" \
    "$PYTHON_BIN" -m mypy .

# 6. Pytest with coverage threshold
run_step "Pytest + Coverage >= 80%" \
    "$PYTHON_BIN" -m pytest --cov --cov-fail-under=80

echo "================================================="
echo "   🎉 All checks passed successfully!"
echo "================================================="
