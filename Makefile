SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --warn-undefined-variables

.DEFAULT_GOAL := help

PYTHON ?= python3
VENV_DIR ?= venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PY) -m pip
VENV_RUFF := $(VENV_DIR)/bin/ruff

RUN_DIR ?= .run
APP_PID_FILE := $(RUN_DIR)/slavikai-ui.pid
APP_LOG_FILE := $(RUN_DIR)/slavikai-ui.log
UI_PID_FILE := $(RUN_DIR)/ui-server.pid
UI_LOG_FILE := $(RUN_DIR)/ui-server.log

.PHONY: help
help:
	@echo "SlavikAI Core"
	@echo
	@echo "Setup:"
	@echo "  make venv            Create venv/ and install requirements"
	@echo "  make activate        Print venv activation command"
	@echo "  make shell           Open an interactive shell with venv activated"
	@echo
	@echo "Quality:"
	@echo "  make lint            ruff check ."
	@echo "  make format          ruff format ."
	@echo "  make format-check    ruff format --check ."
	@echo "  make type            mypy . (strict, tests excluded by config)"
	@echo "  make test            pytest (coverage configured in pyproject.toml)"
	@echo "  make check           lint + format-check + type + test"
	@echo "  make ci              skills lint/manifest + pytest -q (temp candidates)"
	@echo
	@echo "Git:"
	@echo "  make guard-main      Fail if current branch is main"
	@echo "  make git-check       Verify PR branch is pushed and make check passes"
	@echo
	@echo "Run:"
	@echo "  make run             Run UI in foreground"
	@echo "  make up              Run UI in background (pid/log in .run/)"
	@echo "  make down            Stop background UI started by make up"
	@echo "  make status          Show background UI status"
	@echo "  make logs            Tail background UI log"
	@echo
	@echo "UI:"
	@echo "  make ui-install       Install UI dependencies"
	@echo "  make ui-build         Build UI dist"
	@echo "  make ui-dev           Run UI dev server"
	@echo "  make ui-dist-clean    Remove UI dist"
	@echo "  make ui-server        Run UI server in foreground"
	@echo "  make ui-run           Build UI + run UI server"
	@echo "  make ui-up            Run UI server in background"
	@echo "  make ui-down          Stop background UI server"
	@echo "  make ui-status        Show UI server status"
	@echo "  make ui-logs          Tail UI server log"
	@echo "  make ui-clean         Remove UI dist and UI pid/log"
	@echo
	@echo "Cleanup:"
	@echo "  make clean           Remove caches and .run/"
	@echo "  make clean-venv      Remove venv/ (destructive)"

$(VENV_PY):
	$(PYTHON) -m venv "$(VENV_DIR)"

$(VENV_DIR)/.installed: requirements.txt $(VENV_PY)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@touch "$(VENV_DIR)/.installed"

.PHONY: venv
venv: $(VENV_DIR)/.installed

.PHONY: activate
activate: $(VENV_PY)
	@echo "Run:"
	@echo "  source $(VENV_DIR)/bin/activate"

.PHONY: shell
shell: $(VENV_PY)
	@bash -i -c "source \"$(VENV_DIR)/bin/activate\" && exec bash -i"

.PHONY: lint
lint: venv
	"$(VENV_RUFF)" check .

.PHONY: format
format: venv
	"$(VENV_RUFF)" format .

.PHONY: format-check
format-check: venv
	"$(VENV_RUFF)" format --check .

.PHONY: type
type: venv
	"$(VENV_PY)" -m mypy .

.PHONY: test
PYTEST_ARGS ?=
test: venv
	"$(VENV_PY)" -m pytest $(PYTEST_ARGS)

.PHONY: check
check: lint format-check type test

.PHONY: guard-main
guard-main:
	@if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "Not a git repository."; \
		exit 1; \
	fi; \
	branch="$$(git rev-parse --abbrev-ref HEAD)"; \
	if [[ "$$branch" == "HEAD" ]]; then \
		echo "Detached HEAD: switch to a branch."; \
		exit 1; \
	fi; \
	if [[ "$$branch" == "main" ]]; then \
		echo "На main нельзя работать, создайте PR-ветку"; \
		exit 1; \
	fi

.PHONY: git-check
git-check:
	@if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "Not a git repository."; \
		exit 1; \
	fi; \
	branch="$$(git rev-parse --abbrev-ref HEAD)"; \
	if [[ "$$branch" == "HEAD" ]]; then \
		echo "Detached HEAD: switch to a PR branch."; \
		exit 1; \
	fi; \
	if [[ "$$branch" == "main" ]]; then \
		echo "git-check must run on a PR branch (not main)."; \
		exit 1; \
	fi; \
	status_line="$$(git status -sb)"; \
	status_line="$${status_line%%$$'\n'*}"; \
	if [[ "$$status_line" != "## "* ]]; then \
		echo "Could not determine git status."; \
		exit 1; \
	fi; \
	if [[ "$$status_line" != *"..."* ]]; then \
		echo "No upstream for $$branch (branch not pushed)."; \
		git branch -vv; \
		exit 1; \
	fi; \
	if [[ "$$status_line" == *"ahead "* ]]; then \
		echo "Unpushed commits in $$branch."; \
		echo "$$status_line"; \
		git branch -vv; \
		exit 1; \
	fi; \
	$(MAKE) check; \
	echo "OK: PR branch is ready to merge."; \
	echo "Next:"; \
	echo "  git checkout main"; \
	echo "  git rebase origin/main"; \
	echo "  git merge --ff-only $$branch"; \
	echo "  git push origin main"

.PHONY: ci
CI_ARTIFACT_DIR ?= .run/ci-artifacts
ci: venv
	@tmp_dir="$$(mktemp -d)"; \
		artifact_dir="$(CI_ARTIFACT_DIR)"; \
		mkdir -p "$$artifact_dir"; \
		export SKILLS_CANDIDATES_DIR="$$tmp_dir/skills/_candidates"; \
		mkdir -p "$$SKILLS_CANDIDATES_DIR"; \
		"$(VENV_PY)" skills/tools/lint_skills.py >"$$artifact_dir/skills_lint.log" 2>&1; \
		"$(VENV_PY)" skills/tools/build_manifest.py --check >"$$artifact_dir/build_manifest.log" 2>&1; \
		"$(VENV_PY)" -m pytest -q >"$$artifact_dir/pytest.txt" 2>&1; \
		rm -rf "$$tmp_dir"

.PHONY: run
run: venv
	"$(VENV_PY)" main.py

.PHONY: ui-install
ui-install:
	cd ui && npm install

.PHONY: ui-build
ui-build:
	cd ui && npm run build

.PHONY: ui-dev
ui-dev:
	cd ui && npm run dev

.PHONY: ui-dist-clean
ui-dist-clean:
	rm -rf ui/dist

.PHONY: ui-server
ui-server: venv
	"$(VENV_PY)" -m server

.PHONY: ui-run
ui-run: ui-build ui-server

.PHONY: ui-up
ui-up: venv
	@mkdir -p "$(RUN_DIR)"
	@if [[ -f "$(UI_PID_FILE)" ]]; then \
		pid="$$(cat "$(UI_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Already running: pid=$$pid (use: make ui-down)"; \
			exit 1; \
		fi; \
	fi
	@nohup "$(VENV_PY)" -m server >"$(UI_LOG_FILE)" 2>&1 & echo $$! >"$(UI_PID_FILE)"
	@echo "Started: pid=$$(cat "$(UI_PID_FILE)")"
	@echo "Logs: $(UI_LOG_FILE)"

.PHONY: ui-down
ui-down:
	@if [[ ! -f "$(UI_PID_FILE)" ]]; then \
		echo "Not running (no pid file: $(UI_PID_FILE))"; \
		exit 0; \
	fi; \
	pid="$$(cat "$(UI_PID_FILE)")"; \
	if ! kill -0 "$$pid" 2>/dev/null; then \
		echo "Stale pid file (pid=$$pid not running), removing $(UI_PID_FILE)"; \
		rm -f "$(UI_PID_FILE)"; \
		exit 0; \
	fi; \
	cmd="$$(ps -p "$$pid" -o command= 2>/dev/null || true)"; \
	case "$$cmd" in \
		*-m\ server*) ;; \
		*) echo "Refusing to stop pid=$$pid (unexpected cmd: $$cmd)"; exit 1;; \
	esac; \
	kill "$$pid"; \
	for _ in {1..30}; do \
		if kill -0 "$$pid" 2>/dev/null; then sleep 0.1; else break; fi; \
	done; \
	if kill -0 "$$pid" 2>/dev/null; then \
		echo "Still running after SIGTERM: pid=$$pid"; \
		exit 1; \
	fi; \
	rm -f "$(UI_PID_FILE)"; \
	echo "Stopped: pid=$$pid"

.PHONY: ui-status
ui-status:
	@if [[ -f "$(UI_PID_FILE)" ]]; then \
		pid="$$(cat "$(UI_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Running: pid=$$pid"; \
			exit 0; \
		fi; \
		echo "Not running (stale pid file: $(UI_PID_FILE))"; \
		exit 1; \
	fi; \
	echo "Not running"; \
	exit 1

.PHONY: ui-logs
ui-logs:
	@if [[ ! -f "$(UI_LOG_FILE)" ]]; then \
		echo "No log file: $(UI_LOG_FILE)"; \
		exit 1; \
	fi
	@tail -n 200 -f "$(UI_LOG_FILE)"

.PHONY: up
up: venv
	@mkdir -p "$(RUN_DIR)"
	@if [[ -f "$(APP_PID_FILE)" ]]; then \
		pid="$$(cat "$(APP_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Already running: pid=$$pid (use: make down)"; \
			exit 1; \
		fi; \
	fi
	@nohup "$(VENV_PY)" main.py >"$(APP_LOG_FILE)" 2>&1 & echo $$! >"$(APP_PID_FILE)"
	@echo "Started: pid=$$(cat "$(APP_PID_FILE)")"
	@echo "Logs: $(APP_LOG_FILE)"

.PHONY: down
down:
	@if [[ ! -f "$(APP_PID_FILE)" ]]; then \
		echo "Not running (no pid file: $(APP_PID_FILE))"; \
		exit 0; \
	fi; \
	pid="$$(cat "$(APP_PID_FILE)")"; \
	if ! kill -0 "$$pid" 2>/dev/null; then \
		echo "Stale pid file (pid=$$pid not running), removing $(APP_PID_FILE)"; \
		rm -f "$(APP_PID_FILE)"; \
		exit 0; \
	fi; \
	cmd="$$(ps -p "$$pid" -o command= 2>/dev/null || true)"; \
	case "$$cmd" in \
		*main.py*) ;; \
		*) echo "Refusing to stop pid=$$pid (unexpected cmd: $$cmd)"; exit 1;; \
	esac; \
	kill "$$pid"; \
	for _ in {1..30}; do \
		if kill -0 "$$pid" 2>/dev/null; then sleep 0.1; else break; fi; \
	done; \
	if kill -0 "$$pid" 2>/dev/null; then \
		echo "Still running after SIGTERM: pid=$$pid"; \
		exit 1; \
	fi; \
	rm -f "$(APP_PID_FILE)"; \
	echo "Stopped: pid=$$pid"

.PHONY: status
status:
	@if [[ -f "$(APP_PID_FILE)" ]]; then \
		pid="$$(cat "$(APP_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Running: pid=$$pid"; \
			exit 0; \
		fi; \
		echo "Not running (stale pid file: $(APP_PID_FILE))"; \
		exit 1; \
	fi; \
	echo "Not running"; \
	exit 1

.PHONY: logs
logs:
	@if [[ ! -f "$(APP_LOG_FILE)" ]]; then \
		echo "No log file: $(APP_LOG_FILE)"; \
		exit 1; \
	fi
	@tail -n 200 -f "$(APP_LOG_FILE)"

.PHONY: ui-clean
ui-clean:
	rm -rf ui/dist
	rm -f "$(UI_PID_FILE)" "$(UI_LOG_FILE)"

.PHONY: clean
clean:
	rm -rf \
		.coverage \
		htmlcov \
		.pytest_cache \
		.mypy_cache \
		.ruff_cache \
		__pycache__ \
		"$(RUN_DIR)"
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

.PHONY: clean-venv
clean-venv:
	@if [[ -z "$(VENV_DIR)" || "$(VENV_DIR)" == "/" || "$(VENV_DIR)" == "." ]]; then \
		echo "Refusing to remove VENV_DIR='$(VENV_DIR)'"; \
		exit 1; \
	fi
	rm -rf "$(VENV_DIR)"
