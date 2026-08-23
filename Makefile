SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

MAKEFLAGS += --no-builtin-rules
MAKEFLAGS += --warn-undefined-variables

.DEFAULT_GOAL := help

PYTHON ?= python3.12
VENV_DIR ?= venv
PROD_VENV_DIR ?= venv-prod
PROD_VENV_PY := $(PROD_VENV_DIR)/bin/python
LOCK_VENV_DIR := .venv-lock
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_PY) -m pip
VENV_RUFF := $(VENV_DIR)/bin/ruff
LOCK_PY := $(LOCK_VENV_DIR)/bin/python
LOCK_PIP := $(LOCK_PY) -m pip
LOCK_COMPILE := $(LOCK_VENV_DIR)/bin/pip-compile
LOCK_SYNC := $(LOCK_VENV_DIR)/bin/pip-sync

RUN_DIR ?= .run
APP_PID_FILE := $(RUN_DIR)/slavikai-ui.pid
APP_LOG_FILE := $(RUN_DIR)/slavikai-ui.log
PROD_APP_PID_FILE := $(RUN_DIR)/slavikai-prod.pid
PROD_APP_LOG_FILE := $(RUN_DIR)/slavikai-prod.log
UI_PID_FILE := $(RUN_DIR)/ui-server.pid
UI_LOG_FILE := $(RUN_DIR)/ui-server.log
PROD_HOST ?= 0.0.0.0
PROD_PORT ?= 8000

.PHONY: help
help:
	@echo "SlavikAI Core"
	@echo
	@echo "Setup:"
	@echo "  make venv            Create venv/ and install requirements + dev tools"
	@echo "  make venv-prod       Create venv-prod/ and install only production deps"
	@echo "  make venv-embeddings Install embeddings (torch, sentence-transformers) into venv/"
	@echo "  make install-beta    Install production backend, local embeddings, and built UI"
	@echo "  make deps-compile    Generate lock files from *.in via .venv-lock/pip-compile"
	@echo "  make deps-sync       Sync venv/ to requirements.txt (pip-sync, removes extras)"
	@echo "  make activate        Print venv activation command"
	@echo "  make shell           Open an interactive shell with venv activated"
	@echo
	@echo "Quality:"
	@echo "  make lint            ruff check ."
	@echo "  make format          ruff format ."
	@echo "  make format-check    ruff format --check ."
	@echo "  make type            mypy . (strict, tests excluded by config)"
	@echo "  make ui-type         npm run typecheck (ui)"
	@echo "  make ui-test         npm test (ui)"
	@echo "  make test            pytest (coverage configured in pyproject.toml)"
	@echo "  make check           canonical gate: contracts + skills + lint + types + UI + tests"
	@echo "  make check-contracts Validate source-of-truth registry and mechanical contracts"
	@echo "  make skills-check    Validate skill schema and generated runtime manifest"
	@echo "  make ci              skills lint/manifest + pytest -q (temp candidates)"
	@echo
	@echo "Git:"
	@echo "  make guard-main      Fail if current branch is main"
	@echo "  make preflight       Clean local PR branch baseline (upstream not required)"
	@echo "  make git-check       Clean, fully pushed PR branch plus canonical check"
	@echo
	@echo "Run:"
	@echo "  make run             Run UI in foreground"
	@echo "  make run-prod        Run server in foreground for production host/port"
	@echo "  make smoke-prod      Check health and Bearer automation models on a running server"
	@echo "  make up-prod         Run production server in background via venv-prod"
	@echo "  make down-prod       Stop background production server"
	@echo "  make status-prod     Show background production server status"
	@echo "  make logs-prod       Tail background production server log"
	@echo "  make up              Run development server in background via venv"
	@echo "  make down            Stop background development server"
	@echo "  make status          Show background development server status"
	@echo "  make logs            Tail background development server log"
	@echo "  make deploy-example  Print production deployment command sequence"
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

$(PROD_VENV_DIR)/bin/python:
	$(PYTHON) -m venv "$(PROD_VENV_DIR)"

$(LOCK_PY):
	$(PYTHON) -m venv "$(LOCK_VENV_DIR)"
	$(LOCK_PIP) install --upgrade 'pip==24.0'
	$(LOCK_PIP) install 'pip-tools==7.6.0'

$(VENV_DIR)/.installed: requirements.txt constraints.txt $(VENV_PY)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt -c constraints.txt
	@touch "$(VENV_DIR)/.installed"

$(VENV_DIR)/.installed-dev: requirements-dev.txt $(VENV_PY)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-dev.txt -c requirements.txt
	@touch "$(VENV_DIR)/.installed-dev"

$(VENV_DIR)/.installed-embeddings: requirements-embeddings.txt $(VENV_PY)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements-embeddings.txt -c requirements.txt
	@touch "$(VENV_DIR)/.installed-embeddings"

$(PROD_VENV_DIR)/.installed: requirements.txt constraints.txt $(PROD_VENV_DIR)/bin/python
	$(PROD_VENV_DIR)/bin/python -m pip install --upgrade pip
	$(PROD_VENV_DIR)/bin/python -m pip install -r requirements.txt -c constraints.txt
	@touch "$(PROD_VENV_DIR)/.installed"

$(PROD_VENV_DIR)/.installed-embeddings: requirements-embeddings.txt requirements.txt $(PROD_VENV_DIR)/bin/python
	$(PROD_VENV_DIR)/bin/python -m pip install --upgrade pip
	$(PROD_VENV_DIR)/bin/python -m pip install -r requirements-embeddings.txt -c requirements.txt
	@touch "$(PROD_VENV_DIR)/.installed-embeddings"

.PHONY: venv
venv: $(VENV_DIR)/.installed $(VENV_DIR)/.installed-dev

.PHONY: venv-prod
venv-prod: $(PROD_VENV_DIR)/.installed

.PHONY: venv-prod-embeddings
venv-prod-embeddings: $(PROD_VENV_DIR)/.installed-embeddings

.PHONY: install-beta
install-beta: venv-prod venv-prod-embeddings ui-ci ui-build

.PHONY: venv-dev
venv-dev: venv $(VENV_DIR)/.installed-dev

.PHONY: venv-embeddings
venv-embeddings: $(VENV_DIR)/.installed-embeddings

.PHONY: deps-compile
deps-compile: $(LOCK_PY) requirements.in requirements-dev.in requirements-embeddings.in
	$(LOCK_COMPILE) --strip-extras requirements.in --output-file requirements.txt
	$(LOCK_COMPILE) --strip-extras requirements.in --output-file constraints.txt
	$(LOCK_COMPILE) --strip-extras requirements-dev.in -c requirements.txt --output-file requirements-dev.txt
	$(LOCK_COMPILE) --strip-extras requirements-embeddings.in -c requirements.txt --output-file requirements-embeddings.txt

.PHONY: deps-sync
deps-sync: $(LOCK_PY) requirements.txt requirements-dev.txt
	$(LOCK_SYNC) --python-executable "$(VENV_PY)" requirements.txt requirements-dev.txt

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

.PHONY: ui-type
ui-type:
	cd ui && npm run typecheck

.PHONY: ui-test
ui-test:
	cd ui && npm test

.PHONY: test
PYTEST_ARGS ?=
test: venv
	"$(VENV_PY)" -m pytest $(PYTEST_ARGS)

.PHONY: test-behavior
test-behavior: venv
	"$(VENV_PY)" -m pytest --no-cov -m behavior

.PHONY: check
check: check-no-legacy-ui check-contracts skills-check lint format-check type ui-type ui-test test

.PHONY: check-no-legacy-ui
check-no-legacy-ui:
	./scripts/check_no_legacy_ui.sh

.PHONY: check-contracts
check-contracts:
	"$(PYTHON)" scripts/check_source_of_truth.py

.PHONY: skills-check
skills-check: venv
	"$(VENV_PY)" skills/tools/lint_skills.py
	"$(VENV_PY)" skills/tools/build_manifest.py --check

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

.PHONY: preflight
preflight: guard-main
	@if [[ -n "$$(git status --porcelain)" ]]; then \
		echo "Preflight requires a clean worktree."; \
		git status --short; \
		exit 1; \
	fi
	$(MAKE) check
	@echo "OK: clean local PR branch baseline passed."

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
	git fetch --prune origin; \
	if [[ -n "$$(git status --porcelain)" ]]; then \
		echo "git-check requires a clean worktree."; \
		git status --short; \
		exit 1; \
	fi; \
	if ! upstream="$$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null)"; then \
		echo "No upstream for $$branch (branch not pushed)."; \
		git branch -vv; \
		exit 1; \
	fi; \
	expected_upstream="origin/$$branch"; \
	if [[ "$$upstream" != "$$expected_upstream" ]]; then \
		echo "Branch upstream must be $$expected_upstream (got $$upstream)."; \
		exit 1; \
	fi; \
	read -r ahead behind < <(git rev-list --left-right --count "HEAD...$$upstream"); \
	if [[ "$$ahead" != "0" || "$$behind" != "0" ]]; then \
		echo "Branch must match upstream (ahead=$$ahead behind=$$behind)."; \
		git branch -vv; \
		exit 1; \
	fi; \
	if ! git merge-base --is-ancestor origin/main HEAD; then \
		echo "Branch is not based on current origin/main; rebase and push it first."; \
		exit 1; \
	fi; \
	merge_commits="$$(git rev-list --min-parents=2 origin/main..HEAD)"; \
	if [[ -n "$$merge_commits" ]]; then \
		echo "PR branch contains merge commits; rebase it onto origin/main."; \
		echo "$$merge_commits"; \
		exit 1; \
	fi; \
	$(MAKE) check; \
	echo "OK: PR branch is ready to merge."; \
	echo "Next:"; \
	echo "  git checkout main"; \
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
	"$(VENV_PY)" -m server

.PHONY: run-prod
run-prod: venv-prod
	SLAVIK_HTTP_HOST="$(PROD_HOST)" SLAVIK_HTTP_PORT="$(PROD_PORT)" "$(PROD_VENV_PY)" -m server

.PHONY: smoke-prod
smoke-prod:
	"$(PYTHON)" scripts/smoke_beta.py

.PHONY: deploy-example
deploy-example:
	@echo "Production deploy:"
	@echo "  cp .env.example .env"
	@echo "  # Set Cloudflare Access identity variables, SLAVIK_API_TOKEN, and DEEPSEEK_API_KEY"
	@echo "  make install-beta"
	@echo "  make run-prod PROD_HOST=0.0.0.0 PROD_PORT=8000"
	@echo "  SLAVIK_API_TOKEN=*** make smoke-prod"

.PHONY: ui-install
ui-install:
	cd ui && npm install

.PHONY: ui-ci
ui-ci:
	cd ui && npm ci

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
	@nohup "$(VENV_PY)" -m server >"$(APP_LOG_FILE)" 2>&1 & echo $$! >"$(APP_PID_FILE)"
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

.PHONY: up-prod
up-prod: venv-prod
	@mkdir -p "$(RUN_DIR)"
	@if [[ -f "$(PROD_APP_PID_FILE)" ]]; then \
		pid="$$(cat "$(PROD_APP_PID_FILE)")"; \
		if kill -0 "$$pid" 2>/dev/null; then \
			echo "Already running: pid=$$pid (use: make down-prod)"; \
			exit 1; \
		fi; \
	fi
	@nohup env SLAVIK_HTTP_HOST="$(PROD_HOST)" SLAVIK_HTTP_PORT="$(PROD_PORT)" \
		"$(PROD_VENV_PY)" -m server >"$(PROD_APP_LOG_FILE)" 2>&1 & \
		echo $$! >"$(PROD_APP_PID_FILE)"
	@echo "Started production server: pid=$$(cat "$(PROD_APP_PID_FILE)")"
	@echo "Logs: $(PROD_APP_LOG_FILE)"

.PHONY: down-prod
down-prod:
	@if [[ ! -f "$(PROD_APP_PID_FILE)" ]]; then \
		echo "Not running (no pid file: $(PROD_APP_PID_FILE))"; \
		exit 0; \
	fi; \
	pid="$$(cat "$(PROD_APP_PID_FILE)")"; \
	if ! kill -0 "$$pid" 2>/dev/null; then \
		echo "Stale pid file (pid=$$pid not running), removing $(PROD_APP_PID_FILE)"; \
		rm -f "$(PROD_APP_PID_FILE)"; \
		exit 0; \
	fi; \
	cmd="$$(ps -p "$$pid" -o command= 2>/dev/null || true)"; \
	case "$$cmd" in \
		*$(PROD_VENV_PY)*-m\ server*) ;; \
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
	rm -f "$(PROD_APP_PID_FILE)"; \
	echo "Stopped production server: pid=$$pid"

.PHONY: status-prod
status-prod:
	@if [[ -f "$(PROD_APP_PID_FILE)" ]]; then \
		pid="$$(cat "$(PROD_APP_PID_FILE)")"; \
		if ! kill -0 "$$pid" 2>/dev/null; then \
			echo "Not running (stale pid file: $(PROD_APP_PID_FILE))"; \
			exit 1; \
		fi; \
		cmd="$$(ps -p "$$pid" -o command= 2>/dev/null || true)"; \
		case "$$cmd" in \
			*$(PROD_VENV_PY)*-m\ server*) echo "Running production server: pid=$$pid"; exit 0;; \
			*) echo "Not running (pid=$$pid belongs to unexpected process: $$cmd)"; exit 1;; \
		esac; \
	fi; \
	echo "Not running"; \
	exit 1

.PHONY: logs-prod
logs-prod:
	@if [[ ! -f "$(PROD_APP_LOG_FILE)" ]]; then \
		echo "No log file: $(PROD_APP_LOG_FILE)"; \
		exit 1; \
	fi
	@tail -n 200 -f "$(PROD_APP_LOG_FILE)"

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
		.cache \
		build \
		dist \
		__pycache__ \
		ui/dist \
		ui/.vite \
		"$(RUN_DIR)"
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.tsbuildinfo" \) -delete

.PHONY: clean-venv
clean-venv:
	@if [[ -z "$(VENV_DIR)" || "$(VENV_DIR)" == "/" || "$(VENV_DIR)" == "." ]]; then \
		echo "Refusing to remove VENV_DIR='$(VENV_DIR)'"; \
		exit 1; \
	fi
	rm -rf "$(VENV_DIR)"
