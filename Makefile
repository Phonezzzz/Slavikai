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
	@echo
	@echo "Run:"
	@echo "  make run             Run UI in foreground"
	@echo "  make up              Run UI in background (pid/log in .run/)"
	@echo "  make down            Stop background UI started by make up"
	@echo "  make status          Show background UI status"
	@echo "  make logs            Tail background UI log"
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

.PHONY: run
run: venv
	"$(VENV_PY)" main.py

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
