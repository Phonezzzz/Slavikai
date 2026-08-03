---
name: slavikai-dev
description: Ключевые конвенции и правила разработки проекта SlavikAI — Python 3.12, mypy strict, ruff, sandbox, safe-mode, git workflow (режим A), ToolResult, Memory Companion, anti-pseudo audit.
license: MIT
compatibility: opencode
---

# SlavikAI Development Rules

## Когда использовать

Используй этот skill всегда при работе над этим проектом. Skill содержит инварианты, которые нельзя нарушать.

## Язык, типизация, стиль

- Python 3.12 (target-version в `pyproject.toml`)
- `mypy` в режиме `strict = true` (tests исключены)
- `ruff` для линтинга и форматирования (конфиг в `pyproject.toml`)
- UI-часть (`ui/`): TypeScript/JavaScript
- Запрещено «расплывать» типы через `Any`/`cast()` в доменной логике
- Запрещено использовать `Optional[...]` без явной проверки на `None`

## Инструменты: обязательный ToolResult

- Каждый инструмент обязан реализовать `Tool.handle(self, request: ToolRequest) -> ToolResult`
- Ошибки инструмента — через `ToolResult.failure(...)`
- Запрещено проталкивать исключения наружу как основной контроль потока
- Человекочитаемый текст в `data["output"]`

## Sandbox restrictions

- Все операции с путями — в песочнице (`sandbox/`, `sandbox/project/`)
- Использовать существующие хелперы для нормализации путей:
  - `tools/filesystem_tool.py::_normalize_path(...)`
  - `tools/workspace_tools.py::_ensure_in_workspace(...)`
  - `tools/project_tool.py::_normalize_path(...)`

## Safe-mode

- Safe-mode на уровне `ToolRegistry` (`core/agent.py`)
- Инструменты с сетевым/системным доступом выключаются в safe-mode
- Блок-лист: `web`, `web_search`, `shell`, `project`, `tts`, `stt`, `http_client`, `image_analyze`, `image_generate`, `workspace_run`

## Git workflow (режим A)

1. `git checkout main`
2. `git checkout -b pr-<id>-<name>`
3. `make git-check`
4. Реализация + commit + push
5. `make check`
6. `git rebase origin/main` + `git merge --ff-only`
7. `git checkout main`

## Memory Companion: инварианты

- Запрещены авто-апдейты Memory в runtime
- Запрещены авто-создание/изменение PolicyRule в runtime
- BatchReview только вручную

## Anti-pseudo audit

Перед implementation каждого PR выполнить mini non-mutating audit по признакам pseudo-runtime/pseudo-agent behavior. См. `docs/agent/DevRules.md`, раздел «Anti-pseudo audit».

## Тесты и качество

- `pytest` + `pytest-cov`, порог покрытия >= 80%
- `ruff check .` — без ошибок
- `ruff format --check .` — без расхождений
- `mypy .` — strict без ошибок
- Запрещены реальные сетевые вызовы в тестах

## Канонические документы

- `AGENTS.md`
- `docs/agent/DevRules.md`
- `docs/workflow/dev_workflow.md`
- `docs/agent/COMMAND_LANE_POLICY.md`
- `docs/agent/ROUTING_POLICY.md`
- `docs/agent/STOP_RESPONSES.md`
- `docs/architecture/Architecture.md`
