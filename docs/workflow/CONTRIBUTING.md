# CONTRIBUTING

## Базовые требования

- Python 3.12+.
- Установленные зависимости из lock-файла `requirements.txt` (source: `requirements.in`).

## Зависимости (pip-tools)

- Source-set: `requirements.in`.
- Lock/constraints: `requirements.txt`, `constraints.txt` (генерируются из `requirements.in`).
- Обновление lock-файлов:
  - `make deps-compile`
- Синхронизация окружения по lock:
  - `make deps-sync`

## Локальный старт

- `make venv`
- `make run`

## Обязательный workflow

Следуй [порядку работы с PR](dev_workflow.md).

## Проверки качества

- `make check` — обязательный прогон перед финализацией.
- CI: `.github/workflows/check.yml` (запускает `make check`).
- Dependabot: `.github/dependabot.yml` (pip + npm/ui).

## Правила по инструментам

- Любой новый tool возвращает `ToolResult`.
- Для рискованных инструментов:
  - учесть safe-mode (`SAFE_MODE_TOOLS_OFF`),
  - добавить тесты на блокировку.
- Пути и файловые операции должны оставаться в sandbox.

## Документация

- Актуальные документы держать синхронизированными с кодом.
- В рабочем дереве хранятся только действующие документы. Устаревшие документы и завершённые планы удаляются. История изменений сохраняется в Git.
