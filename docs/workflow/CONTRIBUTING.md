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

Смотри `docs/workflow/dev_workflow.md`.
Коротко:

1. `git checkout main` + `git pull --ff-only origin main`
2. `git checkout -b pr-<id>-<slug>`
3. `make preflight`
4. изменения + targeted tests
5. `make check`
6. commit + `git push -u origin HEAD`
7. если `origin/main` продвинулся: rebase PR-ветки, повторный `make check` и push
8. `make git-check`
9. после одобрения: `git checkout main` + `git merge --ff-only <pr-branch>`
10. `git push origin main`
11. `git checkout main`

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
