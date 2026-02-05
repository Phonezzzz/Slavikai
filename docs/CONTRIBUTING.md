# CONTRIBUTING

## Базовые требования

- Python 3.12+.
- Установленные зависимости из `requirements.txt`.

## Локальный старт

- `make venv`
- `python -m server`

## Обязательный workflow

Смотри `docs/dev_workflow.md` и `docs/merge_process.md`.
Коротко:

1. `git checkout main`
2. `git checkout -b pr-<id>-<slug>`
3. `make git-check`
4. изменения + commit + push
5. `make check`
6. `git rebase origin/main`
7. `git merge --ff-only <pr-branch>`

## Проверки качества

- `make check` — обязательный прогон перед финализацией.
- CI: `.github/workflows/check.yml` (запускает `scripts/check.sh`).
- Dependabot: `.github/dependabot.yml` (pip + npm/ui).

## Правила по инструментам

- Любой новый tool возвращает `ToolResult`.
- Для рискованных инструментов:
  - учесть safe-mode (`SAFE_MODE_TOOLS_OFF`),
  - добавить тесты на блокировку.
- Пути и файловые операции должны оставаться в sandbox.

## Документация

- Актуальные документы держать синхронизированными с кодом.
- Исторические roadmap/phase документы хранить в `docs/archive/`.
