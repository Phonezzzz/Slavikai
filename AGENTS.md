# AGENTS — обязательная точка входа

Репозиторий работает по принципу rules-first/policies-first.
Любой агент обязан прочитать и применить правила из этого файла до начала работы.

## Initialization protocol (обязательно)

Перед любыми действиями агент обязан:

1. Прочитать все документы из раздела **Canonical rules**.
2. Определить релевантные контекстные документы и прочитать их.
3. Сформировать **Rules + Context Snapshot** и считать его жёсткими ограничениями на всю сессию.

Работу над задачей начинать только после этого.

## Canonical rules (must read)

- `docs/DevRules.md` — глобальные инварианты проекта.
- `docs/dev_workflow.md` — git-процесс (режим A).
- `docs/merge_process.md` — точная последовательность merge (`ff-only`).
- `docs/COMMAND_LANE_POLICY.md` — границы командного режима.
- `docs/ROUTING_POLICY.md` — маршрутизация chat/mwv.
- `docs/STOP_RESPONSES.md` — единый формат остановки.

## Contextual references (читать по релевантности)

- `docs/Architecture.md`
- `docs/PROJECT_OVERVIEW.md`
- `docs/COMMANDS.md`
- `docs/CONTRIBUTING.md`
- `docs/MWV_CANONICAL_FLOW.md`
- `docs/RUNTIME_MWV_FLOW.md`
- `docs/LIFECYCLE.md`
- `docs/archive/README.md` (если нужен исторический контекст)

## Rules + Context Snapshot (формат)

- Какие канонические правила применяются в текущей задаче.
- Какие контекстные документы прочитаны и почему они релевантны.
- Какие ограничения являются жёсткими (sandbox, safe-mode, MWV, git-flow, approvals).

## Non-negotiable rules

- Работа никогда не ведётся напрямую в `main`.
- Каждая задача выполняется в отдельной PR-ветке.
- Перед началом работы в PR-ветке: `make git-check`.
- Перед завершением: `make check`.
- В `main` только `git merge --ff-only`.
- После merge обязательно вернуться в `main`.
- Если что-то неясно — остановиться и запросить решение у человека.

## Язык (обязательно)

- Все ответы, планы, объяснения и комментарии — на русском.
- Английский допускается только в коде, командах, логах, именах файлов и точных цитатах ошибок/документации.
- Размышления оформлять как: **Краткий план (RU)** + **Проверки/риски (RU)**.

## Workflow (кратко)

1. `git checkout main`
2. `git checkout -b pr-<id>-<name>`
3. `make git-check`
4. реализация + commit + push
5. `make check`
6. `git rebase origin/main` + `git merge --ff-only`
7. `git checkout main`
