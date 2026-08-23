# Dev Workflow — режим A

Цель: линейная история `main`, PR-ветки вливаются без merge-коммитов, история PR сохраняется.

## Правила

- Работа всегда начинается с `main`: `git checkout main`.
- Перед созданием ветки обнови `main`: `git pull --ff-only origin main`.
- На каждый PR создаётся отдельная ветка: `git checkout -b pr-<номер>-<slug>`.
- На чистой новой PR-ветке запускай `make preflight`. Эта цель не требует upstream и
  устанавливает baseline до изменений.
- Работаешь и запускаешь targeted tests.
- Перед commit запускай `make check`, затем коммитишь и пушишь PR-ветку.
- Перед merge обнови remote refs. Если `origin/main` продвинулся, rebase PR-ветку,
  повтори `make check` и push.
- После синхронизации запускай `make git-check`: он сам делает fetch, требует clean
  worktree, upstream без ahead/behind, актуальную базу `origin/main` и повторяет canonical check.
- В `main` вливаешь `git merge --ff-only <pr-branch>`, чтобы не было merge-коммитов.
- После вливания — всегда `git checkout main`.
- Нельзя продолжать новую фичу в старой PR-ветке.

`make preflight` — pre-implementation gate. `make git-check` — последний pre-merge gate. Они
намеренно имеют разные требования и не взаимозаменяемы.
