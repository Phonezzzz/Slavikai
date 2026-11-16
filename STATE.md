# SlavikAI Worker State

**Source of truth:** [ROADMAP.md](ROADMAP.md) и [соглашение Muse](slavikai-how-to-work-with-muse.md). Все значения являются снимком, а не бессрочной гарантией; перед любыми действиями повторить read-only preflight.

## Repository baseline (проверен 2026-11-16)

- Repository: https://github.com/Phonezzzz/Slavikai
- `origin/main`: `092ba39dbd213c0256517caefb5ec0d924f2ed43`
- Единственный локальный worktree: `/home/ki/Project/Slavikai`
- Local branch: `pr-151-chatgpt-subscription`
- Local HEAD: `728c04cd0f9e8e9299ff56eaa6173ed407c3dfd8`
- Working tree: **DIRTY**, есть незакоммиченные изменения; не переключать, не чистить, не stash/reset/rebase.
- PR #151: https://github.com/Phonezzzz/Slavikai/pull/151 — незавершённое пользовательское тестирование и исправления, **исполнитель дефектов Codex, НЕ Muse**.
- Решение владельца по судьбе PR #151: **PENDING**. Muse не вмешивается в этот PR.
- Last merged roadmap ticket / merge SHA: **NONE**.

## Docs bootstrap

- Документы: `ROADMAP.md`, `STATE.md`, `slavikai-how-to-work-with-muse.md`.
- Branch: `docs/muse-roadmap-2026-11-16`, создана напрямую в GitHub от указанного SHA `main`.
- Docs PR: **#152** — https://github.com/Phonezzzz/Slavikai/pull/152, OPEN; ожидание CI и независимого Codex review.
- CI: **PENDING**; Codex independent review: **PENDING**. Не объявлять зелёными до проверок.
- Merge: **ручной merge Muse** только после независимого чистого Codex review, зелёного CI и отсутствия блокеров. **GitHub auto-merge и merge queue запрещены без нового решения владельца**.
- Docs bootstrap разрешён без переключения текущего dirty worktree. Это исключение для публикации документов, не начало параллельного implementation.

## Current implementation ticket

- **ID:** E00-T01
- **Title:** baseline и безопасное освобождение worktree после PR #151
- **Status:** `BLOCKED: PR_151_CODEX`
- **Executor for PR #151 defects:** Codex
- **Current Muse permission:** read-only inspection/reporting, без редактирования PR #151.
- **Required evidence:** branch/HEAD/origin-main SHAs, sanitized status, worktree list, статус PR/CI/review, решение владельца, подтверждение безопасного освобождения.
- **Forbidden:** git checkout/reset/clean/stash над dirty веткой, исправление чужого PR, бесконтрольный merge, архитектурная реализация E01+.

## Queue readiness (62 тикета строго по одному)

| Эпик | ID | Количество | Spec readiness |
|---|---|---:|---|
| E00 | T01–T03 | 3 | READY; PR #151 blocker; docs bootstrap отдельно |
| E01 | T01–T09 | 9 | READY после E00 |
| E02 | T01–T08 | 8 | READY после E01 |
| E03 | T01–T11 | 11 | READY после E02 |
| E04 | T01–T06 | 6 | NEEDS_REFINEMENT |
| E05 | T01–T07 | 7 | NEEDS_REFINEMENT |
| E06 | T01–T06 | 6 | NEEDS_REFINEMENT |
| E07 | T01–T05 | 5 | NEEDS_REFINEMENT |
| E08 | T01–T07 | 7 | NEEDS_REFINEMENT |

- Следующий implementation ticket после снятия блокера: E00-T03, если docs E00-T02 будет принят; иначе E00-T02 завершить по review/CI.
- E01-T01 нельзя начинать до E00 gate.
- Перед E04 (и далее перед E05–E08) — `BLOCKED: ROADMAP_REFINEMENT`, пока все тикеты эпика не станут самодостаточными.
- Любая архитектурная развилка реализации A/B → `BLOCKED: REAL_IMPLEMENTATION_FORK`; Muse представляет варианты, trade-offs и рекомендацию, ждёт решения владельца.

## Review / merge

- `make check`: **NOT RUN** — не запускать в dirty worktree PR #151 ради docs-only публикации.
- CI: **PENDING**.
- Independent Codex review: **PENDING**.
- Blocking findings: **UNKNOWN** до review.
- Last reviewed commit: **NONE**.
- Автоматического merge нет; использовать только явный ручной merge после всех gates.

## Resume / handoff

1. Прочитать инструкции репозитория, ROADMAP.md, соглашение, затем этот STATE.md.
2. Повторить read-only проверки local branch, HEAD, `git status`, `origin/main`. Если они расходятся с STATE — STOP/reconcile.
3. Проверить судьбу PR #151 у владельца и Codex; не начинать реализацию, пока worktree не готов.
4. Одна ветка от актуального main, один тикет/PR, без дополнительных worktrees; не изменять чужой dirty state.
5. После каждого PR записать `make check`, E2E evidence, review, CI, merge SHA, следующий ticket и blocked decisions.
6. Если тикет пока provisional или архитектурное решение не принято — STOP, запрос владельцу с конкретной развилкой.

**Do not repeat:** не исправлять PR #151 силами Muse, не включать GitHub auto-merge, не объявлять DONE без postcondition.
