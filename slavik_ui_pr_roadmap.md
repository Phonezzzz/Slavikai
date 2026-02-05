# Slavik UI — Roadmap (строгая приоритизация)

## Правила выполнения (обязательно)
- Формат: `1 task = 1 PR = 1 merge`.
- Без пропусков: каждый пункт из этого файла обязателен.
- Перед началом работы в PR-ветке: `make git-check`.
- Перед завершением PR: `make check`.
- Merge только `git merge --ff-only`, после merge возврат в `main`.

## PR-0 (P0) — UI контракт и базовые правила
**Цель:** стабилизировать контракт UI <-> backend перед фичами.
- Единый fetch-client с `X-Slavik-Session`.
- Ошибки: `4xx = recoverable`, `5xx/network = service error`.
- `SSE` как основной канал, `polling` как fallback.
- Одна сессия = один `session_id` (без скрытой ротации).

## PR-1 (P0) — Happy path: model -> send -> ответ
**Фиксы:**
- SQLite thread-safety (`agent.respond` + `MemoryCompanionStore`).
- Не превращать 4xx в global `Status: error`.

**Bug mapping:**
- BUG-03 (SQLite thread mismatch).
- BUG-04 (глобальный статус при 4xx).

## PR-2 (P0) — Session continuity
**Фиксы:**
- Persist `session_id` в `localStorage`.
- Hydration через `/ui/api/status` + `/ui/api/sessions/{id}`.

**Bug mapping:**
- BUG-02 (reload ломает модель/историю).

## PR-3 (P1) — Conversations + multi-tab
**Фичи:**
- Подключить `/ui/api/sessions` (list/get).
- Sidebar с реальными чатами.
- Session routing (URL или per-tab storage).

**Bug mapping:**
- BUG-05 (placeholder sidebar).

## PR-4 (P1) — Chat UX
**Фичи:**
- Optimistic user message.
- Нормальный автоскролл (near-bottom logic).

**Bug mapping:**
- BUG-06.

## PR-5 (P1) — Approvals / Safe-mode UI
**Фичи:**
- Approval panel.
- Интеграция `/slavik/approve-session`.
- Retry blocked actions.

**Bug mapping:**
- BUG-07.

## PR-6 (P2) — Debug зона
**Фичи:**
- Event log вынести из main-view.
- Debug tab: events, trace, tool-calls.

## PR-7 (P2) — Settings MVP
**Вкладки:**
- API Keys / Providers.
- Personalization (tone, system prompt).
- Memory (embeddings, limits).
- Tools (safe-mode, registry).
- Import / Export chats DB.

**Bug mapping:**
- BUG-08 (кнопка Settings не no-op).

## PR-8 (P3) — Input расширения
**Фичи:**
- Инструменты прямо в input.
- Attach files.
- Rec/STT.
- Лог «что делает агент» через SSE.

## PR-9 (P3) — Projects & GitHub
**Этапы:**
- Local project indexing.
- GitHub import + approvals.

## PR-10 (P3) — Workspace / Split View
**Фичи:**
- Chat слева.
- Agent Workspace справа.
- Табы: Output, Code, Diff, Logs, Analysis.

## PR-11 (P3) — UI polishing
**Фиксы:**
- Убрать рамки/ограничители.
- Единый язык UI.
- Иконки сообщений (copy/edit/refresh/listen/stop).
- Layout без мусора.

## Строгий порядок выполнения
1. PR-0
2. PR-1
3. PR-2
4. PR-3
5. PR-4
6. PR-5
7. PR-6
8. PR-7
9. PR-8
10. PR-9
11. PR-10
12. PR-11

## Трекер дефектов (без пропусков)

### BUG-01 (P0) — model whitelist
- Симптом: после Set model 409 `model_not_allowed`.
- Причина: whitelist не покрывал provider-path для local/xai.
- Статус: **сделано** (commit `22baa5a`, `config/model_whitelist.py`, `core/agent.py`, `tests/test_model_whitelist.py`).

### BUG-02 (P0) — session continuity
- Симптом: после reload `selected_model=null`, новый `session_id`, 409.
- Причина: UI не персистит/не реиспользует session header на status/hydration.
- План: PR-2.

### BUG-03 (P0) — SQLite thread mismatch
- Симптом: ответ превращается во внутреннюю SQLite ошибку.
- Причина: `agent.respond` выполняется в другом thread относительно SQLite connection.
- План: PR-1.

### BUG-04 (P1) — неправильный global status error
- Симптом: ожидаемые 4xx переводят UI в service-error.
- Причина: общий catch без разделения recoverable/fatal.
- План: PR-1.

### BUG-05 (P1) — пустой Conversations sidebar
- Симптом: только placeholder.
- Причина: UI не использует list/get sessions endpoints.
- План: PR-3.

### BUG-06 (P1) — нет optimistic user-message
- Симптом: user bubble появляется только после ответа backend.
- Причина: нет optimistic enqueue/reconcile.
- План: PR-4.

### BUG-07 (P1) — нет approvals UX
- Симптом: risky flow блокируется без UI approve/retry.
- Причина: backend endpoint есть, UI не интегрирован.
- План: PR-5.

### BUG-08 (P2) — Settings no-op
- Симптом: кнопка Settings ничего не делает.
- Причина: заглушка без handler/route.
- План: PR-7.
