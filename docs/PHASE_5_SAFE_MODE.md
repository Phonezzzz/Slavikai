# PHASE 5 — Safe‑mode + Session Approvals

Этот документ описывает поведение safe‑mode и механики подтверждений на уровне backend.

## 1) Safe‑mode

- По умолчанию safe_mode включён (см. `tools_config` / `DEFAULT_TOOLS` в `core/agent.py`).
- При safe_mode=true опасные действия блокируются до подтверждения.
- При safe_mode=false опасные действия разрешены, но всё равно логируются в trace.

## 2) Категории approvals

Список категорий совпадает с `docs/PHASE_5_DECISION_MATRIX.md`:

- `FS_DELETE_OVERWRITE`
- `FS_OUTSIDE_WORKSPACE`
- `FS_CONFIG_SECRETS`
- `DEPS_INSTALL_UPDATE`
- `GIT_PUBLISH`
- `SYSTEM_IMPACT`
- `SUDO`
- `NETWORK_RISK`
- `EXEC_ARBITRARY`

## 3) Жизненный цикл approvals

1) Клиент отправляет запрос с `session_id`.
2) Backend определяет опасную категорию.
3) Если категория не разрешена — возвращает `approval_required` и trace‑событие.
4) UI вызывает `POST /slavik/approve-session` с `session_id` и `categories[]`.
5) Следующие запросы с тем же `session_id` используют approvals.
6) Новый `session_id` или перезапуск процесса → approvals сброшены.

## 4) Формат approval‑ответа

Ответ при требовании подтверждения — это ошибка:

- HTTP 400
- `error.type = tool_error`
- `error.code = approval_required`
- В `error.details`:
  - `category`
  - `required_categories`
  - `session_id`
  - `prompt` (what/why/risk/changes)
  - `blocked_reason = approval_required`

## 5) Примеры (curl)

### 5.1 Запрос, требующий approval
```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Slavik-Session: session-1' \
  -d '{
    "model": "slavik",
    "messages": [{"role":"user","content":"danger action"}]
  }'
```

### 5.2 Разрешить категории для сессии
```bash
curl -sS -X POST http://127.0.0.1:8000/slavik/approve-session \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "session-1",
    "categories": ["SUDO", "EXEC_ARBITRARY"]
  }'
```

### 5.3 Повторный запрос с той же сессией
```bash
curl -sS -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Slavik-Session: session-1' \
  -d '{
    "model": "slavik",
    "messages": [{"role":"user","content":"danger action"}]
  }'
```

## 6) Trace события

- `approval_required` — когда нужен approve.
- `approval_skipped` — когда safe_mode выключен, но действие рискованное.
- `approval_granted` — когда /approve-session добавил категории.

## 7) Ограничения Phase 5

- approvals живут только в памяти процесса (без персистентности).
- streaming не поддержан.
- UI не содержит логики — только отображение.
