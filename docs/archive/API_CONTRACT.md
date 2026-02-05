# API_CONTRACT - Slavik <-> Open WebUI

Этот документ фиксирует API‑контракт между UI (Open WebUI) и backend‑агентом Slavik.
Контракт задаёт форматы, но не внедряет логику: весь интеллект остаётся в Python‑backend.

## 0) Инварианты контракта
- Open WebUI = только UI. Никакой собственной логики, инструментов, памяти.
- Ошибки не скрываются: любые сбои возвращаются как явные ошибки (HTTP + error payload).
- Агент — единственный источник истины для планирования, инструментов, памяти, safety, feedback.
- Полные детали выполнения доступны как структурированная трасса по `trace_id`.

## 1) Идентификаторы
- `trace_id` — идентификатор одного запроса. Генерируется на каждый запрос и возвращается при успехе; при ошибке возвращается, если успел быть создан.
- `session_id` — идентификатор сессии/чата. Новый чат = новый `session_id`.
- Как UI передаёт `session_id`:
  - рекомендуемый HTTP‑заголовок: `X-Slavik-Session`.
  - альтернативно: поле `session_id` в `slavik_meta` тела запроса.
  - если `session_id` не передан — backend создаёт новый и возвращает его в `slavik_meta`.
- Рекомендуемое соответствие (связано с текущей базой данных):
  - `trace_id` == `ChatInteractionLog.interaction_id` (см. `memory/memory_companion_store.py`).

## 2) Модели и режимы

### Виртуальные модели (OpenAI‑совместимый список)
- `slavik` → основной режим исполнения

## 3) Ошибки (единый формат)

```json
{
  "error": {
    "message": "Human-readable error",
    "type": "invalid_request_error | tool_error | internal_error | not_supported",
    "code": "safe_mode_blocked | validation_error | tool_disabled | tool_not_registered | sandbox_violation | ...",
    "trace_id": "uuid-if-available",
    "details": {}
  }
}
```

## 4) Политика неизвестных полей

### 4.1 Неизвестные sampling‑параметры (generation knobs)
- Игнорировать, НЕ возвращать ошибку.
- Записать warning в trace (доступен через `/slavik/trace/{trace_id}`).
- Примеры ключей: `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `min_p`, `mirostat*`, `num_ctx`, любые `ollama_*`.

### 4.2 Неизвестные структурные/протокольные поля
- Если поле ломает схему запроса или имеет некорректный тип → HTTP 400 `invalid_request_error`.
- Примеры: `messages` не список, `messages[*].role` вне списка ролей, `model` не строка.

## 4) Endpoints

### 4.3 GET /v1/models

**Назначение:** список виртуальных моделей для Open WebUI.

**Response 200:**
```json
{
  "object": "list",
  "data": [
    { "id": "slavik", "object": "model", "owned_by": "slavik" }
  ]
}
```

---

### 4.4 POST /v1/chat/completions

**Назначение:** единая точка входа для UI‑чата (OpenAI‑совместимый формат).

**Request (минимум):**
```json
{
  "model": "slavik",
  "messages": [
    { "role": "user", "content": "..." }
  ],
  "temperature": 0.7,
  "top_p": 1.0,
  "max_tokens": 1024,
  "stream": false
}
```

**Поддерживаемые роли:** `system | user | assistant | tool`.

**Ограничения:**
- `stream=true` не поддержан: HTTP 400, `error.type=not_supported`, `error.code=streaming_not_supported`.
- Если присутствуют `tool` роль или `assistant.tool_calls`, а tool‑pipeline ещё не включён → HTTP 400 `not_supported` с `error.code=tool_calling_not_supported`.
- Неизвестные sampling‑параметры игнорируются (см. раздел 4).

**Response 200 (пример):**
```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "slavik",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "Короткий ответ + план (3–7 шагов) + execution summary + uncertainty notes."
      }
    }
  ],
  "slavik_meta": {
    "trace_id": "uuid",
    "session_id": "uuid",
    "session_approved": false,
    "safe_mode": true
  }
}
```

**Совместимость метаданных:**
- Сейчас `slavik_meta` находится в теле ответа.
- Предпочтительный стабильный путь деталей: `/slavik/*` по `trace_id`.
- В будущем метаданные могут быть продублированы/перенесены в `response.metadata.slavik` (или аналог), но `/slavik/*` остаётся авторитетным источником.

---

### 4.5 GET /slavik/trace/{trace_id}

**Назначение:** структурированная трасса выполнения.

**Response 200 (минимум, соответствует `core/tracer.py`):**
```json
{
  "trace_id": "uuid",
  "events": [
    { "timestamp": "YYYY-MM-DD HH:MM:SS", "event": "user_input", "message": "...", "meta": {} },
    { "timestamp": "YYYY-MM-DD HH:MM:SS", "event": "planning_start", "message": "...", "meta": {} }
  ]
}
```

**Опционально (когда будет доступно):**
- `plan` со списком шагов (`description`, `status`, `operation`, `result`).

---

### 4.6 GET /slavik/tool-calls/{trace_id}

**Назначение:** детальный список вызовов инструментов за конкретный запрос.

**Response 200 (минимум, соответствует `ToolCallRecord`):**
```json
{
  "trace_id": "uuid",
  "tool_calls": [
    {
      "timestamp": "YYYY-MM-DD HH:MM:SS",
      "tool": "workspace_read",
      "ok": true,
      "error": null,
      "args": {},
      "meta": {}
    }
  ]
}
```

---

### 4.7 POST /slavik/feedback

**Назначение:** явный пользовательский фидбэк.

**Request:**
```json
{
  "interaction_id": "uuid",
  "rating": "good | ok | bad",
  "labels": ["too_long", "incorrect"],
  "free_text": "optional text"
}
```

**Response 200:**
```json
{ "ok": true }
```

---

### 4.8 POST /slavik/approve-session

**Назначение:** одноразовое разрешение опасных действий в рамках текущей сессии.

**Request:**
```json
{
  "session_id": "uuid"
}
```

**Response 200:**
```json
{
  "session_id": "uuid",
  "session_approved": true
}
```

## 5) Источники данных (привязка к текущему коду)
- Trace события: `logs/trace.log` (`core/tracer.py`).
- Tool calls: `logs/tool_calls.log` (`tools/tool_logger.py`).
- InteractionLog + Feedback: `memory/memory_companion.db` (`memory/memory_companion_store.py`).
- План/шаги: `core/agent.py` (`last_plan`, `last_plan_original`) — пока в памяти процесса.

## 6) Совместимость
- OpenAI‑совместимый формат для `/v1/chat/completions` и `/v1/models`.
- Любые расширения — строго под ключом `slavik_meta` и через `/slavik/*` endpoints.

## 7) Примеры (минимальные контракт‑тесты)

### 7.1 Обычный запрос (stream=false)
**Request:**
```json
{
  "model": "slavik",
  "messages": [
    { "role": "user", "content": "Сделай краткое резюме проекта." }
  ],
  "stream": false
}
```

**Response 200:**
```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "slavik",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": { "role": "assistant", "content": "..." }
    }
  ],
  "slavik_meta": { "trace_id": "uuid", "session_id": "uuid", "safe_mode": true }
}
```

### 7.2 Неизвестный sampling‑параметр (игнорируется)
**Request:**
```json
{
  "model": "slavik",
  "messages": [
    { "role": "user", "content": "Объясни, что такое VectorIndex." }
  ],
  "top_k": 50
}
```

**Response 200:**
```json
{
  "id": "chatcmpl-uuid",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "slavik",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": { "role": "assistant", "content": "..." }
    }
  ],
  "slavik_meta": { "trace_id": "uuid", "session_id": "uuid", "safe_mode": true }
}
```
Примечание: в trace должна появиться warning‑запись о проигнорированном `top_k`.

### 7.3 stream=true (не поддержан)
**Request:**
```json
{
  "model": "slavik",
  "messages": [
    { "role": "user", "content": "Привет" }
  ],
  "stream": true
}
```

**Response 400:**
```json
{
  "error": {
    "message": "Streaming is not supported.",
    "type": "not_supported",
    "code": "streaming_not_supported",
    "trace_id": "uuid-if-available",
    "details": {}
  }
}
```

### 7.4 Tool-calling при отключённом tool‑pipeline
**Request:**
```json
{
  "model": "slavik",
  "messages": [
    { "role": "user", "content": "Список файлов" },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": { "name": "fs", "arguments": "{}" }
        }
      ]
    }
  ]
}
```

**Response 400:**
```json
{
  "error": {
    "message": "Tool calling is not supported.",
    "type": "not_supported",
    "code": "tool_calling_not_supported",
    "trace_id": "uuid-if-available",
    "details": {}
  }
}
```
