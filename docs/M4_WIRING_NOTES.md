# M4_WIRING_NOTES — подключение MWV в runtime

Этап M4 включает MWV‑контур в runtime с детерминированным routing, без DualBrain‑критика.
HTTP/UI не трогаются. MWV активируется **только** для запросов с явными триггерами.

## Что подключено

- `core/agent.py`:
  - добавлено решение маршрута через `classify_request`.
  - `route="chat"` → обычный ответ LLM (без planner/executor).
  - `route="mwv"` → запуск MWV‑контура.
- MWV‑контур:
  - `ManagerRuntime.run_flow(...)` — оркестрация попыток.
  - `WorkerRuntime` — использует `planner` + `executor` + tool gateway.
  - `VerifierRuntime` — запускает `scripts/check.sh`.

## Где ветвление

Точка принятия решения в `core/agent.py`:
1) `/`‑команды → старые handlers.
2) `classify_request(...)`:
   - `chat` → `_run_chat_response(...)`
   - `mwv` → `_run_mwv_flow(...)`

## Передача данных между ролями

- `RunContext`:
  - session_id, trace_id, safe_mode, approved_categories, max_retries, attempt.
- `TaskPacket`:
  - goal, messages, constraints, context (route_reason, risk_flags).
- `WorkResult`:
  - summary (форматированный план),
  - changes (из workspace‑diffs),
  - diagnostics (ошибки шагов).
- `VerificationResult`:
  - exit_code, stdout/stderr, duration.

## Retry‑модель

- `MAX_MWV_ATTEMPTS = 3` (итого 3 попытки).
- При `FAILED`:
  - добавляется constraint “Исправь только минимальные изменения …”
  - повтор, пока не исчерпан лимит.
- `ERROR` → остановка.
- `WorkResult.FAILURE` → остановка.

## Как отключить MWV (при необходимости)

Временно принудить routing на chat:
- заменить `decision = classify_request(...)` на `RouteDecision("chat", "...", [])`
  в `core/agent.py`.

## Что не трогали

- `agent.py` логика dual/critic не удалена, но в runtime не используется.
- HTTP gateway, UI, OpenWebUI — без изменений.
