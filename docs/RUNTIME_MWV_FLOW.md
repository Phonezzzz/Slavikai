# RUNTIME_MWV_FLOW — текущий рабочий контур

Документ описывает как MWV реально включён в runtime.
Нормативный источник инвариантов: `docs/ARCH_CANON.md`.

## Точка входа

- `Agent.respond(...)` -> `classify_request(...)`.
- При `route="mwv"` вызывается `_run_mwv_flow(...)`.

## Данные контура

- `RunContext`: `session_id`, `trace_id`, `safe_mode`, `approved_categories`, `max_retries`, `attempt`.
- `TaskPacket`: execution contract (см. `TaskPacket v2` в `docs/ARCH_CANON.md`).
- `WorkResult`: summary, changes, diagnostics.
- `VerificationResult`: статус, команда, exit code, stdout/stderr, длительность.

## Последовательность

1. `ManagerRuntime.run_flow(...)` создаёт задачу.
2. `WorkerRuntime` исполняет план и инструменты.
3. `VerifierRuntime.run(...)` выполняет deterministic-проверку.
4. При PASS формируется итог MWV-ответ.
5. При FAIL/ERROR формируется stop-ответ с `MWV_REPORT_JSON`.

## Что не входит в MWV

- Команды `/...` (command lane).
- UI-операции выбора модели/сессии.
