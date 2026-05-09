# MWV_FLOW — current Manager -> Worker -> Verifier

Нормативный источник инвариантов: `docs/architecture/ARCH_CANON.md`.
Этот документ описывает текущий MWV-путь после PR-0..PR-7.

## Точка входа

- Legacy entry: `Agent.respond(...)` → `classify_request(...)`.
- При `route="mwv"` вызывается `_run_mwv_flow(...)`.
- Целевой entry для новых work execution paths — LLM tool calls через `AgentToolLoop`
  и `ToolGateway`; keyword routing не должен расширяться.

## Маршрутизация

- `chat`: объяснение/консультация без явных действий.
- `mwv`: задачи с кодом, файлами, инструментами, командами установки/системы.
- `/...` команды: command lane, вне MWV (см. `docs/agent/COMMAND_LANE_POLICY.md`).

## Данные контура

- `RunContext`: `session_id`, `trace_id`, `safe_mode`, `approved_categories`, `max_retries`, `attempt`.
- `TaskPacket`: execution contract (см. `TaskPacket v2` в `docs/architecture/ARCH_CANON.md`).
- `WorkResult`: summary, changes, diagnostics.
- `VerificationResult`: статус, команда, exit code, stdout/stderr, длительность.

## Канонический цикл

1. `ManagerRuntime.run_flow(...)` строит `TaskPacket` (v2 execution contract).
2. `WorkerRuntime` исполняет только explicit tool requests через gateway/tools.
   `Planner`/`Executor` больше не извлекают аргументы инструментов из prose.
3. `VerifierRuntime.run(...)` выполняет deterministic-проверку. При отсутствии `scripts/check.sh` используется fallback-последовательность (`ruff`, `lint_skills`, `build_manifest --check`, `mypy`, `pytest --cov`).
4. Успех только если одновременно:
   - `work_result.status == success`
   - `verification_result.status == passed`
5. При PASS формируется итоговый MWV-ответ. При FAIL/ERROR — stop-ответ с `MWV_REPORT_JSON`.

## Retry (bounded)

- Количество попыток определяется через `RunContext.max_retries`.
- При verifier fail — ограниченный retry с дополнительными constraints.
- Остановка без retry при: verifier error, worker failure, исчерпании лимита.

## Stop-ответы

Формат соответствует `docs/agent/STOP_RESPONSES.md`.
Канонический формат `STOP_TO_CHAT` — см. `docs/architecture/ARCH_CANON.md`.

## Что не входит в MWV

- Команды `/...` (command lane).
- UI-операции выбора модели/сессии.
- PTY terminal session: интерактивный терминал идёт через `TerminalTool` + yolo-gate,
  не через MWV verifier.
