# MWV_CANONICAL_FLOW — канонический поток Manager -> Worker -> Verifier

Этот документ фиксирует фактический runtime-контур MWV.
Нормативный источник инвариантов: `docs/ARCH_CANON.md`.

## Маршрутизация

- `chat`: объяснение/консультация без явных действий.
- `mwv`: задачи с кодом, файлами, инструментами, командами установки/системы.
- `/...` команды: command lane, **вне MWV** (см. `docs/COMMAND_LANE_POLICY.md`).

## Канонический цикл

1. `ManagerRuntime` строит `TaskPacket` (v2 execution contract).
2. `WorkerRuntime` выполняет задачу через `Planner` + `Executor` + tools.
3. `VerifierRuntime` запускает проверку. При отсутствии `scripts/check.sh` используется fallback-последовательность (`ruff`, `lint_skills`, `build_manifest --check`, `mypy`, `pytest --cov`).
4. Успех только если одновременно:
   - `work_result.status == success`
   - `verification_result.status == passed`

## Retry (bounded)

- Количество попыток определяется через `RunContext.max_retries`.
- При verifier fail используется ограниченный retry с дополнительными constraints.
- Остановка без retry при:
  - verifier error,
  - worker failure,
  - исчерпании лимита.

## Stop-ответы

Формат stop-ответов соответствует `docs/STOP_RESPONSES.md`.
Для канонического формата `STOP_TO_CHAT` см. также `docs/ARCH_CANON.md`.
