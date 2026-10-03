# Task/run admission store — foundation B1

Статус: storage primitive реализован и проверяется `tests/test_task_run_storage.py`.
**Production runtime ещё не использует этот store.** Общий lifecycle contract остаётся target;
B1 не означает durable Ask/Auto/MWV execution, continuation или recovery tool.

## Граница и authority

`core/task_run_storage.py::TaskRunStore` хранит initial accepted request revision,
отдельные task/run/attempt identities, materialized run state и immutable transition history
в одном SQLite DB. Это foundation Task/Run Lifecycle из `TASK_RUN_LIFECYCLE_CONTRACT.md`,
а не cache TinyJuice, Memory или UI session storage. Storage location задаётся доверенным
bootstrap; constructor не является recovery API и не принимает model-selected path.

`admit(scope, request_key, goal, mode)` вызывается только owning runtime после принятия
запроса. Store сам не принимает user intent, не выбирает tools и не разрешает execution.
`RunScope` происходит из verified principal/session boundary; он не является пользовательским
параметром будущего HTTP API. Обладать UUID недостаточно для доступа.

Initial task revision — 1. `task_id`, `task_run_id`, `attempt_id` генерируются независимо;
содержимое goal/session/trace не определяет identity. Initial revision сохраняет exact goal
и execution mode; дальнейшие revision/retry semantics ещё не реализованы.
Admission key unique в principal scope. Повтор exact request возвращает исходный run,
включая его текущий state; другой goal/mode/session с тем же key отклоняется.
Goal ограничен 512 KiB UTF-8; слишком большой request отклоняется до записи, не truncate'ится.
Key ограничен 128 символами. Клиентские wire/idempotency contracts определяются в B2.

## Transactions и состояния

Admission атомарно записывает revision, run и initial history event. Transition атомарно
обновляет state/version и добавляет event с prior state, authority, reason code и timestamp.
`BEGIN IMMEDIATE` сериализует competing writers на SQLite boundary; optimistic expected
version отклоняет stale writer. Время — UTC; state version отличается от content revision.
Read использует consistent transaction snapshot. Ошибки DB не превращаются в RAM fallback.

Разрешены только transitions:

| Из | В | Authority |
| --- | --- | --- |
| admitted | running | runtime |
| running | result_submitted | runtime |
| running / result_submitted | recovery_required | recovery |
| любой non-terminal state | aborted | user / recovery |

`aborted` terminal для данного run, без assertion domain failure. `terminal_at` записывается
в этой же transaction. Late/stale writer не resurrect'ит aborted. Run abort не завершает
logical task revision и не публикует semantic final. `result_submitted` и
`recovery_required` non-terminal; model text, verifier pass или worker done не превращаются
в completed. Завершение с acceptance/disposition и structured runtime signals — следующий
обязательный срез. B1 намеренно не предоставляет completed/failed/cancelled transitions,
resume, new attempt или implicit rerun, для которых ещё нет интегрированной authority.

Restart читает ровно committed state/history. Он сам ничего не помечает completed или
recovery_required: reconciliation требует отдельного controller и fencing живых owners.
Несогласованные history/current state отклоняются, без latest-timestamp эвристики.
Revision/history immutable на SQL boundary; отказ history write откатывает весь transition.

## Isolation и storage lifecycle

Каждое get/events/transition проверяет principal и originating session relation в JOIN.
Foreign и missing IDs дают одинаковый unavailable error. Reattachment в другую session
пока не поддерживается; explicit principal-authorized association относится к B2.
Session prune не вызывает удаление task/run DB: session attachment не является lifetime.

DB создаётся mode 0600, новый parent directory — 0700; file symlink отклоняется.
Existing parent directory должен быть доверенным bootstrap directory. Schema version — 1;
unknown version и unrelated DB отклоняются без migration/compatibility layer.
Никакие raw tool payloads или secrets не копируются в events; reason — bounded code.
Accepted goal может содержать sensitive data, поэтому DB подчиняется private storage scope,
а не operational log permissions. Encryption-at-rest здесь не реализована.

Payload storage, retention/GC, tombstones, paging/recovery и user deletion ещё не реализованы.
Для canonical output действует target из `TOOL_EXECUTION_RESULT_CONTRACT.md`: configurable
7 дней после terminal transition, active/resumable/recovery-pinned защищены от GC,
ручное раннее удаление допускается. Duration metadata/history — отдельное open policy;
B1 не вводит автоматический GC.

## Structured Auto outcome — foundation B2a

`AgentToolsMixin.handle_auto_command` возвращает исходный `AutoRunOutcome` из
`AutoAgent.run_outcome`: status, stop reason, verifier, next steps и text остаются вместе.
Sync/stream routing явно выбирает `.text` только для существующего ответа/history/stream.
Отключённый `/auto` не входит в этот runtime; command-lane projection удалена из facade.
Проверка `tests/test_auto_runtime.py` выполняет настоящий process через registry/gateway,
передаёт diagnostics в следующий model request и проверяет typed success/failure outcome.

Это **не** production adoption store или terminal authority; store не получает runtime
observations. Граница `Agent.respond` / HTTP теперь описана ниже. `resume_auto_run` также пока
теряет typed outcome; текущий `AutoOrchestrator.resume` повторно запускает goal через
`run_v1`, вместо восстановления execution continuation. До durable recovery нужно отдельно
исправить этот путь с учётом уже совершённых effects; повторный execution нельзя выдавать
за replay. B2a не меняет resume semantics и не заявляет canonical output readiness.

## Request-local Agent response — foundation B2b

`core/agent_response.py::AgentResponse` отделяет text projection от исходного
`AutoRunOutcome`. Все sync `Agent.respond` paths возвращают envelope. У остальных paths
пока есть только text; отсутствие Auto outcome не означает success, failure или acceptance.
Существующие HTTP consumers принимают новый typed contract, без string compatibility.

Stream передаёт envelope через внутренний `ResponseProduced` event. HTTP ловит его в
локальной переменной текущего request и не отправляет как provider/UI protocol event.
Ровно один result обязателен для успешного stream; missing/duplicate event — contract error,
без sync rerun и без чтения старого `last_stream_response_raw`. Provider error и cancellation
не требуют fabricated successful result. Provider/tool-loop `Done` — transport signal, не lifecycle completion. Agent удерживает
внутренний `Done` до своей финальной projection: `ResponseProduced`, затем один Agent
`Done`. HTTP принимает результат после исчерпания iterator, а не по первому `Done`.
`last_stream_response_raw` пока остаётся runtime debug projection, не source of truth HTTP.

`_project_agent_response` — HTTP presentation boundary. Для Auto status/stop reason/verifier/
next steps берутся из typed outcome, даже если text содержит противоречащий report.
Остальные report fields и остальные execution profiles остаются legacy presentation;
их запрещено использовать как terminal authority или acceptance evidence.
Envelope не является canonical tool payload, immutable evidence record или durable run.
Store adoption, typed Ask/MWV/Desktop/approval/error observations, run identity propagation,
acceptance и безопасный continuation остаются отдельными обязательными foundation шагами.

## Следующий обязательный срез B2

1. Ввести admission boundary для реальных Ask/Auto/MWV paths с trusted scope, distinct
   request/task/run identities и явной retry/revision association.
2. Передавать run identity через runtime/tool execution context; UI остаётся projection.
3. Typed runtime observations отделить от acceptance/terminal authority; сохранить
   policy denial, approval waiting, cancellation и unknown effects без prose extraction.
4. Фиксировать result/disposition/final согласно ADR-0002/0012; не считать текст completed.
5. Проверить sync/stream/restart/retry integration и scope через реальные entrypoints.

После этого — operation intents/dispatch journaling. До подключения единой authority
не создавать второй authoritative lifecycle в Auto directories, UI snapshots или logs.
Compression не начинается раньше PR F и всех canonical-output readiness gates.
