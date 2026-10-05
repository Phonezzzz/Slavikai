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
`AutoRunOutcome`. Все sync `Agent.respond` paths возвращают envelope. B2c ниже расширяет
поле до `runtime_result`; отсутствие result не означает success, failure или acceptance.
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
Store adoption, typed MWV/Desktop/approval observations, run identity propagation,
acceptance и безопасный continuation остаются отдельными обязательными foundation шагами.

## Typed Ask observations — foundation B2c

`AgentResponse.runtime_result` содержит исходный `AutoRunOutcome`, `LLMResult` или
`AgentToolLoopResult`, отдельно от text projection. Это единственное поле для фактически
возвращённого runtime result; прежнее поле `auto_outcome` заменено. HTTP выбирает Auto projection через
тип результата. Raw result не публикуется в wire/UI протокол и не становится новым log/cache.

Ask sync сохраняет result до review, web-evidence presentation и interaction logging.
Ask stream сохраняет generator return tool loop, включая выполненные tool calls, explicit
error/cancelled observation. При отмене до получения result envelope не фабрикуется.
Отмена после получения result передаёт observation и cancelled Done, без assertion acceptance.
`ResponseFailure` отдельно фиксирует исключение generation/projection/transport; она не меняет
`ToolResult.ok` или Auto outcome. Сбой review/logging после возвращённого result сохраняет
этот result и не вызывает повторную generation/execution. Ошибка stream после tool loop
возвращает request-local envelope, Error и один Done вместо потери result через outer catch.
`ApprovalRequired` проходит через sync/stream Ask наружу к существующему approval handler;
это policy control signal, а не `provider_model_error`. До подтверждения tool не выполняется.
Эта ограниченная current-гарантия зарегистрирована как
`runtime.agent.request_local_response` version 2 в `docs/runtime_contract_claims.json`.

Это volatile boundary, **не full canonical capture или durable evidence**. LLMResult и
вложенные collections не immutable. Если provider бросил exception до возврата loop result,
его частичные observations ещё не гарантированно сохраняются; отдельные tool-stream events
не заменяют dispatch journal/canonical capture. Local web prefetch имеет отдельный legacy
observation path. MWV/Desktop и approval/decision ещё теряют typed observations в projection;
они остаются обязательным следующим B2 срезом до подключения lifecycle authority.

Проверки: `tests/test_agent_response.py` — настоящий Ask tool loop/gateway, ok/failed tool,
sync/stream, fault injection review/logging и сохранение исходного LLMResult native web path;
`tests/test_stream_events.py` — сохранение tool observations после provider Error или
cancellation, один request-local result и Done. Existing HTTP/Auto tests защищают pairing,
request isolation и отсутствие rerun после stream contract error.

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

### Native Ask approval continuation — исправление B2c

`ApprovalRequired` проходит через Ask sync/stream к существующему handler,
не превращаясь в provider failure. Ask approval report имеет `route=chat`.
`ToolLoopContinuation` capture'ит историю с model tool-call IDs, уже выполненные observations,
оставшийся batch, tools/config и точный request subject ДО dispatch. Snapshot копируется
от producer state; это внутренний scoped RAM checkpoint, не durable canonical evidence.
Agent хранит один pending continuation с opaque identity на owning principal/session instance.
Любой принятый UI turn сбрасывает snapshot до append, включая ранние ответы без Agent; отсутствующий snapshot после restart даёт unavailable,
никакой повторной generation или execution. Model/config, mode, workspace root и последний user turn должны совпадать.
После generation admission эти snapshots проверяются повторно под owning Agent lock;
изменение mode/model/root использует тот же session lock.

`chat.tool_continue` decision ссылается на checkpoint и исходный `user_message_id`.
После atomic decision claim, generation admission и owning Agent lock runtime потребляет
checkpoint и исполняет сохранённый call через `ToolGateway.call_approved_once`.
Gateway проверяет exact name/JSON args/session, заново применяет policy, включая DENY,
и создаёт permission context только для этого одного dispatch. Original Gateway/Agent/session
categories не расширяются. Остальные calls batch требуют собственных approvals, даже если
имена/args совпадают. Явное `approve_session` остаётся отдельным более широким разрешением.
После результата продолжает исходный loop с original tool call/result pairing; новый user
turn не создаётся, approval stop-text не используется для восстановления model history.
Resume публикует итоговый text stream; provider token streaming при resume здесь не заявлен.

Busy/config/snapshot failure до начала сохраняет исходный decision pending. После claim
snapshot не используется повторно; double-confirm отклоняется. Provider exception sync loop
несёт `ToolLoopExecutionError.result` с уже завершёнными calls/history, не теряя execution
observations. Provider cancellation также возвращает typed cancelled result. UI transport
успех continuation не определяет `ToolResult.ok` или lifecycle completion.
`edit_and_approve` на этом пути пока отклоняется: edits требуют нового subject/consent.

Legacy `chat.send` approval больше не вызывает ordinary send replay: без сохранённого native
continuation возвращает unavailable. Local web prefetch удалён: explicit web-search request
добавляет инструкции, а модель вызывает доступный read-tool `web` через основной native loop.
Успешный фактический web ToolResult даёт evidence; отсутствие/ошибка web call блокирует
заявления о web-доступе. Provider без native tools не получает отдельного prefetch executor.
MWV approval continuation,
durable pause/restart recovery и terminal authority остаются отдельными foundation gaps.
Gate canonical capture/durability не повышен, TinyJuice/compression не добавлены.

Проверки: `tests/test_tool_loop.py` — sync/stream pause, exact subject/tamper rejection,
same-category remaining batch, provider fault после dispatch; `tests/test_agent_response.py`
— actual Agent/Gateway resume, single use, сохранённый policy context,
review/web evidence/finalization, InteractionLog/short-term и projection fault; `tests/ui_api/test_stream_and_events.py` —
HTTP confirm, busy retry, missing snapshot, mode/model/root admission races,
HTTP cancellation до/после dispatch, web-search approval, original call ID,
один user turn, отсутствие permission leak/wire injection, post-tool provider error.


### Native Desktop approval continuation (volatile)

Desktop pause принадлежит `DesktopRuntime`: исходный tool-loop checkpoint, host lease и
browser/GUI/process resources сохраняются вместе. Execution, resume и cleanup выполняются
на одном runtime-owned потоке, в том числе для thread-affine browser resources. Пауза
ограничена 300 секундами; timeout, reject, новый user turn, выход из Desktop и Agent shutdown
инвалидируют checkpoint и освобождают lease после cleanup через ToolGateway. После cleanup
сохранённое действие не регенерируется и не исполняется.

Confirm продолжает исходный native call/history через тот же loop и Desktop verifier.
Exact request validation не обходит DesktopPolicyRuntime, DENY или descriptor checks.
Session/persistent consent применяет существующие scoped rules; отказ admission удаляет
новые session/persistent rules и сохраняет decision pending без execution. Once-grants
принадлежат exact pending ToolRequest: при нескольких action scopes накапливаются в checkpoint
до admission всего request, затем consumed/discarded после dispatch/cancel/invalidation.
Они не сохраняются как session grants и не переходят следующему tool call. HTTP добавляет только assistant
continuation к исходному user turn. Tests проверяют настоящий Agent/Gateway, file-delete и
verifier, а не fake responder, самостоятельно интерпретирующий approval rules.
Top-level decision response включает актуальные messages/output для обычного UI transport.
Desktop run/resume используют общую финализацию InteractionLog/short-term history;
post-dispatch provider failure в Ask continuation также записывает terminal response,
сохраняя typed tool observations независимо от ошибки generation.
При HTTP cancellation owned decision operation завершает transition: до dispatch pending,
после consumed checkpoint resolved с cancelled response; transport cancellation не оставляет
executing decision и не повторяет tool. Это request-task ownership, не durable run lifecycle.

Это volatile runtime ownership, не durable lifecycle adoption, canonical storage или
restart-safe continuation. После expiry UI decision возвращает unavailable при confirm;
archival/history recovery остаётся отдельной foundation dependency. TinyJuice BLOCKED.
