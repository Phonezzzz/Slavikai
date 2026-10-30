# Task/run admission store — foundation B1

Статус: storage primitive реализован и проверяется `tests/test_task_run_storage.py`.
**Production UI/API Ask/Auto используют admission/start fencing и scoped submission.** Общий lifecycle contract остаётся target;
B1 не означает durable Ask/Auto/MWV execution, continuation или recovery tool.

## Граница и authority

`core/task_run_storage.py::TaskRunStore` хранит initial accepted request revision,
отдельные task/run/attempt identities, materialized run state и immutable transition history
в одном SQLite DB. Это foundation Task/Run Lifecycle из `TASK_RUN_LIFECYCLE_CONTRACT.md`,
а не cache TinyJuice, Memory или UI session storage. Storage location задаётся доверенным
bootstrap; constructor не является recovery API и не принимает model-selected path.

`admit(scope, request_key, goal, mode, request_fingerprint)` вызывается только owning runtime после принятия
запроса. Store сам не принимает user intent, не выбирает tools и не разрешает execution.
`RunScope` происходит из verified principal/session boundary; он не является пользовательским
параметром будущего HTTP API. Обладать UUID недостаточно для доступа.

Initial task revision — 1. `task_id`, `task_run_id`, `attempt_id` генерируются независимо;
содержимое goal/session/trace не определяет identity. Initial revision сохраняет exact goal
и execution mode; дальнейшие revision/retry semantics ещё не реализованы.
Admission key unique в principal scope. Повтор exact request возвращает исходный run,
включая его текущий state; другой goal/mode/session или fingerprint с тем же key отклоняется.
Fingerprint — обязательный SHA-256 exact structured accepted request; он не заменяет goal.
Goal ограничен 1 MiB UTF-8 (включая JSON валидированных attachments); слишком большой request отклоняется до записи, не truncate'ится.
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
Existing parent directory должен быть доверенным bootstrap directory. Schema version — 4;
явно принятые additive upgrades v2→v3 (producer report) и v3→v4 (bounded tool capture)
выполняются транзакционно после проверки исходной shipped schema. Старые IDs/state/history сохраняются; неизвестная/посторонняя
schema отклоняется. Ни reset, ни backfill результата из prose/logs не выполняются.
Никакие raw tool payloads или secrets не копируются в events; reason — bounded code.
Accepted goal может содержать sensitive data, поэтому DB подчиняется private storage scope,
а не operational log permissions. Encryption-at-rest здесь не реализована.

Producer-report и scoped workspace_run capture описаны ниже; полный canonical tool payload
storage для остальных consumers, retention/GC, tombstones,
paging/recovery и user deletion ещё не реализованы.
Для canonical output действует target из `TOOL_EXECUTION_RESULT_CONTRACT.md`: configurable
7 дней после terminal transition, active/resumable/recovery-pinned защищены от GC,
ручное раннее удаление допускается. Duration metadata/history — отдельное open policy;
B1 не вводит автоматический GC.

## Durable producer submission — bounded R2a

`AgentResponse.producer_report()` явно выбирает producer kind по typed runtime variant и
копирует только публичный text + ResponseFailure code/message. Весь runtime object, history,
reasoning, tool args и arbitrary metadata не сериализуются. Report остаётся untrusted content,
не canonical tool evidence, verifier evidence или terminal acceptance. Native loop и Auto
сохраняют свои kind; ни kind, ни слова «готово» не меняют lifecycle authority.

Три existing submission seams (UI Ask/Auto, API Ask/Auto, captured approval continuation)
вызывают `submit_result(scope, run_id, expected_version, report, reason_code)`. Одна
`BEGIN IMMEDIATE` transaction сохраняет immutable envelope с submission/task/revision/run/
attempt/scope/version/time, SHA-256 UTF-8 payload и transition/history reference. Producer не
задаёт эти identities. Generic transition не создаёт новый result_submitted без payload.
SQL binding trigger и deferred FK связывают payload с exact history version; payload/history
immutable. Stale CAS, foreign scope или write failure откатывают все части. State остаётся
non-terminal; failure/cancel/budget report не получает completed из producer status.

Полный JSON envelope ограничен 1 MiB UTF-8; oversize/invalid input не truncate'ится и не
заменяется пустым report. Submission fault после runtime не доказывает отсутствия effects;
повторного execution нет. Сбой до commit может оставить running без report. Сбой после
commit до UI/API projection не уничтожает сохранённый report. Ранее streamed chunks не
становятся authoritative final из-за позднего commit.

`get_submission(scope, run_id)` повторно проверяет principal + originating session, exact
run/attempt/history binding и payload digest. Foreign/missing/unavailable дают одинаковый
unavailable error. У старого v2 result_submitted после upgrade остаётся NULL submission ref:
report явно unavailable; новых fake payload/replay нет. Старые running/admitted records при
новом submission уже обязаны иметь payload. После recovery_required/aborted исходный report
остаётся historical producer report, не новой execution authority.

Текст ответа может быть sensitive: payload хранится в той же private DB, не в event reason
или operational logs. Encryption-at-rest и новый HTTP read/replay/resume endpoint не добавлены.
Нет GC/deletion scheduler; nonterminal данные могут храниться бессрочно. Семидневный terminal
retention canonical tool output остаётся target, этот PR его не заявляет.

Проверки: `tests/test_run_submission.py`, `tests/ui_api/test_durable_submission.py`,
`tests/ui_api/test_task_run_admission.py`. Они проверяют reopen, upgrade/rollback, concurrent
CAS, immutable history/payload, scope, digest, limits, HTTP storage/projection faults,
approval resume/nested approval и post-runtime stream error. Canonical raw bytes capture,
operation intents/unknown-effect settlement, execution recovery и terminal acceptance не
реализованы этим slice; `TOOL_EXECUTION_RESULT_CONTRACT.md` readiness остаётся BLOCKED.

## Durable workspace_run capture — bounded R2b

Только admitted синхронный Auto UI/API и его captured native approval continuation
передают original immutable TaskRun binding через Agent/Auto/loop/Gateway в qualified
ComputerBackend. UI stream доставляет результат того же синхронного Auto execution.
Binding не берётся из model args, auto-* ID или глобального mutable state. Manual
category-only Run и все остальные I0 process paths остаются закрыты.

После policy/path/workspace/environment/revision guards непосредственно перед start
qualified container Store атомарно создаёт operation intent с независимыми opaque
operation_id/tool_attempt_id и original run/attempt/version. Model tool-call ID — только
label. Missing/stale/foreign intent или write fault запрещает start. Сохранённый intent
сам по себе не доказывает execution: crash до/после start остаётся unresolved, без replay.

Единственный существующий backend collector наблюдает separate raw stdout/stderr chunks
до backend view truncation/decode. Retained prefix ограничен 1MiB/stream; observed counts
и EOF раздельны. Existing configured output/timeout/settlement guards не ослаблены:
при их stop сохраняется честный partial/unknown, а не выдуманный full stream/child exit.
Actual child exit отличается от CLI/transport; публикация workspace остаётся отдельной
проверенной boundary. Outcome/publication не переписываются ошибкой записи capture.

До новой lossy ToolResult/model projection original intent получает одну immutable
observation: BLOB prefixes, counts/EOF, actual exit либо unknown, timeout/capture error,
SHA-256 digests, timestamp и opaque result_id. Transaction fault не публикует ref на
отсутствующие bytes; дальнейшие batch/model calls останавливаются без replay. Доступный
RAM outcome не объявляется persisted. Late observation допустима как historical evidence
исходного intent после abort, но не resurrect run и не даёт следующий operation.

Только после commit ToolResult получает opaque refs/completeness и отдельные UTF-8 views
stdout/stderr до32KiB encoded bytes/stream (replacement только в view). Более низкий
existing backend view limit сохраняется. Raw bytes не попадают в model JSON, events или
operational logs. Scoped Python readers проверяют principal/session/run/operation/attempt,
original policy/history binding и digests; foreign/missing одинаково unavailable.

Additive schema4 сохраняет все прежние reports/IDs/history, без fabricated tool evidence
для старых runs. Store не принимает completion/acceptance, не восстанавливает process и
не предоставляет новый HTTP read/replay endpoint. До2MiB/attempt private sensitive bytes
могут храниться бессрочно: aggregate DB quota/retention/GC/deletion этим не добавлены.
Shell/terminal/PTY/Desktop/Ask/Act/MWV/direct HTTP tools, полный canonical capture и
recovery остаются uncovered/target. Production activation/upgrade здесь не выполняются.

Проверки: tests/test_tool_capture_storage.py, tests/test_tool_capture_streams.py,
tests/test_tool_capture_loop.py, tests/test_workspace_capture_projection.py,
tests/test_workspace_python_execution.py, tests/ui_api/test_qualified_workspace_capture.py.
Pytest заменяет только OS/engine boundary и не доказывает OS isolation; real synthetic
qualified byte/HTTP/once/timeout/output-limit/post-effect-store-fault proofs выполняются
отдельно. Producer report R2a остаётся отдельным artifact, не заменяет tool observation.

## Structured Auto outcome — foundation B2a

`AgentToolsMixin.handle_auto_command` возвращает исходный `AutoRunOutcome` из
`AutoAgent.run_outcome`: status, stop reason, verifier, next steps и text остаются вместе.
Sync/stream routing явно выбирает `.text` только для существующего ответа/history/stream.
Отключённый `/auto` не входит в этот runtime; command-lane projection удалена из facade.
Проверка `tests/test_auto_runtime.py` выполняет настоящий process через registry/gateway,
передаёт diagnostics в следующий model request и проверяет typed success/failure outcome.

Это **не** terminal authority; ограниченное production admission описано ниже. Store не хранит runtime
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
Streaming dispatch записывает завершённый ToolResult и исходный call ID до проверки
cancellation, как sync path. Следующий tool не выполняется. HTTP при cancellation
дочитывает owning Agent iterator до request-local ResponseProduced либо его завершения;
отменённые text/tool transport events не публикуются. Полученный typed result остаётся
request-local и не добавляется в wire payload или durable storage.
`ResponseFailure` отдельно фиксирует исключение generation/projection/transport; она не меняет
`ToolResult.ok` или Auto outcome. Сбой review/logging после возвращённого result сохраняет
этот result и не вызывает повторную generation/execution. Ошибка stream после tool loop
возвращает request-local envelope, Error и один Done вместо потери result через outer catch.
`ApprovalRequired` проходит через sync/stream Ask наружу к существующему approval handler;
это policy control signal, а не `provider_model_error`. До подтверждения tool не выполняется.
Эта ограниченная current-гарантия зарегистрирована как
`runtime.agent.request_local_response` version 6 в `docs/runtime_contract_claims.json`.

Это volatile boundary, **не full canonical capture или durable evidence**. LLMResult и
вложенные collections не immutable. При provider exception во время создания или чтения
stream loop возвращает ранее выполненные observations через ToolLoopExecutionError;
Agent сохраняет их в request-local error response без повторного dispatch. GenerationCancelled
сохраняет их в cancelled loop result, как sync path. Незавершённый provider tool call не
становится execution observation. Другие exceptional dispatch/serialization paths и durable
capture ещё не гарантированы; отдельные tool-stream events не заменяют dispatch journal/canonical
capture. Local web prefetch удалён; native approval
continuation описан ниже. MWV/Desktop ещё не имеют полного typed observation contract;
они остаются обязательным следующим B2 срезом до подключения lifecycle authority.

Проверки: `tests/test_agent_response.py` — настоящий Ask tool loop/gateway, ok/failed tool,
sync/stream, fault injection review/logging и сохранение исходного LLMResult native web path;
`tests/test_stream_events.py` — сохранение tool observations после provider Error или
cancellation, один request-local result и Done. Existing HTTP/Auto tests защищают pairing,
request isolation и отсутствие rerun после stream contract error.
`tests/test_tool_loop.py` проверяет отмену внутри реального Gateway dispatch для ok/failed
result, diagnostics/meta, original call ID/history и отсутствие второго dispatch в sync/stream.
`tests/ui_api/test_stream_and_events.py` проверяет тот же путь через настоящий Agent/HTTP:
typed observation дочитывается transport consumer, а response остаётся cancelled.

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

## Typed MWV observation — foundation B2e

`_run_mwv_flow` возвращает `AgentResponse.runtime_result` с исходным
`ManagerRuntime.run_flow` → `MWVRunResult`. Sync routing передаёт envelope напрямую;
stream передаёт тот же envelope через `ResponseProduced` до `Done`. Ошибки formatting,
decision presentation или InteractionLog после manager return сохраняют результат с
отдельным `ResponseFailure(mwv_projection_error)`; tools не запускаются повторно.
Worker failure и verifier PASS остаются разными typed facts; presentation не определяет
execution success. ApprovalRequired остаётся отдельным control signal внешнего handler.

Проверка `tests/test_mwv_response_observation.py` исполняет explicit TaskStepContract через
настоящий Worker/Gateway, затем настоящий `make check` verifier и HTTP stream iterator.
Проверяются обе комбинации worker/verifier, bounded retry, identity manager result и faults
представления. Это volatile observation последнего returned attempt, не canonical payload:
предыдущие attempts, exception до manager return, approval pause и durable lifecycle
остаются foundation gaps. HTTP report вне Auto пока legacy presentation, не authority.
TinyJuice/recovery readiness не повышен.

## Typed Desktop outcome — foundation result boundary

Initial Desktop и native approval resume передают исходный `DesktopRunOutcome`
в `AgentResponse.runtime_result`, включая `verification` и `loop_result`. Presentation
и InteractionLog используют общую `_finalize_desktop_response`; её ошибка сохраняет outcome
с отдельным `ResponseFailure(desktop_projection_error)`. Stream передаёт тот же envelope
через `ResponseProduced`, а presentation error — через `Error` и error `Done`.
`ApprovalRequired` остаётся control signal, не presentation failure.

`tests/ui_api/test_desktop_response_observation.py` использует настоящий Agent, DesktopRuntime,
Gateway, file-delete и verifier, включая initial sync/HTTP stream iterator, approval resume,
provider failure после dispatch, failed tool и presentation/logging faults. Проверяются
identity исходного outcome, original call IDs, tool/verifier facts и освобождение host lease.
Это volatile boundary, не canonical payload/attempt history/restart recovery. Resource owner,
policy/approval и verifier не заменяются новым runtime; readiness TinyJuice не повышается.

## Native Auto approval continuation — foundation

Auto хранит native `ToolLoopContinuation` вместе с original run frame: run identity,
plan, skill, budgets и monotonic start. Resume продолжает exact pending call через тот же
AgentToolLoop/Gateway, не вызывает повторно initial goal и не повторяет completed calls.
Principal/session, mode, runtime root, Brain и config проверяются до consumption checkpoint.
Предварительно отменённый resume оставляет checkpoint невыполненным. HTTP использует общий
native approval controller: approve_once не превращается в category grant для всего batch.
Legacy `auto.run` decision возвращает unavailable; UI state не создаёт execution authority.

`tests/test_auto_native_continuation.py` проверяет настоящий Agent/Gateway, original call IDs,
однократность, сохранение plan/start/run identity, scope mismatch и cancellation до dispatch.
`tests/ui_api/test_auto_native_approval.py` проверяет nested approvals одного risk category
через настоящий HTTP/runtime и duplicate confirmation. Это volatile continuation, не durable
run lifecycle или canonical payload storage. Existing budget enforcement остаётся прежним;
этот срез устраняет reset budgets при resume, не вводит новый dispatch budget controller.

Auto resume перечитывает effective session security под существующим Agent lock до claim;
legacy decision отклоняется до session grant. Отмена до claim сохраняет native approval handle,
reject публикует cancelled projection и финализирует skill state. Native resume выгружает
накопленный progress. Verifier принимает cancellation callback: принадлежащая ему process
group останавливается и дожидается завершения с сохранением доступных stdout/stderr;
Auto проверяет cancellation после tool loop и verifier и не публикует completed при отмене.
Проверки: `tests/test_auto_native_continuation.py`, `tests/test_verifier_runtime.py`,
`tests/ui_api/test_auto_native_approval.py`, `tests/ui_api/test_decision_and_approval.py`.

Session security writer и native claim используют один Agent lock. Session grant при отказе
до claim откатывает только добавленные categories под тем же lock; прежние grants сохраняются.
Pending Auto `/chat/cancel` также синхронизирует workflow/progress. Cancellation перед nested
approval не оставляет новый checkpoint. Drain verifier pipes после остановки owning group
ограничен: escaped descendant не удерживает worker бесконечно; при неполном capture диагностика
явно содержит `verifier pipe capture incomplete`. Это не гарантия полного canonical capture.


## Volatile Act packet approval continuation — foundation

`Agent.run_task_packet` при остановке сохраняет Agent-owned `MWVWorkerCheckpoint`:
исходный packet, context/attempt, exact approval request, выполненный prefix, changes,
счётчики и доступные typed `ToolRequest/ToolResult` observations. UI получает только
opaque checkpoint ID и presentation snapshots. `TaskPacket.context.plan_runner_resume`
отклоняется; HTTP не пересчитывает packet hash ради подстановки выполненных шагов.

Resume проверяет packet, principal/session, mode, root и original brain/config, затем
single-use claim продолжает worker и Manager с исходным attempt. `approve_once` вызывает
`ToolGateway.call_approved_once` только для остановленного request; следующий request
проходит обычную approval проверку. Missing checkpoint/restart возвращает unavailable без
rerun. До dispatch failure сохраняет checkpoint и pending decision; после исполненного
request результат сохраняется отдельно от text/diff projection и не повторяется автоматически.
`MWVRunResult.tool_observations` содержит доступные results, включая failed ToolResult.

HTTP использует existing scoped Agent lock для актуальной workflow/model/security проверки,
claim, session grant и публикации projection. Незапущенный resume откатывает только новые
session categories. Original brain на resume не пересоздаётся. Reject/reset/Agent close
очищают pending checkpoints. Отмена approval request передаётся worker и существующему
cancellable verifier; выполненный ToolResult сохраняется до следующей cancellation проверки.

Проверки: `tests/test_mwv_native_continuation.py` — настоящий worker/Gateway/Manager/verifier,
prefix без повторов, отдельные once approvals, original packet, duplicate claim, pre-dispatch
fault, чужая session, cancellation и post-call projection fault. `tests/ui_api/test_modes_and_plan.py`
проверяет actual Agent reconfiguration/runtime, once/session approvals, rollback с сохранением
старого grant, смену root, reject и explicit forced reset.

Это **volatile Act packet runner**, не durable task/run lifecycle или full byte capture.
Checkpoint не восстанавливается после restart. Existing tool/verifier caps, persisted payloads,
acceptance authority, full retry history и budget enforcement остаются отдельными foundation
задачами. Legacy routed MWV не получает новой гарантии через эту approval boundary.
TinyJuice readiness не повышается.

При завершении Act packet frame scoped Agent удерживает последний typed
`MWVRunResult` в `last_mwv_result` до следующего terminal result или уничтожения Agent.
Это ограниченный volatile handoff, а не durable history или retention store.
Ошибка terminal UI projection не возвращает consumed approval в pending; доступные
`tool_observations` остаются в этом результате. Ошибка публикации логируется;
при недоступном UI storage сохранение terminal projection не гарантируется.
Отмена waiting plan инвалидирует его checkpoint и approval под scoped Agent lock.

Checkpoint хранит исходный packet для exact lineage validation и отдельный фактический
retry packet/context. Продолжение использует owned retry revision; exceptional settlement
сохраняет фактически активные attempt/revision. Изменённый исходный packet отклоняется.
Waiting workflow и decision публикуются одной записью UI storage; при ошибке записи
UIHub откатывает in-memory state, runner удаляет новый checkpoint и публикует failure.
Reject заимствует существующий scoped owner без model resolution/reconfiguration.
Terminal workflow и rejected decision фиксируются одной atomic записью с проверкой
текущих decision identity/status; checkpoint удаляется только после commit. Storage fault
оставляет прежний waiting state и checkpoint доступными для повторного явного решения.
Cancel waiting plan использует ту же atomic CAS-запись terminal workflow и удаления
decision; owned checkpoint удаляется только после успешного commit.

Initial и resumed packet runner регистрируют matching `(principal, session, task_id)`
cancellation token до ожидания Agent lock. Plan cancel сигнализирует token до lock,
затем изменяет только matching всё ещё active task/plan; terminal result не переписывается.
Отмена cooperative: завершённый текущий ToolResult сохраняется, следующие calls останавливаются,
VerifierRuntime прекращает собственный process group. Уже совершённые tool side effects
не откатываются. Отмена до первого dispatch, включая ожидание Agent lock,
проецируется как cancelled. Timeout ожидания runner при shutdown логируется;
остальные cleanup callbacks продолжаются. ScopedAgentProvider удерживает deferred
retirement до фактического выхода borrowers и освобождения Agent lock, включая static
owner; timeout не закрывает Agent ресурсы во время исполнения. Отмена asyncio runner
сигнализирует worker и дожидается thread frame до освобождения lock/borrow. Пока opaque
tool не завершился кооперативно, retirement остаётся pending; принудительное завершение
процесса не даёт новой durability/recovery гарантии. HTTP registry является volatile control
handle, не lifecycle authority.

Cancellation внутри worker сохраняет накопленные completed step results, доступные changes,
file/diff/tool counters и typed observations. Terminal UI проецирует завершённые шаги,
а не возвращает их в todo/waiting_approval.

После committed approval сбой event buffer/delivery логируется отдельно и не инвалидирует
checkpoint: matching state остаётся доступен через UI state API.


## Первое production admission: UI Ask

Server bootstrap создаёт отдельный `.run/task_runs.db`, независимо от Memory и UI session
pruning. UI Ask после validation и root approval, под existing scoped Agent lock, сохраняет
admission и CAS-переход `admitted -> running` до изменения pending continuation и dispatch.
Principal/session берутся из authenticated boundary. Key связывает endpoint/session/client key;
без client key генерируется отдельная request identity. Fingerprint связывает content, attachments,
mode, полную execution model config, grants, effective tools/policy, hash фактических credentials, root и options.
Raw credentials не сохраняются в DB или operational logs. Повтор ключа с другим binding отклоняется. Уже принятый
Ask key также нельзя использовать для dispatch в другом UI режиме. Mode/model/root/grants
проверяются повторно под lock до admission и dispatch; перед dispatch проверяется неизменность
history snapshot, снятого до admission. UIHub под своим lock проверяет исходную историю,
добавляет user message и возвращает snapshot после штатного pruning общего chat/workspace
лимита; повторная проверка перед Agent защищает от последующих mutations. History ещё не
является durable execution checkpoint. Regex web-intent guidance удалена: обычный текстовый
запрос проходит через основной Agent с typed result, без локального command fallback.
Sync/stream Ask строят model context из переданного request history snapshot, а не
накопленного short-term projection. Debug command lane `/...` не является model run:
его локальные ответы не создают admission; принятый Ask key нельзя использовать для команды. Canonical runtime state application получает принятый effective security
snapshot вместо повторной загрузки новых tools после проверки.

UI admission требует атомарного `require_new=True`: existing admitted run тоже нельзя claim'ить
без durable execution context. Проверка проходит внутри transaction, а не предварительным lookup.
При потере volatile HTTP replay cache любой ранее принятый run возвращает
`task_run_continuation_unavailable` до вызова Agent; implicit rerun запрещён. Конкурирующий
CAS claim также не dispatch'ится. DB admission/start fault останавливает request без fallback.
Существующий HTTP replay остаётся presentation cache и не является lifecycle authority.

Доступный typed Agent result переводит run в `result_submitted` до text projection.
Это не acceptance/completion и не durable result capture. При fault/cancellation до submission
run может остаться running/unresolved; автоматической recovery или повторного исполнения нет.
Initial UI Ask approval сохраняет immutable TaskRun binding в принадлежащем Agent native
checkpoint. Перед resume owning HTTP runtime проверяет exact principal/session, state/version
и запись store до category grant/tool dispatch; отсутствующий, foreign или stale run даёт unavailable.
Следующее approval переносит тот же binding. Typed result после resume переводит исходный run
в `result_submitted` до UI/text projection, без нового admission. Это volatile checkpoint binding,
не durable continuation/recovery и не terminal acceptance. Store submission fault после tool effect
оставляет run unresolved; single-use checkpoint не разрешает implicit повтор execution. Project, Desktop и Act admission остаются непокрытыми.
Для `/v1` session grants читаются и применяются только под owning Agent lock перед
dispatch; ошибка context installation останавливает запрос без вызова Agent.
`/v1/chat/completions` теперь использует Ask по умолчанию, а не наследует прежний Agent mode.
Default и explicit Ask имеют один admission/start/submission path.
Без explicit session id API не читает UIHub session: tools берутся из global settings, policy — sandbox, root — canonical workspace. `__principal__` обозначает только scoped runtime/store identity и не используется как UI session lookup. Explicit session требует ownership до чтения context. Fingerprint связывает весь
wire request (включая history), применённый ModelConfig/credential hash, grants, effective tools/policy
и workspace root. Context сверяется перед admission и после применения к Actor перед dispatch; root закрепляется
на dispatch. Admission/claim выполняются до borrow/reconfiguration/context mutation Actor,
поэтому отказ повтора не меняет режим и checkpoint последующего run. API больше не вызывает
старый resolver, который перенастраивал Actor до owning admission.
API без выбранного runtime ModelConfig возвращает `model_not_selected`, даже если bootstrap
передал готовый Agent; исполнения с неявной прошлой конфигурацией больше нет.
`Idempotency-Key` нормализуется общим HTTP helper и namespaces по endpoint/session;
повтор любого committed admission возвращает 409 без повторного model/tool call, даже после
reconstruction приложения. Изменённый binding даёт `idempotency_key_reused`, совпавший —
`task_run_continuation_unavailable`. Без ключа каждая отправка — отдельный run.
Model result переводит тот же run в `result_submitted` до presentation. Debug `/...` не создаёт
model run, принятый Ask key нельзя использовать для команды или Auto. Старый API branch без
runtime setter/default mode удалён; API требует canonical runtime interface. UI workflow mode,
active_plan/task/auto_state больше не применяются к API execution: режим задаётся только request,
permissions/root — принятым snapshot через typed Agent setters. Повтор не сбрасывает native checkpoint. Native Ask approval
сохраняет owned run binding, но публичного `/v1` confirmation/recovery API этот срез не добавляет.
Result byte capture, terminal acceptance и durable recovery остаются target.
Task terminal acceptance/final, evidence references, recovery controller, full capture,
retention/GC и user-visible run inspection остаются обязательными следующими срезами.

Schema v1 storage primitive не имел production consumers; v2 требует request fingerprint.
Existing v1 DB отклоняется, без implicit migration/delete. Тесты нового production пути:
`tests/ui_api/test_task_run_admission.py` — actual Agent, reopen SQLite/RAM replay loss,
changed structured binding, admission/claim storage faults и запрет нового model invocation.
Также проверяются fresh app/Agent с теми же SQLite stores и actual Gateway call, сохранение
pending approval при дубликате, смена режима и context race до admission/dispatch.
Это app reconstruction, не crash/OS process recovery или общий runtime lifecycle guarantee.

Штатный UI передаёт устойчивый `Idempotency-Key`. До dispatch browser сохраняет только
pending request UUID и SHA-256 payload hash; повтор после response loss/reload сохраняет key.
Changed payload (включая потерянные transient modifiers) не заменяет неопределённый key:
до новой отправки требуется explicit discard.
успешно применённый ответ освобождает key для следующей новой отправки. Explicit regeneration
передаёт immutable assistant message identity через `If-Match`; hub под lock удаляет только
соответствующую последнюю пару. Уже отсутствующая identity — successful no-op даже после
restart/lost DELETE response; существующая непоследняя identity возвращает conflict.
После подтверждённого удаления UI не повторяет deletion на retry send.
После ошибочного HTTP send response пользователь может явно отбросить pending key с предупреждением
о неопределённом результате; это не dispatch. Новая отправка требует отдельного действия. Message/attachment bytes
не копируются в этот browser identity cache. Тест actual useSessionTransport проверяет HTTP
header и повтор после lost response. Test bootstrap использует временные per-app stores;
production DB не открывается UI test fixtures.

## UI Auto admission и native run binding

`/ui/api/chat/send` в режиме Auto использует тот же scoped admission/CAS и принятый
request/config/security/history fence, что и Ask. Debug command lane не создаёт model run.
Повтор committed request после RAM loss/reconstruction не исполняется заново; смена режима
не позволяет повторно использовать тот же ключ. Storage fault до claim останавливает dispatch.
Native Auto checkpoint сохраняет исходный TaskRun binding рядом с уже существующим
owned Auto frame/continuation; никаких UI state replay или нового Auto run при confirm.
HTTP resume сверяет точный RunScope и текущую state/version до добавления permissions
и вызова native resume. Следующее approval переносит тот же binding; typed final result
переходит в result_submitted исходного run до presentation. Это не terminal acceptance,
полный durable execution context или restart recovery. Прочие entrypoints и capture
остаются отдельными обязательными срезами.

Auto root approval сохраняет исходную durable request identity ещё до admission и передаёт её
из сохранённого checkpoint при confirm. HTTP decision request не заменяет identity исходного
send; source request сохраняет web_search. Missing/corrupt identity завершает confirm ошибкой
`task_run_continuation_unavailable` до изменения root/decision и dispatch, без генерации нового ключа.
Root approval не является выполнением TaskRun и не разрешает replay уже принятого request.

Root confirm сверяет сохранённые mode/model snapshots под owning Agent lock до изменения root
и decision; resumed admission повторно требует тот же mode/model context. Native Ask/Auto
checkpoint хранит immutable fingerprint принятого effective tools/policy snapshot. HTTP resume
сверяет его до применения runtime state, выдачи grants и вызова continuation; применяет только
проверенный snapshot. Nested approval переносит тот же fingerprint вместе с TaskRun binding.
Изменение session tools/policy требует нового явного run, а не расширяет права старого.

## API Auto admission

Явный `/v1` Auto использует тот же admission/CAS/submission path, что default/explicit Ask.
Mode входит в immutable binding: повтор после reconstruction и смена режима с тем же
ключом не вызывают модель/tools заново. Storage fault останавливает dispatch.
Native Auto checkpoint получает исходный scoped TaskRun и принятый security/grants snapshot;
повтор HTTP-запроса сохраняет существующий checkpoint. Публичного API resume endpoint,
канонического capture, terminal acceptance или restart recovery этот срез не добавляет.
Проверка: `tests/ui_api/test_api_task_run_admission.py` — реальные Agent/ToolGateway,
повтор с reopen store/app, storage faults и security context races для Ask/Auto.
