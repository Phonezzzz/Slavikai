# Task / Run Lifecycle — audit, research и alternatives

Дата исследования: 2026-09-08.

**Статус: research/design rationale, не current runtime contract.** Нормативная target-модель
находится в [`TASK_RUN_LIFECYCLE_CONTRACT.md`](TASK_RUN_LIFECYCLE_CONTRACT.md). Документ не
выбирает database, broker, workflow engine, scheduler, API, persisted enum или process topology и
не утверждает, что durable lifecycle уже реализован.

## 1. Метод и repository surfaces

Исследование проведено target-first:

1. из Context, Memory, Multi-Agent Coordination и ARCH_CANON выведены обязательные boundaries;
2. проверены фактические execution/control flows, а не только classes со словом `Task`;
3. current runtime использован как evidence и источник reuse opportunities, но не как constraint;
4. current failures, cancellation, approval resume и persistence сопоставлены с tests;
5. изучены primary/official sources по durable workflows, reconciliation, leases, actor
   supervision, delivery guarantees и long-running agents;
6. сравнены alternatives и каждой релевантной current abstraction дан explicit verdict.

Проверены repository surfaces:

- `docs/SOURCE_OF_TRUTH.md`, `ARCH_CANON.md`, `Architecture.md`, architecture index и claims;
- Context, Memory и Multi-Agent Coordination contracts/research;
- фактический MWV guide: `docs/agent/MWV_FLOW.md` (пути
  `docs/architecture/MWV_FLOW.md` в repository нет);
- `core/mwv/*`, `core/agent_mwv.py`, `core/agent_routing.py`, `core/agent_tools.py`;
- `core/auto_runtime.py`, `shared/auto_models.py`, `core/tool_loop.py`;
- `core/desktop_runtime.py`, policy/approval и decision models;
- Plan/Act handlers и `server/http/common/workflow_runtime.py`;
- `UIHub`, SQLite UI-session storage, scoped Agent provider, cancellation/idempotency registries;
- tests MWV, Auto, Plan/Act, approvals, cancellation, principal/session persistence и Desktop.

Жёсткие boundaries:

- `principal_id` остаётся security boundary;
- session/context/UI/log не являются lifecycle source of truth;
- private reasoning не требуется для recovery;
- policy, approval и ToolGateway enforcement не делегируются model prose;
- coordination events и long-term Memory остаются отдельными subsystems;
- Plan может адаптироваться во время Act только через versioned control-plane transaction.

## 2. Current identities и фактический state ownership

Единой current lifecycle aggregate нет. Несколько путей используют несовместимые identities:

| Current path | Identity | Где живёт state | Фактическая durability |
| --- | --- | --- | --- |
| Ask/chat | `session_id`, `stream_id`, trace | call stack, Agent, UIHub | messages/session snapshot durable; execution нет |
| Plan | `plan_id`, mutable `plan_revision` | `active_plan` UI snapshot | SQLite session snapshot |
| Act/MWV runner | UI `task_id` + отдельный packet `task_id` | `active_task`, call stack | snapshot durable; runner/attempt нет |
| Direct MWV | `TaskPacket.task_id`, `RunContext.attempt` | synchronous stack | process-local |
| Auto v1 | `run_id` | Agent `last_auto_state`, `_paused_runs` | UI snapshot durable; resumable state process-local |
| Desktop | session/stream + host mutex | Agent/tool loop/process handles | process-local |
| Approval | `DecisionPacket.id`, session scope | UI snapshot + process-local grant store | decision snapshot durable; grants process-local |
| Tool loop | provider `tool_call.id` | loop history/list | process-local |

`TaskPacket` содержит `task_id`, `session_id`, `trace_id` и `packet_revision`, но не различает
logical task, task revision, run, subtask и attempt. `RunContext.attempt` — целое число текущего
MWV loop, не durable identity. `AgentScope(principal_id, session_id)` изолирует mutable Agent, но
не задаёт task/run/agent-participation ownership.

В Plan execute подтверждено identity split: handler создаёт UI `active_task.task_id`, после чего
`compile_plan_to_task_packet(...)` независимо создаёт другой `TaskPacket.task_id`. Runner
выбирается по UI ID, а `WorkResult.task_id` и packet hash относятся к packet ID. Это не просто
неполная metadata, а противоречивая identity boundary.

## 3. Current lifecycle maps

### 3.1. Ask

~~~text
HTTP send + session/stream id
  -> process-local cancellation registration
  -> Agent.respond[_stream] under principal/session Agent lock
  -> optional model/tool-loop control flow
  -> assistant message + UI snapshot persistence
  -> cancellation registration removed
~~~

Ask не создаёт logical task/run record и не имеет authoritative task state. Ошибка превращается
в HTTP/UI error, cancellation — в response field `cancelled=true`, а normal assistant message
фактически завершает request. Session messages переживают restart, execution incarnation — нет.

### 3.2. Plan -> Act

~~~text
Plan draft(plan_id, revision=1, status=draft)
  -> edits/approve mutate snapshot and increment plan_revision
  -> execute creates UI task_id
  -> compiler independently creates packet task_id
  -> UIHub atomically checks mode/plan revision and stores running snapshot
  -> bare asyncio.create_task(plan runner)
  -> synchronous MWV execution in thread
  -> work success + verifier passed => plan/task completed
     otherwise => plan/task failed
~~~

Положительный reuse signal — `start_plan_task_if_possible` выполняет session-local guarded
compare-and-set under lock. Но `plan_revision` повышается и при content edit, и при status-only
transition (`approved`, `running`, `completed`, `failed`), поэтому content revision и lifecycle
version смешаны.

Approval во время шага сохраняет completed step summaries и diff counters внутрь нового packet
hash, оставляет task status `running`, а `execution.status` выставляет `waiting_approval`.
После approval создаётся новый process-local runner. Это bounded resume hint, но не canonical
checkpoint/action journal.

`/plan/cancel` только записывает `cancelled` в UI snapshots. Он не посылает cancellation token
runner'у и не повышает fencing epoch. Уже запущенный runner не перечитывает cancellation перед
финальной записью и может записать `completed`/`failed` поверх `cancelled`. Tests cancellation
этого механизма не найдено.

### 3.3. Auto v1

~~~text
run_id -> idle -> planning -> coding -> verifying -> completed
                                |             |
                                |             -> failed_verifier
                                -> failed_worker / waiting_approval
          budget/internal exception -> failed_internal
~~~

`AutoRunStatus` является одним плоским enum, который смешивает phase, wait condition и failure
category. Current Auto — один native `AgentToolLoop`, несмотря на сохраняемые `pool_size`, shards,
coders и merge structures.

Approval кладёт `_PausedRun` только в process-local dict. `resume(run_id)` повторно вызывает
`run_v1` с тем же run ID и goal, то есть заново строит plan и заново запускает tool loop, а не
продолжает с durable action boundary. Restart теряет `_PausedRun`. `cancel(run_id)` умеет отменить
только paused Auto, не active execution. Budget exhaustion классифицируется как
`failed_internal`, а не отдельный authoritative stop reason.

### 3.4. MWV

~~~text
build one TaskPacket
  -> worker(attempt=N)
  -> verifier(attempt=N)
  -> success, либо retry decision
  -> retry mutates packet_revision/constraints and repeats
~~~

Retry сохраняет `task_id`, что правильно как зачаток logical continuity, но не создаёт stable
`attempt_id` или history record. Retry разрешён только для verifier failure; worker failure и
verifier error останавливают flow. `packet_revision` используется как correction/retry carrier,
хотя target Plan revision, packet snapshot revision и attempt — разные concepts.

Worker `success` + verifier `passed` сразу становится terminal success. Отдельных
`result_submitted`, acceptance authority и verification attempt identity нет.

### 3.5. Background/long-running execution

Plan runner, approval resume и canvas publishing запускаются через bare `asyncio.create_task`.
Task handles, owner identity, lease, heartbeat и checkpoint не сохраняются. Application cleanup
явно останавливает chat generations, terminal manager, embedding download manager и Agents, но
не имеет lifecycle recovery controller для Plan/Auto runs.

SQLite UI session может восстановить `active_plan`, `active_task` и `auto_state`, однако
`UIHub._restore_sessions` не reclaims/reconciles runners. После restart snapshot способен
показывать `running`/`waiting_approval`, когда соответствующего executor/paused-run уже нет.

### 3.6. Failed tool call

`AgentToolLoop` dispatches provider tool call через `ToolGateway`, затем сохраняет `ToolResult` в
process-local loop history. Ошибка может вызвать следующий model iteration, Auto failure или MWV
step failure. Нет durable operation intent, action attempt state или общей taxonomy retriable /
non-retriable / unknown outcome.

### 3.7. User cancellation

Chat/Desktop используют process-local cancellation token с checks между model/tool boundaries.
Запущенный synchronous tool generic contract не обязан прерываться мгновенно. Desktop отдельно
пытается cleanup retained launched processes. Plan cancellation, напротив, только меняет UI
snapshot. Ни один путь не имеет durable cancellation revision, которая fencing'ит late result.

### 3.8. Disconnect и restart

Client disconnect не является explicit task transition. Chat request lifetime, stream и current
Agent borrower связаны с process execution; session messages сохраняются отдельно. Process или
machine restart сохраняет UI/session snapshots, но теряет current call stacks, asyncio tasks,
cancellation tokens, Auto paused map, Agent instances, host mutex и launched process handles.

### 3.9. Retry

Существуют три несогласованных механизма:

- provider retry с backoff (`llm.retry`) внутри одного model request;
- MWV verifier retry через новый packet revision;
- user decision `retry`, который replay'ит source request через chat endpoint.

HTTP `IdempotencyStore` уменьшает duplicate requests в пределах session/endpoint/key, но он
process-local, имеет 90-second window и не является durable semantic operation identity.

### 3.10. Verifier rejection

MWV может выполнить bounded retry; Auto завершает run как `failed_verifier`; Plan runner помечает
task и plan `failed`. Нет общего transition `result_submitted -> rejected -> rework`, отдельного
verification attempt или способа оставить logical task active после terminal run failure.

## 4. Implicit lifecycle и authority

Current lifecycle существует преимущественно в call stack, loops и exceptions:

- начало/конец Ask определяется HTTP coroutine;
- MWV attempt — iteration counter;
- Auto pause — наличие `_PausedRun` в dict;
- active Plan runner — факт существования untracked asyncio task;
- cancellation — process-local event;
- tool outcome — возвращённый `ToolResult`, без started/unknown state;
- completion — branch, который записал final snapshot/message;
- agent ownership — principal/session lock или host mutex, без durable epoch.

Authority также распределена неявно: handlers, Agent, MWV manager, verifier и UIHub могут каждый
записать status своей формы. Model prose не является formal authority, но единого trusted
lifecycle transition boundary нет.

`STOP_TO_CHAT` и DecisionPacket дают полезный typed user-facing reason/choice contract, однако
сейчас STOP в основном является report payload, а не durable run condition/transition. Наличие
STOP response поэтому не доказывает, что task/run можно безопасно восстановить или resume.

## 5. Failure, recovery и external-side-effect audit

Current failure labels покрывают некоторые user-visible причины, но не дают composition rules.
Tool failure, verifier rejection, model error, budget exhaustion и internal exception либо
сливаются в `failed`, либо зашиты в Auto terminal enum. Approval denial Plan path также может
стать `failed`, хотя denial, policy block и execution fault имеют разные recovery semantics.

Критические gaps:

- timeout/crash после external call не моделируется как `unknown outcome`;
- retry не связан со stable action intent/idempotency identity;
- durable checkpoint не фиксирует policy/task/plan revisions, cancellation epoch, ownership,
  pending side effects и accepted evidence как единый recovery boundary;
- restored UI snapshot не доказывает, что owner жив;
- stale worker/result не fenced от cancelled/superseded state;
- final model text и task completion не имеют authoritative ordering contract;
- log/UI event stream не может восстановить state и не является transition history.

## 6. External primary-source findings

### 6.1. Durable execution и identity layers

Temporal различает Workflow ID и уникальный Run ID: одна logical execution может иметь несколько
runs из-за retry/continue-as-new. Отдельно Activity Execution включает chain конкретных Activity
Task attempts. Это поддерживает separation `task_id` / `task_run_id` / `attempt_id`, но SlavikAI
не принимает Temporal persistence/replay model автоматически. [Temporal workflow failures][temporal-workflow],
[Temporal activities][temporal-activity].

Temporal также подчёркивает, что failed workflow task не обязан завершать workflow, а Worker
timeout нужен для takeover другим worker. Это прямое evidence против propagation
`attempt failure -> task failure` и против process ownership как durable truth.

### 6.2. Retry, idempotency и uncertain effects

Temporal отделяет deterministic orchestration от failure-prone Activities и применяет retry
policy к Activities, а не indiscriminately ко всему workflow. Потерянная Activity после dispatch
обнаруживается timeout'ом и может повториться. [Temporal retry policies][temporal-retry].

AWS Durable Execution прямо документирует, что replay и retry могут повторить side effect,
at-least-once требует idempotency, а at-most-once-per-attempt всё равно не гарантирует exactly
once для workflow. Stable operation ID/idempotency token и checkpointed outcome нужны на semantic
action boundary. [AWS idempotency][aws-idempotency], [AWS durable step][aws-step].

LangGraph предупреждает, что node возобновляется с начала, а effects перед interrupt могут
повториться; completed task results нужно checkpoint'ить. Это подтверждает, что resume stale
model context или arbitrary call stack недостаточен. [LangGraph execution][langgraph-execution].

### 6.3. Waiting, background и human-in-the-loop

Microsoft Durable Functions external events позволяют unload worker во время ожидания, затем
wake execution по signal; delivery at-least-once требует event ID/deduplication. Human interaction
добавляет durable timer и explicit timeout branch. [Microsoft external events][azure-events],
[Microsoft human interaction][azure-human].

OpenAI Agents SDK сериализует paused RunState с pending approvals и stable agent identities, но
также fail-closed при неоднозначном output ownership и рекомендует version pending tasks вместе с
agent graph. Полезен именно принцип resumable typed state + compatibility/revalidation, не SDK
schema. [OpenAI HITL][openai-hitl]. OpenAI Responses background mode отдельно демонстрирует, что
provider response может иметь `queued`, `in_progress`, `completed`, `failed`, `cancelled` и
`incomplete` независимо от client connection; это transport/provider job, не полный SlavikAI
task lifecycle. [OpenAI Responses][openai-background].

### 6.4. Ownership, reconciliation и supervision

Kubernetes controllers регулярно сравнивают desired и observed state, а Lease хранит holder и
renewal time для takeover. `resourceVersion` позволяет watch/reconnect без молчаливой перезаписи
нового state. Это поддерживает reconciliation, lease/epoch и optimistic transition guards, не
требуя Kubernetes как runtime. [Kubernetes controllers][k8s-controller],
[Kubernetes leases][k8s-lease], [Kubernetes API versioning][k8s-api].

Akka разделяет expected business failure и supervision, а restart очищает private accumulated
state; delivery across unreliable boundaries не даёт exactly-once, at-least-once допускает
duplicates. Это поддерживает recoverability без private agent context и bounded retry/dedup,
но actor hierarchy не заменяет authoritative task lifecycle. [Akka supervision][akka-supervision],
[Akka delivery][akka-delivery].

## 7. Alternatives

| Alternative | Correctness/recovery | Cancel/retry/multi-agent | Complexity | Verdict |
| --- | --- | --- | --- | --- |
| Implicit call stack/control flow | Теряется при crash; нет durable truth | Ad hoc, stale results не fenced | Низкая сначала, высокая debt | Rejected |
| Один task status enum | Прост для UI, смешивает levels/reasons | Failure propagation и wait states неоднозначны | Низкая | Rejected |
| Hierarchical state machines без history | Хорошее разделение identities | Current state есть, причины/recovery evidence теряются | Средняя | Insufficient alone |
| Полное event sourcing | Сильный replay/audit | Хорошо при строгой determinism discipline | Высокие migration/operational costs | Not required now |
| Выбранный workflow-engine model | Может дать durability | Создаёт premature vendor/runtime constraint | Высокая и vendor-specific | Deferred implementation choice |
| Authoritative hierarchical state + immutable transition history | Явный current truth и audit; history не обязана быть replay engine | Attempts/epochs/checkpoints дают fencing/recovery | Минимально достаточная | **Selected** |

Выбранная target architecture:

> **Hierarchical authoritative lifecycle state for logical task, task run and subtask, with
> immutable transition history, explicit revision/attempt/ownership identities, recoverable
> checkpoints and a side-effect reconciliation journal.**

Это `authoritative current state + durable transition history`, а не обязательный event-sourced
runtime. Current state отвечает «что сейчас», history — «как и кем изменилось», checkpoint — «что
безопасно продолжить», action journal — «что могло произойти с external system».

## 8. Current-runtime verdicts

| Abstraction | Current semantics/evidence | Target mismatch | Verdict | Migration implications |
| --- | --- | --- | --- | --- |
| Current “task” aggregate | Несколько snapshots/call stacks, unified record отсутствует | Нет logical task/run hierarchy | `supersede` | Ввести authoritative aggregate; UI snapshots сделать projections |
| Plan snapshot/revision | Durable session JSON + hash/CAS guard | Revision смешивает content и status | `reuse with modification` | Разделить `plan_revision`, lifecycle version и task revision |
| `TaskPacket` | Immutable hashed execution contract | Нет run/subtask/attempt identity; packet revision несёт retry | `reuse with modification` | Привязать к accepted task/plan/run revisions; retry хранить отдельно |
| `RunContext` | Session/trace/workspace/policy + integer attempt | Process-local bag, нет checkpoint/epoch | `supersede` | Разделить authoritative RunRecord, projection и ephemeral executor context |
| `AgentScope` | Principal/session isolation | Session не равна agent participation/run scope | `reuse with modification` | Сохранить principal boundary, добавить explicit task/run/agent identities |
| `ScopedAgentProvider` | Один mutable Agent + lock per principal/session | Process ownership, не durable worker ownership | `reuse with modification` | Оставить local lifecycle manager; не считать source of truth/lease |
| MWV manager | Worker -> verifier + bounded verifier retry | Implicit attempts; no acceptance/recovery | `reuse with modification` | Сохранить simple mode, подключить common run/subtask/attempt contract |
| Auto v1 execution loop | Native tool loop + verifier | Flat status, restart-unsafe pause/resume/cancel | `reuse with modification` | Reuse tool execution, supersede Auto lifecycle state/controller |
| `AutoRunStatus` | Phase + wait + failures в одном enum | State explosion и неверная propagation | `supersede` | Project target run phase/condition/outcome в UI |
| Auto shard/pool/coder/merge models | Legacy shapes вокруг single tool loop; production references не найдены | Не authoritative decomposition/ownership | `remove/deprecate` | Не мигрировать как lifecycle model; удалить отдельным runtime PR |
| UI `active_task`/`active_plan` | Persisted session snapshots | UI/session-owned truth; cancel can be overwritten | `supersede` | Сделать read projection из lifecycle store; temporary write adapter не target |
| `UIHub` | Session state, notifications, bounded process event buffer | Event stream не durable history/control plane | `reuse with modification` | Оставить projection/notification role, убрать lifecycle authority |
| bare `asyncio.create_task` | Plan/resume/canvas process-local work | Нет ownership, recovery, cancellation | `supersede` | Durable execution controller dispatches replaceable process workers |
| `ChatCancellationRegistry` | One active generation/session, process token | Нет task/run cancel epoch/restart semantics | `reuse with modification` | Оставить transport interrupt; domain cancel идёт через lifecycle authority |
| `DesktopRunCoordinator` | One process mutex for physical host | Нет durable lease/owner epoch/takeover | `reuse with modification` | Local mutex остаётся safety layer; durable action/run ownership отдельно |
| HTTP `IdempotencyStore` | 90-second process-local request dedup | Не semantic/durable operation idempotency | `reuse with modification` | Сохранить edge dedup; lifecycle keys/action journal отдельны |
| `AgentToolLoop` | Model/tool iterations + final gate | Нет durable action attempts/unknown outcome | `reuse with modification` | Dispatch through action intent/attempt journal and recovery checks |
| `ToolGateway` | Typed policy-enforced dispatch | Не владеет task lifecycle или external outcome journal | `reuse as-is` | Остаётся execution boundary; lifecycle wraps, не обходит его |
| Approval/Decision models | Explicit pending decision с TTL; grants session-process scoped | Run/action binding и restart revalidation неполны | `reuse with modification` | Bind decision to exact action/run/revisions; never replay stale grant |
| Policy flow | Deterministic boundary + profiles | Snapshot/live revocation interaction не modelled lifecycle-wide | `reuse with modification` | Persist reference/version and revalidate before side effects/resume |
| Session lifecycle | Persisted conversation/UI attachment | Session pruning/disconnect не должны решать task fate | `unrelated` | Task references session attachments, но имеет independent lifetime |
| Artifact flow | Session JSON metadata/files | Нет canonical artifact revision/integrity lifecycle | `reuse with modification` | Task stores references/outcomes, artifact subsystem owns payload/lifecycle |
| Verifier result | Passed/failed/error evidence | Нет verification attempt/acceptance authority | `reuse with modification` | Version attempts and distinguish reject/inconclusive/accepted |
| Final response path | Writes assistant message after execution branch | Message может masquerade as completion | `reuse with modification` | Render final only from authoritative outcome; progress separately |

## 9. Current-runtime impact / migration implications

Главное расхождение — lifecycle authority сейчас принадлежит session snapshot и control flow.
Правильная target boundary требует отдельного durable lifecycle control plane; адаптация только
`active_task` JSON закрепила бы неправильный coupling с UI/session.

Последовательность будущей migration здесь не проектируется, но semantic implications известны:

- создать distinct identities и authoritative transitions до переноса background execution;
- перестать генерировать два task ID для одного Plan execute;
- разделить plan content revision и lifecycle version;
- UIHub читать/project state и публиковать progress, но не принимать terminal transitions;
- обернуть current tool loops action intent/attempt journal без обхода ToolGateway;
- заменить Auto `_paused_runs` и bare asyncio handles recoverable ownership/checkpoints;
- добавить cancellation/ownership epochs до разрешения recovery/parallel workers;
- классифицировать restored `running` snapshots как legacy/inconsistent, а не автоматически resume;
- deprecated Auto pool/shard shapes не переносить как target domain model.

Не требуется rewrite ради rewrite: ToolGateway, packet immutability/hash, principal isolation,
Plan compare-and-set pattern, verifier evidence и typed decisions являются пригодными building
blocks после корректировки responsibilities.

## 10. Architectural gaps и согласование contracts

Противоречия current runtime с уже принятыми target contracts:

- Context contract требует distinct task/run/tool-attempt identities; current runtime их не имеет;
- Context recovery требует durable checkpoint и unknown-side-effect tracking; UI snapshots этого
  не обеспечивают;
- Coordination contract требует atomic ownership epoch и stale-result rejection; current
  process locks/mutex этого не дают;
- Memory contract запрещает automatic task-result promotion; current lifecycle не должен
  добавлять её как completion side effect;
- ARCH_CANON cancellation/approval/policy safety не может считаться выполненной bare UI status
  mutation и process-local grants.

Между target contracts архитектурного противоречия не обнаружено. Новый contract уточняет их
общую lifecycle boundary и не делает coordinated mode обязательным для single-agent/MWV paths.

## 11. Open/deferred decisions

Implementation-defined остаются:

- DB/storage engine, broker, scheduler, workflow engine и process topology;
- exact persisted enums/serialization, lease durations, timeout и retry counts;
- checkpoint frequency и recovery worker implementation;
- concrete API/UI/notification protocol;
- compensation executor и integration-specific idempotency mechanisms;
- retention/compaction transition history;
- whether current simple Ask creates a persisted logical task by default or only a lightweight
  run record under a policy threshold.

Не implementation-defined: identity separation, authoritative transition boundary, cancellation
fencing, unknown outcome reconciliation, durable background ownership, policy/approval
revalidation, completion authority и separation UI/session/log from source of truth.

## Sources

- [Temporal: detecting Workflow failures][temporal-workflow]
- [Temporal: Activity Execution][temporal-activity]
- [Temporal: Retry Policies][temporal-retry]
- [AWS Durable Execution: idempotency and retries][aws-idempotency]
- [AWS Durable Execution: step semantics][aws-step]
- [LangGraph: execution, replay and idempotency][langgraph-execution]
- [LangGraph: interrupts][langgraph-interrupts]
- [Microsoft Durable Functions: external events][azure-events]
- [Microsoft Durable Task: human interaction][azure-human]
- [OpenAI Agents SDK: human-in-the-loop][openai-hitl]
- [OpenAI Responses API: background response states][openai-background]
- [Kubernetes controllers][k8s-controller]
- [Kubernetes Leases][k8s-lease]
- [Kubernetes API resourceVersion/watch][k8s-api]
- [Akka supervision and monitoring][akka-supervision]
- [Akka message delivery reliability][akka-delivery]

[temporal-workflow]: https://docs.temporal.io/encyclopedia/detecting-workflow-failures
[temporal-activity]: https://docs.temporal.io/activity-execution
[temporal-retry]: https://docs.temporal.io/encyclopedia/retry-policies
[aws-idempotency]: https://docs.aws.amazon.com/durable-execution/patterns/best-practices/idempotency/
[aws-step]: https://docs.aws.amazon.com/durable-execution/sdk-reference/operations/step/
[langgraph-execution]: https://langchain-ai.github.io/langgraph/how-tos/state-reducers/
[langgraph-interrupts]: https://langchain-ai.github.io/langgraph/concepts/breakpoints/
[azure-events]: https://learn.microsoft.com/en-us/azure/azure-functions/durable/durable-functions-external-events
[azure-human]: https://learn.microsoft.com/en-us/azure/durable-task/common/durable-task-human-interaction
[openai-hitl]: https://openai.github.io/openai-agents-python/human_in_the_loop/
[openai-background]: https://developers.openai.com/api/reference/cli/resources/responses/methods/create
[k8s-controller]: https://kubernetes.io/docs/concepts/architecture/controller/
[k8s-lease]: https://kubernetes.io/docs/concepts/architecture/leases/
[k8s-api]: https://kubernetes.io/docs/reference/using-api/api-concepts
[akka-supervision]: https://doc.akka.io/libraries/akka-core/current/general/supervision.html
[akka-delivery]: https://doc.akka.io/libraries/akka-core/current/general/message-delivery-reliability.html
