# Task / Run Lifecycle — target contract

**Статус: target architecture, не описание current runtime.**

Этот документ задаёт semantic lifecycle contract logical task, task revision, task run, subtask,
attempt, verification и background execution SlavikAI. Он не выбирает database, broker,
workflow engine, scheduler, API, exact persisted enum, timeout/retry values или process topology.
Current-runtime audit, alternatives, external research и migration implications находятся в
[`TASK_RUN_LIFECYCLE_RESEARCH.md`](TASK_RUN_LIFECYCLE_RESEARCH.md).

Контракт применяется вместе с [`ARCH_CANON.md`](ARCH_CANON.md),
[`CONTEXT_ARCHITECTURE_CONTRACT.md`](CONTEXT_ARCHITECTURE_CONTRACT.md),
[`MULTI_AGENT_COORDINATION_CONTRACT.md`](MULTI_AGENT_COORDINATION_CONTRACT.md) и
[`MEMORY_ARCHITECTURE_CONTRACT.md`](MEMORY_ARCHITECTURE_CONTRACT.md).

## 1. Capability boundary и target model

SlavikAI должен поддерживать единый lifecycle contract для simple single-agent, Plan -> Act,
MWV, Auto, coordinated multi-agent и background execution. Это не делает все tasks durable,
background или multi-agent по умолчанию: policy/capability выбирает необходимый execution mode,
но одинаковые identity, transition authority, completion и safety semantics действуют везде.

Baseline task/run/coordination scope принадлежит одному `principal_id`. Cross-principal или
collaborative multi-user task потребует отдельного identity, consent и authorization contract и
не допускается неявно текущей моделью.

Основная target architecture:

> **Hierarchical authoritative lifecycle state for logical task, task run and subtask, with
> immutable transition history, explicit revision/attempt/ownership identities, recoverable
> checkpoints and side-effect reconciliation.**

~~~text
Logical Task(task_id) -- accepted Task Revision
          |
          +-- Task Run(task_run_id, plan_revision, policy/budget snapshot)
                 |
                 +-- Subtask(subtask_id, claim_epoch)
                 |      +-- Agent participation(agent_id, agent_instance_id)
                 |      +-- Attempt(attempt_id)
                 |
                 +-- Tool/Action Intent(operation_id)
                 |      +-- Tool Attempt(tool_attempt_id)
                 |
                 +-- Verification Attempt(verification_attempt_id)
                 +-- Coordination Scope(coordination_scope_id)
                 +-- Recoverable Checkpoint
~~~

Для каждого authoritative aggregate логически существуют:

1. current state — ответ на «что сейчас»;
2. immutable transition history — кем, почему и из какой revision state изменён;
3. accepted evidence/references — на чём основан outcome;
4. recoverable checkpoint — что можно безопасно продолжить;
5. derived UI/context/telemetry projections — rebuildable views, не authority.

Полное event sourcing не требуется. Реализация может хранить materialized current state и
append-only transition records; history обязана быть достаточной для audit/fencing/recovery
decisions, но не обязана replay'ить произвольный application code.

## 2. Identity model

Все identity opaque и stable в своём domain. Их нельзя выводить только из prose, array position,
process ID или UI session.

| Identity | Semantic meaning | Revision/retry rule |
| --- | --- | --- |
| `task_id` | Устойчивая пользовательская goal lineage | Не меняется от retry/run restart |
| `task_revision` | Accepted version goal, constraints и completion criteria | Material user/authority change создаёт новую revision |
| `task_run_id` | Одна execution incarnation конкретной task revision | Whole-run restart/re-execution создаёт новый ID; crash recovery может сохранить тот же |
| `plan_revision` | Accepted execution/decomposition strategy | Adaptive replan создаёт новую revision, не обязательно task revision |
| `subtask_id` | Authoritative decomposition unit внутри task lineage/run | Retry не создаёт новый subtask; replacement получает новый ID или supersession link |
| `attempt_id` | Одна попытка выполнить run fragment/subtask | Каждый retry создаёт новый attempt |
| `agent_id` | Stable logical role/participant | Не меняется от process reconstruction роли |
| `agent_instance_id` | Одна runtime incarnation agent | Crash/restart/replacement создаёт новый ID |
| `coordination_scope_id` | Scope typed coordination одного principal/task run | Не равен session; закрывается отдельно от agent |
| `verification_attempt_id` | Одна verifier execution | Recheck/rework создаёт новый ID |
| `operation_id` | Stable semantic tool/action intent | Сохраняется через delivery retry/reconciliation |
| `tool_call_id` | Один accepted model/tool call occurrence | Не является автоматически idempotency key external system |
| `tool_attempt_id` | Одна dispatch attempt operation/tool call | Каждый retry получает новый ID |

`task_revision` и `plan_revision` не взаимозаменяемы. Изменение метода выполнения не меняет
пользовательскую цель; изменение goal/scope/acceptance criteria требует task revision. Status
transition повышает lifecycle state version, но не content revision.

Task revision проходит `proposed -> accepted_current | rejected`; новая accepted revision
supersedes prior current revision, сохраняя lineage. Clarification draft не обязана становиться
revision до acceptance. Retry без изменения goal/criteria revision не создаёт.

`session_id` — attachment/continuity identity из Context contract. Session может породить,
показывать или reattach task, но не определяет task/run lifetime. `trace_id` коррелирует
observability и не заменяет domain identity.

## 3. Независимые lifecycle machines

Нельзя сводить все levels к одному enum. Минимально независимы:

- logical task aggregate и accepted task revision;
- task run;
- subtask/dependency/ownership;
- agent participation/instance;
- operation/tool attempt;
- verification;
- coordination scope;
- approval/decision и artifact lifecycles в owning subsystems.

Каждая transition указывает aggregate identity, prior state version, next state, authority,
reason, timestamp, causal/correlation references и применимые task/plan/cancellation/owner epochs.
Conflicting write отклоняется или проходит explicit reconciliation; blind last-write-wins
запрещён.

Следствия:

- failed attempt не делает subtask или task failed автоматически;
- failed subtask не делает run failed, если strategy допускает replacement/replan;
- `result_submitted` не означает verified/accepted/completed;
- terminal run не запрещает новый run той же logical task;
- закрытие agent instance не закрывает task;
- closed coordination scope не удаляет lifecycle/evidence.

## 4. Logical task и task revision

Logical task — stable lineage пользовательской цели. Его authoritative states:

- **created** — identity существует, goal ещё может требовать definition;
- **defining** — идёт clarification/acceptance criteria formation;
- **ready** — accepted task revision достаточна для planning/execution;
- **active** — существует non-terminal run или authorized work;
- **completed** — current accepted revision удовлетворена и result accepted;
- **failed** — authority решила, что current accepted revision нельзя завершить допустимым run;
- **cancelled** — authoritative user/control/policy cancellation задачи;
- **superseded** — goal заменена другой logical task с explicit lineage.

`completed`, `failed`, `cancelled`, `superseded` terminal для данной accepted task revision и не
откатываются in place. Материальное продолжение создаёт новую accepted task revision и explicit
reopen transition того же task lineage либо новую logical task согласно user intent; прошлый
terminal outcome сохраняется.

Waiting/blocked/paused — typed operational condition open task/run, а не четыре взаимозаменяемых
слова terminal failure:

- `waiting_user_input` обычно удерживает task в `defining` или active run;
- `waiting_approval`, dependency/resource/external-event waits принадлежат run/subtask;
- `paused` означает deliberate resumable stop;
- `blocked` означает отсутствие currently valid progress path до external/control change.

Plan existence не переводит task автоматически в completed/active. Accepted plan revision —
execution contract, а не task state.

## 5. Task run state machine

Task run — одна execution incarnation accepted task revision. Semantic state представляется
phase + condition + typed reason, чтобы не создавать один гигантский enum.

### 5.1. Phases

~~~text
created -> preparing -> executing -> completing -> completed
                         |    |          |
                         |    |          +-> failed | cancelled | aborted
                         |    +------------> failed | cancelled | aborted
                         +-----------------> failed | cancelled | aborted
~~~

- **created** — run identity/intent accepted, execution ещё не prepared;
- **preparing** — фиксируются task/plan revisions, policy/budget baseline, capability, scope,
  coordination mode и initial checkpoint;
- **executing** — attempts/subtasks/actions могут dispatch'иться;
- **completing** — новые ordinary work dispatch запрещены; идёт required verification,
  aggregation, artifact/evidence integrity и acceptance transaction;
- **completed** — completion contract atomically accepted;
- **failed** — run terminal после исчерпания/классификации recovery paths;
- **cancelled** — terminal authoritative cancellation;
- **aborted** — terminal administrative/policy/recovery stop без domain failure assertion.

### 5.2. Non-terminal conditions

`preparing`/`executing`/`completing` имеют одну authoritative condition:

- `runnable` / `running`;
- `waiting` с typed reason;
- `paused`;
- `blocked`;
- `recovery_required`.

`recovery_required` не terminal failure: дальнейший dispatch запрещён до reconciliation.
`blocked` может закончиться resume, replan, cancel или run failure. UI label не заменяет этот
state tuple.

Run считается начатым после atomic creation и durable preparation intent, а не после первого
model token. До первого side effect должны быть accepted task revision, applicable plan/strategy,
principal/scope, policy reference/snapshot, budgets, cancellation epoch и owner protocol.

Coordination scope создаётся только если mode его требует; он открывается до participant claims и
переходит `closing` до run terminal acceptance. Scope closure не может предшествовать учёту
material in-flight outcomes.

## 6. Subtask lifecycle

Этот contract сохраняет semantic baseline Coordination contract:

~~~text
proposed -> ready -> claimed -> running -> result_submitted -> completed
                          \-> blocked | released | reclaimable | failed | cancelled
~~~

Дополнения:

- `claimed` содержит owner `agent_id`/`agent_instance_id`, `claim_epoch` и lease/heartbeat
  metadata, если ownership переживает process;
- `result_submitted` содержит evidence/artifact references, но не даёт completion authority;
- accepted verifier/control transition переводит в `completed` или `rework`/`ready`;
- retry создаёт новый `attempt_id`, сохраняя `subtask_id`;
- replan может сделать subtask `superseded`/`cancelled` и fencing'ит prior epochs;
- dependency readiness вычисляется из authoritative states/revisions, не agent сообщения.

## 7. Agent participation и ownership

Agent participation lifecycle отделён от task/subtask:

`registered -> assigned -> active <-> waiting -> released | lost | terminated`.

Worker может публиковать findings/results и предлагать transitions, но не присваивает себе
authoritative completion/plan/policy authority. Claim/reclaim выполняется atomic transition с
monotonic `claim_epoch`. Result от старого epoch сохраняется как stale evidence, но не меняет
current subtask/run state.

Process-local lock допустим как дополнительный local safety mechanism, но не является durable
ownership. Для takeover нужен observed lease/heartbeat expiry или explicit release, new epoch и
reconciliation in-flight attempts.

## 8. Attempt и action lifecycle

Attempt — bounded execution try, а не synonym run. Минимальная semantics:

~~~text
created -> dispatched -> running
                    -> succeeded | failed | cancelled | timed_out | unknown | stale
~~~

`unknown` — terminal outcome конкретной attempt observation, но unresolved operation condition:
system не знает, произошёл ли external side effect. Parent action/run блокируется для
reconciliation; `unknown` нельзя автоматически трактовать как safe failure.

Для side-effecting tool/action отдельно существуют:

- **operation intent** (`operation_id`) — approved semantic effect, target, arguments hash,
  policy/approval/budget/task/plan revisions и idempotency key;
- **tool/action attempt** (`tool_attempt_id`) — один dispatch с owner/epoch/start/deadline;
- **observed outcome** — success/failure/cancelled/timed_out/unknown;
- **verified outcome** — deterministic observation/reconciliation evidence;
- compensation metadata — available/not available, attempted outcome, без обещания rollback.

Provider `tool_call_id` коррелируется с operation, но model-generated ID один не доказывает
idempotency или authority.

## 9. Transition authority

Authoritative transitions проходят trusted lifecycle control boundary с compare-and-set по
state version/revisions/epochs.

| Actor | Допустимая authority |
| --- | --- |
| User/principal | goal revision, input, explicit pause/resume/cancel, acceptance где policy разрешает |
| Policy/security runtime | require/forbid mode, pause, downgrade, cancel/abort, forbid resume, require verification |
| Orchestrator/control plane | prepare run, decompose, dispatch, wait/block/replan proposals, aggregate, propose completion |
| Lifecycle controller | Validate и atomically commit authoritative transitions/fencing |
| Worker/agent | Claim request, heartbeat, progress/finding, result/failure proposal; не final task authority |
| Verifier | Accepted/rejected/inconclusive evidence; не task completion сам по себе |
| ToolGateway/action controller | Enforce dispatch и record attempt/outcome; не меняет goal/plan |
| Scheduler/executor | Lease/dispatch mechanics; не определяет domain success |
| UI/client | Issues authorized commands and renders projections; UI state не source of truth |

Model prose может предложить transition, но никогда не является state mutation command без
structured validation и authority.

## 10. Typed waiting, pause и resume

Минимальные wait reasons:

| Reason | Кто может unblock | Resume behavior |
| --- | --- | --- |
| `waiting_user_input` | authorized principal/control input | Revalidate task revision/plan |
| `waiting_approval` | canonical approval decision | Bind decision exact operation; revalidate expiry/policy |
| `waiting_dependency` | dependency transition | Auto-resume только при still-current revisions |
| `waiting_external_event` | authenticated/deduplicated event | Correlate event identity and validity |
| `waiting_resource` | scheduler/resource controller | Revalidate budgets/lease/policy |
| `waiting_verification` | verifier outcome | Accept, rework, inconclusive или fail |
| `waiting_recovery` | recovery authority/reconciler | Resume only from accepted checkpoint |

Каждый wait record содержит reason, blocked entity/action, unblock authority, optional deadline,
timeout policy и user-visible explanation. Timeout переводит в typed timeout/recovery decision,
а не автоматически в failure или retry.

Pause отличается от wait: pause — explicit control decision временно прекратить dispatch. Cancel
terminal; pause resumable. Resume того же run разрешён только после проверки:

- current task/plan revisions и completion criteria;
- cancellation/ownership epochs и outstanding claims;
- live policy, approval validity и capability;
- remaining/reallocated budgets;
- artifact/evidence integrity;
- pending/unknown side effects;
- coordination checkpoint/dependencies;
- compatible executor/agent definition.

Resume формирует новый bounded ContextPackage и при необходимости новый attempt/agent instance;
stale model context не продолжается как authority.

## 11. Retry contract

Retry всегда указывает level и failure classification:

- provider/model request retry;
- tool/action attempt retry;
- subtask attempt retry;
- replacement/retry agent instance;
- verification attempt retry;
- whole task-run retry;
- new run for updated task revision.

Правила:

- operation/subtask retry сохраняет logical identity и создаёт новый attempt ID;
- whole-run retry создаёт новый `task_run_id`, но сохраняет `task_id`/task revision, если goal не
  изменилась;
- retriable vs non-retriable определяется typed failure/policy, не строкой model output;
- retry budget bounded и учитывает backoff/external cost, но exact values implementation-defined;
- partial success/timeout сначала проходит idempotency/reconciliation policy;
- retry stale/cancelled/superseded epoch запрещён;
- successful duplicate response может reconciled как prior success, а не новая completion;
- retry никогда не сбрасывает consumed approval/budget/action history молча.

Exactly-once external execution не обещается. Target предпочитает stable semantic idempotency,
at-least-once delivery tolerance, outcome verification и explicit STOP для ambiguity.

## 12. Cancellation contract

Cancellation — first-class authoritative transition с monotonic cancellation revision/epoch,
authority и reason:

- user cancellation;
- orchestrator cancellation в delegated scope;
- policy/runtime cancellation/abort;
- parent task/run propagation;
- subtask/agent/tool-action cancellation.

После commit cancellation:

1. новые dispatch/claims/approvals блокируются;
2. cancellation signal best-effort отправляется active owners/tools;
3. cancellable operations получают safe shutdown window;
4. in-flight external actions классифицируются completed/failed/unknown и reconciled;
5. compensations выполняются только policy-authorized и observable;
6. late old-epoch results могут retained как evidence, но не resurrect state;
7. final user-visible state сообщает cancellation и unresolved effects, если они есть.

Parent cancellation propagates по explicit policy. Non-cancellable external action или cleanup
может продолжить reconciliation после run `cancelled`, но не переводит его обратно в active или
completed. Повторный cancel idempotent.

## 13. Completion и verification

Ни final model message, ни artifact existence, ни worker `done`, ни отсутствие pending tool calls
сами по себе не являются completion.

Semantic flow:

~~~text
work outcome -> result_submitted
  -> required verification: accepted | rejected | inconclusive
  -> final aggregation candidate
  -> authoritative acceptance transaction
  -> run completed
  -> final projection/delivery
~~~

Verification может быть optional по accepted plan/policy, но acceptance decision обязательна.
Completion transaction проверяет применимые criteria:

- current task/plan revisions и required subtasks;
- required verifier outcome;
- artifact/evidence integrity;
- отсутствие unresolved blockers/conflicts/unknown side effects;
- отсутствие pending required approvals/actions;
- budgets/policy не маскируют incomplete result как success;
- final aggregation соответствует accepted evidence.

Verifier rejection создаёт rework/new attempt или terminal run failure по policy; logical task
может остаться active. `inconclusive` не является pass. Completion acknowledgement idempotent и
fenced against cancellation/supersession.

Final response — user-visible projection authoritative state. Progress/commentary, submitted
result и verifier report должны называться своими phases; «готово» запрещено до committed
completion.

## 14. Failure taxonomy и propagation

Failure record включает level, category, retryability, causal refs, observed effects и user
visibility. Минимальные categories:

- model/provider;
- tool/external service;
- policy denial / approval denial-expiry-revocation;
- invalid task/plan/context projection;
- verification rejection/error/inconclusive;
- timeout;
- resource or budget exhaustion;
- process/worker/orchestrator crash;
- corrupt/inconsistent canonical state;
- coordination/ownership failure;
- unreconciled external side effect.

`policy_denied`, `approval_rejected`, `cancelled`, `timed_out`, `blocked` и `failed` не
взаимозаменяемы. Attempt failure сначала меняет attempt и owning subtask/action; propagation на
run/task является отдельной authoritative transition. Budget exhaustion останавливает dispatch и
требует replan/extension/failure/cancel decision, но никогда не success.

## 15. Crash recovery и reconciliation

Canonical recoverable state не зависит от hidden reasoning, in-memory object, websocket, UI
session, log stream или process-local async handle.

Recovery checkpoint минимально фиксирует:

- principal/task/task revision/run/plan/coordination identities;
- authoritative state versions, cancellation and ownership epochs;
- policy reference/snapshot and required live revalidation markers;
- budget reservations/consumption;
- accepted completed subtasks/steps and dependencies;
- outstanding claims/leases and agent instances;
- operation intents, attempts, unknown/pending external outcomes;
- approval requests and decisions, но не stale grants as authority;
- artifact/evidence/verifier references and integrity/version;
- coordination consumption checkpoint;
- optional bounded private working summary permitted by Context contract, never required COT.

Recovery controller:

1. loads authoritative state/history/checkpoint;
2. fences stale owners and claims or verifies still-live lease;
3. reconciles pending/unknown external actions;
4. revalidates task/plan/policy/approvals/budgets/artifacts/dependencies;
5. either resumes same run with new instance/attempt, starts a new run, leaves
   `recovery_required`, or aborts/fails through authority;
6. records decision and exposes user-visible status.

Same run resumes only when accepted revisions and already committed outcomes remain valid and
continuation cannot duplicate an unresolved non-idempotent effect. Otherwise new run or explicit
STOP/reconciliation is required.

## 16. Background execution и session reattachment

Durable background run принадлежит principal/logical task, а не HTTP request, websocket,
browser tab или Python task. Он имеет durable identity/state/checkpoint, policy/budget, owner
lease/epoch, wake conditions, cancellation path и completion/failure notification record.

`disconnect != cancellation`. Если policy разрешает detached execution, run продолжает работу;
иначе disconnect вызывает typed pause/cancel policy transition. Reattachment требует same
authorized principal и explicit association с task/run. Новая session получает bounded current
projection и transition cursor, а не ownership через старый session ID.

Multiple authorized clients могут наблюдать один run. Commands проходят optimistic state version
и idempotency checks; stale client не может overwrite newer state. Final response хранится/
доставляется как outcome projection независимо от того, был ли client connected в момент
completion.

## 17. External side effects и idempotency

Перед side effect создаётся durable operation intent. После dispatch timeout/crash означает
`unknown`, пока provider observation, local verification или authoritative reconciliation не
установит outcome.

Decision path:

~~~text
unknown external outcome
  -> verify/reconcile
  -> confirmed success | confirmed failure | still unknown
  -> continue | bounded retry | compensate | pause/STOP
~~~

Idempotency требуется для task/run creation, claim, action intent/dispatch, artifact creation,
completion, cancellation, approval consumption, background wake/resume и external event intake.
Keys stable на semantic operation scope, principal-bound и не переиспользуются для другого
payload/revision.

Компенсация не равна rollback guarantee. Она отдельная approved operation с attempt/outcome и
может failed/unknown. Non-compensatable или still-unknown effect остаётся visible blocker.

## 18. Budget lifecycle

Run preparation фиксирует budget allocation и policy caps для time, tokens, tool calls, retries,
files, parallel agents и external cost, где применимо. Budget state distinct from prompt
instructions и включает reservations, committed consumption, child allocations и extensions.

- child/subtask budget не может превысить parent available amount;
- parallel reservations предотвращают oversubscription;
- retry consumes explicit retry/resource budget;
- reallocation/replan observable и policy-bound;
- extension требует authority и не mutates historical allocation;
- exhaustion переводит run в typed wait/block/terminal decision, не success;
- recovery reconstructs consumed/reserved budget before dispatch.

Exact units, limits и accounting engine implementation-defined.

## 19. Policy и approval lifecycle

Run хранит policy version/reference и execution snapshot, но live revocation/stronger current
policy имеет precedence перед future side effects. Policy может require/forbid coordinated или
background mode, pause, downgrade capability, cancel/abort, forbid resume или require verifier.

Approval request связывается с exact principal, run, operation intent/arguments hash, policy/task/
plan revisions, scope, expiry and consumption semantics. States acceptance/rejection/expiry/
revocation принадлежат canonical approval subsystem. Pending approval создаёт typed wait.

После restart/resume approval revalidates. Expired/revoked/consumed grant не replayed из packet,
Memory, prompt, checkpoint или session snapshot как действующая authority. Approval denial не
обязан означать failed logical task; orchestrator может replan к безопасному alternative.

## 20. Adaptive replanning

Worker не hidden-mutates active plan. Replan — control-plane transaction:

~~~text
active plan revision N
  -> plan_change_proposed
  -> identify affected work + synchronization barrier
  -> validate task revision, policy, budgets and evidence
  -> accept plan revision N+1
  -> cancel/supersede/reassign affected subtasks and fence attempts
  -> preserve explicitly safe unaffected work
  -> rebuild projections and resume
~~~

Material goal/acceptance change сначала создаёт task revision. Old attempts/results сохраняются с
old revisions и не применяются к N+1 без explicit compatibility validation. Replan не сбрасывает
external actions, budgets, approvals или verifier history.

## 21. Multi-agent и coordination integration

Lifecycle state и coordination history связаны IDs, но остаются разными layers:

- claims/heartbeats/results/failures являются typed coordination events/proposals;
- authoritative claim/completion/cancel/replan transitions коммитит lifecycle boundary;
- `claim_epoch` и cancellation epoch fencing'ят late agent results;
- lost owner делает subtask reclaimable только после reconciliation;
- sibling видит policy-filtered relevant events/state projection, не private context;
- conflicting assertions поступают verifier/control authority, а не решаются recency;
- coordination scope проходит `open -> closing -> closed`; closure учитывает material in-flight
  work и не означает deletion;
- single-agent/MWV execution может не иметь active shared bus, сохраняя тот же lifecycle contract.

## 22. Artifacts, Memory и user communication

Artifact lifecycle принадлежит artifact subsystem. Task/run хранит references на candidate,
partial, verified, superseded или diagnostic artifact revision и integrity evidence. Artifact
existence не completion; cancellation/failure не удаляет artifact автоматически. Cleanup —
отдельная policy-authorized action.

Task lifecycle не пишет long-term Memory автоматически. Accepted result, verified artifact,
outcome или lesson после completion может создать Memory candidate через Memory promotion
boundary. Failure/retry/coordination history также не Memory по умолчанию.

User communication имеет отдельные semantics:

- progress update — observable non-authoritative projection;
- clarification/approval — typed pending decision;
- blocker/wait notification — projection authoritative condition;
- result submitted/verifier outcome — intermediate outcome;
- final response — projection committed completion/failure/cancelled/aborted state.

## 23. Persistence, recovery и derived state

Canonical lifecycle truth состоит из authoritative current aggregates, immutable transition
records, accepted checkpoints и action/approval/artifact references. Конкретное physical storage
не выбрано.

Derived/rebuildable:

- UI `active_task`, mode/status badges и progress feeds;
- context projections/summaries;
- search indexes and dashboards;
- notification queues after acknowledged delivery policy;
- metrics/traces/log views.

Recovery/rebuild обязан обнаруживать inconsistent current state/history/checkpoint/index, не
молча выбирать самый новый timestamp. Partial failed transition не должен публиковать state,
который не имеет committed authority/history. Backups/restores сохраняют fencing epochs и
terminal transitions, чтобы stale delayed data не resurrect cancelled/superseded/completed work.

## 24. Observability и audit

Без private reasoning audit должен отвечать:

- какая task/revision/run/subtask/attempt/agent identity участвовала;
- кто и на каком основании committed transition;
- какие plan/policy/budget/cancellation/ownership revisions применялись;
- что retry/cancel/recovery/replan изменили;
- какие actions имеют success/failure/unknown/compensation state;
- какой verifier outcome и evidence привёл к acceptance;
- почему final response соответствует terminal state.

Transition audit, operational log, coordination events, retrieval/LLM trace и user-visible
explanation — разные projections/retention domains. Sensitive tool payloads и secrets не должны
копироваться в lifecycle history без необходимости; references/redaction сохраняют auditability.

## 25. Invariants

1. Session lifetime не является task-run lifetime.
2. Logical task, task revision, task run, subtask и attempt имеют distinct identities.
3. Retry не создаёт новую logical task автоматически.
4. Plan content revision и lifecycle state version различаются.
5. Result submission, verification, acceptance и completion различаются.
6. Tool timeout/crash не доказывает отсутствие external side effect.
7. Cancellation является authoritative versioned transition.
8. Late stale attempt/old owner epoch не может resurrect cancelled/superseded state.
9. Crash recovery не зависит от private model reasoning или stale prompt.
10. Durable background execution не зависит от process-local async handle, websocket или UI.
11. State transition authority не определяется model prose.
12. Plan changes versioned, observable и fencing'ят affected work.
13. Failed attempt/subtask/run не автоматически означает failed logical task.
14. Budget exhaustion не считается success.
15. Final response соответствует committed authoritative lifecycle state.
16. Silent fallback после coordination/recovery failure запрещён, если меняет correctness,
    isolation, policy или completion semantics.
17. Current state и immutable transition/audit history логически различаются.
18. UI state, log stream и coordination chatter не являются lifecycle source of truth.
19. Completion criteria explicit, attributable и verifiable.
20. Policy/approval revalidates before resume and external side effects; stale authority не replayed.
21. Exactly-once external execution не обещается; idempotency и reconciliation explicit.
22. Unknown external outcome блокирует unsafe retry/completion до resolution или authorized STOP.
23. Coordination mode optional by capability/policy; lifecycle correctness не зависит от swarm.
24. Task completion не автоматически создаёт Memory и не удаляет artifacts.

## 26. Open questions и implementation-defined decisions

Semantic boundary зафиксирована, но остаются open product questions:

- создаётся ли durable logical task для каждого ordinary Ask или только для execution,
  background и explicitly tracked goals;
- какая authority принимает completion там, где verifier optional;
- какие classes tasks разрешены detached/background;
- когда user follow-up является task revision, новым run или новой task;
- retention policy transition history/checkpoints и user-visible run history;
- где policy требует manual reconciliation non-compensatable effects.

Implementation-defined:

- database, schema, broker, workflow engine, scheduler и executor topology;
- exact persisted enums/serialization, transaction/event storage technique;
- timeout, retry count/backoff, lease duration и heartbeat cadence;
- API/UI/notification design;
- concrete idempotency providers, recovery worker and compensation implementation;
- checkpoint cadence/size/compaction;
- verifier implementation и exact budget units.

Нельзя оставить implementation-defined identity separation, transition authority, fencing,
completion contract, unknown-effect handling, policy/approval revalidation, background durability
или session independence.
