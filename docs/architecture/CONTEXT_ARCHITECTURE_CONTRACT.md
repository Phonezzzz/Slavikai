# Context Architecture — target contract

**Статус: target architecture, не описание current runtime.**

Этот документ задаёт semantic contract управления контекстом SlavikAI. Он не выбирает
database, API, transport, tokenizer, embedding model, prompt template или migration sequence.
Current-runtime evidence, alternatives и migration implications вынесены в
[`CONTEXT_ARCHITECTURE_RESEARCH.md`](CONTEXT_ARCHITECTURE_RESEARCH.md).

Нормативные security/runtime invariants из [`ARCH_CANON.md`](ARCH_CANON.md) имеют силу вместе
с этим contract. Multi-agent обмен дополнительно подчинён
[`MULTI_AGENT_COORDINATION_CONTRACT.md`](MULTI_AGENT_COORDINATION_CONTRACT.md).

## 1. Capability boundary

SlavikAI должен поддерживать bounded, provenance-aware context для single-agent, MWV,
hierarchical delegation и coordinated multi-agent execution. Наличие этой capability не
означает, что каждый режим обязан использовать Memory, coordination, compaction или несколько
агентов.

Контекст model call не является source of truth. Это производная, одноразовая проекция
разрешённых данных для конкретного principal, task run, agent и turn:

~~~text
authoritative state + retained evidence + private working state + Memory candidates
                                 |
                    selection / projection / policy
                                 |
                    bounded model-visible ContextPackage
                                 |
                              model call
                                 |
              proposed output / tool request / published result
                                 |
                   validation / enforcement / persistence
~~~

Target model можно кратко выразить так:

> **Authoritative typed runtime state + isolated private agent contexts + explicit context
> projections + separate long-term Memory.**

## 2. Термины

- **Authoritative state** — принятая runtime-запись identity, task, plan, policy, approval,
  ownership, lifecycle или verified outcome. Её нельзя заменить prompt summary.
- **Evidence** — versioned observation: user message, source revision, tool outcome,
  verification result или artifact с provenance и trust metadata.
- **Private working context** — model-facing история, local notes, промежуточные tool-loop
  messages и private summaries одного agent instance.
- **Context source** — разрешённый authoritative/evidence/private/Memory input до selection.
- **Context projection** — выбранное и преобразованное представление source для конкретного
  consumer и purpose.
- **ContextPackage** — immutable manifest плюс model-visible items для одного model call.
- **Lossless reference** — identity/version/hash/locator, позволяющие получить разрешённый
  исходный record без подмены summary.
- **Lossy summary** — производное сокращение, которое может опускать детали и не является
  единственным источником критического состояния.
- **Canonical structured state** — typed state с отдельной mutation/authority semantics;
  например, accepted plan revision или active blocker.
- **Interaction continuity** — пользовательская conversational lineage между turns.
- **Task continuity** — цель, ограничения, progress, blockers и recovery state длительной
  задачи, не сводимые к transcript.
- **Memory projection** — policy-filtered retrieval result из long-term Memory; не копия всего
  Memory store.

## 3. Non-negotiable invariants

1. `principal_id` является hard security boundary для всех context sources, projections,
   packages, artifacts, Memory и recovery state.
2. `session_id` не является universal identity для conversation, task, run, agent, approval,
   tool call или coordination scope.
3. Authoritative state хранится и изменяется вне model-visible prose. Модель может предложить
   transition, но trusted runtime её валидирует и принимает либо отклоняет.
4. Model-visible context является disposable derived view. Его потеря не должна уничтожать
   task truth или единственную копию критического evidence.
5. Каждый agent имеет отдельный private working context. Parent, child и peers не получают
   полный private transcript, scratchpad или chain-of-thought друг друга.
6. Parent-to-child, child-to-parent и peer-to-peer передача выполняется только через явные,
   typed, policy-filtered projections с provenance.
7. Policy/approval enforcement не зависит от того, вспомнила ли модель правило. Prompt
   содержит лишь безопасное объяснение; authoritative check выполняется на trusted boundary.
8. Summary не является единственной копией active constraints, approvals, task state,
   unresolved blockers, verified outcomes или artifact identity.
9. Untrusted source content не может повысить себя до system instruction, policy, approval или
   authoritative state только за счёт попадания в prompt.
10. Критический context не выбрасывается молча. Если mandatory set не помещается, runtime
    обязан сузить работу, пересобрать package, выбрать другой capability profile или STOP.
11. Context selection не расширяет audience, tool capabilities, workspace scope или provider
    egress по сравнению с source policy.
12. Disconnect, process restart и reattachment не сливают private contexts и не возрождают
    истёкшие approvals.

## 4. Identity model

Identity dimensions независимы, даже если current implementation временно хранит некоторые
из них одним значением.

| Identity | Meaning | Lifetime / relation |
| --- | --- | --- |
| `principal_id` | Authenticated security subject | Hard partition; никогда не выводится из model text |
| `session_id` | UI/client attachment и continuity handle | Может disconnect/reconnect; не владеет всеми tasks principal |
| `conversation_id` | Lineage пользовательского диалога | Может пережить несколько sessions/clients; branch/fork имеет явную lineage |
| `task_id` | Logical user goal/commitment | Переживает один или несколько execution runs |
| `task_run_id` | Одна execution incarnation task | Boundary recovery, budgets, policy snapshot и coordination scope |
| `coordination_scope_id` | Shared coordination boundary | Только для coordinated run; один principal + task run |
| `agent_id` | Logical participant/role в task run | Стабилен для assignment; не равен process instance |
| `agent_instance_id` | Конкретная runtime incarnation | Меняется после crash/restart; reattaches только по validated state |
| `subtask_id` | Decomposed unit of work | Имеет assignment/ownership/dependencies отдельно от agent |
| `model_turn_id` | Один model inference turn | Correlation для ContextPackage и outputs |
| `tool_call_id` | Logical requested tool action | Не равен attempt; сохраняет correlation с source turn |
| `tool_attempt_id` | Один execution attempt | Нужен для retry/idempotency/unknown outcome |

`trace_id` остаётся observability correlation и может связывать эти identities, но не заменяет
ни одну из них и не является authorization token.

### 4.1. Disconnect, restart и reattachment

- Client disconnect не завершает автоматически task run и не меняет ownership.
- Process restart создаёт новый `agent_instance_id`; trusted runtime восстанавливает только
  разрешённый checkpoint и canonical state, а не скрыто предполагаемую память процесса.
- Reattachment проверяет principal, task/run status, policy revision, participant membership,
  checkpoint lineage и unresolved/unknown tool actions.
- Expired/revoked approvals повторно проверяются; transcript фраза «разрешаю» не является
  достаточным recovery record.
- Две одновременно восстановленные instances не получают право исполнять один exclusive
  assignment без authoritative ownership/fencing semantics.

## 5. Context classes и ownership

Таблица задаёт semantic classes, а не обязательные классы кода или физические таблицы.

| Context class | Owner / identity | Lifetime | Authoritative source | Writers / readers | Model visibility | Restart / retention |
| --- | --- | --- | --- | --- | --- | --- |
| Interaction context | principal + conversation | Conversation | Versioned user/assistant interaction record | UI/runtime append; authorized agents read projection | Selected recent/exact turns | Durable per history policy; branch lineage preserved |
| Session continuity | principal + session | Attachment | Session registry/current attachment state | Trusted UI/runtime | Обычно не нужен, кроме safe labels | Recreated on reconnect; no authority over task truth |
| Task specification | principal + task + revision | Task | Accepted goal, constraints, acceptance criteria | User/control plane writes; agents read | Mandatory applicable projection | Durable while task/history retained |
| Task-run context | task run | Run | Lifecycle, budgets, assignments, checkpoints | Control plane and validated transitions | Selected status/progress | Durable enough for recovery/audit |
| Plan context | task + plan revision | Versioned task phase | Accepted plan revisions | Plan/control plane writes; workers read | Active revision and relevant rationale | Old revisions retained per audit policy |
| Private agent context | agent + instance + run | Agent execution | Private working record/checkpoint | Owning agent/runtime only | Да, только owning model | Bounded; recoverable summary optional; no automatic sharing |
| Tool/execution context | tool call + attempt | Action lifecycle | Request, policy decision, outcome, verification | Model proposes request; gateway/tools/verifier write records | Bounded request/result projection | Durable for side-effect reconciliation and audit |
| Coordination context | coordination scope + event/checkpoint | Coordinated run | Contract events + authoritative task state | Authorized publishers/control plane; participants consume | Selective event projection at sync points | Material history replayable per coordination contract |
| Verification context | verification assignment | Verification cycle | Evidence package + verifier result | Runtime assembles; verifier reads/writes outcome | Verifier-only package; result may be shared | Durable with task outcome |
| Artifact context | artifact ID + version | Artifact retention | Artifact record/content locator/integrity | Producers write; authorized consumers read | Metadata/snippet or explicit content | Payload retention separate from metadata/provenance |
| Policy/security context | principal + run/action + policy revision | Scope/decision | Trusted policy and approval stores | Trusted runtime only | Safe explanatory projection; never enforcement source | Revocation/expiry evaluated live or by valid snapshot semantics |
| Memory projection | principal + memory scope + retrieval revision | One assembly/turn | Long-term Memory records + retrieval manifest | Memory subsystem selects; assembler consumes | Selected claims/evidence only | Package disposable; Memory has separate lifecycle |
| Background/recovery context | task run + owner epoch/checkpoint | Paused/background run | Durable lifecycle/checkpoint/action journal | Scheduler/control plane/runtime | Minimal resume projection | Survives disconnect/process loss; stale instances fenced |
| Audit/telemetry context | trace/event identities | Policy-defined | Append-only observations/logs | Trusted instrumentation | Не включается автоматически | Separate retention; not task or Memory truth |

Ни один class не становится общим только потому, что физически хранится в одной database или
связан одним `session_id`.

## 6. Source-of-truth layers

### 6.1. Authoritative control state

Authoritative для runtime decisions:

- authenticated identities и membership;
- current task/run/subtask lifecycle и ownership;
- accepted task/plan revisions;
- effective policy, capability scope и approval decisions;
- accepted verification outcome;
- cancellation, completion и recovery transitions.

Модель не редактирует этот слой напрямую. Prose history, summaries, telemetry и Memory
references могут объяснять transition, но не заменяют её.

### 6.2. Evidence и artifacts

Evidence record authoritative только для утверждения «это было наблюдено/получено при таких
условиях». Tool `ok=true` доказывает зарегистрированный return, но не внешний эффект, пока
contract не считает response verified либо отдельная verification не подтверждает результат.

Artifact identity отделена от content projection. Большой, binary или sensitive payload
передаётся модели по explicit authorized reference/snippet, а не дублируется во всех histories.

### 6.3. Private working state

Private context помогает одному agent продолжать reasoning, но не является authoritative
task state. Runtime может сохранять bounded private checkpoint для recovery, если policy это
разрешает; chain-of-thought сохранять или передавать для восстановления не требуется.

### 6.4. Long-term Memory

Memory хранит durable personal/project knowledge по отдельному contract принятия, provenance,
consent, freshness и retention. Interaction, coordination event или tool result не становится
Memory автоматически. Runtime context читает только Memory projection.

### 6.5. Derived representations

Prompt packages, summaries, embeddings, indexes, caches, wiki/views и UI snapshots являются
derived. Они содержат source IDs/revisions, coverage и invalidation status. Их удаление не
должно уничтожать уникальную authoritative информацию.

## 7. Context assembly pipeline

Каждый model call проходит отдельную assembly transaction:

1. **Frame:** определить principal, conversation, task/run, agent, turn, purpose, provider,
   capability profile, language, current time и token/output/tool reserves.
2. **Resolve mandatory state:** загрузить применимые task/plan/policy/approval/lifecycle
   revisions и required coordination checkpoint.
3. **Select eligible sources:** применить principal, audience, sensitivity, temporal,
   freshness, provider egress и purpose filters до retrieval/reranking.
4. **Retrieve candidates:** interaction, private working state, artifacts/evidence,
   coordination и Memory ищутся раздельно; одинаковый ancestor не считается независимым
   подтверждением.
5. **Resolve representations:** exact span, structured state, compact summary или artifact
   reference выбираются по риску и задаче.
6. **Reconcile:** сохранить contradictions, uncertainty, supersession и stale markers;
   молчаливый last-write-wins в prompt запрещён.
7. **Budget:** разместить mandatory items, резервы и затем optional ranked context.
8. **Render:** пометить trust/source boundaries и не позволять untrusted data имитировать
   system/control instructions.
9. **Manifest:** зафиксировать sources/revisions, transformations, omissions, token estimate,
   policy/assembler version и package hash.
10. **Validate:** проверить identity/scope, mandatory coverage, tool-call continuity и provider
    limits; при fail package не отправляется модели.

### 7.1. ContextPackage manifest

Conceptual manifest содержит как минимум:

- package/model-turn identity и package version/hash;
- principal/task/run/agent identities;
- purpose и provider/capability profile;
- source IDs/revisions и projection type;
- trust class, audience, sensitivity и freshness;
- exactness/mandatory markers;
- transformation/summary lineage;
- estimated token cost и reserves;
- explicit omission/truncation/conflict markers;
- effective policy revision/reference.

Это semantic requirement, не финальная wire schema.

## 8. Trust, provenance и prompt-injection boundary

Минимальные trust classes:

- **trusted control:** system/runtime policy, authenticated identity, accepted state;
- **principal-authored:** текущая user instruction/decision с verified principal lineage;
- **verified observation:** evidence, прошедшее defined verification;
- **untrusted observation:** web, files, emails, tool output, OCR, external messages;
- **agent assertion:** finding/hypothesis/summary, не ставшие authoritative transition;
- **derived representation:** retrieval snippet, summary, synthesis, index result.

Trust class относится к происхождению и проверке, а не к роли строки в prompt. Даже если
untrusted document сериализован рядом с system message, его инструкции остаются data.

Каждый projected item сохраняет:

- origin/source identity и revision/hash;
- author/publisher/capture method;
- observed/recorded time и freshness;
- transformation lineage;
- audience/sensitivity/egress constraints;
- epistemic status: observed/reported/inferred/hypothesis/verified/disputed/retracted;
- completeness/truncation markers.

Security decisions не делегируются source metadata, tool annotations, retrieval ranking или
LLM self-assessment. Sensitive raw data минимизируется до передачи provider; redaction не
заменяет access control.

## 9. Parent, child и peer projections

### 9.1. Parent-to-child

Child получает минимально достаточный assignment package:

- task/subtask ID, accepted goal и packet/plan revision;
- scope, constraints, budgets, acceptance/verifier contract;
- safe capability/policy projection и trusted handles для enforcement;
- explicit dependencies/blockers;
- разрешённые user excerpts, facts, artifacts и evidence references;
- coordination checkpoint/subscription, если mode coordinated;
- expected output/publication contract.

По умолчанию не передаются:

- полный parent conversation/private transcript;
- parent scratchpad, hidden reasoning или chain-of-thought;
- unrelated session/Memory context;
- secrets/auth material;
- approvals как prose;
- tool results, не нужные assignment.

### 9.2. Child-to-parent

Child возвращает typed result: status, findings/hypotheses, evidence/artifact refs, blockers,
assumptions, verification state, unresolved uncertainty и requested plan change. Parent не
обязан получать private tool-loop transcript.

### 9.3. Peer consumption

Peers получают только события, разрешённые coordination contract, через controlled sync points.
Event сначала проверяется по scope/audience/publisher/trust, затем преобразуется в bounded
projection. Raw event payload не вставляется асинхронно внутрь незавершённого model turn.

## 10. Tool и execution context

### 10.1. До tool call

Model-visible tool definition — capability advertisement, не authorization. Runtime связывает
request с model turn, task/run, agent, policy revision и logical `tool_call_id`, валидирует
arguments, target scope, approval и idempotency до side effect.

### 10.2. После tool call

Tool outcome хранится как typed evidence:

- requested/started/completed/failed/cancelled/unknown status;
- logical call и attempt identities;
- normalized arguments или protected argument reference;
- result metadata, error, timing и affected resource identities;
- trust/completeness/truncation/sensitivity;
- verification status и follow-up evidence.

Модель получает:

- небольшой structured result целиком, если он разрешён и помещается;
- для большого/sensitive результата — typed summary/exact excerpts плюс artifact reference;
- явный marker, если payload truncated, redacted, stale или incomplete.

Полный raw tool result не должен автоматически копироваться в durable conversation, private
checkpoint, coordination и Memory одновременно.

### 10.3. Retry и unknown outcome

Retry создаёт новый `tool_attempt_id`, но сохраняет logical `tool_call_id`/idempotency intent.
После crash между side effect и recorded result состояние считается `unknown`, пока runtime не
reconcile-ит внешний ресурс. Молчаливое повторение destructive action запрещено.

## 11. Compaction и summaries

Compaction управляет representation, а не authority или trust.

### 11.1. Три вида continuity data

1. **Lossless refs:** IDs, revisions, hashes, artifact locators, exact user decisions.
2. **Canonical structured state:** active goal/constraints/plan/progress/blockers/policy refs.
3. **Lossy summaries:** conversation/private/evidence overviews для экономии context.

Critical state хранится в первых двух видах. Summary помогает selection, но не заменяет их.

### 11.2. Summary provenance

Summary содержит source range/IDs/revisions, scope, generator/model/version, generated time,
coverage, omissions, unresolved conflicts, truncation и superseded-by relation. Повторное
summary предыдущего summary не создаёт новое independent evidence.

### 11.3. Protected semantics

Compression обязана сохранять либо exact reference на:

- отрицания и запреты;
- числа, units, identifiers, paths и deadlines;
- active constraints и acceptance criteria;
- unresolved blockers/conflicts/unknown outcomes;
- conditions, temporal qualifiers, uncertainty и provenance;
- user decisions и policy references.

Если validation не может подтвердить coverage, old authoritative data сохраняется, package
пересобирается или runtime STOP. Failed compaction не должна атомарно заменить единственную
пригодную history.

## 12. Budgeting и overflow

Budget provider-aware и включает не только history:

~~~text
model input limit
  - system/runtime instructions
  - tool schemas/protocol overhead
  - output reserve
  - expected tool-result and correction reserve
  - safety margin
  = allocatable context budget
~~~

Минимальный порядок приоритета:

1. current user intent и trusted system/policy explanation;
2. applicable task/plan revision, active constraints и required state;
3. pending approval/decision, unknown side effect, blockers и cancellation state;
4. latest verified evidence для текущего шага и required coordination updates;
5. private recent turn/tool continuity owning agent;
6. relevant interaction history, artifact excerpts и Memory projection;
7. optional background, examples и neighboring context.

Приоритет не обходит access policy. Динамическое распределение между классами допустимо, но
runtime резервирует output/tool/correction capacity. Если mandatory context больше лимита:

- deduplicate representations по source ancestry;
- заменить permitted detail на structured state + lossless refs;
- разделить task/model turn или выбрать capability с подходящим context limit;
- STOP с наблюдаемой причиной, если корректное выполнение невозможно.

Silent head/tail truncation critical context запрещена.

## 13. Coordination integration

Coordination contract остаётся владельцем event/task-state semantics. Context architecture
определяет только consumption projection:

- material events остаются durable/replayable вне model window;
- consumer хранит authoritative checkpoint, а не только «последний увиденный текст»;
- на sync point runtime получает gap-free eligible events после checkpoint;
- events deduplicate-ятся по identity и ancestry, сортируются с учётом causal/order metadata;
- findings/hypotheses/failures сохраняют epistemic status;
- authoritative `task_claimed`, `plan_changed`, cancellation и completion берутся из control
  state/accepted transitions, не из свободного prose;
- selected updates добавляются в next ContextPackage, после чего checkpoint advance
  фиксируется независимо от того, процитировала ли их модель.

Coordination-to-Memory promotion остаётся отдельным explicit lifecycle.

## 14. Verification context

Verifier получает purpose-built evidence package, а не весь worker private context:

- exact task/subtask and acceptance contract revision;
- claimed result/change set;
- relevant tool outcomes and artifact versions;
- environment identity/freshness;
- deterministic checks and their outputs;
- known limitations, unresolved assumptions и unknown effects.

Verifier result — отдельный authoritative outcome после trusted validation. Уверенный worker
summary не заменяет verification. При недостаточном evidence verifier возвращает
failed/error/unknown согласно своему contract, а не угадывает.

## 15. Background execution и recovery

Для resumable/background run durable semantic checkpoint включает:

- principal/task/run/agent/subtask identities и owner epoch;
- accepted task/plan/policy revisions;
- lifecycle status, budgets consumed/reserved и cancellation state;
- completed step outcomes и remaining dependencies;
- pending approvals без восстановления истёкшего grant;
- tool action journal с unknown outcomes;
- artifact/evidence refs и verifier state;
- coordination consumption/publication checkpoints;
- optional bounded private recovery summary с provenance.

Не требуется сохранять chain-of-thought. Resume строит новый ContextPackage из canonical state,
evidence и разрешённого private checkpoint. Runtime сначала reconcile-ит ownership, policy,
external state и interrupted actions; только затем продолжает model loop.

## 16. Mutation, invalidation и retention

- Source correction/retraction/erasure инвалидирует dependent summaries, indexes, caches и
  future projections.
- Изменение policy/audience немедленно влияет на eligibility; старый package не даёт право на
  новый action.
- Package immutable после отправки конкретному model turn; новый context создаёт новый package.
- Conversation retention, private checkpoint retention, tool evidence, artifacts,
  coordination, Memory и audit имеют независимые policies.
- Erasure проходит dependency closure; нельзя оставить доступный derived пересказ sensitive
  source и заявить, что source забыт.
- Audit хранит manifest/decision metadata по data-minimization policy, но не обязан хранить
  полный private prompt или reasoning.

## 17. Failure semantics

| Failure | Required behavior |
| --- | --- |
| Source unavailable | Использовать только допустимый cached/evidence version с freshness marker либо abstain/refresh |
| Retrieval/reranker failure | Явный degrade; mandatory local state и security filters не обходятся |
| Context overflow | Reproject/split/STOP; не silent-drop critical items |
| Summary failure | Сохранить previous authoritative/history state; не commit partial replacement |
| Conflicting sources | Показать conflict или policy-based resolution с provenance |
| Policy changed after package build | Revalidate before side effect; stale package не authorizes action |
| Coordination gap | Replay/resync before checkpoint advance; не продолжать с придуманным state |
| Agent/process crash | Fence stale instance, reconcile unknown actions, rebuild from durable state |
| Provider rejects payload | Re-render compatible projection без изменения authority/scope |
| Prompt-injection suspicion | Mark/quarantine context, reduce capabilities or require review; data не становится instruction |

## 18. Observability

Без раскрытия private reasoning должны быть доступны:

- package identity/hash, source revisions и projection decisions;
- token estimates/actual usage и reserve/overflow reasons;
- omitted/truncated/redacted/stale/conflict markers;
- policy/egress decision и model/provider identity;
- retrieval/summary/assembly errors и fallback path;
- tool/action/verification correlations;
- recovery and coordination checkpoints.

Telemetry не является authoritative task state, evidence acceptance или Memory.

## 19. Integration with existing runtime contracts

| Contract | Context implication |
| --- | --- |
| Ask | `Stateless` означает no execution-side state mutation, not context-free: Ask получает bounded read-only context projection; Memory write по-прежнему только explicit confirmation |
| Plan | Читает evidence и формирует versioned plan; adaptive replanning создаёт новую accepted revision, не hidden prompt mutation |
| Act | Получает immutable assigned contract projection; policy/scope enforcement остаётся вне prompt |
| Auto | Использует те же assembly/isolation/recovery invariants; context design не делает coordinated mode default |
| MWV | Может продолжать simple hierarchy; packet/messages должны постепенно стать explicit projection, но current mode не объявляется multi-agent |
| Desktop | Live environment observations требуют строгой freshness/identity; remembered state не разрешает host action |
| Multi-agent coordination | Публикует typed shared events; каждый consumer получает selective projection, не общий transcript |
| Memory | Long-term subsystem с отдельным acceptance/retention; runtime получает только scoped projection |

## 20. Runtime relationship

### Current-runtime impact / migration implications

Target contract не зависит от сохранения current abstractions. Полная evidence/verdict table
находится в research document; нормативные implications существенных расхождений:

- current `(principal_id, session_id)` сохраняется как security/attachment anchor, но не может
  подменять conversation/task/run/agent/turn identities;
- one mutable `Agent` и `short_term` на session допустимы для simple current modes, но должны
  быть superseded как модель concurrent isolated agents и durable private context;
- `TaskPacket` revisions/hash/scope/policy/budgets/verifier следует reuse with modification,
  однако arbitrary `messages`/`context` не являются target ContextPackage;
- current `RunContext` должен быть substantially reworked/superseded: local app/security state
  и model-visible input должны иметь отдельные contracts;
- `AgentToolLoop`, `ToolGateway` и verifier boundaries следует reuse, добавив context/evidence
  identity, result projection и recovery semantics;
- UI session history, `active_task`, artifacts и event replay остаются current UI/storage
  surfaces; они не становятся authoritative context/task/evidence store;
- current Memory confirmation/principal partitioning сохраняются, но Memory retrieval выдаёт
  candidates для assembler, а не напрямую authoritative prompt truth;
- `SessionSummarizer`, fixed char slots и mechanical head truncation должны быть superseded как
  target continuity/compaction semantics;
- process-local background/cancellation handles не подходят для durable runs; legacy
  Auto shard/pool shapes не определяют target context taxonomy.

Migration должна сохранять current histories, simple modes и security enforcement, но не через
compatibility layer, превращающий session row, summary или UI event buffer в новый source of
truth. Runtime refactoring и migration sequence не входят в эту architecture-only задачу.

## 21. Implementation-defined decisions

Отложены без ослабления invariants:

- concrete storage, serializer, API и type names;
- tokenizer/token estimator и provider-specific rendering;
- exact token allocations, thresholds и summary triggers;
- retrieval algorithms, embeddings, reranker и cache;
- artifact storage и encryption;
- checkpoint cadence и retention periods;
- summary model/validation method;
- distributed executor/fencing mechanism;
- UI visualization и operator controls;
- migration sequence from current sessions/Agent/TaskPacket.

Эти решения должны реализовывать contract, а не переопределять source of truth через удобный
prompt, session row или cache.
