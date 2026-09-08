# Multi-Agent Coordination Contract — target architecture

**Status: target. Механизм не реализован в current SlavikAI.**

Этот документ является нормативной детализацией discovered requirement из
[`ARCH_CANON.md`](ARCH_CANON.md). Он определяет semantic contract, а не component topology,
transport, database schema, API или wire format. Research и rejected alternatives находятся в
[`MULTI_AGENT_COORDINATION_RESEARCH.md`](MULTI_AGENT_COORDINATION_RESEARCH.md).

## 1. Capability boundary

SlavikAI должен поддерживать shared coordination, когда несколько активных agents реально
взаимодействуют в рамках одного task run. Это optional execution capability, а не обязательный
режим для каждой задачи.

First-class допустимыми остаются:

- single-agent execution;
- current MWV;
- `orchestrator -> isolated worker -> final result -> orchestrator`;
- простая иерархическая delegation;
- coordinated multi-agent execution по этому contract.

Выбор execution mode является policy-governed control-plane decision. Orchestrator оценивает
task structure, dependencies, budgets и risk, но выбирает только среди режимов, разрешённых
effective policy. Trusted policy/runtime может потребовать coordinated mode для определённого
класса задач, запретить или ограничить его из-за security/cost/capability/health constraints,
либо потребовать downgrade/STOP. Model suggestion сама по себе не является authority выбора.
Простая задача не должна превращаться в swarm только потому, что runtime умеет запускать
несколько agents, если обратное явно не предписано policy.

## 2. Target model

Целевая модель:

~~~text
                          user / caller
                               |
                         Orchestrator
            decomposition, membership, policy, lifecycle,
              plan authority, recovery, final aggregation
                               |
              +----------------+----------------+
              |                                 |
      authoritative task state        typed coordination history
      ownership / dependencies         explicit work events / replay
              |                                 |
       +------+------+                    +-----+------+
       |             |                    |            |
   Agent A        Agent B              Agent C      Verifier
 private ctx     private ctx           private ctx   private ctx
       |             |                    |            |
       +------- policy-scoped publish / receive -------+
~~~

Task state и coordination history логически разделены:

- **authoritative task state** определяет принятые lifecycle transitions, ownership,
  dependencies и cancellation;
- **typed coordination history** сохраняет явно опубликованные assertions, results,
  artifacts и notifications;
- **orchestrator** остаётся control-plane authority, но не обязан быть relay для каждого
  сообщения между peers;
- **agent private context** никогда не является частью общего слоя по умолчанию.

Реализация может разместить эти responsibilities в одном или нескольких компонентах, если
семантические границы сохраняются.

## 3. Термины

- **Logical task** — пользовательская задача с устойчивым `task_id` и revision lineage.
- **Task run** — одна execution incarnation logical task. Recovery может продолжить run;
  явный restart после terminal state создаёт новый run identity.
- **Coordination scope** — bounded пространство участников, task state и events одного task
  run.
- **Participant** — orchestrator, worker, specialist или verifier, зарегистрированный в scope
  с agent identity, role и policy-derived permissions.
- **Private working context** — model messages, scratchpad, hidden reasoning, local summaries,
  tool-loop history и иное содержимое, не опубликованное явно.
- **Coordination event** — immutable typed record о явно опубликованном occurrence.
- **Material coordination event** — event, потеря которого может изменить task state,
  ownership/readiness, plan, execution decision, verification, recovery, final aggregation или
  объяснимость существенного результата/conflict. Сюда также относится work information, на
  которое разрешено полагаться другому participant. Progress hint является ephemeral только
  если его потеря не может повлиять ни на одно из этих свойств.
- **Task-state transition** — authoritative изменение состояния task/subtask, принятое
  coordinator/control plane с concurrency precondition.
- **Artifact** — адресуемый результат работы с provenance и integrity metadata; event обычно
  передаёт reference, а не весь payload.

## 4. Non-negotiable invariants

1. У каждого agent отдельный private working context.
2. Никакой transcript, chain-of-thought или hidden reasoning не публикуется автоматически.
3. Agent получает только explicitly published coordination data, разрешённые его audience и
   effective policy.
4. Coordination scope не пересекает `principal_id` boundary.
5. Scope принадлежит одному principal и одному task run. Session является interaction/
   continuity attachment, но session-wide shared chat/event stream не является coordination
   source of truth.
6. Event от model/agent является утверждением publisher, а не автоматически истинным фактом.
7. Agent не может самовольно создать authoritative task-state transition, approval,
   permission или policy exception.
8. Task claiming атомарно: у claimable subtask не более одного active owner в одной ownership
   epoch.
9. Delivery и retries допускают duplicates; consumers обязаны быть idempotent. Contract не
   обещает magical exactly-once processing.
10. Material events и accepted task transitions доступны для replay/recovery до закрытия и
    в течение policy-defined retention scope.
11. Coordination history, task state, Memory, audit log и telemetry — разные logical stores
    и trust domains, даже если implementation физически совмещает их.
12. Shared coordination не расширяет tools, filesystem scope, network access, approvals или
    data visibility ни одного participant.
13. Materiality определяется semantic event type и trusted policy, а не произвольным выбором
    publisher ради обхода durability/audit requirements.

## 5. Scope, membership и visibility

### 5.1. Scope identity

Security identity scope концептуально включает:

`principal_id + task_id + task_run_id + coordination_scope_id`

Эти значения не обязаны буквально повторяться в каждом wire payload. Они должны быть
получены из authenticated runtime context либо защищённого event envelope, а не приниматься
на доверие из model-generated content.

Origin/current `session_id` является обязательной audit/interaction reference, но не заменяет
task-run identity. Task run может пережить client disconnect и быть явно reattached к другой
session того же principal без слияния private agent contexts.

Coordination является task-run scoped:

- session хранит пользовательскую continuity, origin/attachment и может ссылаться на несколько
  task runs;
- logical task связывает revisions/retries;
- task run ограничивает active participants, ownership epochs и replay;
- один global/session group chat для всех tasks запрещён как target source of truth.

### 5.2. Membership

Orchestrator/control plane регистрирует participant до publish/receive. Membership содержит
как минимум agent identity, role, assigned subtasks, allowed event classes/audiences и
effective policy reference. Agent не может добавить себя или sibling в scope через event.

Sibling может видеть события другого sibling, если одновременно выполнены условия:

- оба состоят в одном coordination scope;
- event audience включает receiver;
- type/subtask subscription разрешена;
- security/policy filter разрешает payload или artifact reference.

Orchestrator имеет visibility ко всем coordination events и task-state transitions внутри
своего scope, но не получает private contexts agents. Verifier visibility определяется
verification assignment и может быть шире worker visibility, не пересекая principal/policy
boundary.

### 5.3. Cross-task и cross-session flows

Direct cross-task и cross-principal delivery не входит в baseline target. Cross-session
broadcast также запрещён: authorized reattachment той же session-independent task run не
является пересылкой events в чужой scope. Переиспользование результата другой задачи
происходит только как explicit import/reference, authorized orchestrator action и новый event
в receiving scope с сохранённым provenance. Cross-principal sharing запрещён без отдельного
будущего contract и user-authorized policy.

Baseline намеренно исключает collaborative multi-user task, даже если participants принадлежат
разным разрешённым principals одного deployment. Такая capability потребует отдельного
contract для membership authority, consent, data ownership, approval delegation, revocation и
audit. Ослаблять single-principal boundary локальным implementation exception запрещено.

## 6. Semantic event envelope

Каждый accepted event имеет следующие обязательные semantic fields независимо от wire
encoding:

- stable `event_id` и contract/event type version;
- `event_type`;
- authenticated publisher identity и role;
- coordination scope identity;
- logical task и optional subtask subject;
- audience/routing intent;
- producer-local sequence или equivalent ordering marker;
- occurrence time и authoritative recorded time;
- correlation/trace identity;
- causation/reference IDs, когда event является следствием другого event/transition;
- typed payload;
- provenance/evidence/artifact references, где применимо;
- sensitivity/redaction classification.

Каждая publication attempt имеет отдельный acceptance outcome. Rejected attempt не становится
accepted event и хранится только как redacted audit metadata с reason, если этого требует
policy.

`principal_id`, permissions и publisher role не могут определяться только payload, созданным
LLM. Большие и чувствительные данные передаются через policy-checked artifact references,
а не автоматически копируются в event.

## 7. Минимальные semantic event types

Названия ниже фиксируют семантику; точные wire names остаются implementation-defined.

| Event | Назначение и обязательные type-specific fields | Кто публикует / кто получает | Durability | Task-state effect |
| --- | --- | --- | --- | --- |
| `finding` | Проверяемое observation/assertion; statement, evidence refs, source/freshness, uncertainty | Assigned agent/verifier/orchestrator; relevant subscribers + orchestrator | Durable | Нет; не становится fact автоматически |
| `hypothesis` | Рабочая версия; proposition, rationale, falsification condition или next check | Agent/orchestrator; relevant peers + verifier по assignment | Durable | Нет |
| `failure` | Неуспешная ветка/tool/subtask attempt; operation/branch, error class, evidence, impact, retryability | Agent/runtime/verifier; owners зависимых subtasks + orchestrator | Durable | Может trigger policy decision, но сам не меняет state |
| `artifact` | Созданный/изменённый result; artifact ID/ref, type, producer, integrity/version, status | Producer; explicitly authorized consumers + orchestrator/verifier | Durable metadata; payload по отдельной retention policy | Нет; наличие artifact не означает acceptance |
| `blocker` | Активная зависимость или missing authority/input; affected subtasks, cause, required resolution, severity | Любой participant; affected owners + orchestrator | Durable до resolution/closure | Может сделать dependent subtask not-ready через authoritative transition |
| `blocker_resolved` | Закрытие ранее опубликованного blocker; blocker ID, resolution/evidence refs, resolver | Authorized resolver/control plane; те же affected consumers | Durable | Может позволить authoritative ready transition |
| `task_claimed` | Принятая exclusive ownership transition; subtask, owner, ownership epoch/version, bounded validity/liveness rule | Только authoritative task coordinator после successful atomic claim | Durable; broadcast affected subscribers | Да |
| `task_released` | Добровольный release, expiry/reclaim или owner loss; subtask, prior owner/epoch, reason | Только authoritative task coordinator | Durable | Да |
| `task_result_submitted` | Candidate result владельца; subtask, ownership epoch, result/artifact refs, acceptance evidence | Active owner; orchestrator/verifier/dependents по policy | Durable | Нет; это submission, не completion |
| `task_completed` | Принятая terminal transition; subtask, accepted result refs, verifier/acceptance refs, final revision | Только authoritative task coordinator после validation | Durable; broadcast dependents | Да |
| `task_failed` | Принятая terminal failure; subtask, final attempt/epoch, cause and evidence refs, retry/replan decision | Только authoritative task coordinator | Durable; broadcast dependents | Да |
| `task_cancelled` | Принятая cancellation; subject, reason, authority, ownership epoch/revision | Orchestrator/control plane через task coordinator | Durable; обязательно active owner/dependents | Да |
| `plan_change_proposed` | Предложение изменить decomposition/dependencies/scope; base plan revision, rationale/evidence, affected subtasks | Любой authorized participant; orchestrator и affected reviewers | Durable | Нет |
| `plan_changed` | Принятая существенная смена плана; old/new plan revision, reason, affected subtasks | Только orchestrator или authorized planner through control plane | Durable; affected participants | Да, через versioned plan/task transitions |
| `verification` | Verification outcome; subject result/artifact, verifier identity/profile, status, evidence refs | Assigned verifier/runtime; orchestrator + affected owners/dependents | Durable | Не меняет state самостоятельно; coordinator использует его для acceptance |

Agent может предложить replan, completion или claim, но authoritative transition появляется
только после control-plane acceptance. Это отделяет speech act модели от принятого состояния.

## 8. Trust и epistemic status

Coordination layer не является truth engine.

- `finding` — attributed assertion с evidence, а не canonical fact.
- `hypothesis` — явно speculative statement.
- `artifact` — созданный object, но не доказательство correctness.
- `task_result_submitted` — candidate result владельца.
- `verification` — результат определённого verifier profile и evidence, не абсолютная истина.
- `task_completed` — authoritative workflow decision, что acceptance contract выполнен; это не
  превращает все текстовые claims внутри результата в вечное знание.

Contradictory findings не перезаписывают друг друга по last-write-wins. Они сохраняются как
разные attributed events. Orchestrator или verifier может опубликовать verification/resolution
с explicit references. Неопределённый конфликт остаётся видимым и влияет на aggregation.

## 9. Routing и consumption

### 9.1. Routing modes

Contract поддерживает:

- task-scope broadcast для событий, влияющих на всех participants;
- targeted delivery к конкретным roles/agents;
- subtask/dependency routing;
- subscriptions по разрешённым event types и subjects.

Publisher выражает routing intent, но authoritative policy filter определяет фактическую
audience. Orchestrator задаёт membership и routing policy, однако coordination layer может
доставлять accepted events peers напрямую без LLM/orchestrator relay.

### 9.2. Live delivery и replay

Push/wakeup является latency optimization, а не единственным источником correctness. Agent
имеет checkpoint и может восстановить пропущенные material events через replay плюс current
task-state snapshot.

Новые events вводятся в model context только в controlled synchronization points — между
model turns, перед следующим tool action, после tool result или при explicit interrupt. Нельзя
асинхронно смешивать raw event payload с незавершённым private reasoning.

Urgent control events (`task_cancelled`, `task_failed`, scope closure, policy revocation)
должны останавливать
новые действия на ближайшей safe boundary. Уже начатое external atomic action завершается или
компенсируется по его собственному execution contract.

## 10. Authoritative task coordination

### 10.1. Subtask state

Coordination task state различает как минимум:

`proposed -> ready -> claimed -> running -> result_submitted -> completed`

и terminal/exception paths:

`blocked`, `released/reclaimable`, `failed`, `cancelled`.

Точный набор persisted enum values implementation-defined, но следующие различия обязательны:

- ready не равно claimed;
- result submitted не равно completed;
- failure attempt не равно terminal task failure;
- lost owner не равно successful release;
- cancellation не равно failure;
- blocked dependency не равно active ownership.

### 10.2. Claim invariants

- Claim принимается одной atomic compare-and-set-like transition относительно expected
  subtask revision/state.
- Одновременно успешен только один competing claimant.
- Accepted claim получает ownership epoch/version. Late events старой epoch не могут менять
  current task state.
- Claim имеет bounded liveness semantics: owner обязан подтверждать жизнеспособность способом,
  определённым runtime policy, либо supervisor должен иметь наблюдаемое основание считать его
  lost.
- Reclaim выполняется только authoritative transition после expiry/loss/cancel/release; новый
  owner получает новую epoch.
- Completion принимается только от current owner epoch и только при выполнении acceptance/
  verification policy.

Lease, heartbeat, supervised process handle или иной mechanism не фиксируются этим документом.
Фиксируется только отсутствие вечного abandoned ownership и защита от late stale owner.

### 10.3. Dependencies и duplicate work

Subtask имеет explicit dependencies и conflict/resource scope. Agent не должен claim-ить
not-ready subtask. Duplicate research может быть разрешён orchestrator намеренно для
independent verification; в остальных случаях authoritative claim предотвращает случайное
дублирование ownership.

Semantic duplicate events не удаляются только по одинаковому text/hash: два независимых
observations могут быть разными evidence occurrences. Technical redelivery определяется по
event identity; semantic consolidation требует отдельного attributed decision.

### 10.4. Adaptive replanning во время Act

Начало Act не делает plan навсегда immutable. `plan_change_proposed`, material finding,
failure, blocker, policy change или verification result могут запустить bounded adaptive
replanning transaction.

- Worker/participant не изменяет active plan, свой TaskPacket или policy самостоятельно; он
  публикует proposal/evidence относительно конкретной base plan revision.
- Orchestrator либо другой explicitly authorized planner выполняет replanning как отдельную
  control-plane transaction без внешних execution side effects.
- Accepted change создаёт новую plan revision и authoritative `plan_changed`; in-place
  mutation старой revision запрещена.
- Затронутые agents останавливают новые действия на safe boundary и revalidate assignment,
  dependencies, budgets, policy и packet revision до продолжения.
- Незатронутые agents могут продолжить работу только если control plane установил, что их
  dependency/resource/policy scopes не изменились.
- Rejected или stale proposal не меняет active plan и остаётся наблюдаемым с reason.

Следовательно, «Plan остаётся execution read-only» означает отсутствие external side effects
в planning transaction. Это не означает запрет менять план во время Act через новую
authoritative revision.

## 11. Concurrency, ordering и idempotency

- Global total order всех agent events не требуется.
- Для одного producer сохраняется monotonic sequence или equivalent gap detection.
- Authoritative transitions одного subtask сериализуются по revision/ownership epoch.
- Causal links выражаются explicit references; wall-clock timestamps не доказывают causality.
- Delivery может быть at-least-once на semantic consumer boundary; duplicate event identity
  обрабатывается idempotently.
- Publish retry сохраняет logical event identity. Новый independent occurrence получает новый
  identity.
- Side effects не повторяются только из-за повторной доставки coordination event; consumer
  сначала проверяет current state, event identity и policy.
- Conflicting plan revisions или stale claims отклоняются наблюдаемо, а не merge-ятся
  last-write-wins.

Distributed consensus не является requirement baseline, пока один authoritative coordinator
владеет task-state partition. Если будущий deployment создаст несколько concurrent writers
без единого authority, это потребует отдельного architecture decision.

## 12. Isolation, security и approvals

### 12.1. Private context boundary

Автоматически запрещено публиковать:

- chain-of-thought, hidden reasoning и model scratchpad;
- полный private transcript/tool-loop history;
- secrets, raw credentials и auth material;
- unrelated user/session context;
- unrestricted tool output, если достаточно redacted summary/reference.

Publication является explicit runtime action с typed payload, provenance и policy check.
Model-generated summary остаётся untrusted input до validation/redaction boundary.

### 12.2. Capability boundary

Receive permission не даёт execute permission. Agent, увидевший artifact path, URL, command
или approval event, не получает право читать, запускать или изменять соответствующий resource.
Все tool actions продолжают проходить через `ToolGateway`/future equivalent execution
boundary с собственным effective policy.

Permissions participant фиксируются при spawn и могут быть только сужены без нового
authorized policy decision. Sibling и orchestrator не могут делегировать больше полномочий,
чем разрешено principal/task policy.

### 12.3. Approval boundary

Approval:

- принадлежит authenticated principal и конкретному scope/action;
- не переносится между agents через prose/event;
- не наследуется sibling автоматически;
- не может быть создан, расширен или подтверждён model output;
- после revocation блокирует новые затронутые actions независимо от buffered events.

Coordination может публиковать `blocker`/failure о required approval и уведомлять
orchestrator, но user approval проходит только canonical approval lane.

## 13. Persistence и data boundaries

Выбран hybrid persistence contract:

- ephemeral delivery допустима для progress hints, deltas и wakeups;
- material coordination events и authoritative transitions durable/replayable;
- current task-state snapshot является projection authoritative transitions либо
  транзакционно согласованным state, но не заменяет history;
- artifacts имеют отдельный lifecycle и references;
- closed scope становится immutable для обычных participants.

Все semantic event types из раздела 7 являются material по умолчанию. Implementation может
вводить дополнительные ephemeral progress types, но обязано доказать, что их потеря не влияет
на correctness, state, recovery, verification, aggregation или audit. Оно не может понижать
указанные semantic events до ephemeral только ради стоимости хранения.

Logical boundaries:

| Data class | Назначение | Может влиять на execution correctness | Автоматически Memory |
| --- | --- | --- | --- |
| Coordination history | Явно опубликованные work events и causality | Да, через consumers/policy; assertions не authoritative | Нет |
| Authoritative task state | Ownership, dependencies, accepted transitions | Да | Нет |
| Audit log | Кто/что/по какой policy принял или отклонил действие | Для расследования и compliance | Нет |
| Observability telemetry | Latency, queue depth, tool timing, health | Не является domain truth | Нет |
| Long-term Memory | Отобранное знание между задачами | Только через отдельный retrieval/use policy | Только explicit promotion |

Promotion из coordination history в Memory требует отдельного policy/consent path,
provenance, epistemic status и, где нужно, verification. Scope retention не должна зависеть от
Memory retention.

## 14. Failure semantics

### Agent crash / lost worker

- active ownership не исчезает молча;
- supervisor отмечает owner lost/expired, делает subtask reclaimable новой transition;
- late old-epoch result сохраняется для audit при policy allowance, но не завершает subtask;
- partial artifacts остаются candidate/partial и не принимаются автоматически.

### Timeout

Timeout публикуется как failure/lifecycle evidence. Retry создаёт новую attempt identity, а
reclaim — новую ownership epoch. Timeout не доказывает отсутствие внешнего side effect;
verification/compensation contract остаётся обязательным.

### Orchestrator failure

Новые claims, plan changes и final aggregation приостанавливаются до восстановления control
plane authority. Agents завершают только уже начатое atomic action и останавливаются на safe
boundary, если план явно не разрешает independent offline-safe continuation. Durable scope
state должен позволять authorized recovery без private contexts.

### Malformed или policy-rejected event

Event не попадает в accepted coordination history и не влияет на task state. Rejection
наблюдаем с reason code и correlation metadata, но sensitive rejected payload не должен
размножаться в logs.

### Coordination layer unavailable

Silent fallback к uncoordinated parallel execution запрещён. Допустимы только:

- pause/STOP с сохранением ownership и checkpoint;
- orchestrator-authorized downgrade к hierarchy/single-agent, если нет unresolved shared
  dependencies и downgrade зафиксирован как plan change;
- завершение явно отмеченной independent offline-safe unit до ближайшей safe boundary.

### Abandoned task и cancellation

Abandoned ownership становится reclaimable по authoritative decision. Cancellation имеет
revision/epoch, доставляется active owners и блокирует новые actions. Completion, пришедший
после cancellation, не воскресит task автоматически.

## 15. Observability и audit

Для debugging/recovery должны быть доступны без private reasoning:

- scope lifecycle и membership changes;
- accepted/rejected event metadata и reason codes;
- task-state transitions, expected/current revisions и claim epochs;
- delivery checkpoint/gap/resync signals;
- causation/correlation links;
- artifact/verification references;
- policy decision identity и approval reference без secret payload;
- agent health, timeout, cancellation и recovery outcomes.

Trace связывает coordination с tool/runtime traces, но telemetry не становится source of
truth. Redaction применяется до durable logging; audit visibility также principal/policy
scoped.

## 16. Lifecycle

Baseline lifecycle допускает ветвления и retries, но сохраняет следующие boundaries:

1. Control plane применяет effective policy; orchestrator выбирает разрешённый execution mode
   либо выполняет policy-mandated coordinated/single/hierarchical mode или STOP.
2. Для coordinated mode создаются task run, coordination scope, plan revision и policy
   snapshot/reference.
3. Orchestrator регистрирует participants и ready subtasks.
4. Agents атомарно получают ownership и работают в private contexts.
5. Agents явно публикуют material findings, hypotheses, failures, blockers и artifacts;
   relevant peers получают их live или через replay.
6. Owners submit results; verifier проверяет там, где требует contract.
7. Coordinator принимает completion и разблокирует dependencies.
8. Orchestrator агрегирует accepted results, unresolved conflicts и verification state в final
   response.
9. Scope закрывается; normal publishing прекращается, history остаётся в retention boundary.
10. Отдельная explicit процедура может promoted selected material в Memory.

Orchestrator может добавлять/отзывать participants, менять план новой revision или отменять
scope. Эти действия являются observed authoritative transitions, а не hidden prompt changes.

## 17. Integration с SlavikAI contracts

| Existing concept | Target integration |
| --- | --- |
| Ask | По умолчанию single-agent/stateless и не создаёт coordination scope. Переход к coordinated execution идёт через Plan/Auto decision, а не скрытый Ask side effect |
| Plan | Формирует initial plan и может выполнять bounded adaptive replanning transaction до или во время Act. Остаётся read-only относительно external execution, но accepted change создаёт новую authoritative plan revision |
| Act | Выполняет assigned packet/subtask в isolated context; worker не меняет plan/policy самостоятельно. После `plan_changed` затронутый Act продолжает только с revalidated assignment/new packet revision |
| Auto | Policy/runtime может require, forbid или constrain coordinated mode; orchestrator выбирает среди разрешённых режимов. Multi-agent не является default только из-за доступности capability |
| MWV | Остаётся допустимым current hierarchy. Worker result и verifier outcome могут стать typed events при embedding MWV в coordinated run; verifier authority сохраняется |
| `TaskPacket` | Сохраняются immutable scope/policy/budget/verifier principles; logical task, task run, subtask assignment и coordination policy должны быть представлены явно, даже если target использует новый envelope |
| `RunContext` | Current shape недостаточен. Target execution context обязан нести authenticated principal, agent/run/scope identity, policy reference и event checkpoint либо безопасные handles к ним |
| Session context | Session является interaction/continuity attachment, но не shared context window или identity task run. Scope может пережить disconnect и controlled reattachment в пределах того же principal. Current единого `SessionContext` type нет; target semantics не зависят от его появления |
| `principal_id` | Hard security partition для scope, events, artifacts, audit и recovery |
| Execution policy | Это target concept effective policy, а не заявление о current class. Он объединяет неизменяемые для run ограничения publish/receive/tools/data/approvals; enforcement остаётся на trusted boundaries |
| Approvals | Только canonical user approval lane; no transfer/elevation через agent events |
| Isolation modes | Каждый agent имеет отдельный model context и capability snapshot. Shared filesystem/workspace не предполагается; если разрешён, resource conflict/ownership задаётся отдельно |
| Background tasks | Должны иметь participant identity, bounded ownership, replay checkpoint и recoverable lifecycle; process-local async handle недостаточен как contract |
| Final response | Orchestrator агрегирует accepted results, artifacts, verification и unresolved conflicts, но не private transcripts |
| Memory | Отдельная long-term subsystem; coordination-to-Memory только explicit promotion с provenance/policy |

## 18. Current-runtime impact / migration implications

Normative target не зависит от сохранения current abstractions. Детальная evidence table и
verdicts находятся в research document. Краткие последствия:

- `AgentScope` security anchor можно сохранить, но current one-Agent-per-session provider
  должен быть superseded для concurrent isolated participants;
- `TaskPacket` полезен как immutable contract principle, но current one-task/one-worker shape
  требует extension или replacement;
- current `RunContext` правильнее substantially rework/supersede, чем перегружать optional
  dictionaries;
- UI `active_task` и `UIHub` не являются authoritative coordination state/history; UI может
  быть projection consumer;
- MWV остаётся поддерживаемым mode и не обязан становиться swarm;
- fragmented policy/approval plumbing можно reuse только за trusted effective-policy boundary;
- residual Auto pool/shard models не должны определять target types и подлежат отдельному
  deprecation decision;
- current session DB, event buffer и background tasks не выбирают target storage/executor.

Migration должна сохранять current simple modes, histories и security behavior, но не через
compatibility layer, который снова сделает UI/session chat source of coordination truth.
Конкретный migration plan не входит в этот документ.

## 19. Open questions и implementation-defined decisions

До implementation design отложены:

- local-only или distributed process topology;
- конкретный storage/transport/broker и transaction boundary;
- wire schema, serialization, schema evolution tooling;
- retention periods и limits;
- liveness mechanism и timeout values;
- scheduler/agent selection;
- artifact store и shared workspace/merge strategy;
- exact policy language и sensitivity classifier;
- recovery leader mechanism, если появится multi-writer deployment;
- UI и operator controls;
- compatibility/migration sequencing.

Ни одно из этих решений не может ослабить invariants scope, atomic ownership, private context,
policy enforcement, replayable material history и explicit epistemic status.
