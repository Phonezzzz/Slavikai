# Multi-Agent Coordination — research и alternatives

Дата исследования: 2026-09-08.

**Статус: research input, не runtime contract и не описание реализованной возможности.**
Нормативная target-модель вынесена в
[`MULTI_AGENT_COORDINATION_CONTRACT.md`](MULTI_AGENT_COORDINATION_CONTRACT.md) и включена в
канон ссылкой из [`ARCH_CANON.md`](ARCH_CANON.md). Этот документ не выбирает transport,
broker, database, wire format или implementation sequence.

## 1. Метод и design priority

Исследование разделяет три уровня:

1. обязательные инварианты SlavikAI: private agent context, principal isolation, policy и
   approvals на execution boundary, проверяемый результат;
2. target semantics, необходимые для устойчивой координации нескольких агентов;
3. current runtime как источник фактов, reuse opportunities и migration implications, но не
   как ограничение target design.

При конфликте между качеством target architecture и удобством адаптации текущего кода
приоритет имеет target architecture, если обратное не требуется подтверждённым compatibility
или security invariant. Sunk cost не является основанием сохранять abstraction.

Проект исследован на коммите `d0d842a`. Прочитаны `SOURCE_OF_TRUTH`, `ARCH_CANON`, current
`Architecture.md`, `runtime_contract_claims.json`, MWV flow и модели, Ask/Plan/Act workflow,
`AgentScope`/`ScopedAgentProvider`, UI session state/storage, UI events, Auto state,
`ComputerActivityLog`, approval policy, Desktop policy и `ToolGateway`. В репозитории нет
`docs/architecture/MWV_FLOW.md`; фактический документ находится в
[`docs/agent/MWV_FLOW.md`](../agent/MWV_FLOW.md).

## 2. Проверенные факты current runtime

### 2.1. Identity, session и isolation

- Mutable Agent создаётся и блокируется на scope `(principal_id, session_id)` через
  `AgentScope`/`ScopedAgentProvider`; persistent Memory разделена по principal.
- Session state хранит mode, active plan/task, Auto state, policy profile и UI messages.
- Единого current-типа `SessionContext` с multi-agent membership или coordination scope нет.
- `RunContext` содержит `session_id`, `trace_id`, workspace root, safe-mode, approved
  categories и retry attempt. В нём нет `principal_id`, agent identity, coordination scope,
  membership или event checkpoint.

Это подтверждает существующие security anchors, но не задаёт правильную гранулярность
multi-agent coordination.

### 2.2. Task и orchestration

- `TaskPacket` является immutable execution contract с `task_id`, `session_id`, `trace_id`,
  revision/hash, steps, policy, scope, budgets, approvals и verifier.
- Current MWV — строгий `ManagerRuntime -> WorkerRuntime -> VerifierRuntime`; один worker
  возвращает `WorkResult`, затем verifier принимает или отклоняет результат. Sibling
  communication и shared task claiming отсутствуют.
- UI Plan/Act хранит один `active_task`; lifecycle использует draft/approved/running/
  completed/failed и waiting-approval execution state. Это не модель нескольких одновременно
  owned subtasks.
- Current Auto v1 фактически использует один native tool loop. Оставшиеся `AutoShard`, coder
  pool и merge-shaped модели не подтверждают работающий multi-agent runtime: прежний
  `planner -> coder pool -> merge -> verifier` удалён из runtime entrypoints.

### 2.3. Events, replay и background execution

- `UIHub` публикует UI/session events подписчикам, имеет bounded in-memory buffer и короткий
  replay по event ID. При overflow события могут coalesce/drop, а consumer получает
  `session.resync_required`.
- UI event buffer не сохраняется как durable event history. Session storage сохраняет
  messages и snapshots active plan/task/Auto, а не coordination log.
- `ComputerActivityLog` записывает tool telemetry и затем drain-ится в session state. Это
  observability data, не агентское знание и не authoritative task state.
- Background execution существует как локальные async tasks, cancellation registries и
  session-scoped locks; durable ownership/recovery после process loss не определены.

Следовательно, наличие pub/sub mechanics не делает `UIHub` semantic coordination layer.
Использование его текущей delivery model как source of truth потеряло бы material findings и
task transitions.

### 2.4. Policy, approvals и verification

- В коде нет единого типа `ExecutionPolicy`. Эффективная политика распределена между
  `TaskPacket.policy/scope/approvals`, `ApprovalContext`, session tool state,
  `DesktopPolicyRuntime` и enforcement в `ToolGateway`/registry.
- Approval связан с конкретным execution context; Desktop persistent rules дополнительно
  имеют `subject_principal_id`. Событие от агента не может законно создать approval.
- MWV считает успехом только сочетание successful work и passed verifier.

Target contract поэтому должен сохранить enforcement boundary и verifier authority, но не
копировать раздробленное policy plumbing как идеальную модель.

### 2.5. Memory boundary

Current Memory требует отдельного confirm/edit-and-confirm для записи. Runtime coordination,
долгосрочное знание, audit и telemetry имеют разный срок жизни и разные основания доверия.
Coordination event не должен автоматически становиться Memory fact.

## 3. Внешние patterns и primary sources

Исследованы следующие первичные материалы:

- OpenAI Agents SDK различает manager-style agents-as-tools, handoffs и code orchestration;
  manager сохраняет final-answer ownership, а handoff передаёт управление и по умолчанию
  conversation history. [Agent orchestration][openai-orchestration],
  [handoffs][openai-handoffs].
- LangChain описывает supervisor/subagents как централизованный path с context isolation, а
  при multi-agent handoff требует явно проектировать, какие сообщения переходят между
  subgraphs. [Subagents][langchain-subagents], [handoffs][langchain-handoffs].
- AutoGen `Swarm` является group chat: participant responses broadcast, и все agents share
  one message context. Это полезный contrast case, но противоречит private-context invariant.
  [AutoGen Swarm][autogen-swarm].
- Anthropic production research system использует parallel orchestrator-worker и отдельно
  отмечает ограничение synchronous hierarchy: lead не может steer subagents во время работы,
  subagents не могут координироваться, а async добавляет state consistency/error propagation
  complexity. [Multi-agent research system][anthropic-research].
- A2A 1.0 разделяет stateful Task, Message, Artifact и status/artifact streaming, поддерживая
  opaque execution без раскрытия внутренних мыслей. Спецификация также предупреждает, что
  transient Messages нельзя считать надёжным каналом critical information.
  [A2A specification][a2a].
- Blackboard architecture организует heterogeneous knowledge sources вокруг общего
  problem-solving state и отдельного control problem. [Hayes-Roth 1985][blackboard].
- Actor model даёт isolated state и asynchronous message passing; practical actor delivery
  показывает, что transport-level send не равен business completion, а ordering обычно
  ограничено sender-receiver pair. [Hewitt, Bishop, Steiger 1973][actor-paper],
  [Akka delivery semantics][akka-delivery].
- CloudEvents 1.0.2 полезен только как reference для event identity/type/source, duplicate
  recognition и separation event format from transport; он не задаёт agent/task semantics.
  [CloudEvents 1.0.2][cloudevents].
- Kubernetes Lease и resourceVersion демонстрируют bounded ownership/liveness и optimistic
  concurrency без требования distributed consensus внутри одного authoritative control
  plane. Это reference для semantics, не предлагаемая dependency.
  [Leases][k8s-leases], [API concurrency][k8s-api].

## 4. Сравнение архитектурных вариантов

Оценка дана относительно инвариантов SlavikAI, а не как универсальный рейтинг.

| Pattern | Isolation / pollution | Coupling и routing | Observability / replay | Races, duplicates, ownership | Scalability | Policy / approvals | Совместимость и итог |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hierarchical orchestrator-worker | Сильная; low pollution | Tight parent coupling, простой routing, central bottleneck | Parent trace понятен; peer progress виден поздно, replay зависит от parent state | Parent может сериализовать claims, но duplicate research узнаёт при aggregation | Хорош для малого fan-out; lead context/cost/latency ограничивают рост | Сильный central enforcement | Близок MWV; сохранить как простой mode, не единственную topology |
| Group chat / shared conversation | Слабая: общий transcript загрязняет и раздувает context | Loose role coupling, trivial broadcast, трудно target/filter | Conversation replay прост, но prose плохо объясняет accepted state | Ownership и conflict resolution неявны; races скрыты в сообщениях | Context растёт для каждого participant, fan-out дорог | Сложно фильтровать secrets, approvals и capabilities per recipient | Противоречит private-context invariant; отклонить как base |
| Blackboard | Private reasoning можно отделить, но shared state широк | Knowledge sources связаны общей ontology/control strategy | Сильная inspectability, если entries typed/versioned; mutable board без history плохо replay-ится | Требует authority/CAS; иначе last-writer ambiguity и duplicate work | Хорош для heterogeneous agents, но hot shared state/contention растут | Mediation возможна; общий mutable board увеличивает blast radius | Использовать shared problem-state идею, не свободно изменяемый board |
| Typed event bus / pub-sub | Сильная при explicit publication/audience | Low producer-consumer coupling; subscriptions и routing policy сложнее | Сильные audit/replay при durable log и correlation | Redelivery/order требуют idempotency; event alone не даёт exclusive ownership | Хороший fan-out и async progress при bounded scopes | Type/scope/publisher хорошо policy-filter-ятся; approval всё равно отдельный | Подходит как communication primitive, недостаточен без task state |
| Shared task board / task-state | Private contexts не затрагивает | Agents coupled к общей state machine; query/watch просты | Current snapshot хорош; transition history нужна отдельно | Лучший вариант для atomic claim, dependencies, cancel/reclaim | Partition по task scope; contention локализован на hot subtasks | Central transition policy и authority естественны | Подходит для concurrency, но не передаёт findings/hypotheses |
| Actor/message passing | Сильная state isolation | Targeted routing естественен; discovery/broadcast требуют registry/router | Mailbox trace возможен; durable replay/business ack не встроены автоматически | Per-actor serialization помогает локально, но delivery/ownership across actors требуют protocol | Хорошая process distribution; operational complexity растёт | Capability references полезны, но approvals всё равно external authority | Возможная implementation model, не достаточный domain contract |
| Hybrid: orchestrator + typed events + authoritative task state | Сохраняет private contexts и selective sharing | Events decouple peers; orchestrator контролирует membership, не relay каждого event | Durable semantic history + state projections дают replay/debug | Versioned atomic transitions решают claims; event IDs/idempotency решают redelivery | Task partitioning и selective subscriptions ограничивают fan-out/context | Policy mediates publish/receive/state; peers не получают elevation | **Рекомендуемый target: совместим с инвариантами без привязки к current classes** |

### 4.1. Почему не hierarchy-only

Hierarchy-only хорошо соответствует current MWV и bounded specialist calls. Она остаётся
правильной для задач без взаимодействия peers. Но для параллельного исследования превращает
orchestrator в relay каждого finding/blocker; информация приходит другим workers слишком
поздно, duplicate work обнаруживается только при aggregation, а failure одного child может
блокировать parent wait.

### 4.2. Почему не shared group chat

Shared chat повышает awareness ценой автоматической передачи чужой истории. Он смешивает
work events, instructions, speculation и private scratchpad; сложнее применить per-event
audience, provenance и policy. Это прямо нарушает target invariant: agents не являются одним
LLM с общим context window.

### 4.3. Почему не pure blackboard или pure event bus

Свободно изменяемый blackboard рискует смешать assertion и authoritative state. Pure event
stream, наоборот, хорошо сообщает о claim, но два одновременных publishers всё ещё могут
считать себя owner, если нет атомарного state transition. Нужны разные semantics:

- append-only typed events для опубликованного знания, результатов и уведомлений;
- authoritative task-state projection для ownership, dependencies и lifecycle;
- orchestrator для decomposition, membership, policy, final aggregation и recovery decisions.

### 4.4. Почему не pure actors

Actor isolation полезна, но actor mailbox не определяет epistemic status finding, durable
replay, shared subtask ownership или final verification. Actor runtime может однажды стать
одной из implementations выбранного contract, но не должен быть самим domain contract.

## 5. Выбранная модель

Предварительная гипотеза подтверждена с важным дополнением:

> **Orchestrator + isolated agent contexts + task-scoped typed coordination history +
> authoritative shared task state.**

Это hybrid, а не обязательный swarm mode. Coordination включается только когда в одном task
run действительно работают несколько взаимодействующих agents. Single-agent, MWV,
hierarchical delegation и manager-owned final result остаются first-class paths.

Typed history отвечает на вопрос «что явно опубликовано и почему это увидели consumers».
Task state отвечает на вопрос «кто имеет право выполнять subtask и какая transition принята».
Orchestrator отвечает за control plane, но не обязан пересылать каждое peer event вручную.
Подробная normative semantics определена в target contract.

### 5.1. Уточнённые authority и lifecycle boundaries

- Orchestrator не имеет безусловной свободы включать coordination: effective policy/runtime
  может require, forbid или constrain mode; orchestrator выбирает только допустимый вариант.
- Single-principal coordination является осознанным security baseline. Collaborative
  multi-user tasks не запрещены навсегда, но требуют отдельного будущего contract, а не
  локального exception.
- `material` формализован через последствия потери event для state, ownership, plan,
  correctness, recovery, verification, aggregation и audit; publisher не может сам понизить
  событие до ephemeral.
- Plan остаётся read-only относительно external side effects, но не frozen: adaptive
  replanning во время Act разрешён как новая versioned authoritative transaction.

## 6. Current-runtime impact / migration implications

Эта таблица применяется после выбора target design и не ограничивает его.

| Current abstraction | Verdict | Расхождение и архитектурная причина | Impact / migration implication |
| --- | --- | --- | --- |
| `AgentScope(principal_id, session_id)` и principal storage partitioning | `reuse with modification` | Даёт current security anchor, но не agent/task/run identity; session не должна подменять durable task run | Сохранить principal boundary и session access checks; добавить отдельные participant/run/scope identities и controlled reattachment без ослабления isolation |
| `ScopedAgentProvider` one Agent + lock per session | `supersede` для multi-agent runs; `reuse as-is` для simple modes | Один mutable Agent и один session lock сериализуют работу и не моделируют isolated concurrent agents | Multi-agent runtime нуждается в отдельных agent instances/contexts и scoped lifecycle; simple current path может остаться |
| `TaskPacket` | `reuse with modification` | Сильный immutable execution contract, но один packet/worker и нет subtask ownership/dependency/coordination policy | Сохранить immutable policy/scope/budget/verifier principles; расширение или новый task envelope должно отделить logical task, subtask и run revision |
| `RunContext` | `supersede` либо существенно rework | Не содержит principal, agent identity, coordination membership/checkpoint; approved categories не являются полной policy snapshot | Ввести целевой execution context с explicit security and coordination identity; compatibility adapter допустим только на границе old MWV, не как target model |
| UI Plan/Act `active_task` lifecycle | `supersede` для coordination; `reuse as-is` для single task UI | Один mutable snapshot не поддерживает concurrent subtask claims, dependency graph и recoverable ownership | Новая authoritative task-state semantics должна быть независима; UI может проецировать её, но не владеть ею |
| Current MWV manager-worker-verifier | `reuse as-is` как допустимый mode; `reuse with modification` только при optional integration | Корректная hierarchy и verifier boundary, но peers отсутствуют | Не ломать MWV ради swarm. Если MWV участвует в coordinated run, его result/verification публикуются через новый contract |
| `UIHub` pub/sub и bounded replay | `unrelated` | UI delivery допускает drop/coalesce, short in-memory replay и resync; это несовместимо с durable material events/state transitions | Coordination contract не строится на `UIHub`. Он остаётся UI delivery surface; отдельная будущая projection может переводить coordination state в derived UI events |
| `ComputerActivityLog` и trace/tool logs | `unrelated` | Telemetry описывает действия, но не является agent assertion, task ownership или verification | Correlate через IDs для observability; не использовать как coordination truth и не повышать telemetry до Memory |
| `TaskPacket.policy`, `ApprovalContext`, session tool state, `DesktopPolicyRuntime` | `reuse with modification` | Enforcement полезен, но policy fragmented и нет единого agent-to-agent publish/receive authorization | Coordination должен использовать effective immutable policy snapshot/concept; события не выдают approvals и не расширяют capabilities |
| UI/session SQLite storage | `unrelated` к target storage choice | Current schema хранит chat/workflow snapshots, не durable semantic history | Не добавлять coordination tables автоматически; storage technology и migration выбираются отдельным implementation design |
| Background `asyncio` tasks/cancellation registries | `supersede` для recoverable background agents | Process-local task handle не даёт durable ownership/restart semantics | Target lifecycle должен переживать disconnect/process loss на semantic уровне; конкретный executor остаётся implementation-defined |
| `AutoShard`/coder pool/merge-shaped legacy models | `remove/deprecate` как архитектурное основание | Они отражают удалённый pipeline и могут создать ложное впечатление shipped coordination | Не расширять их. Удаление/совместимость — отдельная implementation задача; target types выводить из нового contract |
| Long-term Memory | `unrelated` к runtime coordination; explicit promotion interface в будущем | Иной trust, retention и consent lifecycle | Не писать coordination events в Memory автоматически; promotion требует отдельной policy/verification/provenance операции |

Ни один verdict в таблице не является implementation plan. Он фиксирует direction и не
разрешает runtime refactoring в рамках этого architecture-only изменения.

## 7. Обнаруженные gaps и конфликты

Прямого противоречия с `ARCH_CANON` нет: requirement уже помечен `target`, а current MWV и
single-agent paths явно остаются допустимыми. Обнаружены gaps, которые нельзя скрывать как
готовую capability:

- current `(principal_id, session_id)` Agent ownership слишком крупный для concurrent isolated
  agents;
- current session identity одновременно служит runtime lifetime boundary; target task run
  должен переживать disconnect и не зависеть от UI event buffer;
- один `active_task` не выражает shared subtask state;
- `UIHub` replay недостаточно durable для correctness;
- current policy plumbing не образует единого `ExecutionPolicy` contract;
- current `RunContext` не переносит principal/agent/coordination identity;
- residual Auto pool models расходятся с documented current Auto v1 и должны считаться
  legacy-shaped data, а не reuse mandate.

Compatibility conflict с текущими пользовательскими histories/API не найден: target contract
не меняет current storage или endpoints.

## 8. Deferred research decisions

Намеренно не выбраны:

- process topology и local/distributed deployment;
- storage engine, broker, transport, API и wire serialization;
- exact retention periods, lease durations и capacity limits;
- event payload schemas и schema registry technology;
- scheduling algorithm и agent selection strategy;
- shared-workspace implementation и merge strategy;
- migration order для current sessions/Auto/MWV;
- UI representation.

Эти решения требуют отдельного implementation design после принятия semantic contract.

[openai-orchestration]: https://openai.github.io/openai-agents-python/multi_agent/
[openai-handoffs]: https://openai.github.io/openai-agents-python/handoffs/
[langchain-subagents]: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents
[langchain-handoffs]: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs
[autogen-swarm]: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/swarm.html
[anthropic-research]: https://www.anthropic.com/engineering/multi-agent-research-system
[a2a]: https://a2a-protocol.org/v1.0.0/specification/
[blackboard]: https://doi.org/10.1016/0004-3702(85)90063-3
[actor-paper]: https://www.ijcai.org/Proceedings/73/Papers/027B.pdf
[akka-delivery]: https://doc.akka.io/libraries/akka-core/current/general/message-delivery-reliability.html
[cloudevents]: https://github.com/cloudevents/spec/blob/v1.0.2/cloudevents/spec.md
[k8s-leases]: https://kubernetes.io/docs/concepts/architecture/leases/
[k8s-api]: https://kubernetes.io/docs/reference/using-api/api-concepts/
