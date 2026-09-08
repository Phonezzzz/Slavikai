# Context Architecture — audit, research и alternatives

Дата исследования: 2026-09-08.

**Статус: research/design rationale, не current runtime contract.** Нормативная target-модель
находится в [`CONTEXT_ARCHITECTURE_CONTRACT.md`](CONTEXT_ARCHITECTURE_CONTRACT.md). Документ не
выбирает database schema, API, transport или implementation sequence и не утверждает, что
target mechanism уже существует.

## 1. Метод и design priority

Исследование проведено target-first:

1. определены correctness, isolation, recovery, security и context-budget requirements;
2. проверены current code/data flows и existing contracts;
3. изучены primary external sources;
4. сравнены alternatives;
5. только после выбора target каждому current component дан verdict.

Current runtime рассматривается как evidence и источник reuse opportunities, а не ограничение
target architecture. Compatibility имеет приоритет только там, где подтверждён security,
product или public contract; sunk cost не является invariant.

Исследованы:

- `SOURCE_OF_TRUTH.md`, `ARCH_CANON.md`, `Architecture.md`, architecture index и claims;
- Multi-Agent Coordination research/contract и current MWV flow;
- пользовательский research input
  `CONTEXT_MEM_MEMORY_ARCHITECTURE_RESEARCH.md` (непринятый contract);
- `Agent`, routing/memory/tools mixins, model message types и provider boundary;
- `TaskPacket`, `RunContext`, Manager/Worker/Verifier, Plan/Act и Auto lifecycle;
- `AgentScope`, `ScopedAgentProvider`, session domain, `UIHub` и SQLite session storage;
- tool loop/results, artifacts, approvals/policy, cancellation/background handles, tracing;
- Memory capture, canonical atoms, retrieval, session summary и context slots;
- cloning, trimming, serialization, persistence и restoration paths.

## 2. Current context flow: проверенная карта

### 2.1. UI interaction и session history

~~~text
authenticated request
  -> principal + UI session resolution
  -> UIHub session snapshot (messages/workflow/model/policy/artifacts)
  -> append user message + persist whole session
  -> all messages for selected lane -> LLMMessage[]
  -> scoped mutable Agent(principal_id, session_id), one session lock
  -> Agent appends user/assistant messages to short_term (last 20)
  -> memory/context slots + policy instruction + optional web context
  -> Brain / AgentToolLoop
  -> assistant message + output/artifacts/workflow snapshot -> UIHub/SQLite
~~~

Факты:

- `AgentScope` содержит только `principal_id` и `session_id`; `ScopedAgentProvider` владеет одним
  mutable Agent и lock на этот scope.
- UI session хранит messages, output, files/artifacts, model, mode, active plan/task/Auto,
  workspace root, policy profile, tool state и pending decision.
- `ui_chat` перед каждым ответом читает всю lane history из `UIHub`, преобразует её в
  `LLMMessage[]` и вызывает scoped Agent.
- Agent сохраняет в `short_term` только user/assistant и механически оставляет последние 20
  сообщений. System/tool messages не попадают в durable private history через этот путь.
- UIHub по умолчанию держит до 500 сообщений на session и до 200 sessions; default TTL
  session — семь дней. При превышении message limit удаляется head истории.
- SQLite storage сохраняет session/workflow snapshot и при каждом save удаляет/reinserts
  `chat_messages`/`workspace_messages` в одной transaction. Это conversation/workflow storage,
  не append-only evidence/task-run history.
- После process restart `UIHub` восстанавливает messages и workflow snapshots. Новый Agent
  создаётся лениво; его `short_term` заново наполняется из UI history на следующем request.
- `Agent.conversation_id` создаётся случайно в constructor, но UI path связывает continuity
  прежде всего с `session_id`; durable mapping conversation/task/run identity не определён.

### 2.2. Current prompt assembly

`AgentMemoryMixin._build_context_messages` строит один дополнительный system message перед
`short_term`. Общий budget измеряется characters и заполняется фиксированным порядком:

1. pinned canonical atoms;
2. последние session summaries;
3. legacy recent notes;
4. negative feedback hints;
5. user preferences;
6. canonical Memory capsule;
7. project code/docs vector snippets;
8. current workspace file/selection.

Каждый slot обрезается по character limit и остаточному общему budget. Slot содержит label, но
не общий manifest source revisions, sensitivity, audience, freshness, transformation lineage
или explicit omission reason для model consumer. `canonical_memory` показывает confidence,
support/contradict counts и status; project snippets/legacy notes не имеют равной provenance
semantics. После этого policy mixin может добавить ещё system instructions, а web path — ещё
evidence. Единого финального token-budget validation для полного provider payload не найдено.

Current mechanism полезен как bounded injection baseline, но не является target context
assembler: порядок slots фактически задаёт приоритет, character count не provider token budget,
а mandatory task/policy/recovery state не отделено от optional retrieved context.

### 2.3. Tool-loop context

`AgentToolLoop` копирует входной message list в process-local history. После каждого model turn
он добавляет assistant message с tool calls/reasoning field, затем добавляет каждый `ToolResult`
как JSON `role="tool"` с `trust="untrusted_observation"`. JSON включает целиком `data`, `error`
и `meta`; общего size/sensitivity/artifact projection contract в loop нет.

Tool-loop history существует только на время вызова и возвращается в `AgentToolLoopResult`, но
обычный chat finalization сохраняет в short-term/UI только финальный assistant text. Поэтому
последующий turn обычно не получает exact prior tool call/result chain, даже если final answer
зависел от неё. Отдельные trace/tool logs и Computer events дают observability, но не полный
recoverable evidence package.

Положительный current invariant: model tool request проходит `ToolGateway`/registry и approval
checks; наличие instruction в prompt само по себе не authorizes action.

### 2.4. TaskPacket, RunContext, MWV и verifier

- `TaskPacket` — frozen versioned/hashable execution contract с goal/messages/steps,
  constraints, policy, scope, budgets, approvals, verifier и generic context.
- Current MWV builder копирует переданные messages в packet и может добавить Memory capsule в
  `TaskPacket.context`. Parent-to-worker projection как отдельный semantic operation отсутствует.
- `RunContext` содержит session/trace/workspace/safe mode/approved categories/retry attempt. В
  нём нет authenticated principal, conversation/task-run/agent/subtask identity, policy
  revision, context package identity или recovery/coordination checkpoint.
- Manager запускает одного Worker, затем Verifier. Retry создаёт новую packet revision и
  добавляет summarized verifier failure в constraints.
- Verifier получает `TaskPacket` + `RunContext` и запускает deterministic command/profile. Его
  `VerificationResult` содержит command, exit code, stdout/stderr, duration и excerpt, но нет
  общего evidence-package manifest или artifact lineage.
- UI Plan/Act хранит один `active_plan` и один `active_task` snapshot. Plan revision/hash
  проверяются при transition. Task payload может сериализовать packet и execution snapshot для
  approval resume, однако это не general crash-recovery journal.

### 2.5. Auto, background и recovery

Current Auto v1 создаёт новый process-local native tool loop из system prompt + goal, не из
общего ContextPackage. Auto state сохраняет run ID, goal, plan-shaped state, tool call summaries,
budgets, verifier/approval/error. Pause/resume после approval запускает bounded runtime logic,
но current semantics зависят от process/session structures.

Background execution/cancellation реализованы локальными `asyncio` tasks, registries, Events и
session locks. Chat cancellation registry хранит одну active generation на session только в
RAM. UI event replay — bounded in-memory buffer (512 events, десять минут); при overflow
возможны coalesce/drop и `resync_required`. Это подходит для UI streaming, но не для durable
task recovery или critical context delivery.

### 2.6. Artifacts, evidence, audit и telemetry

- UI output artifacts извлекаются из model response/code fences или создаются как canvas text.
  Session сохраняет artifact dictionaries, иногда включая полный text/file content.
- Tool-produced file paths и model-produced artifacts не объединены общей artifact version,
  integrity, provenance и audience model.
- `ComputerActivityLog` сохраняет bounded tool activity summaries/events в session state;
  `Tracer` и tool logs обеспечивают диагностику. Это не authoritative task/evidence store.
- Session export/import сериализует messages, workflow snapshots, policy profile и tool state,
  но не переносит полноценную execution/approval/recovery authority.

### 2.7. Memory и summaries

- Long-term Memory отделена от UI session DB и principal-scoped. Explicit write требует
  confirm/edit-and-confirm; это обязательный current contract.
- Canonical retrieval фильтрует active/conflict atoms по type/confidence/recency, затем vector
  rank/fallback и character packing. Current default скрывает conflicts в runtime capsule.
- `SessionSummarizer` берёт до 40 последних user/assistant messages, обрезает каждое до 500
  characters и сохраняет LLM summary как explicit canonical `FACT` с новым `session:*` key.
  Summary не хранит полный source range/revisions/coverage и затем может попасть в prompts.
- `short_term` head truncation, UI message truncation, session-summary truncation и slot
  truncation — разные несогласованные loss boundaries.

Исследование `CONTEXT_MEM_MEMORY_ARCHITECTURE_RESEARCH.md` полезно отделением evidence,
accepted knowledge, derived representations и selected runtime context. Оно также показывает,
почему freshness, provenance, temporal state, compression, privacy и retention нельзя свести к
retrieval score. Этот input не является принятым target contract и не определяет runtime design.

### 2.8. Current-code evidence map

Карта выше проверена по следующим implementation surfaces; список фиксирует evidence points,
а не делает структуру каталогов частью target contract:

| Область | Current source paths |
| --- | --- |
| Agent/session ownership | `server/agent_provider.py`, `core/agent.py`, `core/agent_tools.py` |
| UI chat/history/restoration | `server/http/handlers/ui_chat.py`, `server/ui_hub.py`, `server/ui_session_storage.py`, `server/http/common/session_transfer.py` |
| Prompt/context construction | `core/agent_memory.py`, `core/agent_routing.py`, `memory/memory_retrieval.py` |
| Tool-call context/results | `core/tool_loop.py`, `shared/models.py`, `core/desktop_runtime.py` |
| Task/Plan/MWV/verifier | `core/mwv/models.py`, `core/agent_mwv.py`, `server/http/common/workflow_runtime.py` |
| Auto/background lifecycle | `core/auto_runtime.py`, `server/http/handlers/ui_chat.py` |
| Policy/approval enforcement | `core/approval_policy.py`, `core/desktop_policy.py`, `core/tool_loop.py` |
| Artifacts/evidence/telemetry | `server/http/common/ui_artifacts.py`, `core/computer_activity_log.py`, `core/tracer.py` |
| Memory/summaries | `memory/canonical_atom_store.py`, `memory/canonical_aggregator.py`, `memory/memory_retrieval.py`, `memory/session_summarizer.py` |

## 3. Current source of truth vs derived data

| Current data | Current authority | Derived/cache status | Gap relative to target |
| --- | --- | --- | --- |
| Verified principal + principal storage path | Security authority | Нет | Нужно распространить principal identity на task/run/package |
| UI session messages | Primary current chat history | Model input — derived subset | Message history смешана с session lifecycle; bounded destructive trimming |
| UI active plan/task/Auto JSON | Current workflow snapshot | UI events — projection | Snapshot не даёт полный transition/recovery history |
| `TaskPacket` revision/hash | Execution contract внутри MWV/Plan | Worker prompt/messages — projection-like, но не explicit | Identity/context generic; whole messages copied |
| Approval stores/session categories | Enforcement input | Prompt explanation/decision UI — derived | Fragmented semantics и session overloading |
| Tool result | Outcome объекта tool call | Serialized model tool message/log/event | Raw vs verified effect и retention не унифицированы |
| Verifier result | Current acceptance signal | Excerpts/reports/UI — derived | Нет general evidence manifest |
| Canonical Memory atom | Current accepted Memory record | Vectors/capsules/summaries — derived | Current epistemic/provenance model ограничена |
| UIHub event buffer | UI delivery history | Ephemeral replay cache | Не durable и допускает loss/coalescing |
| Traces/Computer events | Observability | Summaries/UX projection | Не task truth и не Memory |

## 4. External primary-source findings

### 4.1. Context — finite selected input, не вся накопленная state

Anthropic определяет context engineering как повторяющуюся curation всего model input, включая
instructions, tools, external data и message history, и отмечает diminishing returns от
увеличения context. Это поддерживает per-turn assembly и отказ от «положить всё в общий чат».
[Effective context engineering][anthropic-context].

MemGPT демонстрирует virtual context management и tiers как способ работать за пределами
fixed context window. Это подтверждает необходимость различать stored state и active context,
но не доказывает конкретную tier/storage модель для SlavikAI. [MemGPT paper][memgpt].

### 4.2. Conversation history, app context и model context различаются

OpenAI Agents SDK явно отделяет local run context, который не отправляется LLM, от
conversation state и model-visible history. SDK позволяет фильтровать final model input и
handoff input; по умолчанию handoff может передать всю conversation history, поэтому isolation
требует explicit filter/projection. [Context management][openai-context],
[Handoffs][openai-handoffs].

OpenAI Sessions сохраняет conversation items между runs и отдельно позволяет ограничивать или
переупорядочивать history для model call без повторного сохранения old items. Compaction session
делает recoverable replacement и предупреждает о concurrent mutation. Это strong reference для
разделения retained history, selected input и atomic compaction, но не готовый domain contract.
[Sessions][openai-sessions].

### 4.3. Compaction недостаточно для long-running correctness

Anthropic сообщает, что compaction сама по себе не обеспечивает устойчивую длительную работу:
agents нужны incremental progress и explicit artifacts, чтобы следующий context не угадывал
неполное состояние. Это поддерживает canonical task state + artifacts/evidence вместо
summary-only recovery. [Long-running agent harness][anthropic-long-running].

### 4.4. Durable checkpoints и side-effect idempotency

LangGraph разделяет thread checkpoints и cross-thread long-term store; документация также
предупреждает, что resume может re-execute node, поэтому side effects должны быть idempotent,
а completed task results — checkpointed. Это подтверждает отдельные recovery/action journal и
Memory boundaries. [Persistence][langgraph-persistence],
[Determinism and idempotency][langgraph-idempotency].

### 4.5. Security не может жить только в prompt

MCP specification требует consent/access control и считает tool descriptions/annotations
untrusted без доверенного server; сам protocol не обеспечивает enforcement. OpenAI Agents SDK
также различает capability visibility и authorization конкретных arguments/resources.
[MCP specification][mcp], [OpenAI context management][openai-context].

Anthropic показывает, что внешние pages/documents являются prompt-injection surface и что
model-level robustness не является полной гарантией. Следовательно, context item должен иметь
trust/source boundary, а tool/policy enforcement остаётся вне model text.
[Prompt-injection research][anthropic-injection].

## 5. Alternatives

| Alternative | Strengths | Failure modes | Verdict |
| --- | --- | --- | --- |
| Full transcript as context | Простой continuity/replay | Context growth/rot, leaks private agent data, stale instructions, no typed state | Не target; только bounded interaction source |
| One mutable `SessionContext` object | Простая integration | Session становится universal ID, responsibilities и lifecycles смешиваются, races/recovery unclear | Отклонить как central abstraction |
| Summary-only rolling memory | Дешёвый prompt | Loss of constraints/evidence/provenance, compounding hallucinations, no exact recovery | Отклонить для authoritative/critical state |
| Pure retrieval over all persisted data | Flexible recall | Ranking подменяет authority/ACL/freshness; no mandatory coverage | Использовать только как candidate stage |
| Event sourcing everything including tokens | Полный forensic replay | Высокая стоимость/privacy, COT retention, сложно отделить semantics | Не требуется; durable material state/evidence selectively |
| Shared multi-agent chat | Awareness | Нарушает private contexts, broadcast pollution and prompt injection fan-out | Отклонить как default coordination context |
| Typed state + evidence + per-turn projections + separate Memory | Clear authority/isolation, bounded input, recovery, provider neutrality | Требует explicit taxonomy/manifest/invalidation | **Выбранный target** |

### 5.1. Почему не universal `SessionContext`

Session нужна как attachment и conversation continuity, но task может пережить disconnect,
одна conversation может запускать несколько tasks/runs, а один coordinated run — несколько
agents. Policy/tool attempts/artifacts имеют собственные identities/lifetimes. Универсальный
object либо разрастается optional fields, либо снова делает session security/task/model boundary.

### 5.2. Почему authoritative state и model context разделены

Модель должна видеть goal, constraints и progress, но prose representation не подходит для
atomic ownership, approval expiry, retry dedup или crash recovery. Runtime source records дают
correctness; ContextPackage даёт model ровно необходимое представление. Модель может предложить
change, но не self-authorize его.

### 5.3. Почему Memory отдельно

Task progress и exact tool outcomes нужны для recovery, но не обязательно полезны через месяц.
Personal preference может быть важна across tasks, но не authorizes текущий action. Один store
или физическая DB возможны, однако acceptance, authority, retention и projection semantics
остаются разными.

## 6. Selected target

Выбран **Context Projection Architecture** как концептуальное описание, не обязательное имя
компонента:

~~~text
Authoritative control state       Evidence / artifacts       Long-term Memory
 task/plan/policy/lifecycle        observations/outcomes      accepted knowledge
             \                           |                         /
              \                    eligibility                    /
               +------> selection + projection + reconciliation
                                     |
                               ContextPackage
                          principal/task/run/agent/turn
                                     |
                               isolated model call
                                     |
                         validated outputs/transitions
~~~

Key design decisions:

- authoritative records, evidence, private working context, Memory и derived views — разные
  logical layers;
- one immutable ContextPackage per model turn;
- explicit identities вместо перегрузки `session_id`;
- agent-private context и explicit parent/child/peer projections;
- mandatory context coverage + provider-aware reserves;
- summaries с lineage; critical state никогда не summary-only;
- policy/approval enforced locally and separately from prompt;
- tool results become typed evidence/artifacts before selective model projection;
- recovery from durable state/checkpoints without chain-of-thought;
- coordination consumption follows durable event checkpoint and control-state authority.

Подробные invariants и taxonomy находятся в target contract.

## 7. Current-runtime impact / migration implications

Verdicts следуют из target design; они не являются implementation plan.

| Current abstraction | Verdict | Current mismatch | Architectural implication |
| --- | --- | --- | --- |
| `AgentScope(principal_id, session_id)` | `reuse with modification` | Strong principal anchor, но session заменяет conversation/task/run/agent granularity | Сохранить principal isolation; добавить independent semantic identities и controlled reattachment |
| `ScopedAgentProvider` | `reuse as-is` для simple current modes; `supersede` для concurrent agents | One mutable Agent + lock per session сериализует participants и связывает private context с session | Target agent instances/contexts должны иметь own lifecycle; compatibility path может остаться для single-agent |
| `Agent.short_term` | `supersede` как target private-context model | Только last 20 user/assistant, process-local, механическое head trim, нет source/package lineage | Нужен scoped private working state с explicit projection/checkpoint semantics |
| `Agent.conversation_id` | `supersede` | Random process-local ID не связан durable с UI conversation/session/task | Target `conversation_id` имеет stable lineage; current field не переименовывать в authority |
| `LLMMessage` list | `reuse with modification` как provider representation | Полезный neutral message DTO, но смешивает retained history и final model input; provenance/trust только внутри tool JSON | Сохранить adapter role, но ContextPackage/manifest должен существовать до rendering |
| `_build_context_messages` slot builder | `supersede` | Fixed character slots/order, no final token validation, provenance/audience/freshness inconsistent | Заменить semantic assembler contract; отдельные retrieval helpers могут reuse позже |
| `TaskPacket` | `reuse with modification` | Хороший immutable execution principle, но copies messages и generic context; no package/run/agent identities | Сохранить revisions/hash/policy/scope/budget/verifier; explicit projection/envelope должен заменить arbitrary context bag |
| `RunContext` | `supersede` либо major rework | Не model-visible, что полезно, но недостаточно identities/policy/checkpoints; approved categories слишком узки | Target execution context отделяет trusted local dependencies/security from model package |
| Current MWV manager/worker/verifier | `reuse as-is` как simple hierarchy; `reuse with modification` для target integration | Correct bounded flow, но parent-child context projection/evidence package не explicit | MWV не ломать; adapters могут формировать ContextPackage и verifier package |
| UI Plan/Act state | `reuse with modification` для current UI; `supersede` как target task truth | Version/hash полезны, но one mutable active task snapshot, limited recovery history | UI становится projection consumer; authoritative task/run revisions имеют independent lifecycle |
| `SessionMode`/session domain | `reuse with modification` | Modes полезны, но session объединяет chat/workspace/runtime attachment | Оставить UX mode; не использовать как context/task identity |
| UIHub session snapshot | `reuse as-is` как current UI surface; `unrelated` к target context truth | Conversation/workflow persistence есть, но TTL/head trim/full rewrites | Может показывать projections; не source of recovery/coordination/evidence authority |
| UIHub event pub/sub/replay | `unrelated` | Drop/coalesce, ten-minute in-memory replay | Только UI transport; не context/evidence history |
| UI/session SQLite schema | `unrelated` к target storage choice | Physical split messages + snapshots, no semantic package/evidence history | Не расширять схему автоматически; future storage follows contract |
| `AgentToolLoop` | `reuse with modification` | Provider-neutral loop/gateway path strong; raw full result history is process-local and unbudgeted | Сохранить loop/gateway; добавить typed result projection/package boundaries, no raw auto-copy |
| `ToolResult` | `reuse with modification` | Structured envelope есть, но no call/attempt/provenance/sensitivity/completeness identity | Evolve or wrap as evidence; raw payload may move to artifact |
| Tool registry / `ToolGateway` | `reuse as-is` enforcement principle; `reuse with modification` for context integration | Correct local enforcement, policy inputs fragmented | Keep trusted boundary; bind calls to run/turn/package/policy IDs |
| `ApprovalContext`, session categories, `DesktopPolicyRuntime` | `reuse with modification` | Security behavior valuable, but lifecycle/identity distributed and prompt/session may obscure source | Unified effective-policy semantics; approvals never recovered from summary/history |
| Verifier runtime/result | `reuse with modification` | Deterministic authority useful; evidence input/output not generalized | Build purpose-specific evidence package and retain result lineage |
| UI artifacts | `reuse with modification` | Useful UX object, but model text/code-fence extraction and duplicated content lack integrity/version/ACL contract | Promote artifact identity/provenance semantics independent of UI representation |
| `ComputerActivityLog` / `Tracer` / tool logs | `unrelated` | Observability, not accepted evidence/task state; bounded/drained | Correlate identities, retain separate lifecycle; no silent promotion |
| Canonical Memory stores | `reuse with modification` | Principal isolation/explicit confirmation are good; current atoms/provenance/temporal semantics limited | Keep explicit acceptance boundary; Memory redesign governed separately, expose projections only |
| Memory retrieval/capsule | `reuse with modification` as candidate retrieval | Confidence/recency/vector rank + char pack do not ensure authority/coverage/ACL manifest | Feed candidates to assembler after policy/temporal filters; do not render directly as truth |
| `SessionSummarizer` | `supersede` as critical continuity | Summary saved as explicit fact without robust source lineage and may become sole accessible old detail | Summaries become derived versioned views; task state/evidence refs remain separate |
| Auto v1 runtime | `reuse with modification` | Strong gateway/verifier path; starts with system+goal only and process/session recovery state | Use same ContextPackage/recovery semantics without forcing multi-agent mode |
| `AutoShard`/coder-pool/merge legacy shapes | `remove/deprecate` as architecture basis | Residual model from removed runtime can bias target taxonomy | Do not derive context/identity model from it; removal separate |
| Background asyncio tasks/cancellation registries | `supersede` for durable work; `reuse as-is` for ephemeral UI operations | Process-local handle, no restart ownership/checkpoint | Durable task runs need semantic checkpoints/fencing; UI cancellation may stay local |
| Session export/import | `reuse with modification` for user history portability | Serializes snapshots but cannot transfer live authority/approval/tool action safely | Separate history export from resumable-run transfer contract |

## 8. Conflicts and gaps

Прямого противоречия с current canon не найдено. Target усиливает уже принятые boundaries:

- Ask остаётся non-writing; read-only context projection не даёт execution side effects.
- Plan read-only означает no external side effects, но допускает versioned adaptive replanning.
- Act сохраняет immutable accepted contract и local enforcement.
- Multi-agent contract уже запрещает shared private transcript и требует selective events.
- Explicit Memory confirmation не обходится automatic context capture.

Обнаружены implementation gaps, которые нельзя называть shipped capability:

- нет durable independent conversation/task/task-run/agent/model-turn identities;
- model input собирается не из explicit authoritative manifest;
- session/history/private working state имеют разные silent truncation boundaries;
- tool evidence и prior tool-loop continuity теряются либо дублируются неявно;
- summary lineage/critical-state preservation не определены;
- final provider token budget/reserves/overflow contract отсутствует;
- artifact identity/provenance/sensitivity не унифицированы;
- crash recovery и unknown side-effect reconciliation не определены;
- policy/approval context fragmented;
- UI replay не durable и не подходит для context/coordination correctness.

Есть осознанное терминологическое напряжение: current `ARCH_CANON` называет Ask stateless, но
Ask читает session history/Memory. Canonical definition теперь однозначна: `stateless` означает
no execution-side state mutation, not context-free. Новый document не меняет shipped
zero-side-effects claim.

## 9. Open / deferred decisions

Architecture определяет semantics, но не выбирает:

- concrete storage/schema/API/transport;
- whether conversation and session initially remain 1:1;
- exact tokenizer, budgets, thresholds и summary triggers;
- retrieval/index/reranker technologies;
- artifact store and encryption;
- checkpoint frequency, retention и erasure windows;
- whether private working checkpoints are persisted for every mode;
- distributed executor/fencing implementation;
- provider-specific prompt rendering;
- UI controls and migration sequence.

Эти implementation-defined решения не могут превращать model prompt, summary, UI session row,
event cache или Memory retrieval score в substitute authoritative state.

## 10. Sources

- [Anthropic: Effective context engineering for AI agents][anthropic-context]
- [Anthropic: Effective harnesses for long-running agents][anthropic-long-running]
- [OpenAI Agents SDK: Context management][openai-context]
- [OpenAI Agents SDK: Sessions and compaction][openai-sessions]
- [OpenAI Agents SDK: Handoffs and input filters][openai-handoffs]
- [Model Context Protocol specification: security and trust][mcp]
- [Anthropic: Mitigating prompt injection in browser use][anthropic-injection]
- [LangGraph: Persistence][langgraph-persistence]
- [LangGraph: Determinism and idempotency][langgraph-idempotency]
- [MemGPT paper][memgpt]

Local sources: [`ARCH_CANON.md`](ARCH_CANON.md),
[`Architecture.md`](Architecture.md),
[`MULTI_AGENT_COORDINATION_CONTRACT.md`](MULTI_AGENT_COORDINATION_CONTRACT.md),
[`MULTI_AGENT_COORDINATION_RESEARCH.md`](MULTI_AGENT_COORDINATION_RESEARCH.md),
`CONTEXT_MEM_MEMORY_ARCHITECTURE_RESEARCH.md` (user-provided research input),
[`SOURCE_OF_TRUTH.md`](../SOURCE_OF_TRUTH.md),
[`MWV_FLOW.md`](../agent/MWV_FLOW.md) и `docs/runtime_contract_claims.json`.

[anthropic-context]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
[anthropic-long-running]: https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
[openai-context]: https://openai.github.io/openai-agents-python/context/
[openai-sessions]: https://openai.github.io/openai-agents-python/sessions/
[openai-handoffs]: https://openai.github.io/openai-agents-python/handoffs/
[mcp]: https://modelcontextprotocol.io/specification/2025-03-26/index
[anthropic-injection]: https://www.anthropic.com/research/prompt-injection-defenses
[langgraph-persistence]: https://docs.langchain.com/oss/python/langgraph/persistence
[langgraph-idempotency]: https://docs.langchain.com/oss/python/langgraph/functional-api
[memgpt]: https://arxiv.org/abs/2310.08560
