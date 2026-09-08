# Memory Architecture — target contract

**Статус: target architecture, не описание current runtime.**

Этот документ задаёт semantic contract долгосрочной Memory SlavikAI. Он не выбирает database,
schema, API, vector store, embedding/reranking model, transport, UI или migration sequence.
Current-runtime audit, alternatives, external research и migration implications находятся в
[`MEMORY_ARCHITECTURE_RESEARCH.md`](MEMORY_ARCHITECTURE_RESEARCH.md).

Контракт применяется вместе с [`ARCH_CANON.md`](ARCH_CANON.md),
[`CONTEXT_ARCHITECTURE_CONTRACT.md`](CONTEXT_ARCHITECTURE_CONTRACT.md) и
[`MULTI_AGENT_COORDINATION_CONTRACT.md`](MULTI_AGENT_COORDINATION_CONTRACT.md).

## 1. Capability boundary

Memory — отдельная long-term knowledge subsystem. Она переживает отдельный process, session и
task run, но не является conversation transcript, private agent context, task/plan state,
coordination history, summary, audit log, telemetry или artifact storage.

Model-visible context получает только bounded, policy-filtered Memory projection для конкретного
principal, purpose, task/run, agent и turn. Наличие записи в Memory и её retrieval не означают,
что запись истинна, применима сейчас или обязана попасть в prompt.

Целевая модель:

~~~text
authorized sources / retained evidence
                 |
        propose Memory candidate
                 |
 validation + policy + consent + conflict check
                 |
     accepted versioned Memory records --------> Sensitive Vault domain
                 |                                      |
        derived indexes / views                 restricted indexes/views
                 |                                      |
 scoped retrieval -> filtering -> ranking -> conflict/freshness reconciliation
                 |
       bounded attributable Memory projection
                 |
          ContextPackage for one model turn
~~~

Основная target architecture:

> **Versioned structured Memory records with provenance and epistemic/temporal state, an explicit
> promotion boundary, hybrid retrieval over rebuildable representations, and a separate Sensitive
> Vault trust domain.**

Episodic evidence может породить validated semantic, preference или procedural Memory, но source
evidence не становится accepted Memory автоматически и не уничтожается lossy consolidation.

## 2. Термины и logical layers

- **Source evidence** — адресуемое наблюдение в owning subsystem: user message, verified tool
  result, artifact revision, coordination event или external source. Memory хранит reference и
  bounded supporting excerpt/metadata, а не присваивает себе ownership исходных данных.
- **Candidate** — предложение создать, изменить, объединить, подавить или удалить Memory. Candidate
  не участвует в ordinary retrieval как active knowledge.
- **Memory identity** — стабильная logical identity утверждения/опыта/procedure независимо от
  revisions и derived representations.
- **Memory revision** — immutable accepted version content и metadata. Исправление создаёт новую
  revision либо explicit status transition; provenance history не переписывается.
- **Active Memory** — accepted revision, допустимая для retrieval с учётом scope, policy,
  sensitivity, epistemic и temporal state.
- **Projection** — bounded read result с attribution/status/freshness, подготовленный для одного
  ContextPackage. Projection disposable и не является новой Memory.
- **Consolidation** — traceable derivation, которая объединяет повторяющиеся observations или
  создаёт более общий claim без удаления исходной lineage.
- **Vault** — отдельный trust/access domain для sensitive Memory. Это не просто tag или ranking
  penalty.

Logical layers не обязаны соответствовать отдельным databases, но их semantics нельзя смешивать:

1. source evidence в owning subsystem;
2. Memory candidates и validation outcomes;
3. accepted versioned Memory records;
4. sensitive accepted records в Vault boundary;
5. derived lexical/vector/entity/summary representations;
6. retrieval manifests и model-visible projections;
7. mutation audit и ranking telemetry.

## 3. Taxonomy

Taxonomy разделяет content kind, scope и sensitivity. `project/domain` является scope/domain
dimension, а `sensitive` — security classification; они не дублируют content types.

| Content kind | Purpose | Typical owner/scope | Write authority | Retrieval/update semantics | Default prompt eligibility |
| --- | --- | --- | --- | --- | --- |
| Semantic | Устойчивые факты, concepts и relationships | Principal; optional project/workspace | User confirmation либо trusted Memory control по explicit standing policy | Claim-level retrieval; version/supersession/conflict/temporal validity | Да, если active, fresh, relevant и разрешена |
| Preference | Устойчивые user choices о стиле/способе работы | Principal, иногда project override | User; inference обычно остаётся candidate | Specific-over-general, temporary exception не переписывает durable preference | Да, но не как system policy |
| Episodic | Значимый опыт/event, полезный за пределами исходной task/session | Principal/project, редко agent class | Explicit user acceptance или governed promotion из verified evidence | Time/task/entity retrieval; event не превращается в timeless fact | Только если опыт релевантен текущей задаче |
| Procedural | Проверенный способ работы, workflow, gotcha или diagnostic lesson | Principal/project/domain | User либо controlled validation с evidence | Versioned procedure; descriptive by default; current policy/capability always wins | Да как advisory knowledge, не authority |

| Content kind | Lifetime / default retention | Confidence/provenance | Typical sensitivity |
| --- | --- | --- | --- |
| Semantic | До correction/supersession либо freshness/retention boundary | Claim-level sources, validity and verification required | От ordinary до Vault в зависимости от subject |
| Preference | Пока user не изменит/подавит; периодическая confirmation допустима | User statement preferred; inference всегда distinguishable | Обычно ordinary/restricted, но intimate preference может быть Vault |
| Episodic | Policy-defined; не каждый event достоин долгого retention | Event/task/artifact references and occurrence time required | Зависит от source; raw private experience может быть restricted/Vault |
| Procedural | До invalidation новой policy/environment/version | Verified outcomes, applicability conditions and failures retained | Обычно ordinary/project-restricted; secrets из procedure исключаются |

Дополнительные dimensions:

- **Domain/scope:** principal-wide, project/workspace, named domain и future shared project scope.
  Conversation/task-local и agent-private данные остаются в context/task subsystems, пока explicit
  promotion не создаст long-term Memory.
- **Sensitivity:** ordinary, restricted и vault-classified. Classification влияет на storage,
  access, egress, logging, retention и projection независимо от content kind.
- **Temporal shape:** timeless-by-contract, valid-time fact, event или procedure revision. Timeless
  допускается только когда domain semantics это оправдывают.

`Project/domain memory` не выделяется отдельным payload kind: это semantic, episodic, preference
или procedural record с project/domain scope. `Sensitive memory` также не content kind: одна и та
же preference или fact может находиться в ordinary или Vault domain.

## 4. Scope, ownership и isolation

Каждая candidate, Memory identity, revision, projection и audit mutation связана с authenticated
`principal_id`. Baseline запрещает cross-principal retrieval, merge, deduplication и sharing.

Поддерживаемые semantic scopes:

- **principal-wide** — применимо к одному principal независимо от project;
- **project/workspace** — применимо только к stable project/workspace identity;
- **domain** — named bounded domain с explicit membership/policy;
- **future shared project/team** — только по отдельному multi-user contract, не baseline;
- **conversation/task-local** и **agent-private** — не long-term Memory scopes; это source scopes,
  из которых возможна explicit promotion.

Правила:

1. Scope берётся из authenticated runtime и authoritative project/task state, а не только из
   model-generated payload.
2. Более узкая applicable revision может override общую preference, но это не удаляет общую.
3. Агент одного task не получает автоматически всю principal Memory; retrieval policy учитывает
   assignment, purpose, audience, tool/provider egress и least privilege.
4. Private agent context не становится Memory и не доступен siblings без explicit publication,
   promotion и policy checks.
5. Physical co-location данных разных scopes не ослабляет logical access boundary.

## 5. Write and promotion contract

Нахождение информации в context, transcript, task state, coordination, summary, artifact или tool
output не является promotion.

Semantic lifecycle write path:

~~~text
observed source (outside accepted Memory)
  -> candidate proposed
  -> source/scope/sensitivity validation
  -> authority + consent decision
  -> contradiction/duplication/temporal reconciliation
  -> accepted revision
  -> active | quarantined | disputed | suppressed
~~~

### 5.1. Кто может предложить candidate

- пользователь;
- orchestrator, worker или verifier в пределах task assignment;
- trusted runtime observer;
- background consolidation/revalidation process;
- import/export controller;
- external connector только как untrusted sourced candidate.

Proposal authority не равна acceptance authority. Model/agent никогда не повышает собственное
утверждение до trusted Memory только потому, что оно уверенно сформулировано.

### 5.2. Source eligibility

| Source | Candidate допустима | Direct acceptance |
| --- | --- | --- |
| Explicit user request «запомни» | Да | Только после target preview/confirm semantics или эквивалентного явного user action |
| User-stated durable fact без просьбы сохранить | Да, если полезность и durability правдоподобны | Baseline нет; confirmation либо explicit standing policy |
| Conversation/model inference | Да, с `inferred` provenance | Нет по умолчанию |
| Verified task result | Да, как result-derived candidate | Только при explicit standing policy для узкого low-risk class; иначе review |
| Verified artifact | Да, reference + integrity/version | Не автоматически; artifact не Memory |
| Coordination event | Да, если material и source разрешён | Нет; event publisher не Memory authority |
| External/web/API/file content | Только quarantined/untrusted candidate | Никогда напрямую |
| Model-generated summary/consolidation | Да, как derived candidate с lineage | Никогда как user-stated/verified без validation |

### 5.3. Acceptance authority и confirmation

- Explicit user confirmation является baseline authority для personal facts, preferences и
  sensitive Memory.
- Automatic acceptance разрешается только explicit policy для узкого class, scope и source, с
  deterministic eligibility checks, retained provenance и reversible mutation. Отсутствие такой
  policy означает candidate/confirmation.
- Credential/secret payload не является eligible Memory даже при user confirmation. Остальные
  sensitive, intimate, high-harm, legal/medical/financial, identity/access, permission или
  security-control data никогда не принимаются автоматически.
- Agent-generated procedure не принимается как prescriptive instruction; verified workflow может
  быть accepted только как descriptive/advisory Memory.
- Ambiguous intent, material uncertainty или высокий false-positive cost переводят candidate в
  confirmation/quarantine, а не в silent write.

### 5.4. Inferred durable eligibility

Повторение само по себе недостаточно. Candidate должна одновременно иметь:

- ожидаемую полезность между sessions/tasks;
- semantic durability дольше текущего interaction/task;
- достаточную source quality и confidence;
- понятные owner и scope;
- допустимую sensitivity/retention;
- отсутствие признаков temporary exception, joke, quotation, role-play или third-party claim;
- acceptable harm/embarrassment cost при ошибке;
- user intent, совместимый с сохранением.

При существенном сомнении запись остаётся candidate. Future automatic promotion может быть
policy-enabled, но не становится общим default.

## 6. Provenance, evidence и lineage

Каждая accepted revision сохраняет semantic equivalents следующих полей:

- source type и source reference;
- authenticated author/origin, а для model output — model/agent role;
- observation/event time и recorded time;
- candidate creation и accepted time;
- last confirmed/revalidated time;
- evidence references и bounded excerpts/digests, где допустимо;
- derivation/consolidation parents;
- Memory identity, revision и supersession links;
- acceptance authority/policy revision;
- epistemic status/confidence basis;
- temporal validity и sensitivity classification.

Provenance является dependency graph, а не одной строкой `source`. Duplicate observations с
общим ancestor не считаются независимым подтверждением. Model inference, external assertion,
system observation, user statement и verified result всегда различимы.

Удаление или недоступность source не обязаны автоматически удалить Memory, но делают lineage
broken/redacted и запускают policy-defined revalidation, suppression или source-aware deletion.

## 7. Epistemic status и confidence

Acceptance, truth и confidence — разные dimensions. Минимальная semantic status model:

- **asserted** — заявлено authoritative для высказывания source, но не независимо проверено;
- **inferred** — derived conclusion;
- **verified** — подтверждено подходящим verifier/source contract;
- **disputed** — есть unresolved contradiction;
- **stale** — current applicability требует revalidation;
- **superseded** — заменено принятой revision;
- **retracted** — source/authority отозвал утверждение;
- **quarantined** — не допускается ordinary retrieval;
- **suppressed** — существует для lineage/audit, но исключено из use;
- **removed** — больше не доступно как Memory согласно deletion contract.

Confidence отражает качество конкретного claim и evidence, но не заменяет status. Автоматическое
повышение допускается только по независимым evidence и deterministic policy; repeated rendering,
retrieval, summary или agents с общим source не увеличивают confidence.

`disputed`, `stale`, `inferred` и partially verified records нельзя projected как безусловный
current fact. Projection либо исключает их, либо явно показывает qualification и конфликт.

### 7.1. Quality lifecycle

Нормальный lifecycle не сводится к одному status enum:

~~~text
candidate
  -> rejected | quarantined | accepted
  -> active
  -> revalidated | disputed | stale | suppressed
  -> superseded | retracted | archived | removed
~~~

Merge/consolidation создаёт traceable candidate или derived view; demotion не уничтожает prior
evidence. Revalidation может подтвердить текущую revision либо создать новую, но не переписывает
историю acceptance. Возврат suppressed/stale record в active требует той же authority, которая
достаточна для соответствующего content/sensitivity class.

## 8. Update, contradiction и temporal semantics

Blind last-write-wins запрещён.

- Same logical fact с изменённым значением создаёт новую candidate/revision и relation к prior
  revision.
- Preference change обычно supersedes previous revision в совпадающем scope; временное исключение
  получает valid interval/narrow scope и не переписывает durable preference.
- Uncertain contradiction переводит identity в disputed/conflict set; обе стороны и evidence
  сохраняются.
- Source conflict разрешается acceptance authority или trusted verifier policy, не recency alone.
- Obsolete fact становится superseded/stale с сохранённым historical validity interval.
- Correction/retraction связывается с исправляемой revision и invalidates derived representations.

Temporal model различает, когда применимо:

- `event_time` — когда произошло событие;
- `valid_from` / `valid_to` — когда утверждение было/является истинным;
- `learned_at` — когда SlavikAI получил source;
- `accepted_at` — когда revision принята;
- `confirmed_at` — последняя validation;
- `superseded_at` / `retracted_at` — lifecycle transition.

Отсутствующее valid time означает unknown, а не «истинно навсегда».

## 9. Freshness, decay и revalidation

Конкретные TTL остаются implementation-defined. Policy задаёт semantic freshness class:

- stable until corrected;
- expected to change and periodically revalidate;
- event-bound/historical;
- valid for explicit interval;
- short-lived and expire;
- unknown freshness.

Device ownership, employment, location, project status, schedules и temporary plans не должны
использоваться как current fact после freshness boundary без qualification/revalidation.
Preferences могут быть stable, но recent explicit correction имеет semantic precedence.

Decay влияет на retrieval eligibility/ranking, но не удаляет provenance. Pinned/protected status
не отменяет staleness, conflict, sensitivity или user suppression.

## 10. Forget, delete, suppress и retention

Операции имеют разные semantics:

- **suppress** — не использовать в retrieval/projection, сохраняя governed record/lineage;
- **supersede** — заменить current applicability новой revision;
- **retract** — отметить, что source/authority отозвал assertion;
- **archive** — исключить из ordinary active set, сохранив historical access по policy;
- **redact** — удалить/скрыть sensitive payload, сохранив минимальную допустимую structural audit;
- **forget** — user intent прекратить future use; должен немедленно suppress active and derived
  retrieval, затем выполнить policy-defined deletion closure;
- **delete/remove** — удалить canonical Memory payload/revisions в допустимой retention boundary;
- **retention expiry** — lifecycle action по policy, не случайное исчезновение index entry.

User correction/suppression обязана отражаться в subsequent retrieval до ответа об успехе.
Critical operation меняет canonical state и invalidates indexes/caches; удаление только vector
entry недостаточно.

Source-aware deletion учитывает:

- одна Memory может иметь несколько независимых sources;
- удаление одной source occurrence убирает её support и может demote/suppress claim;
- derived candidates/revisions, зависящие только от удалённого source, входят в deletion closure;
- исходные transcripts/artifacts/audit/backups принадлежат своим retention contracts; Memory не
  обещает уничтожить их без отдельной authorized operation;
- linked artifact deletion делает reference broken/redacted и вызывает revalidation.

Backup и audit не должны использоваться для ordinary retrieval или восстановления forgotten
Memory. Их retention/access/deletion определяются отдельной legal/security policy.

`forget`/`delete` создают durable anti-resurrection tombstone, который проверяется при restore,
rebuild, replication, обработке delayed events и re-import. Старая revision не может снова стать
active только из-за появления её payload или derived representation; для этого нужна новая
explicit authorized acceptance operation с новой auditable revision/lineage. Tombstone содержит
только минимально достаточные non-payload identity/version/deletion metadata и не сохраняет
удалённый content, source excerpt, credential или иной sensitive payload.

## 11. Retrieval contract

Target pipeline:

~~~text
query + ContextFrame
  -> deterministic scope/access/sensitivity/status filters
  -> candidate generation (structured + lexical + semantic + optional entity relations)
  -> deduplication by identity/lineage
  -> temporal/conflict/freshness reconciliation
  -> ranking/reranking
  -> sufficiency and budget selection
  -> attributable bounded projection + retrieval manifest
~~~

Deterministic/trusted stages применяют principal, project/domain, agent purpose, provider egress,
sensitivity, suppression/deletion, status и policy before content leaves its trust domain.
Model judgement может помогать query expansion, semantic reranking, contradiction suggestion и
compression, но не может ослабить access policy, принять candidate или скрыть conflict.

Ranking учитывает независимо:

- exact structured match и entity/relation match;
- lexical и semantic relevance;
- task/user relevance;
- scope specificity;
- epistemic status и evidence quality;
- temporal applicability/freshness;
- importance/utility;
- diversity и common lineage;
- conflict/suppression state;
- budget cost.

Vector similarity — candidate signal, не truth, authority или deletion mechanism. Graph/entity
traversal является optional derived retrieval path, а не обязательным canonical store.

## 12. Projection into model context

Memory projection следует Context Architecture Contract:

- имеет отдельный token/attention budget после critical system/user instructions, current task,
  policy, pending approvals/blockers и required evidence;
- содержит stable Memory/revision references, content kind, source class, epistemic status,
  freshness/validity и conflict qualification в достаточной для model форме;
- не маскируется под текущую user message или system policy;
- не включает весь Memory store, raw transcripts, large artifacts или unrestricted Vault;
- дедуплицирует related revisions/derived summaries по lineage;
- при budget pressure сначала сохраняет mandatory qualifications/conflict markers, затем наиболее
  полезные optional records; omission фиксируется в manifest;
- не повышает authority из-за placement в system message;
- не создаёт новую accepted Memory при каждом read/render.

Projection manifest фиксирует query/purpose, selected identities/revisions, applied filters,
transformations, exclusions/omissions, budget и policy/retriever version. Это retrieval trace, не
Memory data и не prompt transcript.

## 13. Sensitive Vault

Vault нужен как отдельный security domain, потому что ordinary similarity search, shared index,
logs и agent-wide projection не обеспечивают least privilege.

Vault-classified categories включают sensitive long-term knowledge: health, personal,
identity/location/contact, financial/legal/intimate data и иные user-designated private facts,
preferences, episodes или procedures. Конкретная classification policy остаётся расширяемой.

**Sensitive Memory Vault не является Credential/Secret Store.** Passwords, authentication/session
tokens, API keys, recovery secrets/codes, private keys, seed phrases и raw credentials не являются
допустимым Memory payload или candidate и не должны попадать в Memory storage, indexes или
projection. Если отдельный credential service когда-нибудь появится, Memory может хранить только
policy-authorized opaque reference и минимальную non-secret metadata, из которых нельзя
восстановить credential.

Инварианты Vault:

1. Запись требует explicit informed consent и purpose/scope; automatic promotion запрещена.
2. Credential/secret payload не сохраняется в Memory вообще; approvals, permissions и temporary
   capabilities не сохраняются или replayed как active authority.
3. Read требует explicit authorized purpose, principal match, policy decision и egress check.
4. Ordinary retrieval/index не видит plaintext Vault records и не может выбрать их по semantic
   similarity.
5. Model visibility не следует из read permission автоматически; предпочтителен capability/tool
   use без раскрытия payload, где возможно.
6. Cross-agent visibility минимальна и assignment-scoped; sibling не наследует доступ.
7. Logs, traces, errors, projections и audit по умолчанию redact payload.
8. Export/delete требуют explicit operation и complete derived-index closure.
9. Encryption boundary обязательна концептуально; конкретные crypto/key mechanisms deferred.

## 14. Multi-agent semantics

- Worker/orchestrator/verifier могут создавать candidates с authenticated publisher и source
  references.
- Worker не может directly commit accepted Memory или Vault record. Authoritative promotion идёт
  через Memory policy/control boundary и нужную user/trusted authority.
- Finding, hypothesis, failure, artifact или verification event из coordination остаётся
  coordination/evidence; promotion всегда отдельная операция.
- Sibling читает только explicit task-purpose projection, не весь principal store и не candidates
  другого agent.
- Orchestrator не получает private agent contexts; он может видеть candidate metadata и принимать
  только те control decisions, которые разрешены policy.
- Conflicting agent assertions создают linked candidates/dispute, а не last-writer winner.
- Duplicate candidates объединяются по logical identity и common lineage без искусственного роста
  confidence.

## 15. Poisoning и prompt injection boundary

Web pages, files, tool outputs, external APIs, agent messages, model summaries и retrieved text —
untrusted data для Memory acceptance. Instruction-like content не становится preference,
procedure, policy или system instruction без соответствующей authority.

Required controls:

- authenticated source envelope и immutable provenance references;
- source-class policy before extraction/promotion;
- quarantine для external/model-inferred candidates;
- separation content from instructions during validation/projection;
- sensitivity and secret detection before storage/indexing/egress;
- contradiction and anomaly checks;
- explicit confirmation/verifier for high-risk categories;
- bounded excerpts and artifact references вместо uncontrolled payload copying;
- audit of proposal, rejection, acceptance and later retrieval без unnecessary sensitive content.

Sanitization снижает risk, но не превращает external content в trusted assertion. Model verifier
также остаётся fallible и не может единолично выдать authority.

## 16. Procedural Memory, policy и approvals

Procedural Memory по умолчанию descriptive/advisory: «этот workflow раньше сработал при таких
условиях». Prescriptive policy: «это действие обязательно/разрешено» живёт в canonical policy
subsystem.

- Procedure всегда подчиняется current policy, approvals, capability availability и task scope.
- Memory не может grant tool access, approve side effect, ослабить sandbox или отменить `DENY`.
- Historical approval может храниться только как qualified historical fact, если retention это
  допускает; enforcement использует current authoritative approval state.
- Credential/token payload не хранится в Memory; expired/revoked approval или temporary capability
  никогда не replayed из Memory как действующая authority.

## 17. Artifacts, entities и consolidation

Memory может ссылаться на artifact identity/version/integrity/provenance, но не копирует large
payload. Artifact не становится Memory автоматически. Изменённый, удалённый или inaccessible
artifact переводит reference в changed/broken state и запускает revalidation.

Canonical baseline — structured records с optional entity references, не mandatory knowledge
graph. Entity layer целесообразен для stable disambiguation persons/devices/projects/
organizations/locations и relationship/temporal queries. Flat records остаются допустимы для
простых preferences/procedures. Graph traversal и entity summaries являются derived views, пока
отдельный future contract не докажет необходимость graph как authority.

Consolidation:

- связывает output со всеми source revisions/evidence;
- не считает derived output независимым support;
- не удаляет original evidence автоматически;
- reversible/recomputable, когда output является derived view;
- создаёт новую candidate/revision, если меняет accepted semantic claim;
- invalidates/rebuilds downstream representations после correction/deletion.

## 18. Durability, recovery и consistency

Canonical durable state: accepted Memory identities/revisions/status transitions, candidate
decisions, required provenance references и Vault access metadata. Source evidence и artifacts
остаются canonical в своих subsystems.

Lexical/vector/entity indexes, embeddings, summaries, retrieval caches и prompt projections —
derived. Они должны быть rebuildable из canonical Memory + доступных source references.

Write semantics обеспечивают:

- atomic accepted revision/status transition;
- idempotent proposal/acceptance operation identity;
- no half-accepted state при failed write;
- explicit index lag/failed state без rollback canonical truth;
- recovery/reconciliation между canonical state и derived indexes;
- corruption isolation, backup/restore и integrity verification;
- process restart без зависимости от private agent context.

Read path обязан fail closed для access/sensitivity ambiguity и soft-degrade для unavailable
derived index, используя допустимый deterministic fallback либо возвращая no Memory. Stale index
не может воскресить suppressed/deleted/retracted revision.

## 19. User control и explainability

Target capabilities позволяют principal:

- увидеть active, candidate, stale/disputed, suppressed и Vault-classified Memory в разрешённом
  виде;
- увидеть source/derivation и reason acceptance;
- понять, почему конкретная Memory была retrieved/used;
- подтвердить, исправить, narrow scope, supersede, suppress или забыть;
- запретить automatic candidate creation/promotion по category/source;
- export допустимого Memory state отдельно от transcripts/audit/artifacts.

UI/API deferred, но semantic operations и observable outcomes обязательны. Ответ об исправлении
или forgetting нельзя возвращать до canonical transition и invalidation active retrieval path.

## 20. Observability и audit

Memory data, mutation audit, retrieval trace, ranking telemetry и user explanation — разные
records/retention domains.

Audit должен позволять установить:

- кто/что создал candidate;
- какие source/evidence использованы;
- какая authority/policy приняла или отклонила изменение;
- когда и почему изменились revision/status/scope/sensitivity;
- почему record была eligible, ranked, projected или excluded;
- какие derived representations invalidated/rebuilt.

Sensitive payload не дублируется в audit/telemetry. Retrieval telemetry по возможности хранит IDs,
scores/reasons и redacted metadata, а не content.

## 21. Non-negotiable invariants

1. Memory не является conversation transcript.
2. Memory не является model prompt; prompt получает bounded projection.
3. Memory не является authoritative task, plan, policy, approval или permission state.
4. Retrieval не означает truth, freshness или current applicability.
5. Model inference не становится trusted Memory автоматически.
6. Provenance и derivation lineage не теряются при consolidation.
7. Sensitive Memory имеет отдельный access/index/egress boundary, но Vault не является
   credential/secret store.
8. Cross-principal sharing, retrieval и deduplication запрещены baseline policy.
9. Agent private context не становится long-term Memory автоматически.
10. User correction/suppression отражается в subsequent retrieval до success acknowledgement.
11. Expired/revoked approval, credential или capability не восстанавливается через Memory.
12. Promotion из task/coordination/artifact data explicit и policy-governed.
13. Critical deletion/update не зависит только от vector/entity index.
14. Indexes, caches, embeddings, summaries и projections являются derived, не canonical truth.
15. Blind last-write-wins и recency-as-authority запрещены.
16. Pinned status не обходит conflict, freshness, sensitivity, suppression или policy.
17. Descriptive procedural Memory не повышается до prescriptive policy.
18. External content и agent messages не создают trusted instructions через Memory.
19. Forget/delete оставляет durable non-payload tombstone: restore, rebuild, replication, delayed
    event или re-import старой revision не может resurrect forgotten Memory без новой explicit
    authorized acceptance operation.

## 22. Open and implementation-defined decisions

Без отдельного design/implementation PR остаются открытыми:

- concrete DB/storage topology и schema;
- vector database, embedding model, reranker и indexing engine;
- exact confidence formula и thresholds;
- TTL/freshness values и revalidation schedules;
- concrete crypto/key management для Vault;
- UI, API, batch/consolidation jobs и transport;
- entity resolution implementation и необходимость authoritative graph;
- exact projection/token budgets;
- migration sequence current stores;
- future multi-user shared Memory contract.

Эти решения не могут ослабить semantic boundaries и invariants этого документа.
