# Memory Architecture — audit, research и alternatives

Дата исследования: 2026-09-08.

**Статус: research/design rationale, не current runtime contract.** Нормативная target-модель
находится в [`MEMORY_ARCHITECTURE_CONTRACT.md`](MEMORY_ARCHITECTURE_CONTRACT.md). Документ не
выбирает database schema, API, transport или implementation sequence и не утверждает, что target
mechanism уже существует.

## 1. Метод и design priority

Исследование проведено target-first:

1. из Context и Multi-Agent contracts выведены границы authority, isolation и projection;
2. проверен фактический current data flow, а не названия classes;
3. read-only просмотрены local SQLite schemas/counts без чтения user payload;
4. изучены primary papers, official framework docs и repositories;
5. сравнены alternatives;
6. выбрана минимальная достаточная target-модель;
7. каждой релевантной current abstraction дан explicit verdict.

Current runtime является evidence и источником reuse opportunities, но sunk cost не ограничивает
target design. `CONTEXT_MEM_MEMORY_ARCHITECTURE_RESEARCH.md` использован как пользовательский
research input, а не как принятый contract.

Проверены repository surfaces:

- source-of-truth hierarchy, `ARCH_CANON.md`, `Architecture.md`, architecture index и claims;
- Context Architecture и Multi-Agent Coordination contracts/research;
- current `Agent`, routing, tool/MWV, policy/approval, principal/session lifecycle;
- Memory managers/stores, claim extraction, aggregation, vectors, retrieval и summarization;
- confirmation decision, HTTP routes, UI decision panel и tests;
- local `memory/*.db` schemas/counts и active processes.

## 2. Current Memory map

### 2.1. Фактические stores и ownership

`principal_storage_paths(...)` создаёт для owner legacy root paths, а для members — hashed
principal directory. В каждом principal scope логически существуют пять stores:

| Current store | Фактическая роль | Current authority |
| --- | --- | --- |
| `canonical_atoms.db` | Mutable canonical claim rows | Current accepted Memory для retrieval |
| `vectors.db` | Embeddings для atoms, code и docs namespaces | Derived index; не source of truth |
| `memory.db` | Legacy notes/preferences/project facts | Legacy durable records, read by prompt path |
| `memory_categories.db` | Auto-written inbox + manually triaged categories | Operational candidate/inbox store, не canonical atoms |
| `memory_companion.db` | Interaction/tool log, feedback, policy rules/candidates, batch reports | Audit/feedback/policy state, не long-term Memory |

Read-only local inspection показал существующие schemas:

- `canonical_atom`: 11 columns, включая stable key/value/confidence/counts/status/pinned, 0 rows;
- `memory`: 6 columns, 0 rows;
- `memory_item`: 12 columns, 0 rows;
- `memory_companion`: interaction/policy/feedback/batch tables, только schema metadata row;
- `vectors`: 6 columns, 5 rows, все содержимое не читалось.

В момент проверки active SlavikAI server/process не обнаружен. Counts фиксируют только этот local
snapshot и не являются product guarantee.

### 2.2. Explicit accepted Memory flow

~~~text
user message starts with remember/запомни
  -> AgentRouting detects explicit request before ordinary Ask/Auto routing
  -> ClaimExtractor regex + optional LLM enrichment
  -> memory_save DecisionPacket with text + claim preview
  -> UI shows confirm | edit_and_confirm | reject (reject default)
  -> confirm path calls ToolGateway(memory_save_confirmed, confirmed_decision=True)
  -> server reconstructs claims from preview
  -> CanonicalAggregator upserts canonical_atom rows in transaction
  -> each atom best-effort syncs into vectors namespace=atoms
  -> interaction/tool observation stored in memory_companion.db
~~~

Положительные properties:

- preview не пишет canonical/vector data;
- reject не вызывает Memory write;
- confirmed canonical batch write transactional;
- canonical write остаётся successful при vector failure, а result сообщает `partial`;
- principal-specific Agent получает principal-specific stores.

Ограничения:

- preview позволяет LLM extraction сформировать type/key/value/summary/confidence, но acceptance не
  делает независимую source/sensitivity/contradiction validation;
- reconstructed claim всегда `is_explicit=True`, поэтому происхождение extraction/inference
  теряется;
- `CanonicalAtom` не сохраняет source kind/id, evidence, accepted authority или revisions;
- одинаковый stable key mutates одну row; prior value/provenance теряются;
- conflict сохраняет старое value и только увеличивает counter;
- repeat автоматически повышает confidence, даже если observations имеют общий ancestor;
- vector sync не имеет обязательной reconciliation/rebuild guarantee после partial failure.

### 2.3. Другие write paths

`capture_memory_claims_from_text(...)` способен извлекать и сразу применять explicit и некоторые
non-explicit preference/environment/fact claims. Production call site не найден; метод вызывается
тестами. Это dormant unsafe capability, а не shipped automatic Memory flow.

`/end-session` является реальным command path:

~~~text
last <= 40 user/assistant messages, each <= 500 chars
  -> LLM SessionSummarizer
  -> Claim(type=FACT, explicit=True, source=session.summary)
  -> direct CanonicalAggregator upsert + vector sync
  -> clear process-local short_term
~~~

Этот path автоматически принимает lossy derived summary как canonical fact без отдельной
`memory_save` confirmation и без source coverage/revision lineage. Он противоречит broad current
claim «любая Memory write только после отдельного confirm» и target Context/Memory boundary.

`MemoryInboxWriter` автоматически сохраняет sanitized descriptions неизвестных requests и
повторных tool failures в `memory_categories.db`. Это durable operational inbox с TTL/count/rate
limits, не accepted Memory. Manual triage только переносит ту же row между categories
`notes/facts/preferences/rules/glossary`; dangerous rules/preferences требуют `allow_dangerous`,
но triage не создаёт canonical atom и не имеет полноценного promotion contract.

`MemoryCompanionStore` автоматически пишет raw chat input/response, tool args/output previews,
feedback, batch review reports и policy candidates. Approved policy candidate становится
`PolicyRule`, а не Memory. Этот контур полезен как пример candidate/approval, но физически смешивает
audit, feedback и authoritative policy в database с названием Memory и может сохранять sensitive
payload без общего redaction contract.

Legacy `MemoryManager` поддерживает direct notes/preferences/project facts and hard delete. В
production current code найдено чтение notes/preferences для prompt, но не найден current runtime
write call. `FeedbackManager` имеет только tests/callable legacy implementation; current Agent
использует feedback из `MemoryCompanionStore`.

### 2.4. Current retrieval and ranking

~~~text
query
  -> list <= 400 ACTIVE/CONFLICT canonical atoms
  -> filter status (conflicts default excluded), fixed claim types,
     confidence >= 0.45, last_seen <= 365 days
  -> vector cosine search in atoms namespace, top_k expansion
  -> append deterministic confidence/recency/support fallback
  -> take <= 8
  -> character-pack <= 1800 chars, stop at first overflow
~~~

Gaps:

- no principal filter inside DB query; isolation relies on choosing correct physical DB;
- no project/domain/agent/purpose/sensitivity/access filters;
- no provenance, valid-time, source quality, conflict explanation or common-lineage handling;
- `last_seen_at` conflates observation, confirmation and temporal validity;
- hard-coded 365-day recency silently excludes records rather than marks stale/revalidate;
- vector similarity controls order; fallback is deterministic but not lexical/entity retrieval;
- character packing stops at first oversized item and can waste remaining budget;
- no sufficiency/abstention signal or retrieval manifest;
- vector DB also stores code/docs namespaces and prunes by insertion order/record limits.

### 2.5. Current context projection

`AgentMemoryMixin._build_context_messages` builds one system message and fills a total 12,000
character budget in fixed order:

1. pinned canonical atoms;
2. three active session summaries;
3. legacy notes;
4. negative feedback hints;
5. legacy preferences;
6. canonical capsule;
7. code/docs vector snippets;
8. workspace content.

Это bounded baseline, но не target projection:

- fixed order is an implicit priority policy;
- characters are not provider token/attention budget;
- each slot truncates silently from model perspective;
- same fact can appear in pinned, summary, legacy and capsule forms;
- pinned query does not filter status, поэтому deprecated/conflict record может попасть первым;
- provenance/sensitivity/freshness/validity are absent from rendered Memory;
- feedback, project retrieval, summaries and accepted Memory share one system-message authority
  surface;
- critical policy/task/current user instructions do not have an integrated final budget contract.

### 2.6. Short-term/session identity and durability

`short_term` is process-local last 20 user/assistant messages. UI session history is separately
persisted and can repopulate a new Agent. Neither is long-term Memory. `conversation_id` is random
per Agent construction; Memory source falls back to session ID or this process-local ID.

Runtime reset clears transient `short_term`/workflow/approval fields and intentionally leaves all
Memory databases. This confirms the current persistence boundary, but not correctness of the
stored taxonomy or recovery/index consistency.

### 2.7. Current user management and API/UI surface

Browser UI routes позволяют:

- list conflicts;
- resolve conflict через in-place `activate`, `deprecate` или `set_value`;
- list/pin/unpin atoms;
- preview/apply/undo inbox triage;
- confirm/edit/reject explicit `memory_save` DecisionPacket.

Не найден production surface для полного списка/search accepted Memory, source/provenance view,
general correction with revision history, sensitivity/scope management, explanation why-used,
user suppression/forget/export или source-aware deletion. `CanonicalAtomStore.delete_atom(...)`
делает soft-deprecate, но production caller/route не найден. Legacy `MemoryManager.delete(...)`
делает hard delete одной row, но runtime call site также не найден.

### 2.8. Current source of truth vs derived/cache

| Current data | Classification | Problem |
| --- | --- | --- |
| `canonical_atom` row | Current accepted Memory truth | Mutable row, no evidence/revision/temporal lineage |
| `vectors` atom entry | Derived index | Partial sync possible; no proven reconciliation scheduler |
| `memory` note/preference | Legacy parallel Memory truth | Duplicates canonical concepts with different semantics |
| `memory_item` inbox/category | Candidate/operational store | Triage category may look accepted without promotion authority |
| session summary atom | Derived summary incorrectly promoted | Lossy content becomes current fact |
| `short_term` and UI chat | Conversation/private context | Not Memory; separate trimming and retention |
| interaction/tool logs | Audit/evidence-like observations | Mixed with policies; raw sensitive payload risk |
| policy rules/approvals | Authoritative policy/security state | Must remain outside Memory |
| artifact/file | Artifact state | Memory may reference; payload is not Memory |
| context slot/capsule | Derived projection | No manifest/lineage and mixed authority labels |

## 3. External primary-source findings

### 3.1. Taxonomy and memory actions

CoALA models language agents with modular memory and explicit internal/external actions. Its
semantic/episodic/procedural distinction is useful vocabulary, but does not by itself define
SlavikAI consent, provenance or security. [CoALA paper][coala].

LangGraph official documentation similarly separates thread-scoped short-term state from
cross-session long-term Memory and discusses semantic, episodic and procedural classes plus hot
path/background writes. This supports taxonomy and write-timing separation, not a ready-made
acceptance policy. [LangGraph Memory overview][langgraph-memory].

OpenAI Agents SDK documentation explicitly separates agent memory distilled from prior sandbox
runs from conversational Session memory. This directly supports the SlavikAI boundary that history
and Memory are not synonyms. [OpenAI Agent memory][openai-agent-memory],
[OpenAI Sessions][openai-sessions].

### 3.2. Bounded/hierarchical context

MemGPT demonstrates virtual context management across memory tiers rather than treating the fixed
context window as storage. It supports the storage-versus-projection distinction, but its tier
mechanism is not adopted as a domain model. [MemGPT paper][memgpt].

Generative Agents stores observations, derives higher-level reflections and retrieves dynamically
for planning. This validates the usefulness of episodic evidence and consolidation, while also
showing why model-generated reflection must remain derived/candidate until governed acceptance.
[Generative Agents paper][generative-agents].

### 3.3. Retrieval, updates and temporal correctness

LongMemEval evaluates extraction, multi-session reasoning, temporal reasoning, knowledge updates
and abstention. It reports substantial degradation from simply feeding long history and finds
benefit in decomposition, multi-key indexing and time-aware query expansion. This argues against
conversation-summary-only and vector-only designs. [LongMemEval paper][longmemeval].

Zep/Graphiti demonstrates episodes with provenance, entity relationships and temporal validity
windows, plus hybrid retrieval. These are strong patterns for changing facts. They do not prove
that a graph must be SlavikAI canonical storage, so entity/graph remains optional derived access.
[Zep paper][zep], [Graphiti repository][graphiti].

Microsoft GraphRAG combines extracted entities/relationships/claims, summaries and embeddings,
and distinguishes local, global and basic vector search. It confirms that graph representations
are expensive derived pipelines with multiple query modes, not a free replacement for canonical
records. [GraphRAG indexing][graphrag-index], [GraphRAG query][graphrag-query].

### 3.4. Poisoning and trust boundaries

NIST's 2026 agentic-AI threat material identifies persistent Memory/context poisoning: forged or
misleading content can be retrieved as trusted truth in later tasks. It also links insecure
inter-agent messaging to spoofing/replay. This supports authenticated provenance, quarantine,
separate acceptance authority and not treating coordination/external content as trusted Memory.
[NIST agentic threats][nist-agentic].

### 3.5. Internal reference audit

The user-provided `CONTEXT_MEM_MEMORY_ARCHITECTURE_RESEARCH.md` independently found useful ideas:
retained occurrences, accepted canonical knowledge, derived representations and bounded runtime
context should be separate; provenance/authority/freshness/popularity are independent; temporal
validity and deletion closure matter. It also found concrete weaknesses in the reference system:
global content-hash dedup, TTL/pinned interaction, heuristic contradiction/decision extraction,
archive semantics, Cyrillic cache-key issues and uneven privacy. Therefore it informs requirements
but is not reused as target architecture.

## 4. Alternatives

| Alternative | Strengths | Critical limitations | Verdict |
| --- | --- | --- | --- |
| 1. Conversation summary as Memory | Minimal implementation and prompt cost | Lossy, weak provenance, conflicts/updates/deletion unsafe, history conflated with Memory | Reject |
| 2. Flat vector store of text memories | Simple semantic recall | Similarity is not authority; weak exact update, temporal, deletion, conflict and explainability | Reject as primary; retain vector as derived index |
| 3. Structured records + vector retrieval | Explicit fields, manageable complexity, hybrid exact/semantic access | Needs evidence lineage, temporal versions, policy and Vault added | Accept as core after extension |
| 4. Entity-centric/graph Memory | Relationship/temporal queries and consolidation | Entity resolution complexity, graph poisoning, expensive lifecycle/deletion; overkill for simple preferences | Do not require as canonical baseline; optional derived view |
| 5. Episodic event store + derived semantic Memory | Excellent provenance/history/replay | Raw event volume, privacy and acceptance ambiguity; event log is not user Memory by itself | Use source-evidence pattern, not one universal store |
| 6. Hybrid versioned Memory | Supports facts/preferences/episodes/procedures, lineage, temporal state, hybrid retrieval and Vault | Higher control-plane complexity than flat store | Selected: minimum adequate for correctness/safety |

### 4.1. Rejected architecture shortcuts

- **One giant user profile document:** concurrent updates and source-aware deletion become unsafe;
  one bad inference contaminates a large unit.
- **One universal event log as Memory:** task/coordination/audit retention and Memory consent have
  different authorities.
- **Knowledge graph everywhere:** graph is valuable for entity relationships, not necessary for
  every preference/procedure; derived graph can be added without making it source of truth.
- **Automatic promotion after N repeats:** repeats can share one source, repeat an error, quote an
  attacker or describe temporary behavior.
- **LLM judge as sole validator:** useful as candidate/reranking signal, not trusted acceptance or
  access-control authority.
- **Last-write-wins:** destroys temporal history, source conflict and correction explainability.
- **Pinned means always inject:** pinning cannot override stale/conflict/sensitive/policy filters.

## 5. Selected target architecture

Выбран contract:

~~~text
source evidence remains in owning subsystem
        -> explicit candidate with source references
        -> deterministic eligibility + sensitivity + authority checks
        -> user/policy validation and contradiction reconciliation
        -> immutable accepted revision + active/status transition
        -> rebuildable lexical/vector/entity representations
        -> scope-first hybrid retrieval and reranking
        -> conflict/freshness-aware bounded Memory projection
~~~

Почему это minimum adequate:

- structured identity/revisions нужны для correction, supersession и source-aware deletion;
- provenance and epistemic metadata нужны для poisoning defense и explainability;
- explicit promotion нужен, потому что source presence не означает user intent/authority;
- hybrid retrieval нужен для exact preferences/project scope, lexical names, semantic relevance и
  optional relationships;
- Vault нужен, потому что ordinary store/index/log filters недостаточны для sensitive data;
- graph не обязателен: entity links/graph traversal могут быть derived only when query value
  оправдывает cost.

Target taxonomy оставляет четыре content kinds: semantic, preference, episodic и procedural.
Project/domain — scope; sensitivity/Vault — security axis. Это избегает duplicated types и
позволяет одинаковые lifecycle/retrieval rules комбинировать с разными scopes.

## 6. Threat analysis

| Threat | Current exposure | Target boundary |
| --- | --- | --- |
| External prompt injection becomes preference/policy | Extracted/model text can shape preview or summaries | External/model content only quarantined candidate; authority and source labels retained |
| Agent hallucination becomes canonical fact | Dormant direct capture and `/end-session` summary path | Agent may propose only; acceptance boundary separate |
| Repetition boosts false confidence | Aggregator increments support/confidence | Independent lineage required; common ancestor deduped |
| Conflict hidden from model | Default retrieval excludes conflicts; pinned may bypass status | Conflict reconciliation before ranking; qualification or exclusion recorded |
| Stale fact treated current | Single last_seen + 365-day filter | valid-time/freshness class/revalidation semantics |
| Sensitive data or credentials leak into embeddings/logs | Shared vector DB and raw interaction/tool logs | Sensitive Memory uses Vault-specific index/access/egress and redacted observability; credential/secret payload is excluded from Memory and may only be referenced opaquely through a separate service |
| Deleted data resurrects from index/cache/backup | No complete deletion closure | Canonical transition + derived invalidation + durable non-payload tombstone block restore, rebuild, replication, delayed events and re-import until a new authorized acceptance |
| Cross-agent overexposure | No target agent-purpose Memory policy | Per-agent purpose projection and least privilege |
| Approval replay through procedure/history | Policy and feedback live near Memory-named data | Memory cannot grant authority; policy/approval store remains canonical |
| Broken artifact silently grounds Memory | No artifact revision/integrity link in atom | Versioned reference and broken/stale state |

## 7. Current-runtime impact / migration implications

Verdicts describe target fit, not an implementation plan.

| Current abstraction | Current semantics / evidence | Mismatch | Verdict | Migration implication |
| --- | --- | --- | --- | --- |
| `PrincipalStoragePaths` / hashed member roots | Physical principal partition for five DBs | Scope only physical; no project/Vault/record-level semantics | `reuse with modification` | Preserve hard principal boundary; extend logical scopes and separate Vault |
| `MemoryManager` / `memory.db` | Flat mutable legacy notes/preferences/facts; prompt reads | Parallel truth, no provenance/version/status; no production write found | `remove/deprecate` | Stop projecting after accepted data migration policy; do not preserve as target API |
| `CategorizedMemoryStore` | Mutable inbox/categories with fingerprint and TTL | Operational signals mixed with memory-like facts/rules; no acceptance lineage | `supersede` | Replace with explicit candidate/evidence semantics; operational inbox remains separate |
| `MemoryInboxWriter` | Automatic unknown/tool-error writes | Useful observability, not user Memory; source/category naming misleading | `supersede` | Route signals to observability/candidate subsystem, never active Memory directly |
| `ClaimExtractor` | Regex + optional LLM extraction; explicit source envelope | Extraction output can define key/type/confidence; no sensitivity/entity validation | `reuse with modification` | Keep as untrusted candidate producer with richer source labels/validators |
| `capture_memory_claims_from_text` | Direct apply method; no production caller found | Unsafe latent automatic promotion | `remove/deprecate` | Eliminate direct acceptance path; all callers use promotion boundary |
| `Claim` model | Has source kind/id and explicit flag | Loses source after aggregation; no evidence/temporal/acceptance metadata | `supersede` | New candidate/revision contract, not additive fields assumed here |
| `CanonicalAtom` / `CanonicalAtomStore` | One mutable row per stable key | No revisions, evidence, validity, sensitivity, scope or lineage | `supersede` | Canonical data requires new semantic model; migrate with provenance limitations marked |
| `CanonicalAggregator` | Equality boosts confidence; conflict keeps old value | Common-source amplification, mutable history, recency/confidence heuristics | `supersede` | Replace with version/conflict/consolidation policy; do not emulate old counters as authority |
| `VectorIndex` | Shared SQLite vectors for atoms/code/docs; bounded pruning | Derived status is correct, but scope/Vault/model-version/rebuild semantics weak | `reuse with modification` | Reuse search mechanics only behind memory-specific derived-index contract |
| `AtomEmbeddingIndex` | Upsert/delete atom representation; manual rebuild exists | No guaranteed reconciliation, revisions or Vault separation | `reuse with modification` | Keep adapter pattern; index revision IDs and support deterministic rebuild/invalidation |
| `memory_retrieval` | Confidence/recency filter + vector order + fallback + char pack | Missing access/sensitivity/project/temporal/conflict/provenance/sufficiency pipeline | `supersede` | Introduce staged target retrieval; low-level vector search may be reused |
| `SessionSummarizer` | Lossy summary converted to explicit FACT | Summary is derived context, not accepted Memory | `reuse with modification` | Reuse summarization only as derived view/candidate with coverage lineage |
| `/end-session` canonical write | Direct summary promotion and vector sync | Bypasses confirmation and conflates history summary with Memory | `remove/deprecate` | End-session may clear transient context/create summary, but must not direct-commit Memory |
| `memory_save` DecisionPacket/UI | Preview, edit, confirm, reject default | Good baseline; lacks source/sensitivity/conflict/update/delete explanation | `reuse with modification` | Generalize semantic confirmation while keeping explicit user control |
| `memory_save_confirmed` ToolGateway boundary | Requires confirmed decision before write | Couples generic tool enforcement to current atom payload | `reuse with modification` | Preserve trusted acceptance boundary; target operation/schema remain separate |
| Memory conflict/pin HTTP handlers | List/resolve conflicts; pin/unpin | In-place resolution, no lineage; pin can expose non-active record | `supersede` | User operations target revisions/status; pin remains preference, not policy bypass |
| Memory triage handlers | Preview/apply/undo category moves | Category move can resemble promotion but is not canonical acceptance | `supersede` | Candidate review semantics replace triage-as-move |
| `MemoryCompanionStore` interaction/feedback | Durable logs, feedback and policy candidates/rules | Not Memory; raw payload and mixed responsibilities increase security coupling | `unrelated` | Keep outside target Memory; likely split/rename under their own contracts later |
| BatchReview policy candidates | Evidence-backed proposed -> approved policy flow | Policy is not Memory and must not be inferred as such | `unrelated` | Pattern informs review, but canonical policy remains separate authority |
| Feedback hints in context | Negative feedback transformed into system slot | Derived guidance without accepted preference Memory | `supersede` | Context may use scoped feedback evidence; durable preference needs Memory promotion |
| Agent `short_term` / UI session history | Process/session conversation continuity | Not long-term Memory and independently truncated | `unrelated` | Keep under Context/Session contract; only explicit promotion crosses boundary |
| `TaskPacket` and `RunContext` | Execution contract and transient identity/context | Memory refs/purpose/retrieval manifest absent | `reuse with modification` | Carry policy-scoped projection/reference only; never Memory payload as task authority |
| MWV / orchestrator / workers | Isolated execution modes | No shared Memory control boundary | `reuse with modification` | Agents propose candidates; Memory controller owns acceptance; modes remain valid |
| Coordination events/history | Explicit task-run work exchange (target) | Not yet implemented; could be mistaken for shared Memory | `reuse as-is` | Keep current contract boundary: promotion is separate policy-governed operation |
| Policy/approval stores | Canonical execution authority | Memory-named companion DB obscures boundary | `reuse as-is` | Enforcement always reads policy/approval state, never Memory replay |
| Artifact/session output storage | Payload and UI references | No general version/integrity linkage to Memory | `reuse with modification` | Memory stores only versioned reference; artifact subsystem keeps payload ownership |

## 8. Architectural gaps and conflicts

### 8.1. Confirmed current contradiction

`docs/SOURCE_OF_TRUTH.md` broadly stated that any Memory write requires a separate
confirm/edit-and-confirm. The explicit «запомни» flow satisfies that invariant, but `/end-session`
directly writes a `session.summary` canonical fact. Documentation is narrowed in this change to
describe the verified explicit-request contract; the runtime bypass remains debt and is not
declared fixed.

### 8.2. Current gaps, not contract contradictions

- current principal partition is strong baseline, but project/domain/Vault and agent-purpose
  isolation are missing;
- current canonical atoms are accepted records but lack required provenance/revisions/temporal
  semantics;
- current derived vector index can be rebuilt manually, but no durable reconciliation contract was
  found;
- current UI has explicit save confirmation, conflict resolution and pinning, but no complete
  inspect/source/explain/correct/suppress/forget/export lifecycle;
- current sensitivity protection exists for tool approvals/paths, not Memory classification,
  embedding, retrieval or deletion;
- current context slots are bounded but mix unrelated sources and omit a retrieval/projection
  manifest;
- `memory_companion.db` name and physical responsibility mix can mislead future implementation into
  treating audit/feedback/policy as Memory.

No contradiction was found with Context or Multi-Agent target contracts after selecting separate
promotion and projection boundaries. The new contract deliberately strengthens their existing
requirements rather than changing them.

## 9. Open / deferred decisions

Architecture intentionally does not choose:

- DB/schema/migration path;
- vector engine, embedding model, lexical index or reranker;
- exact confidence formula, thresholds, TTLs or schedules;
- concrete Vault crypto/key management;
- API/UI/transport and background jobs;
- exact token budgets;
- whether entity graph remains derived or later becomes a separately governed authority;
- shared multi-user Memory semantics.

Before implementation, separate design work must define acceptance operation idempotency,
revision model, source-reference retention/deletion closure, Vault threat model, index
reconciliation and migration handling for provenance-poor legacy atoms. None of those are
implemented by this documentation change.

## 10. Sources

- [Cognitive Architectures for Language Agents (CoALA)][coala]
- [Generative Agents: Interactive Simulacra of Human Behavior][generative-agents]
- [MemGPT: Towards LLMs as Operating Systems][memgpt]
- [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory][longmemeval]
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory][zep]
- [Graphiti official repository][graphiti]
- [LangGraph Memory overview][langgraph-memory]
- [OpenAI Agents SDK: Agent memory][openai-agent-memory]
- [OpenAI Agents SDK: Sessions][openai-sessions]
- [Microsoft GraphRAG indexing methods][graphrag-index]
- [Microsoft GraphRAG query overview][graphrag-query]
- [NIST: Agentic AI Emerging Threats, Mitigations, and Challenges][nist-agentic]

[coala]: https://arxiv.org/abs/2309.02427
[generative-agents]: https://arxiv.org/abs/2304.03442
[memgpt]: https://arxiv.org/abs/2310.08560
[longmemeval]: https://arxiv.org/abs/2410.10813
[zep]: https://arxiv.org/abs/2501.13956
[graphiti]: https://github.com/getzep/graphiti
[langgraph-memory]: https://docs.langchain.com/oss/python/concepts/memory
[openai-agent-memory]: https://openai.github.io/openai-agents-python/sandbox/memory/
[openai-sessions]: https://openai.github.io/openai-agents-python/sessions/
[graphrag-index]: https://microsoft.github.io/graphrag/index/methods/
[graphrag-query]: https://microsoft.github.io/graphrag/query/overview/
[nist-agentic]: https://csrc.nist.gov/csrc/media/presentations/2026/agentic-ai-emerging-threats,-mitigations,-and-cha/1.3-agentic_ai-sotiropoulos.pdf
