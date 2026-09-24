# Capability Discovery — boundary proposal for review

**Status:** research proposal, incomplete, 2026-09-24. This is not the accepted
`system/capability-map.md`, an implementation order, or a runtime-completion
claim. Current State evidence is in
[current-state-flow-ownership-matrix.md](current-state-flow-ownership-matrix.md).
Normative inputs are `ARCH_CANON.md`, the current Communication and Verification
contracts, ADR-0001–0010, and **unintegrated** lifecycle, Context, Memory and
Multi-Agent contracts at commits `6040640`, `4b9a453`, `f883893` and `4ee7516`.
The production checkout examined is `920d5f3`.

## Proposed capability owners

An owner below means the proposed authority for its own facts and decisions,
not a prescribed service, process, package or database. Rows need further
cross-domain review before becoming Target.

| Candidate owner | Own facts/decisions | Boundary that must remain explicit | Plan coverage |
| --- | --- | --- | --- |
| User intent and planning | Clarification and intent proposals, proposed goal/criteria revisions and execution strategies | Task/run control records accepted goal/criteria and plan revisions with authorized user/policy input; a plan cannot grant tool authority or declare completion | 4, 5, 18 |
| Work initiation and triggers | Target includes user-preapproved schedules/external-event subscriptions (ADR-0003); Work Initiation owns rules, firing history and deduplicated task-creation intent (ADR-0004) | Lifecycle accepts task creation after current principal/policy/criteria/budget validation. Background execution owns continuation of existing work. Detailed firing contract remains open | Target beyond 1–20; 4, 9, 12, 14 |
| Task/run control | Accepted goal/criteria and plan revision references; logical task, run, attempt and subtask identity; transitions, ownership, wait, cancel and recovery decisions | Worker/model/UI reports do not mutate authoritative task truth without a validated transition | 4, 10, 14 |
| Agent coordination | Typed coordination history, membership proposals, dependency exchange and conflict reports for simultaneous agents | Optional execution mode; task/run control accepts claims and ownership epochs, while private agent context remains separate | 1, 4, 12 |
| Tool/action execution | Typed operation, attempt, target, outcome and external-effect reconciliation | Policy grants permission; tool output is an observation, not authority or verification | 7, 10 |
| Verification and acceptance | Criteria evaluation, evidence packages, typed outcomes and acceptance decision interface | Verifier reports evidence; lifecycle commits completion with authorized acceptance | 6, 4, 16 |
| User communication | Semantic progress, clarification, waiting, final and notification identities and delivery | Projects accepted state; tokens, UI events and worker text are not task truth | 5, 13 |
| Context | Bounded model-visible projection for a principal/task/run/agent/turn | Projection is disposable; Memory and lifecycle remain separate authorities | 2, 3, 9 |
| Memory | Versioned accepted long-term knowledge, provenance, conflict/correction and forgetting | Not transcript, context package, artifact store, approval or credential vault | 3, 2, 16 |
| Result and artifact records | Identity, content/version/integrity, provenance, acceptance references and retention for produced objects | Existence or a named file in assistant text does not imply verified result | 16, 6, 11 |
| Model access | Route eligibility, capability evidence, model/provider/protocol selection and explicit fallback | Access entitlement, transport, credentials and billing/privacy boundary stay distinguishable; a local HTTP client does not own model installation or engine supervision | 18, 15, 19 |
| Local inference operations | Target host engine identity/configuration, authorized lifecycle, readiness/health, process and failure reconciliation (ADR-0005) | Model Access owns route qualification/selection; policy, resource and action authorities constrain host effects; provisioning/tuning scope remains open | 18, 8, 9, 15, 19 |
| Resource governance | Target shared allocation/reservation decisions across task/run/children and model/tool/host/verification use (ADR-0006) | Lifecycle owns task budget baseline and exhaustion transition; adapters meter/enforce locally; policy controls increases | 15, 4, 10, 18 |
| System evaluation | Target cross-task cases, datasets, scored runs and improvement findings (ADR-0006) | Evaluation does not accept a particular task result; criteria-bound Verification and Lifecycle retain that authority | 17, 6, 20 |
| Interaction and modality adapters | Request-scoped text, STT/TTS, image and attachment ingress/egress (ADR-0008) | Transcription/rendering/transport do not own task or communication semantics; direct UI media routes need an explicit policy boundary; no continuous session owner | 5, 7, 16 |
| Reusable procedures | Skill/procedure discovery, version, eligibility and provenance | Current manifest matching yields instructions, not permissions; a skill cannot elevate principal authority | Candidate beyond 1–20; 7, 8, 19 |
| External extension governance | External provider identity, capability catalog, versions, trust, permission grants and revocation (ADR-0009) | Provider metadata is untrusted input; each call remains subject to typed Tool Execution and effective principal/policy/approval | Candidate beyond 1–20; 7, 8, 9, 19 |
| Operations and quality feedback | Deployment/configuration lifecycle, health, telemetry, audit and system-level evaluation | Quality eval and telemetry do not substitute for per-task verification | 15, 17, 19, 20 |

## Candidate classification, not yet a domain decision

| Concern | Proposed classification | Reason and unresolved boundary |
| --- | --- | --- |
| Lifecycle, execution, verification, communication, Context, Memory, artifacts and model access | Capability owners | Each holds a distinct durable fact, decision or externally visible outcome in the target contracts; exact component topology remains open. |
| Identity/policy/approval and credentials | Cross-cutting authority with dedicated security owners | Every action needs a common principal and effective policy. Approval and credential state have their own authority; neither may be inferred from Context or tool output. [System-level audit](principal-policy-credential-acceptance-boundaries.md) traces selected paths; ADR-0007 fixes owner-key default/delegation, while full role/credential data-flow remains open. |
| Human oversight and result acceptance | Interaction interface crossing goal, policy, verification and lifecycle owners | `approve_once`, accepted partial result, trigger-rule approval and cancellation have different subjects and effects. The UI collects a choice; the relevant authority must validate and commit it. Current shared acceptance transaction is unproven. |
| Persistence, recovery, causal events and provenance | Cross-domain contracts applied by the owners (ADR-0006 for material history) | A universal store or event bus is not required by the evidence; each aggregate must state what is canonical, replayable and reconstructible. Need transaction-boundary and retention inventory. |
| Resource budgets | Cross-domain constraint with Resource Governance allocation owner (ADR-0006) | Parent/child, model, tool and verification budgets interact. [Selected-path audit](resource-event-evaluation-boundary-audit.md) finds Auto/Plan limits, not a shared ledger; reservation/accounting details need section 15. |
| UI streams, chat messages, Computer activity and notifications | Delivery/visibility projections | They can show state or carry user input but must not independently decide task success or policy. Semantic communication artifact is a separate target owner. |
| Voice/image/browser/GUI modalities | Request-scoped interaction adapters plus media/artifact evidence (ADR-0008) | Current STT/TTS and attachment paths convert formats; media interpretation and generated media need provenance and post-condition rules. Continuous session control is outside current Target. |
| Skills and third-party extensions | Separate reusable-procedure and external-provider governance owners (ADR-0009) | Current manifest yields selected instructions, not permissions. Installation, version trust, revocation and external-tool identity require later audits. |
| Local inference operations | Target owner split accepted by ADR-0005 | Current `LocalHttpBrain` calls an endpoint, and a UI route can launch `ollama serve` and poll discovery. Neither is a full manager. Provisioning, tuning, host-action authorization and resource contracts remain OQ-MA-12/13. |
| Work initiation | Target capability and owner boundary accepted in ADR-0003/0004 | In the inspected HTTP routes, work starts through browser/API calls; `RuleEngine` matches an existing user message and changes pre-generation instructions, not task creation. A repository-wide absence claim is not established. Rule consent, expiry and deduplication details remain open. |
| System evaluation | Accepted quality-feedback owner separate from task verification (ADR-0006) | [Selected-path audit](resource-event-evaluation-boundary-audit.md) finds feedback and batch review, not criteria-bound task acceptance or a Target eval system. Evals compare behavior across cases/releases; evidence/data-retention boundary remains section 17. |

## Shared contracts that cross owners

- **Identity, policy and approval (8, 9, 19):** principal, actor, task and
  action identities must propagate across every owner. A security authority
  decides permission and approval scope; it does not own the task result.
  Credential and sensitive-data storage need their own trust boundary.
- **Persistence and recovery (10, 11):** each owner names canonical state,
  version, transaction boundary, derived projections and retention. Recovery
  reconciles those owners; a UI snapshot cannot become a second task authority.
  Current lifetime evidence is in
  [state-authority-inventory.md](state-authority-inventory.md).
- **Events and causal history (12):** producer, ordering, deduplication and
  replay follow the owning aggregate. A durable domain transition, UI stream,
  audit record and telemetry event have different guarantees.
- **Resource governance (15):** time, token, monetary, tool and external
  resource budgets constrain plans, execution, verification and child work.
  Exhaustion is a stop or decision input, never evidence of accepted success.
- **Provenance and version binding (2, 3, 6, 16–18):** any model-visible fact,
  result, artifact or capability claim identifies source, authority, revision,
  sensitivity and validity where material.

## Proposed directed dependencies

```text
principal + policy + approved goal/criteria
              ↓
task/run control ←→ planning/coordination
              ↓
typed tool/action execution → result/artifact records
              ↓                         ↓
       verification evidence ←──────────┘
              ↓
task acceptance/terminal transition
              ↓
semantic communication → delivery adapters
```

Context draws **read-only projections** from authorized task state, evidence,
artifacts and Memory into a model call; model output returns as a proposal.
Model access selects a qualified route subject to policy. Persistence records
each owner's state; observability/evaluation observe and assess without
becoming the authority for task completion.

ADR-0003 includes autonomous triggers in Target. ADR-0004 assigns the
preapproved rule and firing ledger to Work Initiation, which sends a stable
intent to the **task-creation boundary before this diagram**. It cannot mint
a principal, approval or accepted goal by itself. Detailed firing semantics
remain pending OQ-CD-04.

The most change-amplifying interfaces appear to be (a) principal/task/run and
revision identity, (b) action attempt and unknown-effect semantics, (c)
evidence-to-acceptance-to-terminal transition, and (d) canonical-versus-derived
state with recovery. This is an architectural inference from the Current State
splits and target contracts, not a validated implementation sequence.

## Contract reconciliation before promotion

The four snapshot commits above are **not ancestors** of the current
`architecture/target` HEAD (`e7cc2b0`), as checked with Git. They remain
source material, not accepted documents in this checkout. The checks below
compare semantic boundaries; they do not integrate those files.

| Topic | What is established | Remaining decision/evidence |
| --- | --- | --- |
| Terminal cause and partial result | ADR-0002 fixes cause × disposition and acceptance authority; Communication/Verification now refer to it | Integrate a reconciled lifecycle contract; snapshot logical-task states omit `aborted` while ADR-0002 requires it for an administratively terminated task revision |
| Completion authority | Communication and Verification agree that worker text, tool success and artifact existence cannot complete a task | Specify the actual lifecycle acceptance transaction, persisted revision checks and final publication handoff |
| Recovery | Current UIHub persists workflow snapshot; Plan/Act has step resume and Auto reruns the goal from a process-local pause | Decide canonical checkpoint/effect reconciliation boundary across all lanes; do not infer durable execution from a saved `running` label |
| Context/Memory | Unintegrated contracts separate ephemeral context projection from accepted long-term knowledge; current principal-specific stores and confirmed save exist | Check full retrieval/write scope and consent, correction, forgetting and sensitive-vault authority before importing contracts |
| Coordination | Unintegrated contract distinguishes optional shared coordination from task and private context | Reconcile claims/epochs with lifecycle, principal isolation and acceptance; current MWV is not proof of the target shared coordination mode |
| Results/communication | Target semantic communication references versioned results; current UI artifacts are session JSON derived from response text | Determine artifact owner, integrity/version, retention and exact link to evidence and final delivery |
| Model access | ADR-0001 requires a separate subscription-backed route and forbids silent cross-boundary fallback | Compliant entitlement/auth/transport is open; route taxonomy and capability qualification remain research |
| Modality, skills and local inference | Current STT/TTS routes, text-serialized attachments, manifest skill selection, local HTTP model client and narrow Ollama launcher are confirmed in code | ADR-0008/0009 decide request-scoped media and separate external-provider governance; ADR-0005 assigns local engine lifecycle separately from model routing. Detailed contracts still need section 7/18/19 audits. None is proven complete by these paths. |

| Snapshot contract | Boundary check against current contracts/ADR | Result |
| --- | --- | --- |
| Multi-Agent Coordination `4ee7516` | Separates authoritative task state/ownership from typed coordination history and private agent context; approval remains canonical policy lane; material events durable. Consistent with current Communication/Verification authority split. | Provisionally aligned. Membership, claim epoch and recovery handoff still need a single accepted lifecycle owner and revision checks. |
| Context `4b9a453` | Makes model-visible package a derived projection; task/plan/policy, verification and artifacts remain in their owning stores; Memory is separate. Consistent with the candidate owner split and no-completion-from-prose rule. | Provisionally aligned. Context identity and invalidation must bind to the reconciled task/run and policy revisions; no runtime-completion claim. |
| Memory `f883893` | Requires governed promotion/acceptance, principal scope and provenance; excludes task/approval/policy truth. Sensitive Memory Vault explicitly excludes credentials and secret tokens. Consistent with the proposed Context/Memory/security owner separation. | Snapshot policy-based automatic acceptance needs reconciliation with DevRules §11 and current explicit confirm (OQ-MEM-01). Consent, deletion closure, vault/credential and source-artifact handoffs also remain open. |
| Lifecycle `6040640` | Owns task/run transitions and separates attempts/results/verification; aligns with no-final-before-acceptance. Its logical task terminal enum omits ADR-0002's `aborted` cause. | Normative conflict open (B-2026-09-24-01); cannot integrate unchanged. |

## Gate 0 questions reviewed during system-map acceptance

1. Validate the proposed owner rows against missing mature-system concerns.
   Current State probes now cover voice/attachments, skill selection and the
   local model client and UI-invoked Ollama launcher; Target boundaries for
   request-scoped media semantics, extension trust, inference-operation scope, credentials,
   human review and long-running
   autonomy remain open. Split or merge rows only when one owner can maintain
   a coherent invariant.
2. Decide which cross-cutting contracts require a dedicated authoritative
   subsystem and which are rules enforced by each owner. Identify every
   canonical-to-derived state handoff and trust boundary.
3. Audit remaining target claims and unintegrated contracts for contradictions
   with ADR-0002 and the selected owners. Register blockers and decisions;
   do not treat Git integration as semantic acceptance.
4. Record the accepted coarse `system/` map and its limits in the
   [Gate 0 closure matrix](gate-0-closure-matrix.md). Detailed Target contracts
   and runtime enforcement remain later work.

The candidate table covers all 20 numbered plan sections; this was checked
against their headings, not their full implementation acceptance criteria.
**Next step:** review the cross-owner handoffs in
[system-flows-and-trust-boundaries.md](system-flows-and-trust-boundaries.md),
reconcile Lifecycle and record remaining engineering decisions before
promoting any system document. The four snapshot contracts have a provisional
boundary comparison above; none has been integrated by that comparison.
