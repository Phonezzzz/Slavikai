# Gate 0 — contract and claim status inventory

**Status:** current checkout audit, 2026-09-24. Architecture HEAD
`e7cc2b0` with the uncommitted ADR-0002–0010/research edits listed in Git status;
production reference HEAD `920d5f3`. This inventory distinguishes document
authority from executable runtime evidence. It does not promote any claim.

## Accepted or present in this architecture checkout

| Document | Current role | Runtime evidence boundary |
| --- | --- | --- |
| `ARCH_CANON.md` | Binding Ask/Plan/Act/Auto/Desktop invariants and Target design under `docs/SOURCE_OF_TRUTH.md` | Mixed current and Target statements; read its per-section status and claims registry, not the title alone. |
| `USER_INTERACTION_COMMUNICATION_CONTRACT.md` | Target semantic communication; updated for ADR-0002 | Registry claim `runtime.communication.semantic_artifacts` is `target`; current UIHub/streaming does not close it. |
| `VERIFICATION_ARCHITECTURE_CONTRACT.md` | Target evidence/acceptance; updated for ADR-0002 | Registry claim `runtime.verification.authoritative_acceptance` is `target`; existing Auto/MWV verifier passes are narrower. |
| ADR-0001 | Accepted product requirement for subscription-backed OpenAI/ChatGPT access | Registry claim `model_access.openai_subscription` is `target`; compliant auth/transport remains undecided. |
| ADR-0002 (working tree) | Accepted owner decision on terminal cause, partial acceptance and final mapping | No corresponding runtime claim or implementation in the current registry; lifecycle snapshot needs reconciliation. |
| ADR-0003 (working tree) | Accepted product scope for preapproved schedule/event-initiated work | No current scheduler, subscription or trigger registry claim is established by this ADR; ADR-0004 assigns ownership and OQ-CD-04 tracks detailed firing design. |
| ADR-0004 (working tree) | Accepted Target owner split: Work Initiation owns rules/firings; Lifecycle accepts task creation | No runtime implementation claim; exact rule/firing and cross-owner recovery protocol remain for later work. |
| ADR-0005/0006 (working tree) | Accepted Target boundaries for Local Inference Operations, Resource Governance, material history and System Evaluation | No runtime implementation claim; detailed contracts remain later work. |
| ADR-0007 (working tree) | Accepted owner-only provider-key use by default, with explicit delegation rule for another principal | The inspected resolver lacks principal/delegation context; full route audit and enforcement remain open. |
| ADR-0008 (working tree) | Accepted request-scoped media Target, without continuous voice/visual sessions | Current STT/TTS and attachment paths do not prove unified consent, egress or artifact handling. |
| ADR-0009 (working tree) | Accepted governed external provider Target, with permissions and versions owned separately from skills | No current extension-governance runtime claim; protocol and call contract remain open. |
| ADR-0010 (working tree) | Accepted supersession status/lineage without a separate final solely for replacement | Lifecycle snapshot needs reconciliation and no current runtime implementation is claimed. |
| ADR-0011 (working tree) | Accepted system-level one-owner-per-fact and cross-cutting protocol classification | Detailed contracts and all runtime claims remain separate; pending snapshots are not promoted. |

`docs/SOURCE_OF_TRUTH.md` orders runtime security enforcement, schemas and
`docs/runtime_contract_claims.json`, `ARCH_CANON.md`, Current State description
and guides. `docs/architecture/` is the source of truth for this long-term
architecture work; the claims registry is the machine-readable index of
**runtime** claim status and verification references. A registry label or test
path is not proof that this architecture turn executed the test.

## Relevant claims currently recorded

| Claim | Registry status | What Gate 0 may infer |
| --- | --- | --- |
| `auth.browser.cloudflare_access`, `identity.principal_isolation` | `implemented` | Scoped current mechanisms are recorded; cross-domain identity still needs its section 9 audit. |
| `runtime.ask.zero_side_effects`, `runtime.plan.read_only_tools`, `runtime.auto.v1_tool_loop` | `implemented` | Specific current lanes exist; no universal task lifecycle/acceptance follows. |
| `runtime.auto.ask_plan_act_fsm` | `target` | The intended Auto FSM is separate from current Auto v1. |
| `runtime.communication.semantic_artifacts`, `runtime.verification.authoritative_acceptance` | `target` | The contracts are accepted Target, not shipped behavior. |
| `memory.explicit_confirmation`, `skills.per_run_instructions` | `implemented` | Narrow confirmed mechanisms; not full Memory or extension-governance Target. |
| `model_access.openai_subscription` | `target` | Mandatory route remains research/implementation work. |

The registry has no claim for an integrated Task/Run Lifecycle contract or
general durable task recovery at this HEAD. That absence is a status/inventory
finding, not a proof that no individual runtime mechanism exists; the
[Current State matrix](current-state-flow-ownership-matrix.md) records those
mechanisms.

## Unintegrated source contracts

| Commit | Contract | Current status for this checkout |
| --- | --- | --- |
| `6040640` | Task / Run Lifecycle | **Pending reconciliation**, not ancestor of `e7cc2b0`; conflicts with ADR-0002 on logical task `aborted` and needs ADR-0010 status/lineage versus final mapping; block B-2026-09-24-01. |
| `4ee7516` | Multi-Agent Coordination | **Pending reconciliation**, not ancestor; provisionally compatible with ownership separation. Resolve lifecycle membership/epoch and event acceptance before deciding integration; not accepted here or implemented. |
| `4b9a453` | Context Architecture | **Pending reconciliation**, not ancestor; provisionally compatible with derived projection boundary. Bind package/recovery identity to accepted lifecycle revisions before deciding integration; not accepted here or implemented. |
| `f883893` | Memory Architecture | **Pending reconciliation**, not ancestor; Context/Memory/security owner split is provisionally compatible. Snapshot §5.3–5.4 allows policy-based automatic acceptance, while current DevRules §11 bans runtime auto-updates without explicit approve; OQ-MEM-01 records the unresolved authority boundary. Resolve consent, deletion closure and artifact/credential handoff before integration; not accepted here or implemented. |

Compatibility notes and the proposed domain split are in
[capability-boundary-proposal.md](capability-boundary-proposal.md). No snapshot
should be merged mechanically, and no source contract should become a runtime
`implemented` claim without path-specific executable evidence.

## Missing accepted domain contracts

The 20-section plan still calls for Tool Execution, Approval/Policy, Artifact,
Persistence/Recovery, Event, Budget, Observability, Model Routing and Security
audits/specs. Some related invariants exist in `ARCH_CANON.md` or other
documents, but this inventory found no accepted standalone Target contract
for each of these domains in the present `docs/architecture/` tree. Gate 0
must set their system boundaries; detailed contracts and runtime claims belong
to their later sections. Modalities, extensions and system evaluation still
need explicit classification so they are not lost merely because they lack
numbered sections. ADR-0008/0009 classify request-scoped media and external
provider governance; ADR-0005 classifies local inference operations as a
distinct Target owner. Their detailed contracts and runtime claims remain open.

**Next step:** use this status inventory during each domain reconciliation.
The system-level map is accepted in Gate 0; resolve the lifecycle mismatch
before integrating its contract, and keep Target and runtime claim statuses
separate throughout the roadmap.
