# Gate 0 closure matrix — Capability Discovery

**Status:** Gate 0 accepted at system level, checked 2026-09-24 against architecture
checkout `e7cc2b0` and production checkout `920d5f3`, both reachable on
`origin/architecture/target` and `origin/codex/vision-evidence-grounding`. The
unintegrated snapshot objects `6040640`, `4ee7516`, `4b9a453` and `f883893` are
preserved on `origin/archive/slavikai-transfer-snapshot-2026-09-16`. The source
20-section work list is summarized in
[`../roadmap/master-roadmap.md`](../roadmap/master-roadmap.md). This is an
architecture acceptance record, not evidence that Target runtime exists.

| Gate requirement | Evidence now | Verdict | Remaining work after Gate 0 |
| --- | --- | --- | --- |
| Mature-system capability space, including missing areas beyond the 20 headings | [Candidate inventory](capability-discovery-inventory.md), [20-section owner review](plan-coverage-and-owner-review.md), [boundary proposal](capability-boundary-proposal.md), [primary-source boundary check](mature-system-boundary-check.md) and [sources.md](sources.md). All 20 headings have an explicit owner/cross-cutting classification; media, extensions, local inference, work initiation and data consent/egress appear beyond them. ADR-0003–0009 decide several Target boundaries. | **Met at system level.** No unowned major area was found in the reviewed plan, flows and references. This does not assert that no future capability will be discovered. | Carry OQ-CD-04 and OQ-MA-12/13 details to later sections; validate data consent/egress enforcement against remaining runtime flows. |
| Current State and ownership anchored to production code | [Flow/ownership matrix](current-state-flow-ownership-matrix.md), [state lifetime inventory](state-authority-inventory.md), [principal/policy/credential audit](principal-policy-credential-acceptance-boundaries.md), [media/extension/operations path audit](media-extension-operations-boundary-audit.md), [resource/event/evaluation audit](resource-event-evaluation-boundary-audit.md) and [trust-boundary draft](system-flows-and-trust-boundaries.md) cite actual entry, execution, policy and storage paths. | **Met for capability discovery.** Representative entry, execution, policy and storage paths anchor the split; the audit does not certify all runtime routes. | Carry exhaustive path and enforcement audits into their numbered sections; do not infer production deployment from static code. |
| Existing Target/ADR/claim status is separated from runtime | [Contract/claim status inventory](contract-status-inventory.md) distinguishes accepted/present docs, registry labels, working-tree ADR-0002–0011 and four unintegrated snapshots; no snapshot is represented as shipped. | **Inventory criterion met:** all four snapshots are explicitly pending with reconciliation work. Contract integration remains open. | Record final disposition after boundary review; later update claims only with runtime evidence. Gate 0 need not implement those contracts. |
| Lifecycle/Communication/Verification contradiction is known and resolved where product choice exists | [Reconciliation audit](lifecycle-communication-verification-reconciliation.md), ADR-0002/0010/0012, updated Communication and Verification contracts; [blocker B-2026-09-24-01](../roadmap/blockers.md). | **Met for Gate 0.** The accepted normative rule is unambiguous; the conflicting snapshot remains explicitly pending, not accepted. | The reconciled contract is in `../TASK_RUN_LIFECYCLE_CONTRACT.md` with owner acceptance confirmed; the lifecycle target claim is recorded; section 4 runtime is next. Do not integrate snapshot text as-is. |
| Accepted system capability map, owner boundaries and trust boundaries | [Capability map](../system/capability-map.md), [boundaries](../system/boundaries.md), [owner review](plan-coverage-and-owner-review.md) and ADR-0011 classify accepted owner boundaries, pending detailed Target contracts and cross-cutting protocols, including request-scoped media and external-provider governance. | **Met at system level.** One owner per fact and the major trust handoffs were checked against ARCH_CANON, Communication, Verification, ADR-0001–0011 and all four pending snapshots. | Detailed owner transactions, security enforcement and schemas remain for their sections. |
| Accepted major data flows and directed dependencies | [Flows](../system/data-flows.md) and [dependencies](../system/dependencies.md) cover ingress → task → action → evidence → final, trigger, Memory/Context, model egress, media/extensions, deployment and canonical/derived state. | **Met at system level.** Major handoffs and directed design dependencies are recorded; wire and recovery protocols are still open. | Specify and verify the exact handoffs in dependent sections. |
| Open questions, blockers and discovered work are visible | OQ-CD-01/02/03/05 are settled by ADR-0007–0010; `open-questions.md` retains detailed OQ-CD-04 and OQ-MEM-01. `roadmap/blockers.md` records Lifecycle and Memory snapshot integration blocks; [discovered-work.md](../roadmap/discovered-work.md) tracks work outside the active route. | **Met.** Remaining questions are assigned to later domain audits; neither changes the coarse map. | Preserve the blocks until contracts agree. |
| Checkable Gate 0 Done record and dependent implementation order | This matrix and [master roadmap](../roadmap/master-roadmap.md) record the accepted coarse map and its limits; [dependencies](../system/dependencies.md) defines which contracts precede runtime work. | **Met.** Gate 0 closes without claiming any numbered runtime section Done. | Start the first dependent contract reconciliation, then section work through audit → Target/ТЗ → verified runtime slices. |

## Decisions that control promotion

- **Product scope:** ADR-0008 keeps media interaction request-scoped without a
  continuous session owner; ADR-0009 includes governed external providers with
  a separate owner from skills. ADR-0003 accepts preapproved schedule/event
  initiation, and ADR-0004 assigns the owner;
  OQ-CD-04 now concerns the detailed firing contract. ADR-0005 assigns the
  local-engine owner; OQ-MA-12/13 retain its acquisition and operations scope.
  ADR-0007 settles owner-key default/delegation, while implementation remains
  for sections 8/9/18/19.
- **Cross-contract semantics:** ADR-0010 requires supersession status and
  lineage without a separate final solely for replacement. ADR-0002 fixes
  `aborted` and accepted partial work. ADR-0012 makes every accepted ordinary
  Ask a durable logical task with one semantic final. The reconciled Lifecycle
  draft now reflects all three; owner acceptance was confirmed 2026-09-24.
- **Engineering boundary decisions:** ADR-0011 and the four system documents
  assign the coarse owners of task/run, action effect, artifact, approval,
  communication and trigger facts. Exact transaction and recovery contracts
  remain for their numbered sections.

The local Ollama route is a concrete cross-boundary audit item for sections
8/9/18/19: `auth_gate_middleware` authenticates `/ui/api/`, while
`handle_ui_local_ollama_start` does not inspect the resulting role before
calling a host-process launcher. The allowed-role policy and runtime behavior
need a targeted audit; this matrix does not label it a vulnerability or change
that endpoint.

**Gate verdict: accepted at system level on 2026-09-24.** The four system files
are the coarse Target map, boundaries, flows and dependencies. They do not
accept pending domain snapshots or prove runtime. B-2026-09-24-01 blocks
Lifecycle contract integration/section 4 specification; B-2026-09-24-02 blocks
Memory snapshot promotion/automatic writes. B-2026-09-24-01 is resolved at the
contradiction level by the reconciled contract; owner acceptance and the
lifecycle target claim are complete. No numbered section is runtime-Done because
of this research; section 4 runtime is the next dependent work.
