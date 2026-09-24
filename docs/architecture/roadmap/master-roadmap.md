# Master Roadmap

- **Для чего:** задавать верхнеуровневую последовательность архитектурных работ.
- **Сюда:** milestones, зависимости, последовательность, acceptance gates, статус.
- **Не сюда:** детальная разбивка задач, research-содержимое.
- **Обновлять:** при изменении приоритетов или последовательности.

## Completed milestone — Gate 0 Capability Discovery (2026-09-24)

**Status: accepted at system level.** The 20-section work list is in
`/home/ki/Desktop/plan.md`; its section names are not yet the accepted
capability decomposition. Current evidence and a candidate owner/dependency
map are in `../research/capability-discovery-inventory.md`,
`../research/current-state-flow-ownership-matrix.md` and
`../research/capability-boundary-proposal.md`. The [20-section owner review](../research/plan-coverage-and-owner-review.md)
classifies each plan heading and additional system areas. Contract/claim status is in
`../research/contract-status-inventory.md`; a source-grounded draft of major
flows and trust boundaries is in
`../research/system-flows-and-trust-boundaries.md`. The Lifecycle/Communication/
Verification contradiction was audited, and ADR-0002 resolved the terminal
cause × partial-result decision. The lifecycle snapshot remains unintegrated.
The current durable/ephemeral state split is inventoried in
`../research/state-authority-inventory.md`.
Principal, approval, credential and acceptance handoffs have a selected-path
audit in `../research/principal-policy-credential-acceptance-boundaries.md`;
the [media/extension/operations audit](../research/media-extension-operations-boundary-audit.md)
records direct STT/TTS routes, static tool/skill surfaces and the limit of
deployment evidence. Memory snapshot promotion authority needs a separate
reconciliation with DevRules §11 (OQ-MEM-01; blocker B-2026-09-24-02).
ADR-0007 now fixes owner-key use to owner only by default with explicit
delegation to another principal. Current key resolution does not receive
principal/delegation and requires a full route audit before implementation.
Auto/Plan budgets, UI event replay, traces and quality feedback were compared
in `../research/resource-event-evaluation-boundary-audit.md`; ADR-0006 assigns
the Target resource, material-history and evaluation owner split. This is not
a runtime claim.
The criterion-by-criterion status is in
`../research/gate-0-closure-matrix.md`.
The accepted product choices and remaining engineering boundaries are summarized in
`../research/gate-0-decision-brief.md`.
Capability Discovery identified preapproved work initiation/triggers as a
Target capability beyond the original 20 headings (ADR-0003); ADR-0004 assigns
its owner, while OQ-CD-04 retains detailed firing design. Local inference
Current State includes a narrow UI-invoked Ollama launcher, not just an HTTP
client. ADR-0005 assigns the Target host-engine lifecycle owner separately
from Model Access; acquisition, tuning and detailed operations remain open.

Gate 0 is done only when the system-level capability map, boundaries, major
data flows and directed dependencies are accepted in `../system/`; every
unintegrated target contract is marked accepted, rejected or pending with
reconciliation work; architectural blockers/open questions are registered;
and no normative contradiction is silently treated as implemented behavior.
The four `../system/` files now contain the accepted coarse map, boundaries,
flows and dependencies. The [closure matrix](../research/gate-0-closure-matrix.md)
records the criterion-by-criterion decision. Pending snapshots and detailed
domain handoffs remain open; the Lifecycle conflict blocks section 4 contract
integration, not the accepted system map. No numbered section or Target runtime
is marked Done by this milestone.

**Next step:** the Lifecycle snapshot `6040640` is now reconciled as a draft in
`../TASK_RUN_LIFECYCLE_CONTRACT.md` against ADR-0002/0010/0012 and the
Communication/Verification contracts (see
`../research/lifecycle-communication-verification-reconciliation.md`). Remaining
before section 4 runtime work: record the lifecycle target status in the claims
registry, confirm cross-document acceptance, then take each dependent section
through Current State audit → Target/ТЗ → verified runtime PRs. Memory promotion
OQ-MEM-01 remains assigned to section 3.
