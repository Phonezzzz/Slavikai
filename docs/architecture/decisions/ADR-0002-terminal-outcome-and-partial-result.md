# ADR-0002 — Terminal cause and accepted partial result are separate

- **Status:** accepted
- **Date:** 2026-09-24
- **Decision owner:** SlavikAI product/architecture owner
- **Affected domains:** task/run lifecycle, verification/acceptance, user communication,
  artifacts, approvals/policy, persistence/recovery, multi-agent coordination

## Context and problem

The unintegrated task/run lifecycle contract in snapshot `6040640` distinguishes
terminal run states `completed / failed / cancelled / aborted`. The communication
target contract instead lists `final_partial` alongside finals for terminal causes.
Without a mapping, an accepted usable fragment after cancellation or failure can
hide why work stopped, while budget exhaustion or artifact existence can be
mistaken for accepted completion. See
[`../research/lifecycle-communication-verification-reconciliation.md`](../research/lifecycle-communication-verification-reconciliation.md).

## Decision

1. **Terminal cause** and **result disposition** are orthogonal, revision-bound
   facts. Terminal cause is `completed / failed / cancelled / aborted` for a
   user-facing terminal task revision. A run may reach a terminal state while
   its logical task remains open for a new run; that run does not emit a task
   final. `aborted` means an administrative, policy or recovery stop without
   asserting domain failure. A logical task terminated for that reason must
   retain `aborted` as its cause rather than be relabelled `failed`.
2. Result disposition is `full / accepted_partial / none`, bound to exact
   task, criteria, result and artifact revisions. Only the authorized user or
   criteria explicitly approved by that user before execution can accept an
   incomplete result. Budget exhaustion, model prose, artifact existence,
   verifier pass, policy block or worker report cannot confer acceptance.
3. The semantic final records **both** cause and disposition. Presentation
   leads with the cause when cause is `failed`, `cancelled` or `aborted`;
   accepted partial work is disclosed separately. A single result never gets
   two competing logical finals. The mapping is:

   | Terminal cause | Disposition | Communication class |
   | --- | --- | --- |
   | `completed` | `full` | `final_success` |
   | `completed` | `accepted_partial` | `final_partial` |
   | `failed` | `none` or `accepted_partial` | `final_failure`; disclose accepted work if present |
   | `cancelled` | `none` or `accepted_partial` | `final_cancelled`; disclose accepted work if present |
   | `aborted` | `none` or `accepted_partial` | `final_aborted`; disclose accepted work if present |

   `completed + none` and non-completed `+ full` are invalid terminal
   combinations until a reconciliation explicitly establishes a different
   authoritative outcome. `final_partial` is a communication class for
   accepted partial completion, never a lifecycle state.
4. Verification provides typed, attributed evidence and outcomes; a separate
   lifecycle acceptance transaction validates the applicable current
   criteria, partial acceptance authority, unresolved side effects and
   revisions. A failed or cancelled run can preserve an accepted partial
   artifact without converting its terminal cause to `completed`.
5. One logical final is published for a **user-facing terminal task revision**
   whose cause is one of the four mapped causes. Run termination that leaves
   the task active is represented through status or a meaningful
   progress/waiting artifact. [ADR-0010](ADR-0010-supersession-lineage-without-separate-final.md)
   makes `superseded` an explicit status/lineage transition without a separate
   final for the old revision; previously delivered finals remain historical.

## Alternatives considered

- Make `partial` a lifecycle terminal state: rejected; it loses the
  distinction between accepted scope and cause of termination.
- Emit `final_partial` whenever any usable work exists: rejected; cancellation,
  failure and administrative stop would be obscured.
- Let the coordinator infer acceptance from nonempty output or budget exhaustion:
  rejected; it bypasses the user's acceptance criteria.
- Keep cause and accepted disposition separate, with cause-first finals:
  selected by the product owner.

## Consequences and boundaries

- Communication and verification contracts must use the mapping above.
- The snapshot lifecycle contract may be integrated only after reconciliation
  with this ADR and the accepted Gate 0 system boundaries; this ADR does
  not approve a mechanical merge of `6040640`.
- Runtime needs revision-bound result/disposition evidence, an explicit
  acceptance authority, and terminal/final idempotency. This is Target, not
  a claim that current runtime implements it.
- Exact wire schema, storage, UI wording and timeout policy remain
  implementation-defined.
