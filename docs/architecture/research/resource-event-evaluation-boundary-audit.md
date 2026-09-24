# Gate 0 — resource, event and evaluation boundary audit

**Status:** read-only system research, incomplete, 2026-09-24. Inspected
production checkout `920d5f3`; no runtime or restart test was run. This
distinguishes observed mechanisms from Target owner boundaries accepted in
[ADR-0006](../decisions/ADR-0006-resource-event-evaluation-authority.md).
Detailed section 12/15/17 audits are still required.

## Current State evidence

| Concern | Observed path | What the observation does not prove |
| --- | --- | --- |
| Auto run limits | `core/auto_runtime.py::AutoBudgets` resolves environment-based runtime, tool-call, file, token and retry limits; `run_v1` records these in its local run state and returns `budget_runtime`/`budget_tool_calls` stops on inspected paths. | A local run limit is not a durable, principal/task/child shared allocation ledger. A budget stop does not establish accepted result or final-success criteria. Each declared limit needs a path-specific enforcement audit. |
| Plan/Act attempts | `server/http/common/workflow_runtime.py` derives `max_retries` from `TaskPacket.budgets.max_attempts`; `core/mwv/manager.py` loops worker and verifier attempts up to that bound. | This controls one retry dimension, not model spend, external cost, concurrent child reservation or resources consumed by unknown external effects. |
| UI events | `server/ui_hub.py` stores a bounded per-session `event_buffer` in memory and replays from a matching event ID; missing history requests resync. | Delivery replay is not a durable task transition or action-effect ledger. A server restart loses this buffer even if UI session snapshots restore. |
| Trace | `core/tracer.py` appends sanitized records to a rotating `logs/trace.log` with a size cap. | A diagnostic trace is not the authoritative causal log for task/run state, action attempts or policy grants. Its timestamps and rotation alone do not prove ordered/replayable domain events. |
| Feedback/quality review | `server/http/handlers/slavik.py` records feedback for a supplied interaction; the UI path verifies trace ownership. `core/batch_review.py` summarizes interactions/feedback and creates policy-rule candidates. | User feedback and batch review are quality inputs, not criteria-bound verification of the exact task/result revision. A candidate rule is not accepted policy merely because feedback produced it. |

## Target boundary analysis behind ADR-0006

1. **Resource Governance** must own the allocation decision for shared time,
   tool, token, monetary and host resources across a task and its child work.
   A task/run owns its limits and consumption references; execution/model/host
   adapters report measured use and may enforce local stops. A reservation or
   budget increase requires current principal/policy authority. The exact
   ledger, units, accounting for uncertain effects and cost provider are
   section 15 questions. The owner split is accepted by ADR-0006, not observed
   runtime.
2. **Domain transitions and causal records** remain with the authority that
   changes the fact: Lifecycle for task transitions, Work Initiation for
   firings, Policy for grants, Tool Execution for attempts/effect
   reconciliation, and Coordination for its accepted shared history. Section
   12 may define a shared event envelope and delivery
   infrastructure, but a generic bus cannot become a second authority for
   those states. UI replay, telemetry and audit exports have different
   guarantees. Whether a common durable event log is required remains open.
3. **System Evaluation/Quality Feedback** owns evaluation definitions,
   datasets/cases, scored runs, provenance and improvement findings across
   tasks/releases. It may consume sampled task evidence subject to consent and
   retention, but does not accept a particular user's task result. Section 6
   Verification/Acceptance stays criteria-bound to that task/result revision.
   Current feedback and batch review are inputs, not proof of this Target.
4. **Observability/Audit** projects events and operational measurements with
   principal, sensitivity, provenance and retention rules. Security audit,
   diagnostic traces, user progress and product quality evaluation are
   separate products of authoritative facts; none may promote a trace line
   into task truth or policy authority.

## Dependency and open scope

- Gate 0 classifies Resource Governance and System Evaluation as distinct
  Target owners under ADR-0006, and events/persistence/observability as
  cross-owner contracts. The unintegrated Lifecycle `6040640` records policy/budget
  baselines and remaining/reallocated budgets but does not define a shared
  allocation ledger; Multi-Agent `4ee7516` owns typed coordination history,
  not Lifecycle transitions. This supports a split, not integration of those
  snapshots. No competing evaluation owner was found in this targeted check.
- Sections 10–12 must define crash, replay, ordering and idempotency at the
  owning aggregate before a UI stream can claim durable delivery. Section 15
  needs parent/child reservations, concurrent consumption, compensation and
  exhaustion semantics. Section 17 needs eval privacy, dataset provenance,
  regression thresholds and feedback-to-policy review gates.
- No repository-wide claim that a global ledger/event bus/evaluation system
  is absent follows from these selected paths. The detailed audits must
  inspect other processes and deployments.
