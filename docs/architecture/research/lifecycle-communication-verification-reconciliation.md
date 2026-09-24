# Lifecycle / Communication / Verification reconciliation

**Status:** audit finding, 2026-09-24. Normative resolution:
[`../decisions/ADR-0002-terminal-outcome-and-partial-result.md`](../decisions/ADR-0002-terminal-outcome-and-partial-result.md).
Supersession is resolved separately by
[`ADR-0010`](../decisions/ADR-0010-supersession-lineage-without-separate-final.md):
status/lineage without a separate old-revision final. Согласованный draft
находится в [`../TASK_RUN_LIFECYCLE_CONTRACT.md`](../TASK_RUN_LIFECYCLE_CONTRACT.md)
со статусом «черновик согласования Target»; он ожидает cross-document acceptance и не является
runtime claim. The audit itself is not a runtime claim.
**Scope:** compare the unintegrated lifecycle research/contract in Git snapshot
`6040640` with the communication and verification target contracts on
`architecture/target` at `e7cc2b0`. Production checkout observed at `920d5f3`.
No merge, code change, or contract status promotion was performed.

## Shared invariants confirmed

- Worker/model/tool completion, result submission, artifact existence and UI text
  cannot by themselves complete a task. Lifecycle owns the authoritative
  transition; verification reports evidence; communication projects accepted
  lifecycle state. All three documents agree on this boundary.
- Cancellation request, cancellation in progress and accepted terminal
  cancellation are distinct. Unknown external side effects require reconciliation
  before unsafe retry or an unqualified success claim.
- Progress is not final. A final must identify the current task/run revision and
  relevant verification/result/artifact state.

Sources: snapshot `6040640:docs/architecture/TASK_RUN_LIFECYCLE_CONTRACT.md`,
sections 5, 13, 17 and 22;
`USER_INTERACTION_COMMUNICATION_CONTRACT.md`, sections 1, 5, 8 and 9;
`VERIFICATION_ARCHITECTURE_CONTRACT.md`, sections 1, 4 and 6.

## Normative mismatch

| Question | Lifecycle snapshot | Communication / Verification target | Required decision |
| --- | --- | --- | --- |
| Terminal state | Run: `completed / failed / cancelled / aborted`; logical task has `completed / failed / cancelled / superseded`. Partial is not a state. | Communication names `final_partial` beside success/failure/cancelled; section 5 says “completed, failed, cancelled and partial outcomes”. Verification also mentions `final_partial`. | Define a separate result disposition and map each terminal state + disposition to one final artifact. |
| Accepted incomplete result | Completion transaction accepts criteria, but neither the run nor task model defines an accepted partial disposition. | `final_partial` includes budget exhaustion, policy-limited scope, mixed subtask outcomes, unavailable dependency and user termination after usable result. | Specify who may accept incomplete scope, whether task revision is narrowed, and how a cancelled/failed run can carry a usable result without being called completed. |
| Aborted run | Run has `aborted` for administrative/policy/recovery stop without a domain failure assertion. | No `final_aborted` class or explicit mapping. | Decide user-visible final class and required disclosure for an aborted run. |
| Final cardinality | Snapshot describes final projection after terminal acceptance. | One logical final per task revision, but a task revision may have multiple runs and a run can terminate without ending the task. | Specify whether a terminal run without terminal logical task gets a progress/waiting artifact or a final. |
| Verification outcome | Snapshot compresses required verification to accepted/rejected/inconclusive in its flow. | Target verification defines passed/failed/inconclusive/stale/not_applicable/blocked/verifier_error and a distinct acceptance decision. | Use the richer typed verification outcome at the boundary; acceptance remains a later lifecycle transaction. |

These are semantic mismatches. Textual merge resolution cannot decide them.

## Recommended contract shape for decision

1. Preserve terminal lifecycle state and result disposition as orthogonal
   concepts. Candidate disposition: `full / accepted_partial / none`, bound
   to exact task/criteria/result/artifact revisions and acceptance authority.
   `accepted_partial` requires explicit acceptance of what remains incomplete;
   it is not inferred from a nonempty artifact or exhausted budget.
2. One authoritative final for the **logical task revision** only when that
   revision reaches a terminal outcome. A terminal run that will be retried or
   replaced does not generate task final. Run outcome remains available in status
   and meaningful progress. This follows the existing one-final-per-revision rule.
3. Include both terminal state and disposition in the semantic final payload.
   Resolve presentation class by explicit mapping, not by treating
   `final_partial` as a lifecycle state. At audit time the precedence cases
   `cancelled + accepted_partial`, `failed + accepted_partial`, and
   `aborted + accepted_partial/none` were unresolved; ADR-0002 resolves them
   by preserving terminal cause in the class and disclosing accepted partial
   work as a separate result field.
4. Verification emits its seven typed outcomes against exact revisions.
   Coordinator/control plane alone accepts a full or partial result and commits
   lifecycle transitions. `inconclusive`, `stale`, `blocked` and
   `verifier_error` cannot be silently treated as `passed`.

The product owner selected cause-first finals and user/preapproved-criteria
partial acceptance. ADR-0002 records the complete mapping. The snapshot
`6040640` still must not be integrated mechanically. Gate 0 accepted only
the coarse system boundaries; the detailed lifecycle contract remains open.

## Section 4 follow-up: internal snapshot ambiguity

The snapshot's section 3 says a newly accepted task revision supersedes the
previous current revision within a logical task. Section 4 instead defines
logical-task `superseded` as replacement of the goal by **another logical
task**. ADR-0010 covers old/new accepted revision status and lineage, so the
reconciled contract must distinguish same-task revision replacement from
cross-task replacement. Both need explicit identity and stale-work fencing;
neither produces an old-revision final merely because replacement occurred.
The snapshot also routes `result_submitted → verification → acceptance → run
completed → final projection` in section 13. It must insert the separate
logical-task terminal decision: a completed or stopped run can leave the task
active, and only a user-facing terminal task revision gets a logical final.
This is a contract correction, not evidence of current runtime behavior.

## Next evidence and acceptance gate

- Reconcile the lifecycle contract from snapshot `6040640` with ADR-0002,
  ADR-0010 and the updated communication/verification contracts after the
  system-level boundary review and before section 4 runtime specification;
  preserve the source snapshot.
- Recheck the accepted Gate 0 boundaries and current runtime paths before
  writing a lifecycle implementation spec. Preserve the cross-task versus
  same-task replacement distinction and the run/task final boundary.

## Section 4 contract edit checklist (2026-09-24)

This checklist is the narrow semantic redline for adapting the **existing**
snapshot, not permission to copy its other pending cross-domain contracts as
accepted. Current-path evidence and limits are in
[current-state-flow-ownership-matrix.md](current-state-flow-ownership-matrix.md).

| Snapshot location | Required reconciliation | Acceptance check |
| --- | --- | --- |
| Intro and §§1–3 | Reference the accepted system map, ADR-0002/0010, Communication and Verification directly. Mark Multi-Agent, Context and Memory snapshots pending. Keep task, revision, run, subtask, operation and attempt IDs distinct. | No pending snapshot becomes normative by citation; no UI session ID substitutes for task identity. |
| §4 task/revision | Add logical-task `aborted` user-facing terminal cause. Keep `superseded` as a revision status/lineage transition outside the four terminal causes. Distinguish replacement within one `task_id` from explicit cross-task lineage. | Four causes map to finals; revision replacement alone maps to status/correction. Prior delivered final remains historical. |
| §§5, 9–12 | Keep run `completed/failed/cancelled/aborted` independent of task disposition. Authority and fencing validate principal, task/revision, plan revision, policy, approval, budget and effect state at resume/retry. Work Initiation owns new-task triggers; `waiting_external_event` for an existing run does not own a trigger rule. | A terminal run may leave the task open; stale worker or approval cannot revive old work. |
| §13 completion | Replace compressed `accepted/rejected/inconclusive` verifier step with the seven Verification outcomes. Separate verification, authorized full/partial result acceptance, run transition, **task-revision terminal decision**, final readiness and delivery. Bind `full/accepted_partial/none` to exact criteria/result/artifact revisions; only the user or preapproved criteria may accept incomplete scope. | `result_submitted` and verifier `passed` alone never produce `completed` or final; invalid cause/disposition combinations are rejected. |
| §§14–17 failure/effects | Preserve failure level and effect uncertainty. A failed/cancelled/aborted run can retain accepted partial work without changing its cause; unknown side effects require reconciliation or an explicit unresolved disclosure before terminal projection. | Retry does not infer no effect from timeout; budget or policy stop does not become success. |
| §§20, 22–25 replan/communication | Version replacement and fence old attempts; retain historical artifacts/evidence. For `failed/cancelled/aborted` final, cause leads and accepted partial work follows. `final_partial` means only `completed + accepted_partial`; notification/progress is distinct from final. | One logical final per user-facing terminal revision, none merely for supersession or a terminal run whose task remains open. |
| §26 open choices | Move resolved cause, partial-acceptance and supersession decisions to ADR references. Durable ordinary Ask is resolved by ADR-0012. Keep detached-task eligibility, retention and non-compensatable-effect reconciliation explicitly open until decided. | No implementation spec silently assumes an answer to a remaining product/authority question. |

The first runtime slice can use one actual Plan/Act or Auto path after this
contract passes the cross-document check. It must prove identity/revision and
transition authority before adding durable waits, effect retry or final
publication. The order is a dependency, not evidence that any slice exists.

### Current-path check for this redline

Production checkout `920d5f3` still has `TASK_STATUSES =
{running, completed, failed, cancelled}` in
`server/http/common/ui_runtime.py`; `workflow_runtime._run_plan_runner` writes
`completed` when work reports success and its verifier passes. The status is
stored as `active_task` under a UI session in `server/ui_hub.py`; no separate
task-revision acceptance transaction is established by these paths. Auto's
`AutoOrchestrator._paused_runs` is an in-process dictionary, and `resume`
re-enters `run_v1` with the old goal/run ID. These source observations support
the identity, recovery and effect-reconciliation gaps; they do not prove a
particular production failure occurred. A focused run of workflow, Auto,
decision/approval and chat-cancellation tests on this checkout passed 58 tests
with `--no-cov`. The same focused selection with the repository-wide 80%
coverage gate enabled had 58 passing tests but exited 1 because focused
coverage was 20.88%; it was rerun without that unrelated global gate.
