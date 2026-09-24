# Verification Architecture Contract

**Status:** target architecture; not current runtime.

Result disposition and terminal communication follow
[`decisions/ADR-0002-terminal-outcome-and-partial-result.md`](decisions/ADR-0002-terminal-outcome-and-partial-result.md).

This contract defines how SlavikAI determines whether a result may be accepted by the
authoritative lifecycle. It complements
[`USER_INTERACTION_COMMUNICATION_CONTRACT.md`](USER_INTERACTION_COMMUNICATION_CONTRACT.md)
and [`VERIFICATION_ARCHITECTURE_RESEARCH.md`](VERIFICATION_ARCHITECTURE_RESEARCH.md).

## 1. Authority and core invariants

- `result_submitted` is not `completed`.
- Worker/model `done`, tool success, artifact existence, approval, UI state or verifier
  prose do not independently authorize completion.
- The producer is not automatically the authoritative verifier.
- Verification reports evidence/outcomes; it does not mutate lifecycle state by itself.
  The coordinator/control plane proposes completion, rework, failure, escalation or
  STOP; the authoritative Task/Run Lifecycle alone accepts the terminal transition.
- Approval answers whether an action is permitted; verification answers what happened and
  whether criteria are met. Approval success is not execution success.
- Required verification and acceptance are prerequisites for `final_success`.
  Gaps may yield waiting, rework, failure, cancellation or abort; an
  `accepted_partial` disposition requires user acceptance or preapproved
  criteria, not merely an incomplete verification result.

## 2. Taxonomy and profiles

The target distinguishes execution, result, artifact, state, policy and acceptance
verification. Epistemic verification qualifies evidence across them as verified,
accepted, inferred, unresolved, disputed or partial.

The coordinator selects a profile from risk, side effects, reversibility, artifact
importance, uncertainty, policy and budget:

- **minimal:** low-risk/read-only operation with no material acceptance concern;
- **self-check:** local deterministic check where correlated failure risk is low;
- **deterministic:** schema/test/post-condition/hash/state check is sufficient;
- **independent:** separate verifier/evidence path for material risk or producer bias;
- **external-confirmation:** authoritative external read-after-write/service state;
- **user-confirmation:** human acceptance for subjective/unobservable criteria.

Profiles are policy decisions. Independent verification is selective, not universal, and
required only when profile/risk/policy justifies it. Required strength cannot be silently
downgraded for budget or convenience.

## 3. Acceptance criteria and evidence

Material tasks/subtasks have explicit criteria bound to current task, plan and result
revision. Replan versions criteria; ambiguous criteria cause clarification, bounded
minimal acceptance or escalation, not vague model completion. The verifier receives the
exact current criteria revision.

Each attempt receives a bounded evidence package containing, as applicable, task/plan/
criteria revisions, result and artifact refs/versions, relevant tool observations,
external state reads, deterministic outputs, policy context, provenance, timestamps and
sensitivity classification. Private chain-of-thought, irrelevant transcript, private
sibling context and secrets are not passed automatically.

Evidence trust is attributed: authenticated runtime result, deterministic check,
external source, user assertion, agent assertion, artifact content and verifier
judgement do not have equal authority. Malicious artifact/tool output is untrusted input.

## 4. Lifecycle and outcomes

```text
result_submitted → verification_required? → verification_attempt
                 → outcome → accept / rework / fail / escalate / STOP
```

Semantic outcomes:

- **passed:** evidence supports the exact criterion/revision;
- **failed:** evidence demonstrates non-compliance;
- **inconclusive:** evidence insufficient or conflicting without proving wrongness;
- **stale:** target revision is obsolete;
- **not_applicable:** criterion/profile does not apply, with reason;
- **blocked:** required verification cannot proceed due to permission, dependency,
  unavailable evidence or user input;
- **verifier_error:** verifier failed, timed out or was unavailable.

Verifier error is not result failure. Inconclusive is not passed. Stale is not a valid
acceptance basis. Blocked requires wait/escalation/STOP, not silent success.
When an accepted task revision is superseded, its verification records remain
historical evidence bound to that revision. They cannot accept the replacement
revision's result without fresh revision-bound evaluation. Supersession status
and lineage follow ADR-0010 and do not themselves publish a separate final.

Retries distinguish rerunning a deterministic check, a new verifier attempt, reworking
the result and creating a new result revision. Repetition does not itself increase
confidence. Changed task/result/artifact/policy/external state can invalidate prior
verification.

## 5. Deterministic, semantic and external checks

Deterministic checks are preferred where sufficient: tests, schema/JSON/YAML parsing,
syntax/type/lint, file existence/hash/content, diff cleanliness, HTTP/DB/process state,
post-action reads and stable external identifiers. Semantic/LLM verification is for
meaning, quality, subjective or otherwise non-deterministic criteria.

Code verification distinguishes passed, failed, not-run and unavailable; scope/risk
selects targeted, integration, regression or full checks. Browser/GUI execution is not
visual success: use DOM/API/state post-condition where available, screenshots as bounded
evidence and user confirmation only for inherently subjective/unobservable criteria.

For side-effecting tools, distinguish intent, execution attempt, observed result and
authoritative external-state verification. Timeout/exception means unknown outcome, not
absence of side effect:

```text
unknown → reconcile/read-after-write → accept / retry / compensate / STOP
```

Write acknowledgement is sufficient only when the external contract makes it authoritative;
otherwise use a follow-up read with stable ID and account for eventual consistency. Retry
must be idempotent or guarded by existing-state verification.

## 6. Multi-agent, lifecycle and communication

- Workers publish result/evidence, not completion authority.
- Verifier outcomes are typed and attributed; private worker context is not automatic
  verifier input.
- Contradictory outcomes are retained and resolved by coordinator policy through re-run,
  independent verifier, rework or escalation; never last-write-wins.
- Acceptance is authoritative only after required profile outcomes resolve.
- Verification projects to communication as relevant evidence; it is not user chat and
  does not become Memory automatically.
- `final_success` requires required verification/acceptance. `final_failure` may follow
  terminal verification failure. Incomplete, inconclusive or blocked verification
  does not itself authorize `final_partial`; that class requires
  `completed + accepted_partial`. Failed/cancelled/aborted finals may disclose
  separately accepted usable work without changing their terminal cause.
- Verification history is bounded/projected into context and does not replace canonical
  task/artifact state.

## 7. Budgets, policy, security and auditability

Verification consumes tokens, tool calls, time and external cost. Policy fixes the minimum
profile and bounds budget; exhaustion cannot silently downgrade required verification.
The task may remain waiting, fail, be cancelled/aborted, or reach an
explicitly accepted partial disposition according to authoritative policy
and the user's acceptance criteria.

Verifier access is least-privilege and scoped to task/artifact/principal. It does not
grant approvals/capabilities, access private CoT/secrets by default or trust unvalidated
tool/artifact prose. Material verification must be auditable without private reasoning.

Audit evidence must identify what was checked, criteria/profile, verifier authority,
evidence refs/provenance, exact revisions, outcome/reason, retries/escalation and final
acceptance decision. Concrete storage/schema is implementation-defined.

## 8. Target invariants

1. Worker `done` is not verification; result submission is not completion.
2. Approval is not verification; verification does not provide permission.
3. Deterministic verification is preferred when sufficient.
4. Independent verification is selective, not universal.
5. Verification binds to exact task/criteria/result/artifact/policy revision.
6. Changed artifacts invalidate stale verification.
7. Timeout does not prove absence of external side effect.
8. Unknown side effects require reconcile/verify before material retry.
9. Verifier does not receive private CoT automatically.
10. Verification event does not change task state by itself.
11. Coordinator/control plane proposes completion; the authoritative Task/Run
    Lifecycle accepts the terminal transition.
12. Verifier execution failure is not result failure; inconclusive is not passed.
13. Contradictory outcomes are not resolved by last-write-wins.
14. Material evidence has provenance and verification is auditable.
15. Final success requires required verification/acceptance.
16. Verified result is not an automatic Memory write.

Concrete verifier model, runner, scoring threshold, queue, DB schema, retry count,
screenshot comparison and UI are implementation-defined.
