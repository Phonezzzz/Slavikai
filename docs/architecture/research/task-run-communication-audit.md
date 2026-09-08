# Task / Run Communication Architecture Audit

**Статус:** research/evidence and rationale для normative target contract. Runtime
implementation не изменяется. Normative contract:
[`../USER_INTERACTION_COMMUNICATION_CONTRACT.md`](../USER_INTERACTION_COMMUNICATION_CONTRACT.md).

**Scope:** lifecycle-facing communication для progress, status, waiting/approval,
terminal results, delivery/replay и user interruption. Полный UI/chat protocol и wire
schema остаются будущим implementation design.

## 1. Current-runtime evidence

| Surface | Current evidence | Architectural verdict |
|---|---|---|
| Workflow state | `running/completed/failed/cancelled`; plan-step `waiting_approval`, `blocked`, `failed`, `done` | Partial lifecycle state; не достаточно для communication authority без acceptance/terminal gate |
| Assistant response | `ui_chat` получает model text/stream result и сохраняет assistant message в UIHub | Current response path; model text не является доказательством authoritative completion |
| Token streaming | `chat.stream.start/delta/done`, auxiliary tool/usage/error events | Presentation/transport surface; не semantic progress |
| UIHub activity | `agent.respond.*`, `response.ready`, context/tool activity and auto-progress publishing | Delivery/visibility infrastructure; не communication source of truth |
| Auto progress | Agent events drained and published during active stream | Existing operational signal; no semantic filtering, artifact identity or stale suppression |
| Computer activity | Separate `computer_events`, bounded and summarized independently from chat messages | Operational visibility lane; not conversational progress |
| Replay/resync | In-memory event buffer with bounded size/TTL, `get_events_since`, subscriber overflow resync signal | Useful delivery primitive; not durable communication replay or logical deduplication |
| Request idempotency | HTTP idempotency store replays request responses by endpoint/key/fingerprint | Request-level protection; not logical final/progress artifact delivery identity |
| Persistent history | UIHub/session storage persists chat messages and session state | Conversation projection; no typed communication-artifact store |

### Current final-path verdict

The current path treats a returned model text (or synthesized error/cancel text) as
the assistant response and appends it to chat history. Auto/tool result and verifier
signals influence surrounding workflow state, but there is no single final publication
gate proving that authoritative completion, required verification, acceptance and
stable artifact references have all been satisfied. This is a target gap, not a claim
that the current runtime is incorrect for its present scope.

Failure, cancellation and approval denial can produce user-visible text or decision
payloads, but their semantic relation to terminal task state is not represented by a
typed communication artifact. `cancel requested` and `cancelled` must remain distinct;
a failed attempt must not automatically become terminal failure communication.

## 2. Alternatives

| Alternative | Correctness | Noise | Recovery/replay | Multi-agent | Separation of truth/presentation | Verdict |
|---|---|---|---|---|---|---|
| Raw token/tool stream as progress | Weak: generation/tool completion is not lifecycle acceptance | High and provider-shaped | Chunks are disposable and hard to deduplicate | Poor: private peer events leak or duplicate | Weak | Reject |
| Every internal event user-visible | Weak and often misleading | Unacceptable spam | Large, stale event log | Poor: fan-out duplicates | Weak | Reject |
| Orchestrator textual updates | Better comprehension, but prose remains a weak source identity | Medium with filtering | Needs reconstructed identity/gates | Better aggregation, fragile provenance | Partial | Viable presentation strategy, not canonical model |
| Typed artifacts from lifecycle state | Strong; explicit prerequisites and provenance | Supports filtering/coalescing | Stable identity, replay and supersession | Strong aggregation boundary | Strong | Select |
| Hybrid chat + operational activity lane | Useful visibility, but risks second conversational lane | Medium | Activity replay differs from result replay | Good for telemetry | Partial unless artifacts remain canonical | Reuse activity lane only as separate projection |

## 3. Selected target architecture

Select:

```text
authoritative task/run state
        + coordination / verification / approval signals
        |
communication decision
        |
typed semantic communication artifact
        |
rendering / streaming / notification / replay
```

Lifecycle owns truth. The communication decision layer decides whether a meaningful,
audience-safe semantic output should exist. The artifact owns logical identity and
provenance. Rendering/delivery may produce chat text, a status card, a notification or
a replay projection. Token streaming is only a presentation/transport detail. Computer
activity remains an operational lane, while Chat remains the only conversational lane.

This is a bounded seam: it hides publication eligibility, coalescing, supersession and
delivery identity without making UI rendering responsible for lifecycle decisions.

## 4. Semantic communication artifact

The future contract should have a typed semantic record, without fixing a wire schema:

- logical `communication_id` and correlation to `task_id`, `task_run_id` and task/run
  revision;
- `communication_type`, audience/principal and sensitivity/redaction class;
- lifecycle state/revision and accepted source/result/artifact references used to create it;
- creation time and optional `supersedes`/replacement relation;
- delivery class (`conversation`, `status`, `approval`, `notification`) and provenance;
- semantic content/projection reference, separate from rendered assistant text.

Rendered text is not necessarily canonical. One artifact may have chat, notification
and reload projections; the same rendered text must not create a second logical artifact.
Communication artifacts do not own result artifacts and do not grant approvals or
capabilities.

## 5. Minimal semantic catalog

| Type | Creation authority | Prerequisite | Durable/replayable | Multiplicity/supersession |
|---|---|---|---|---|
| `progress_update` | communication decision layer | meaningful accepted delta; non-terminal run | bounded durable/replayable; coalescible | multiple per run; replace/suppress stale entries |
| `clarification_request` | lifecycle/orchestrator decision | input required to proceed | durable until resolved; replayable | multiple, task/revision-scoped |
| `approval_request` | canonical approval subsystem | approval required and still valid | durable enough for recovery; replayable | canonical approval identity; stale render cannot revive it |
| `waiting_notice` | lifecycle/communication layer | authoritative waiting/blocked state | replayable while current | superseded by resume/cancel/terminal state |
| `status_response` | communication layer | explicit user status query | usually non-authoritative projection | one per query; does not change task state |
| `final_success/failure/cancelled/partial` | communication layer after final gate | accepted terminal outcome + required checks | durable and replayable | one logical terminal result per task revision |
| `notification` | notification policy | event worth notifying outside active chat | durable pointer to result/status | optional; never substitutes for full final |

Types are semantic classes, not UI components. Exact names may change during detailed
contract design.

## 6. Publication gates

### Final gate

Final publication requires all applicable conditions:

1. authoritative terminal transition is accepted for the current task/run revision;
2. required verification and acceptance criteria are resolved;
3. external side effects are reconciled or explicitly surfaced as unresolved;
4. stable artifact/result references are available, or their absence is explicit;
5. policy allows disclosure to the target principal/audience;
6. generated communication is bound to the current revision and has a logical identity.

`completed`, `failed`, `cancelled` and `partial` each permit a different final type.
Worker prose, last model message, raw tool completion, UI state and artifact existence
alone never authorize final publication. If completion transition is not accepted,
the system may expose progress/waiting/error state but must not claim completion.

### Progress gate

Progress requires a meaningful user-relevant delta, accepted provenance, no sensitive
or private content, no duplicate/equivalent current artifact, no stale revision and no
better representation already present in the activity lane. Candidate causes include
accepted intermediate result, lifecycle transition, blocker/wait, recovery, replan,
partial verification and degraded mode. Event occurrence alone is insufficient.

## 7. Coalescing, freshness and correction

- A semantic coalescing window is implementation-defined; it groups related events
  before publication rather than exposing `tool started -> retry -> subtask done` as spam.
- Equivalent updates are deduplicated; newer accepted state supersedes stale progress.
- Reconnect may show latest meaningful progress plus current authoritative state, not
  the entire raw event stream. Progress after replan/cancel/final is suppressed unless
  explicitly qualified as historical correction.
- A correction uses an explicit supersession/correction relation and does not silently
  rewrite important history. Unresolved findings retain epistemic status: verified,
  accepted, unverified, conflicted or partial evidence.
- Final is self-contained and preserves significant decisions/limits; progress may be
  compacted to milestones, while final and user-visible decisions remain durable.

## 8. Status, interruption and concurrency

- A status request is a read-only projection of authoritative state, latest accepted
  results, blockers/waits, current plan phase and relevant activity. It does not mutate
  task state and is distinct from autonomous progress.
- Clarification, requirement change, cancel, status request and unrelated question are
  distinct intents. A user message does not implicitly cancel a run.
- A requirement change binds to an explicit task/run and creates a task revision or
  replan decision. Obsolete work may be cancelled/superseded; an old final cannot be
  delivered as the current result.
- The model supports multiple concurrent task runs semantically: every progress/final,
  cancel and status query is task/run scoped. No global `active_task` may be the only
  correlation authority if concurrency is enabled.
- Detached completion is `terminal state -> final artifact -> optional notification`.
  Notification may say only completed/failed/needs input; the full final remains
  available on reattach.

## 9. Recovery, delivery and retention

- Final created before crash: recover the existing pending artifact; do not generate a
  new logical final.
- Final sent but acknowledgement lost: retry by `communication_id`, never create a
  duplicate logical message.
- Queued progress after accepted final: suppress it as stale.
- Delayed old-revision communication: suppress or mark explicitly historical; never
  present it as current.
- Final, clarification, approval and meaningful progress have durable/replayable
  semantics appropriate to recovery; raw token deltas and activity telemetry may be
  ephemeral. Rendering caches are rebuildable.
- UIHub's bounded in-memory replay/resync and request idempotency are reusable delivery
  primitives, but must be extended or wrapped by durable typed artifact identity before
  they can satisfy this contract. Verdict: **reuse with modification**, not source of
  truth.

## 10. Integration and security boundaries

- **Multi-agent:** workers/siblings emit typed internal coordination events. Orchestrator
  aggregates accepted results, conflicts and provenance; only the communication decision
  layer publishes user-facing artifacts by default.
- **Context:** communication history is a bounded conversation projection/reference,
  never authoritative task state and never automatic duplication into private contexts.
- **Memory:** communication does not auto-promote content to Memory.
- **Verification:** accepted verification/acceptance is a final gate input; raw worker
  claims are epistemically marked.
- **Approvals:** approval artifacts project canonical approval state; expiration/revocation
  invalidates stale rendering and cannot be revived by replay.
- **Artifacts:** result artifacts and communication artifacts are distinct; references
  include version/stability information.
- **Observability:** logs/telemetry/activity are not conversational truth.
- **Security:** exclude private reasoning, redact secrets/tool payloads, enforce principal
  and task scope, prevent cross-task mixing, and never promote prompt-injected or
  unvalidated agent prose to trusted communication.

## 11. Target invariants

1. Progress and final are distinct semantic output classes.
2. Two-phase means two classes, not exactly two messages.
3. Token streaming is not progress semantics.
4. Computer activity is not a second conversational lane.
5. User-facing communication is not lifecycle source of truth.
6. Final publication requires authoritative lifecycle permission.
7. Worker `done` and artifact creation do not authorize final.
8. Progress cannot masquerade as completion.
9. Final is self-contained.
10. Stale progress is not delivered after newer terminal/revision state.
11. Workers/siblings do not publish user chat directly by default.
12. Clarification, approval and waiting have distinct semantics.
13. Cancel requested is not cancelled terminal.
14. Failed attempt is not terminal failure response.
15. Delivery retry is idempotent on logical communication identity.
16. Reconnect/replay does not resurrect stale communication.
17. Communication does not grant approval/capability.
18. User correction binds to explicit task/revision semantics.
19. Notification is not the full conversational final.
20. UIHub/activity/streaming are delivery/presentation surfaces, not authoritative
    communication truth.

## 12. Research conclusion

The preliminary hypothesis is confirmed and refined: choose **authoritative lifecycle
state + typed semantic communication artifacts + separate rendering/delivery**. Keep
UIHub with modification as delivery/projection infrastructure; keep token streaming and
Computer activity as presentation/operational surfaces. The original discovered
requirement remains recorded as `discovered requirement` with an explicit link to the
normative target contract; the repository has no separate incorporated/resolved roadmap
status. Runtime work is still required before the contract can be treated as implemented.
