# User Interaction / Communication Contract

**Status:** target architecture; not current runtime.

**Source of truth:** this document defines the target semantic boundary for user-facing
communication. Runtime/lifecycle state remains authoritative for task truth. Current
evidence and alternatives are in
[`research/task-run-communication-audit.md`](research/task-run-communication-audit.md);
the lifecycle-facing summary is in [`ARCH_CANON.md`](ARCH_CANON.md).

The repository does not yet contain separate
`MULTI_AGENT_COORDINATION_CONTRACT.md`, `CONTEXT_ARCHITECTURE_CONTRACT.md` or
`MEMORY_ARCHITECTURE_CONTRACT.md`; this contract records their required integration
boundary without inventing those future contracts.

## 1. Semantic pipeline and boundaries

```text
authoritative lifecycle state
        ↓
communication decision
        ↓
typed semantic communication artifact
        ↓
rendering / streaming / notification / replay
```

- Chat remains the only conversational entrypoint.
- User-facing communication is a projection of authoritative lifecycle state, never
  its source of truth; communication does not mutate lifecycle state.
- Token streaming is presentation/transport behavior, not progress semantics.
- Computer activity is an operational visibility lane, not a second conversational lane.
- UIHub may be reused as delivery/projection infrastructure, but is not communication
  truth, lifecycle truth or durable replay authority.
- Workers/siblings publish internal coordination events by default; they do not write
  directly to user chat. Internal events become user-facing only through the
  communication decision layer.

Concrete wire schema, transport, queue, database, UI layout, notification provider,
serialization and retry timing are implementation-defined.

## 2. Communication classes

| Class | Purpose/authority | Prerequisite | Replay/retention | Multiplicity |
|---|---|---|---|---|
| `progress_update` | Meaningful accepted user-visible delta; communication decision layer | Active current revision | Bounded durable/replayable projection; may compact | Multiple; coalesce/deduplicate/stale-suppress |
| `clarification_request` | Input required to proceed | Authoritative waiting-for-user | Durable until resolved; replayable | Multiple per revision; resolve/supersede explicitly |
| `approval_request` | Projection of canonical approval subsystem | Valid approval required | Durable enough for recovery; replayable | Canonical approval identity; expiry/revocation supersedes |
| `waiting_notice` | Current waiting/blocker state | `waiting_user_input`, `waiting_approval` or significant blocker | Replay while current | Superseded by resume/replan/cancel/terminal |
| `status_response` | Explicit read-only status answer | Current authoritative state | Query-scoped projection | One per query; does not mutate state |
| `final_success` | Self-contained successful result | Accepted completion and required checks | Durable and replayable | One logical final per task revision |
| `final_failure` | Self-contained terminal failure | Authoritative terminal failure | Durable and replayable | One logical final per revision |
| `final_cancelled` | Self-contained terminal cancellation | Authoritative cancelled state | Durable and replayable | One logical final per revision |
| `final_partial` | Self-contained incomplete result | Terminal outcome with incomplete scope | Durable and replayable | One logical final per revision |
| `notification` | Minimal detached/background alert or pointer | Notification policy | Durable pointer/status as policy requires | Optional; never replaces full final |

These are semantic classes, not wire names or UI components. Each artifact has stable
logical identity, task/run/revision correlation, source lifecycle revision,
audience/principal, sensitivity classification, provenance and optional
supersession/correction relation. It may reference result/artifact versions without
owning those artifacts. Rendered assistant text is not necessarily canonical.

## 3. Two-phase communication

Two-phase communication means two semantic classes, not exactly two messages:

```text
progress -> progress -> final
```

is valid, as is `final` for a short task. Progress may be absent. A final for a
user-facing terminal task is always separate semantic completion communication. Progress
never means completion.

## 4. Progress contract

`progress_update` requires a meaningful, user-relevant, accepted delta. Candidate causes
are significant intermediate result, lifecycle phase change, blocker/wait, adaptive
replan, recovery, execution-mode change, partial verification or material remaining-work
change. The decision layer applies semantic deduplication, coalescing, sensitivity
filtering and revision/freshness checks.

The following do not create progress automatically: every tool call/retry, token delta,
coordination event, trace/log entry or Computer activity event. Low-level noise remains
outside Chat. Timing/rate limits are implementation-defined. Progress after a newer
terminal state, final or task revision is suppressed unless an explicit correction or
historical projection is intended.

## 5. Final publication gate

Final publication is permitted only when the authoritative lifecycle permits the
corresponding terminal outcome and all applicable conditions hold:

1. terminal state is accepted for the current task/run revision;
2. required verification and acceptance are resolved;
3. external side effects are reconciled or explicitly surfaced as unresolved;
4. result/artifact references are stable enough for delivery, or absence is explicit;
5. disclosure/security policy permits the audience;
6. communication is generated against the current revision with stable identity.

Worker `done`, model generation end, artifact existence, tool-loop end, no pending tool
calls, UI state or progress prose never authorize final by themselves. `completed`,
`failed`, `cancelled` and `partial` outcomes require corresponding final classes.

## 6. Final, epistemic and outcome semantics

Every final is self-contained: what was done, what was not done, significant result,
limitations, unresolved issues, relevant artifacts and material verification status.
Progress is supplementary visibility, not required context for understanding final.

Claims preserve qualifications such as verified, accepted, inferred, unresolved,
disputed or partial. `final_failure` describes actual incomplete work, possible success,
material cause and side-effect/artifact state. `final_partial` is first-class for budget
exhaustion, policy-limited scope, mixed subtask outcomes, unavailable dependencies or
user termination after an accepted usable result; it must not masquerade as success.

## 7. Waiting, clarification, approval and status

- `waiting_user_input`, `waiting_approval` and significant blockers are user-visible;
  short internal resource waits need not be.
- A clarification request is not final. Its answer binds to the outstanding request,
  task and revision and may resume or trigger adaptive replan. A changed goal creates
  an explicit revision rather than silently becoming an answer.
- Approval communication only projects canonical approval state. A stale card cannot
  create, revive, approve or revoke authority after expiry, denial or revocation.
- A status request is a read-only projection of current state, latest accepted result,
  blocker/wait, plan phase and relevant activity. It is not autonomous progress.

## 8. Cancellation, interruption and concurrency

`cancel_requested`, cancellation in progress and authoritative `cancelled` are
distinct. No cancelled final is published before terminal cancellation is accepted
when an external/non-cancellable action is still unknown. Late stale completion cannot
replace the cancelled final.

Interruption is classified as status, clarification answer, requirement change,
correction, pause, cancel or unrelated message. A new user message does not implicitly
cancel an active run. Requirement changes bind to task/plan revision, obsolete work and
adaptive replan; final is generated only for the latest accepted revision.

Multiple concurrent runs are supported semantically: progress, final, cancel and status
are task/run/revision scoped, with no cross-task mixing. A single `active_task`
projection is not the target correlation authority.

## 9. Delivery, recovery, replay and correction

The semantic lifecycle distinguishes artifact creation, required persistence, delivery
pending and delivered/replayed without prescribing an enum or transport:

- crash after final creation recovers and delivers the same logical final;
- delivery retry reuses the same communication identity and cannot duplicate it;
- delayed progress after accepted final is suppressed;
- old-revision communication is suppressed or explicitly historical;
- reconnect restores sufficient current representation: current wait/blocker, latest
  meaningful progress and terminal final/notification state, not necessarily raw events;
- material correction is explicit/traceable through supersession/correction relation,
  never a silent history rewrite.

Detached execution follows `terminal outcome -> durable final artifact -> optional
notification -> full final on reattach`. Notification is not the full final.

## 10. Integration and security boundaries

- **Multi-agent:** workers emit typed internal events; orchestrator/control plane
  aggregates accepted results, blockers, conflicts and verification outcomes.
- **Context:** communication history is a bounded conversation projection/reference,
  never authoritative task state; replay duplicates must not look like new facts.
- **Memory:** progress/final are not automatically promoted to Memory; promotion uses
  the separate Memory contract and explicit rules.
- **Artifacts:** communication and result artifacts are distinct; messages reference
  stable artifact versions and may summarize several artifacts.
- **Observability:** logs, traces, telemetry and Computer activity are not conversational
  truth.
- **Security:** private reasoning/CoT, secrets and raw sensitive tool payloads are
  excluded or redacted; principal/task scope is enforced; communication never grants
  capability or approval and unvalidated prompt-injected prose is not trusted output.
- **Rendering:** one semantic artifact may render as text, stream, voice, notification
  or card; token chunks are never canonical records.

## 11. Current-runtime relationship

| Surface | Current confirmed semantics | Target relationship |
|---|---|---|
| Assistant response path | Model/error/cancel text is appended to UIHub chat history | Pass semantic final/progress gate before classification |
| Token streaming | UIHub publishes stream deltas and auxiliary tool events | Rendering only; chunks disposable |
| Auto progress | Agent progress events are drained/published during stream | Candidate input; add filtering, identity and freshness |
| UIHub | Pub/sub, bounded event buffer, resync and session projection | Reuse with modification; not truth or durable replay authority |
| Computer activity | Separate operational events and summaries | Separate lane; may inform progress decision |
| Approval UI | Decision/approval payload projection | Mirror canonical approval; stale render cannot revive |
| Failure/error | Text and payloads can surface before unified terminal semantics | Bind final outcome to authoritative state |
| Cancel flow | Cancellation token/request and cancelled responses exist | Separate request/in-progress/terminal cancellation |
| Reconnect/replay | Bounded UIHub replay/resync | Add logical artifact replay and stale suppression |
| Session history | Persisted chat/workflow projections | Preserve semantic finals/decisions; raw event log not required |
| Background notification | No complete artifact contract | Optional notification points to durable full final |
| `active_task` | Session payload may contain active task | Not sole authority when concurrency is enabled |

## 12. Status and implementation-defined boundary

This contract is **target architecture; not current runtime**. Concrete wire schema,
transport, storage, queue, UI, notification provider, retry timing and token protocol
remain implementation-defined. The discovered requirement **Two-phase communication for
long-running tasks** is covered by this contract, but remains recorded as
`discovered requirement` because no `incorporated/resolved` roadmap transition exists.
This link is not a runtime implementation claim.
