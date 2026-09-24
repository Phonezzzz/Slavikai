# ADR-0006 — Separate resource allocation, domain history and system evaluation

- **Status:** accepted architecture boundary
- **Date:** 2026-09-24
- **Decision owner:** SlavikAI architecture workstream
- **Affected domains:** resource governance, task/run lifecycle, coordination,
  tool execution, work initiation, events, observability, evaluation,
  verification, policy

## Context

The [Gate 0 audit](../research/resource-event-evaluation-boundary-audit.md)
found lane-local Auto/Plan attempt limits, a bounded in-memory UI event
buffer, rotating diagnostics and interaction feedback/batch review. These have
different lifetimes and authorities. The unintegrated Lifecycle snapshot
`6040640` requires policy/budget baseline and recovery checks; Multi-Agent
snapshot `4ee7516` requires durable material coordination history. Neither
turns UI replay or diagnostics into authoritative task truth.

## Decision

1. **Resource Governance** owns shared allocation and reservation decisions
   across tasks, runs and child work, subject to principal/policy authority.
   Lifecycle records the accepted budget baseline and transition on exhaustion;
   model, tool, verifier and host adapters meter or enforce their local usage
   and report it. They cannot independently increase a shared grant.
2. **Each domain authority owns its own durable material facts and transition
   history.** Lifecycle owns task/run transitions; Work Initiation owns trigger
   firings; Policy owns grants; Tool Execution owns action attempts/effect
   reconciliation; Coordination owns accepted shared coordination history.
   A shared event envelope, bus or storage may distribute these records but
   does not acquire their decision authority. UI streams, traces and telemetry
   remain projections with their own retention and replay guarantees.
3. **System Evaluation/Quality Feedback** owns cross-task cases, datasets,
   scored runs and improvement findings. It may consume authorized evidence
   with provenance and retention controls. It never accepts a specific task
   result: criteria-bound Verification reports evidence, and Lifecycle commits
   accepted completion under the user/preapproved criteria.
4. Exact budget units, accounting, reservation transaction, durable event
   transport, replay protocol, evaluation methodology and datasets remain for
   sections 12/15/17. This ADR does not claim those mechanisms exist now.

## Why

Allocation spans children, models, tools, verification and host resources;
lane-local counters cannot decide a shared grant consistently. Domain history
must remain tied to the authority that changed the fact, otherwise a replayed
message can appear to create task truth. Quality scores across cases answer a
different question from acceptance of one user's exact result.

## Alternatives considered

- Put all budget decisions in each runner: rejected because concurrent child
  work and shared resource/cost limits need one allocation authority.
- Treat a generic event bus as the owner of all state: rejected because
  transport/append mechanics do not validate task, policy or effect semantics.
- Use feedback/eval score as task verification: rejected because its cases and
  criteria are not necessarily the current task/result revision.

## Consequences

- Gate 0 can include Resource Governance and System Evaluation as explicit
  Target owners, while event delivery and observability are cross-owner
  contracts. Coordination remains owner of its own material shared history.
- Sections 10–12, 15 and 17 must define exact transaction/replay, accounting
  and quality-evaluation contracts; no runtime completion follows from this
  decision. The lifecycle snapshot remains blocked separately by ADR-0002's
  terminal mapping conflict.
