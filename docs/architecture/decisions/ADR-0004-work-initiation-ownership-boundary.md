# ADR-0004 — Work initiation owns trigger rules and firing, Lifecycle owns tasks

- **Status:** accepted architecture boundary
- **Date:** 2026-09-24
- **Decision owner:** SlavikAI architecture workstream
- **Affected domains:** work initiation, task/run lifecycle, identity/policy,
  events, background execution, persistence/recovery, budgets, communication

## Context

ADR-0003 includes user-preapproved schedule/external-event initiated work in
the mature Target. The unintegrated Lifecycle snapshot `6040640` assigns the
Lifecycle controller authoritative transitions and describes a
scheduler/executor as lease/dispatch mechanics. It also distinguishes an
existing run waiting for an external event from a new logical task. Current
UI/API request handlers and message-triggered `RuleEngine` do not supply a
durable trigger-rule or firing contract. [Temporal Schedules](../research/mature-system-boundary-check.md)
is a reference for distinct schedule registration/action/overlap concerns, not
a selected implementation.

## Decision

1. **Work Initiation** is the owner of accepted trigger-rule identity,
   revision, enabled/revoked status, authorized source/schedule definition,
   principal binding and firing history. A rule references the user approval
   that permits creation of a bounded class of new work; it does not store or
   confer tool approvals.
2. An ingress adapter authenticates/normalizes an external event or clock
   occurrence and submits it to Work Initiation. Event content is untrusted
   task input. Work Initiation validates rule eligibility and emits a stable,
   deduplicated **task-creation intent** with rule/firing/source references.
3. The **Task/Run Lifecycle** authority alone accepts or rejects that intent
   and creates the logical task/revision/run under the current principal,
   criteria, policy, budget and cancellation constraints. It must revalidate
   the rule/revision and effective authority at acceptance so a revoked or
   changed rule does not grant stale work. The accepted task has its own
   identity and lifecycle independent of the trigger.
4. A scheduler/executor may wake and dispatch but cannot approve a rule,
   create authoritative task state on its own, grant effects or declare
   completion. A background worker continues an existing accepted task/run;
   a waiting run's external-event unblock remains a lifecycle transition,
   separate from firing a rule to create new work.
5. The boundary must reconcile crash/retry between firing record and task
   creation using stable intent identity and idempotent acceptance. The exact
   storage/transaction protocol, overlap, catch-up and notification policy
   remain for the detailed contract.

## Why

Rules may outlive individual tasks and may create several tasks, while one
task may have retries/runs independent of the rule that initiated it. Keeping
rule history and task truth under separate owners gives each one a coherent
identity, revision and recovery boundary. Policy remains a third authority;
neither an event nor a scheduler can turn a prior user approval into broader
permission at firing time.

## Alternatives considered

- Put the rule and firing ledger inside a UI session: rejected because a
  session is an attachment/projection and may expire while the approved rule
  is still valid.
- Make Lifecycle own schedule/event registration: rejected because rule
  revision, missed firings and overlap exist independently of any task and
  have a different lifetime.
- Let an Event bus or Background executor create tasks directly: rejected
  because delivery/dispatch is not task-creation or policy authority.

## Consequences

- Gate 0 can put Work Initiation in the system capability map as a distinct
  Target owner rather than an optional candidate. ADR-0003 still defines the
  product scope and user-preapproval requirement.
- OQ-CD-04 narrows to exact rule/firing contract: source identity, approval
  binding, expiry/revocation, overlap/catch-up, duplicate handling and
  transaction/reconciliation across initiation and Lifecycle.
- Sections 4, 8, 9, 10–12, 14–15 and 17 must honor this boundary. No current
  runtime implementation or scheduler adoption is claimed here.
