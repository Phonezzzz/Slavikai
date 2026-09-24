# ADR-0003 — Preapproved autonomous work initiation is in Target

- **Status:** accepted
- **Date:** 2026-09-24
- **Decision owner:** SlavikAI product owner
- **Affected domains:** work initiation, task/run lifecycle, identity/policy,
  events, background execution, budgets, communication, audit

## Context

The 20-section plan includes background continuation but does not explicitly
decide whether a new task may start without a fresh user message. The current
inspected UI/API entry points start work from requests; `core/rule_engine.py`
matches approved rules against an existing user message and changes
pre-generation instructions, so that path does not answer the Target question.
The [Gate 0 boundary check](../research/mature-system-boundary-check.md)
shows why schedule registration/firing and an existing run's continuation are
different responsibilities.

## Decision

1. The mature SlavikAI Target **includes** new work initiated by a schedule
   or an external event without a new user message.
2. Such work may start **only under rules approved in advance by the user**.
   An event payload, model text, tool output or extension metadata cannot
   approve its own rule or expand its authority.
3. A trigger firing is a proposal to create new work under the currently
   effective principal, rule, policy and resource scope. The task/run authority
   accepts a new task/run only after validating those references. A trigger
   does not itself mark work completed or grant tool permission.
4. Work initiation, continuation of an existing run and delivery of a later
   notification remain distinct semantic handoffs. Their physical services,
   stores and scheduling mechanism are not selected here.

## Why

The product owner selected preapproved autonomous initiation for the mature
system. Making the approval boundary explicit preserves user control over
when new work may begin and prevents an untrusted event from becoming a task
or permission solely because it arrived.

## Consequences and open design

- Capability Discovery must include work initiation and its dependency on
  identity, effective policy, accepted task criteria, budget and lifecycle.
- The detailed contract must define rule registration/revision/revocation,
  principal and source binding, schedule/event semantics, expiry, missed-event
  handling, overlap, deduplication, restart recovery and notification policy.
  ADR-0004 assigns owner authority; OQ-CD-04 tracks the remaining protocol work.
- This is a Target decision, not a claim that a scheduler, event subscription
  or durable trigger registry exists in current SlavikAI runtime.
- No particular scheduler or protocol (including Temporal or MCP) is chosen.

## Alternatives considered

- Only start work from a current user/UI/API request: rejected by the product
  choice for the mature Target.
- Let arbitrary external events or model output create tasks: rejected because
  it would bypass the chosen preapproved-rule boundary.
