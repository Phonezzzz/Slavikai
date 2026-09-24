# ADR-0005 — Local inference operations own the host engine lifecycle

- **Status:** accepted architecture boundary
- **Date:** 2026-09-24
- **Decision owner:** SlavikAI architecture workstream
- **Affected domains:** local inference operations, model access, tool/action
  execution, resource governance, policy/approval, observability, deployment

## Context

The current `LocalHttpBrain` is a client of a local model endpoint. Separately,
`server/http/handlers/sessions.py` exposes a UI action that can launch
`ollama serve` in a detached process and poll model discovery. This demonstrates
two different responsibilities, but does not establish a managed engine
lifecycle: no provisioning, process ownership across restart, ongoing health
or resource arbitration has been verified. The model-access research also
needs to qualify a route's model, protocol, capabilities and entitlement; a
successful HTTP response alone is not such qualification.

## Decision

1. The mature Target has a distinct **Local Inference Operations** owner for
   the lifecycle of a host inference engine used by local routes: declared
   runtime identity/configuration, authorized start/stop, readiness and health,
   process ownership, failure/restart reconciliation and actual resource use.
   This is an ownership boundary, not a choice of service topology or engine.
2. **Model Access** owns route eligibility and selection, model/capability
   qualification, protocol compatibility and explicit fallback. It consumes
   versioned local-runtime availability/capability observations; it cannot
   infer engine health from a configured URL or declare a model qualified
   because the engine started.
3. Effective principal/policy/approval authority governs host actions and
   downloads; Resource Governance grants or denies resource reservations;
   Tool/Action Execution records externally visible attempts and effects.
   Local Inference Operations cannot grant itself these authorities. UI and
   API surfaces request operations and project their state; they do not become
   the engine's lifecycle owner.
4. Model acquisition, automatic engine selection/tuning, installation and
   compilation are **not accepted by this ADR**. Their product scope,
   supply-chain rules, rollback and exact operational contract remain open
   in OQ-MA-12/13 and the section 18/19 audits.

## Why

An engine may be running while no model route is qualified, and a qualified
model route may become unavailable when its engine exits or resources change.
These facts have different lifetimes and evidence. One owner for engine state
lets restart/reconciliation have a canonical answer without turning a UI
handler or model client into the authority for a host process.

## Alternatives considered

- Put engine lifecycle inside Model Access: rejected because route/capability
  selection and host process state have different authorities, lifetimes and
  recovery paths.
- Keep launch/readiness in a UI handler: rejected for the Target because a UI
  request does not own the process after disconnect or server restart.
- Make a generic tool result the canonical engine state: rejected because a
  successful launch attempt is only an observation; sustained readiness and
  resource ownership require later reconciliation.

## Consequences

- Gate 0 can classify local inference operations as a distinct Target owner.
  Current runtime remains a client plus a narrow UI launcher, not a completed
  manager.
- Section 18 must audit route qualification and its dependency on local-runtime
  observations. Sections 8/9/15/19 must define host-action authorization,
  resource control, runtime recovery and deployment responsibilities.
- OQ-MA-11's owner split is closed by this ADR. OQ-MA-12/13 still control the
  scope and detailed contract; no installation, tuning or supervision is
  claimed implemented.
