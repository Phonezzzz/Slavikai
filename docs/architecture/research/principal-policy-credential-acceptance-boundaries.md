# Gate 0 — principal, policy, credentials and human acceptance boundaries

**Status:** read-only system audit, incomplete, 2026-09-24. Production source
inspected at `920d5f3`; no live authentication, restart or external-effect test
was run. This is Current State evidence and Target boundary analysis, not a
section 8/9/19 implementation specification or a vulnerability verdict.

## Current State evidence

| Fact or decision | Inspected source | Established limit |
| --- | --- | --- |
| Ingress identity | `server/http/common/auth.py::auth_gate_middleware` resolves `RequestIdentity(principal_id, role, auth_method)` for `/ui/api/` and `/v1/` or `/slavik/`; Cloudflare UI identity may be `owner` or `member`, bearer automation has `automation` role. | A principal and role exist at ingress; that does not prove every downstream tool or egress path enforces the same effective role. Local unauthenticated mode deliberately maps to `local_unauth`. |
| UI session ownership | `auth.py::_ensure_session_owned` asks UIHub for access by current principal; `server/agent_provider.py::AgentScope` includes principal and session. | Session ID is a scope key, not an approval or a new principal. The agent provider also has a static-instance construction path, so path-specific isolation needs verification. |
| Human decision packet | `server/http/handlers/decision.py` checks session ownership, decision identity, pending state and expiry before applying a choice. `always_allow` separately requires owner role. | A persisted decision packet describes a proposed action and user choice. It cannot by itself prove that the external effect happened or that task criteria passed. The detailed race/expiry and replay audit belongs to sections 8/9/10. |
| Action permission | `core/tool_gateway.py::ToolGateway.call` evaluates `core/approval_policy.py::decide_request` for agent tool calls before registry dispatch, with a special path for `confirmed_decision_only` tools. | ToolGateway is an observed agent-tool policy lane, not proof that every direct UI route or background path passes through it. A `ToolResult` is an execution report, not acceptance evidence. |
| Approval lifetimes | `server/http/common/runtime_contract.py::SessionApprovalStore` holds category and Desktop once/session grants under `AgentScope` in memory. `core/desktop_policy.py::DesktopPolicyStore` persists scoped rules with subject principal. | Restored UI decisions, current once/session grants and persistent rules are different facts. Restart, principal change, rule revocation and action-time revalidation need explicit contracts. |
| Provider credentials | `server/http/handlers/settings.py::handle_ui_settings_update` requires owner role before provider-key changes; `config/api_keys.py` saves provider keys in an application-level JSON file with mode `0600` on write. `server/http/common/ui_settings.py::_resolve_provider_api_key` accepts provider/key source, not principal or delegation; UI chat and STT call it. `llm/*` also supports provider environment/config keys. | The inspected store/resolver does not enforce ADR-0007's owner-only default at key resolution. This source trace does not establish all egress routes or a live exploit. File mode does not settle role-based use, rotation, revocation or secret exposure. No key values were read. |
| User acceptance of results | Current Verification contract requires evidence against exact criteria/revisions, and ADR-0002 gives incomplete-result acceptance to the user or preapproved criteria. UI `approve_once`/`confirm` decisions authorize specific actions such as plan execution or Memory save. | Action/plan approval is not result acceptance. Current routes do not establish a common persisted task-result acceptance transaction; this remains a Target-to-runtime gap. |

## Proposed Target owner split for Gate 0 review

1. **Identity/Principal authority** resolves actor, principal, role and
   delegation/automation origin. A task, run, action, approval, artifact,
   credential use and user acceptance must retain that binding. UI session
   identity is an attachment, not the root of authority.
2. **Policy/Approval authority** owns effective rules, user grants, expiry,
   scope and revocation. The action executor evaluates permission against the
   concrete action and current authority. A preapproved trigger rule grants
   only the bounded task-creation class from ADR-0003/0004, never arbitrary
   future tool effects.
3. **Credential authority** owns secret material, allowed principal/route,
   use and revocation. Model Access decides route eligibility, and execution
   uses credentials only within that decision. Memory Vault, Context and UI
   settings projections cannot become a general credential store by accident.
4. **Human oversight** is an interaction and decision interface over several
   authorities. It may submit goal/criteria changes, grant a scoped action,
   accept an incomplete result, cancel work or approve a trigger rule, but
   each choice commits through its respective owner. A generic `approved`
   boolean would conflate these distinct decisions.
5. **Verification/Acceptance** owns evidence and the acceptance decision
   interface for exact criteria and result revisions. Lifecycle alone commits
   accepted task state and terminal outcome; Communication projects that
   outcome. ADR-0002 governs partial disposition and final cause.

This split is a boundary proposal, not an accepted deployment topology. It
keeps security decisions independent of model text, retrieved content,
scheduled events and UI projections while preserving the user's ability to
authorize specific work.

## Carry forward

- Gate 0 must give identity, policy/approval and credentials explicit owners
  in the system map, even if their later service/storage topology is open.
- Sections 8/9/19 must audit every call path, credential source, role and
  action-time check; section 11 must test approval/decision recovery; section
  6 must define exact result acceptance and its durable handoff to Lifecycle.
- [ADR-0007](../decisions/ADR-0007-owner-provider-credential-delegation.md)
  resolves owner-key use: owner only by default; another principal requires
  an explicit delegation rule. Current application-level key resolution does
  not receive that principal/delegation. Audit every use path before designing
  the credential handle and rollout in sections 8/9/18/19.
