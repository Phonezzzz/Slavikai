# Gate 0 — current state authority and lifetime inventory

**Status:** read-only system-level audit, incomplete, 2026-09-24. Production
checkout `/home/ki/Project/Slavikai` at `920d5f3`. This inventory identifies
observed storage and process lifetime; it does not claim transaction safety,
complete domain coverage or successful restart tests. It is input to later
Persistence/Recovery audits, not their Target specification.

| State/fact | Observed owner and storage | Lifetime/rebuild evidence | Authority limit / dependency |
| --- | --- | --- | --- |
| Browser session, chat rows, mode, plan/task/Auto snapshots, decisions and UI artifacts | `UIHub` writes `PersistedSession` through `SQLiteUISessionStorage` (`server/ui_hub.py::_persist_session_locked`, `server/ui_session_storage.py`) | `_restore_sessions` reloads snapshots; sessions are pruned by TTL/count. Artifact JSON follows that session record. | The session is a durable UI/workflow snapshot. Restoring `running` does not reconstruct the worker, effect log or accepted task result. |
| Plan/Act worker and approval resume | `handlers/plan.py`/`handlers/decision.py` schedule `_run_plan_runner` as an asyncio task; `TaskPacket.context.plan_runner_resume` records completed steps in the UI workflow snapshot. | Process-local worker dies on process exit. A decision path can schedule resume; examined app startup has no general runner reattachment. | Step skip is not proof of exactly-once action effects. Canonical run checkpoint, effect reconciliation and ownership lease are Target dependencies. |
| Auto execution and pause | `AutoOrchestrator` holds `_paused_runs`; UIHub stores latest `auto_state` projection. | `_paused_runs` is in memory. `resume()` starts `run_v1` again from the original goal in the same process. | Saved `auto_state` does not reconstruct execution after agent recreation; repeated effects are a control-flow risk, not an observed duplicate. |
| UI live events | `UIHub` maintains bounded `event_buffer` and subscriber queues. | In memory; reconnect may request resync when the buffer cannot replay. | Transport/delivery state, not a durable domain transition or semantic final record. |
| Pending decision packet | UIHub session snapshot stores `decision_packet`; decision handler checks session owner, decision ID, state and expiry. | Persisted with UI session; handler may move it through `pending`/`executing`/resolution. | Packet persistence does not make approval consumption and external effect one transaction. |
| Once/session approvals | `SessionApprovalStore` maps `AgentScope(principal_id, session_id)` to categories and Desktop rules (`server/http/common/runtime_contract.py`). | In memory, constructed by `create_app`; scope rules may be cleared on mode/session changes. | Must not infer that a saved decision packet restores grants, or that prior permission remains effective after restart. |
| Persistent Desktop rules | `DesktopPolicyStore` stores principal-scoped rules in `.run/desktop_approvals.json` (`core/desktop_policy.py`). | File-backed; writes through temporary file and replace. Invalid rules fail load; an explicit reset path exists. | Separate policy authority from session approvals and task state; revalidate scope at action time. |
| HTTP duplicate-request cache | `IdempotencyStore` is an in-memory `(endpoint, session_id, key)` map with default 90-second expiry, used in chat send and Plan execute (`server/http/common/idempotency.py`). | Lost on restart; replay only within retained process/window and for matching request fingerprint. | Request replay suppression is not durable action idempotency and cannot reconcile an unknown external effect. |
| Global and session model selection | `config/model_store.py` persists global `ModelConfig`; `RuntimeModelStateStore` holds runtime global/session overrides in memory. UIHub persists session provider/model metadata; chat handler rebuilds the runtime override from that metadata before resolving the agent. | Global state hydrates at app creation. Session metadata restores with UIHub, while runtime override is reconstructed on the inspected UI chat path. | Persisted selection metadata and active runtime route are separate facts. Other lanes need path-specific verification; a selected alias does not prove route capability/readiness. |
| Long-term Memory | `server/principal_storage.py` gives scoped agents principal-specific database paths; `MemoryManager` and related stores own their records. | File-backed databases, with owner legacy path and hashed other-principal directories. | Memory is not an approval, task checkpoint or result acceptance store; full retrieval/write/forget scope remains for section 3. |
| Workspace files and external effects | Tool Registry dispatches tool calls; filesystem, browser, service or host tool owns the resulting external object/effect. UIHub may separately record a response-derived artifact. | External lifetime is tool/target-specific. The inspected generic `ToolResult` does not provide a universal durable effect ledger. | Artifact existence, ToolResult success and accepted task outcome must remain separate. An interrupted effect needs target-specific observation or idempotency evidence before retry. |

## System-level implications

1. **Durability is per fact, not per screen.** A restored UI session can display
   `running` while the process-local runner is gone. Target task/run authority
   must state how to reconcile that condition before delivery or retry.
2. **Policy has several lifetimes.** Once/session approvals, persistent Desktop
   rules and a pending decision packet are different records. Recovery cannot
   treat any one as a substitute for current effective permission.
3. **Idempotency is layered.** A short HTTP request cache helps within one
   process, while action intent, dispatch, effect observation and acceptance
   need stable principal/task/operation identities and recovery semantics.
4. **Model and Memory states are outside the UI task aggregate.** Route
   selection/egress and knowledge retrieval must be revalidated at the point
   of use; UI metadata and a context projection cannot authorize them.

**Next step:** use this inventory with
[system-flows-and-trust-boundaries.md](system-flows-and-trust-boundaries.md)
to decide accepted system boundaries and directed dependencies. Section 11
must later check transaction, migration, corruption, retention and replay in
detail; those properties are unverified here.
