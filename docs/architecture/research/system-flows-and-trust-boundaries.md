# Gate 0 — system flows and trust boundaries

**Status:** preserved research draft, 2026-09-24. This is a source-grounded
Current State trace plus proposed Target handoffs. The accepted coarse Target
flows/boundaries and their limits are in `system/` and the Gate 0 closure matrix. Production checkout:
`/home/ki/Project/Slavikai` at `920d5f3`; no runtime test was run here.

## Current State: authority and data movement

| Flow | Observed path and state handoff | Boundary and unresolved issue |
| --- | --- | --- |
| Browser request | `server/http/common/auth.py::auth_gate_middleware` resolves `RequestIdentity` for `/ui/api/`; session access is checked against its principal through UIHub. `handlers/ui_chat.py::_handle_ui_send_impl` enters a scoped agent, streams a response, then writes UI messages, decisions and optional response-derived artifacts. | Authentication and UI session ownership are established at ingress. The assistant message is a UI projection, not an accepted logical-task result. |
| Automation request | The same middleware resolves bearer identity for `/v1/` and `/slavik/`. `/v1/chat/completions` enters a principal/session-scoped agent, but supports `ask|auto` without Plan/Act or streaming parity. | Shared principal/session identity must survive the different entry adapters; response shape cannot serve as a shared task-completion authority. |
| Plan/Act | Plan handlers compile `TaskPacket`, UIHub starts a task after mode/plan checks, and a process-local `_run_plan_runner` executes MWV. Runner outcome and verification are written back to UIHub; events/messages project that state to the browser. | The saved `running` snapshot is distinct from a live worker. On examined startup, no automatic scheduling of an interrupted runner was found. Plan step resume skips recorded complete steps, but interrupted external effects need reconciliation. |
| Auto | `AutoOrchestrator.run_v1` runs a model/tool loop and selected verifier, while UIHub receives progress and latest `auto_state`. Approval pause lives in `_paused_runs`; resume invokes `run_v1` again with the original goal. | The UI snapshot does not reconstruct a paused executor after agent recreation. Repeated external effects after a same-process resume are a control-flow risk inference, not an observed incident. Verifier profiles differ in evidence strength. |
| Tool action | `core/tool_loop.py` calls `core/tool_gateway.py::ToolGateway.call`, which evaluates approval/policy and dispatches via `tools/tool_registry.py`; the caller receives `ToolResult`. | Model/skill text is an action proposal. ToolGateway is the observed policy boundary for agent tool calls; `ToolResult.ok` is reported execution success, not a checked external postcondition. Direct UI TTS has a distinct path and needs a separate policy review. |
| Memory/context | `server/principal_storage.py` supplies principal-specific database paths; scoped Agent selects memory for a model call. An explicit remember request uses preview/decision and confirmed save. | Retrieved content is model input, not task/policy authority. Principal paths and one confirmed-save path do not prove full cross-lane consent, provenance or correction coverage. |
| Artifacts/delivery | `ui_artifacts._build_output_artifacts` derives session JSON records from response text, UIHub persists them in SQLite, and download handlers check session ownership. UI events have an in-memory bounded replay buffer. | A session artifact is neither a workspace file nor a verified task result; deletion follows session retention. Event replay is a delivery mechanism, not a canonical event log. |
| Media/model access | STT sends audio to a configured OpenAI transcription endpoint; TTS directly calls `TtsTool`; UI attachments become JSON text for model input. `LocalHttpBrain` calls a configured local HTTP endpoint. A separate UI endpoint can launch `ollama serve` and poll model discovery. | Provider egress, raw media retention, user-initiated TTS policy and attachment provenance need explicit ownership. The Ollama endpoint crosses into host process execution but does not provide a full inference-management lifecycle. |

Detailed code locations and evidence limits are in
[current-state-flow-ownership-matrix.md](current-state-flow-ownership-matrix.md).

## Cross-check against unintegrated Target snapshots

The snapshots below are source material, not accepted contracts in this
checkout. This check concerns handoffs and authority, not runtime behavior.

| Snapshot | Handoffs that fit the proposal | Reconciliation still required |
| --- | --- | --- |
| Lifecycle `6040640` | Distinct task/revision/run/attempt IDs, task-controlled terminal acceptance, durable operation intent and unknown-effect reconciliation, session reattachment and artifact references fit the proposed boundaries. | Its logical task terminal list omits ADR-0002 `aborted`. It lists `superseded` as terminal for an accepted revision but does not express ADR-0010's status/lineage without a separate final. See B-2026-09-24-01. |
| Multi-Agent `4ee7516` | Principal/task-run coordination scope, trusted membership, typed events and policy-filtered audience keep coordination distinct from task truth and private Context. Worker result is a submission; coordinator accepts completion. | Bind event and ownership epochs to the reconciled lifecycle revision; decide durable publication/acceptance boundary without treating the UI event buffer as that log. |
| Context `4b9a453` | ContextPackage is a derived projection of typed owner state and evidence, with principal, purpose, revision, source and invalidation references. Model text cannot grant policy or alter accepted state. | Define the accepted task/run/plan/policy revision identifiers and artifact source interface before importing this contract; reattachment must not reuse stale authority. |
| Memory `f883893` | Versioned accepted knowledge, consent/promotion, source provenance and scoped retrieval remain outside task, artifact and policy authority. Memory cannot grant tools or approvals. | Define source-artifact deletion/retention handoff and credential-versus-sensitive-Memory boundary; verify that principal/consent rules cover every current retrieval and write path. |

The only identified **normative mismatch** in this cross-check is the lifecycle
terminal mapping. Other rows identify dependencies or evidence still needed;
provisional compatibility is not acceptance or implementation.

## Candidate Target trust boundaries

These are proposed invariants for review, not claims that runtime implements
them. An arrow crosses an authority boundary; each handoff needs typed identity,
revision and audit/evidence rules before implementation.

```text
user / automation caller
  → ingress identity + session authorization
  → accepted task goal / criteria / policy revision
  → plan and model-visible Context (derived proposals)
  → action request + effective approval/policy
  → tool attempt + external effect + artifact record
  → verification evidence + authorized acceptance decision
  → terminal task transition + semantic final/progress identity
  → UI / API / voice / notification delivery
```

1. **Caller → authority:** browser owner/member, local mode and bearer automation
   have different authentication semantics. Every task, delegated actor, tool
   call, approval and artifact access must bind to the effective principal and
   allowed scope. A session ID alone is not a grant.
2. **External or retrieved content → agent:** user attachments, webpages, tool
   output, memory candidates and model output have provenance and sensitivity,
   but cannot change policy, accepted goal or completion state by instruction.
   Context is a bounded projection with revision/source references.
3. **Agent proposal → effect:** policy and approval apply to the concrete
   action/target under its caller/task context. The action attempt needs an
   identity and unknown-effect state so recovery can reconcile an interrupted
   call before retrying it. An approval response cannot silently broaden scope.
4. **Effect/result → acceptance:** a tool's reported success, a generated file,
   verifier score and user acceptance are different facts. Evidence must bind
   to the goal/criteria and result revision being judged. Only the lifecycle
   authority may commit a terminal outcome; ADR-0002 governs terminal cause and
   partial-result disposition.
5. **Canonical state → delivery:** a semantic final/progress record projects an
   accepted task state to each channel. Token streams, chat rows, notifications
   and UI artifacts can fail or replay without silently changing that state.
6. **Local state → providers/extensions:** external model/STT/integration
   requests cross egress and credential boundaries. Eligibility, data scope,
   provider route and fallback need independent decisions. Skill instructions
   or third-party extension output cannot grant their own permissions.

## Directed dependency implications

- Define principal/task/run/attempt IDs and revision checks before unifying
  Plan/Act, Auto, `/v1` and detached execution. Keep UI session ID as a channel
  scope unless explicitly bound to a task.
- Define action/effect reconciliation and result/artifact identity before
  promising durable retry, exactly-once behavior or verifiable delivery.
- Define criteria/evidence/acceptance and ADR-0002 terminal mapping before a
  common final publication gate. This is why the unreconciled lifecycle
  snapshot remains a blocker for section 4 specification, not Gate 0 research.
- Define canonical-versus-derived persistence and recovery for each owner
  before making event streams, Context caches or UIHub snapshots authoritative.
- Route model and request-scoped media access through principal/policy/egress
  decisions under ADR-0008. Govern external provider identity, versions and
  permissions separately from skills under ADR-0009.

**Next step:** reconcile the unintegrated Lifecycle contract with ADR-0002/0010.
Detailed runtime handoffs remain for the relevant numbered sections.
