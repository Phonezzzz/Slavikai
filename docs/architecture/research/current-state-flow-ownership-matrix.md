# Current-state execution flow and ownership matrix

**Status:** Gate 0 source audit, bounded rather than exhaustive, 2026-09-24. Read-only source audit of
`/home/ki/Project/Slavikai` at `920d5f3`. File/function references below are
observed code paths, not runtime test evidence or approved Target contracts.

## Entry points and control flow

| Path | Observed flow | State and execution owner | Limit relevant to Target discovery |
| --- | --- | --- | --- |
| Browser chat | `server/http/routes.py` → `handlers/ui_chat.py::_handle_ui_send_impl` → scoped `Agent.respond_stream` → response/report split → UIHub message, artifact, decision and workflow updates | UIHub session is the persisted UI projection; `ScopedAgentProvider` owns a mutable agent and lock per principal/session; `ChatCancellationRegistry` controls an active generation | A streamed model response and appended assistant message are not an accepted task outcome. Some responses are UI guidance without agent execution. |
| OpenAI-compatible API | `/v1/chat/completions` → `handlers/chat.py::handle_chat_completions` → scoped agent `respond` | Agent scope derives from principal and session; explicit UI session can supply workflow state | Supports `ask|auto`, rejects `plan|act`, and rejects streaming. This route does not supply the same UI workflow contract as browser Plan/Act. |
| Plan/Act | `handlers/plan.py` drafts/edits/approves a plan; execute compiles a `TaskPacket`, conditionally starts the task in UIHub, then schedules `_run_plan_runner` with `asyncio.create_task` | `UIHub.start_plan_task_if_possible` atomically checks mode, plan identity/revision and running-task conflict; `workflow_runtime.py::_run_plan_runner` executes MWV and writes plan/task outcome back to UIHub | The persisted task status and the runner's process-local asyncio task are different things. Source inspection has not established automatic restart of an interrupted runner. |
| Auto | `core/auto_runtime.py::AutoOrchestrator.run_v1` runs native tool loop, then `_run_verifier`; `core/agent.py` records progress; UI chat drains progress and persists latest `auto_state` | AutoOrchestrator owns in-process run state and `_paused_runs`; UIHub stores latest normalized state and publishes progress events | `COMPLETED` follows the selected verifier profile, not a general user acceptance gate. Approval resume calls `run_v1` again with the same goal, rather than continuing a persisted tool-loop checkpoint. |
| Tool effects | `core/tool_loop.py` → `core/tool_gateway.py::ToolGateway.call` → `tools/tool_registry.py` | ToolGateway applies approval/policy before registry dispatch; registry/tool owns actual effect; MWV/Auto consume result | `ToolResult.ok` proves reported dispatch success, not necessarily a verified external postcondition. |
| Memory | `server/http/app.py::create_app` passes `principal_storage_paths` to each scoped Agent; `core/agent.py` constructs `MemoryManager`, categorized/inbox and companion stores; `core/agent_memory.py` contains claim capture, preview and confirmed save paths | Memory subsystems own their own persisted data; agent owns selection/use during a response | Separate stores and selected retrieval do not yet prove a unified provenance, correction or accepted-knowledge contract. |

## State, evidence and communication

| Concern | Observed owner and evidence | What the evidence does **not** establish |
| --- | --- | --- |
| Identity and isolation | `server/agent_provider.py::AgentScope` is `(principal_id, session_id)` and scopes agent/lock; UI session resolution checks principal ownership in the HTTP handlers. Auth lanes are implemented in `server/http/common/auth.py`. | A single universal principal model across all tools, background tasks and external integrations. Trace the context through each execution lane. |
| UI session persistence | `server/ui_hub.py::_persist_session_locked` saves messages, artifacts, decision, mode, active plan/task and auto state through `UISessionStorage`; `_restore_sessions` reloads them. | Durable execution or atomic coupling of task transitions, effects and communication. |
| Live events | `UIHub.publish` appends to a bounded in-memory event buffer and subscriber queues; replay can demand resync. UI chat publishes `auto.progress` and chat stream events. | A canonical durable event log or recoverable semantic communication artifact. |
| Verification | `AutoOrchestrator` maps verifier pass to `COMPLETED`; `_run_verifier` passes `response_only` for nonempty text without tool calls and `tool_outcomes` for successful calls outside a repository with canonical verifier. `workflow_runtime._run_plan_runner` marks Plan/Act completed only with work success plus `VerificationStatus.PASSED`. | Universal verification of every user goal, acceptance of partial results, or consistent evidence strength between verifier profiles. |
| User-facing result | `ui_chat.py` appends an assistant message and may create output artifacts from response text/tool calls; `handlers/chat.py` returns proxy response and report metadata. | A single final publication gate tied to an authoritative logical task revision, acceptance decision and durable delivery identity. |
| Approval | `ToolGateway` may raise `ApprovalRequired`; UI handlers persist decision packets and Plan/Act resume data in UIHub. | That a decision packet alone can safely replay arbitrary effects after restart; needs dedicated audit. |
| UI artifacts | `server/http/common/ui_artifacts.py::_build_output_artifacts` derives file/text records from the assistant response; `UIHub.append_session_artifact` stores them in the session, and download handlers check session ownership before reading them. | Identity or retention independent of the UI session, linkage to a verified task result, or that named output equals a real workspace file. |

## Additional Gate 0 boundary probes

- **Voice and attachments:** `server/http/routes.py` exposes separate
  `/ui/api/stt/transcribe` and `/ui/api/tts/speak` routes. The STT handler
  posts audio to the configured OpenAI transcription endpoint and returns
  text; the TTS handler directly invokes `TtsTool.handle`, checks the returned
  file is under the sandbox audio root, and serves bytes. UI chat attachments
  are validated and serialized into a JSON text block by
  `server/http/common/chat_payload.py::_ui_messages_to_llm`. These are current
  transport/conversion paths. They do not establish a unified multimodal
  semantic input, result, provenance or task lifecycle contract. The direct
  TTS handler is also a distinct invocation path from agent ToolGateway calls;
  its policy and post-condition requirements need a separate Tool Architecture
  audit, with user-initiated speech distinguished from autonomous tool action.
- **Skills:** `core/skills/manifest.py` loads a versioned manifest;
  `core/skills/index.py` matches requests and resolves dependencies. Its
  `SkillResolution.system_instruction` explicitly says skill text grants no
  tools, permissions or approval. `core/agent_routing.py::_resolve_skill_run`
  applies the selection to Auto. This establishes a bounded instruction
  selection mechanism, not an accepted extension trust, installation,
  revocation or capability-delegation contract.
- **Local models:** `llm/local_http_brain.py::LocalHttpBrain` sends compatible
  chat/tool requests to a configured HTTP endpoint (default localhost).
  `config/model_store.py` persists selected model configuration and
  `server/http/common/runtime_model_state.py` holds active global/session
  selection. Separately, `/ui/api/local/ollama/start` in
  `server/http/handlers/sessions.py` probes the local provider and, if it is
  unavailable, starts `OLLAMA_BIN serve` as a detached host process, polls model
  discovery for up to eight seconds and reports availability. This is a narrow
  UI-invoked launcher/readiness path, not model selection, weight provisioning,
  resource reservation, process supervision or recovery. It crosses a host
  process-execution boundary; authorization/ownership of that action needs a
  dedicated audit before a Target manager is specified.

## Restart and approval-resume trace

- `server/http/app.py::create_app` builds `UIHub` with SQLite session storage.
  `UIHub.__init__` restores persisted plan/task/auto state, including a
  previously `running` task. The inspected app construction has no startup
  scheduling of `_run_plan_runner`; the found scheduling sites are Plan
  execute and decision-resume handlers. A persisted `running` value therefore
  does not establish that work is currently executing after a process restart.
- Plan/Act approval pause stores completed step snapshots, changes and counts
  in `task_packet.context.plan_runner_resume`
  (`workflow_runtime._task_packet_with_resume_state`). The resumed worker in
  `core/agent_mwv.py` skips step IDs recorded as `done`. This is a bounded
  step-level resume mechanism; it does not prove exactly-once external effects
  for a step interrupted between an effect and its recorded completion.
- Auto approval pause stores `_PausedRun` only in `AutoOrchestrator._paused_runs`.
  `resume(run_id)` removes that object and calls `run_v1` with the original
  goal/run ID. The decision handler calls `Agent.resume_auto_run`; after agent
  recreation the saved UI `auto_state` cannot reconstruct `_PausedRun`. In the
  same process, prior successful effects may be requested again when the goal
  is rerun. This is a risk inference from control flow, not an observed
  duplicate-effect incident.
- `handlers/decision.py::handle_ui_decision_respond` checks session ownership,
  decision identity, pending status and expiry before moving a decision to
  `executing` and dispatching resume. Some validation failures restore
  `pending`; success resolves the decision. This protects decision selection
  but does not by itself couple decision transition, background execution
  scheduling and external effects atomically.

## Memory and artifact scope trace

- `server/principal_storage.py::principal_storage_paths` keeps the owner under
  the legacy `memory/` root and gives other principals hashed subdirectories.
  `create_app` passes those database paths to the agent factory. The agent
  retrieves user preferences and a canonical memory capsule for context in
  `core/agent_memory.py`; explicit remember requests create a preview/decision,
  and `handlers/decision.py::_resolve_memory_save_decision` calls the agent's
  confirmed-save ToolGateway path. This demonstrates a scoped path and a
  confirmation route, not comprehensive proof that every retrieval/write lane
  enforces the same principal and provenance rules.
- `ui_artifacts._build_output_artifacts` creates fresh UUIDs from named file
  blocks in response text, or a canvas text result. Its `files_from_tools`
  argument is discarded. UIHub appends the whole record to the session, and
  `SQLiteUISessionStorage` serializes artifacts as JSON. Download routes
  verify session ownership and serve the stored text content. These UI records
  are distinct from workspace files and from evidence-backed task results.
- `UIHub._prune_sessions_locked` removes sessions after the configured TTL or
  max-session limit and `_drop_sessions_locked` deletes their persisted session
  records. The default TTL is seven days and max count is 200. The inspected
  artifact records have no separate retention owner; this does not imply that
  workspace files referenced elsewhere are deleted.

## Implications for Capability Discovery

1. **Separate owners:** UI session state, Plan/Act runner, Auto run, tool effects,
   verifier and user-visible communication are distinct current mechanisms. The
   Target needs explicit authority and handoff contracts between them.
2. **Recovery is a foundation question:** persisted UI task/auto snapshots and
   process-local execution coexist. Plan/Act has a step-level approval resume;
   Auto has a process-local restart-from-goal resume. The Target must specify
   interruption reconciliation, idempotency and effect evidence before
   promising durable task/run continuation.
3. **Evidence strength varies:** verification/acceptance must name the profile,
   evidence and authority; the same `completed` label cannot imply identical
   proof on all lanes.
4. **Multi-entry consistency is open:** browser chat, `/v1`, Plan/Act and Auto
   have different execution and result paths. A system map must distinguish
   adapters/projections from canonical task, run and communication ownership.
5. **Memory and artifact ownership differ:** principal-scoped memory databases
   and session-scoped UI artifacts have different lifetimes. The Target must
   name provenance, acceptance and retention owners for persistent knowledge
   and delivered results separately.
6. **Adapters do not imply full capabilities:** voice endpoints, attachment
   serialization, skill instruction matching, a local model HTTP client and
   a narrow Ollama launcher are useful Current State pieces, but cannot close
   the corresponding Target domains without accepted semantic, trust and
   lifecycle contracts.

**Next step:** use these Current State findings in section-specific audits.
The coarse system map was accepted at Gate 0; section 4 must reconcile the
Lifecycle snapshot before its implementation specification. This source audit
does not certify every runtime path or restart behavior.
