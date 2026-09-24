# Gate 0 — mature-system boundary reference check

**Status:** external research, 2026-09-24. These are primary-source examples
of concerns a mature agent may need to own. They neither define SlavikAI
product scope nor prove current runtime behavior. No framework adoption is
proposed by this document.

| Reference and observed contract | Architectural inference for SlavikAI | SlavikAI disposition |
| --- | --- | --- |
| [Temporal Schedules](https://github.com/temporalio/documentation/blob/main/docs/develop/go/workflows/schedules.mdx) separate a schedule's identity/spec/action from the workflow executions it starts and expose pause, update, trigger and overlap policy; the [Schedule policy API](https://typescript.temporal.io/api/interfaces/proto.temporal.api.schedule.v1.SchedulePolicies._Properties) also describes catch-up windows and pause-on-failure. | ADR-0003 includes preapproved recurring/event-initiated work, and ADR-0004 assigns rule/firing ownership to Work Initiation, separately from Lifecycle. Background continuation of an existing run does not cover this boundary. | OQ-CD-04: exact firing, overlap and catch-up contract remains open. This does not select Temporal or its policies. |
| The [MCP tools specification, 2026-07-28](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/server/tools.mdx) says clients must treat tool annotations as untrusted unless supplied by a trusted server; tool names are unique only within a server. The [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/seps/2663-tasks-extension) requires authentication/authorization on each task request and does not make a task a higher-trust channel. | An extension catalog may describe capabilities, but the effective principal, server identity, operation scope and policy remain trusted SlavikAI decisions. Remote task handles and tool descriptions must not be promoted into local lifecycle or approval authority. | ADR-0009 includes external providers with governance separate from skills; protocol choice and detailed call contract remain open. No decision to adopt MCP. |
| [LiveKit AgentSession](https://docs.livekit.io/agents/logic/sessions/) gives voice interaction its own listening/thinking/speaking state, media I/O and turn handling. Its [tool interruption documentation](https://docs.livekit.io/agents/logic/tools/definition/) says user speech can interrupt the agent while a tool continues until it returns unless application code cancels it. | A continuous media session would need turn/interruption and delivered-speech state distinct from the logical task/run and external tool effect. | ADR-0008 excludes continuous voice/visual sessions from the current Target; separate request-scoped media operations suffice. Current STT/TTS endpoints are only selected Current State paths. |

## What this changes in Gate 0

1. Keep **work initiation**, **task continuation** and **user-facing
   notification** as separate handoffs under ADR-0003. Trigger
   overlap/catch-up is about creating work, not completing a run.
2. Keep **skill instructions**, **external tool metadata**, **effective policy**
   and **task authorization** distinct even if one extension protocol transports
   several of them.
3. Keep request-scoped media effects and task/effect state under their
   respective owners; ADR-0008 does not require realtime turn state.

**Next step:** use these references to judge owner completeness in the
[boundary proposal](capability-boundary-proposal.md). ADR-0008/0009 settle
media and extension product scope; ADR-0003/0004 settle trigger inclusion and
ownership but not the detailed firing contract.
