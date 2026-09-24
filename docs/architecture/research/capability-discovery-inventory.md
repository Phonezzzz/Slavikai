# Capability Discovery — initial system inventory

**Status:** preserved discovery research, 2026-09-24. Candidate inventory only;
the accepted coarse map and owner boundaries are in `system/` and the
[Gate 0 closure record](gate-0-closure-matrix.md).
Current-state spot checks use production checkout `920d5f3`; target-document
checks use `architecture/target` at `e7cc2b0` plus explicitly named
unintegrated commits. No code or runtime was changed for this inventory.

## Why Gate 0 required more than this inventory

At the start of this inventory `system/capability-map.md`, `boundaries.md`,
`data-flows.md` and `dependencies.md` were templates; they now contain a
coarse accepted map with explicitly open domain details. The 20-item desktop plan is a useful
work list, but mixes capabilities, cross-cutting contracts, audits and
implementation concerns. It cannot by itself establish a complete system map.
The unintegrated lifecycle snapshot and existing communication/verification
contracts also show that cross-domain semantics must be reconciled before
domain-by-domain implementation.

## Candidate capability space

| Candidate area | Evidence of current or target relevance | Discovery question |
| --- | --- | --- |
| User interaction and modalities | Current UI/chat plus `tools/stt_tool.py`, `tts_tool.py`, `image_analyze_tool.py`; target communication contract | Is voice/vision a separate perception/output domain or an interface over shared task/communication state? |
| Goal understanding, planning and reasoning | Current Ask/Plan/Act/Auto, `core/plan_compiler.py`, `core/auto_runtime.py` | Where do intent, strategy/plan and adaptive replan live, and who accepts a changed goal? |
| Task/run control and background autonomy | Current Auto/MWV, unintegrated lifecycle contract `6040640` | What is authoritative task/run state across sessions, restart and detached work? |
| Work initiation and triggers | ADR-0003 includes preapproved schedule/event-initiated work in Target; current UI/API request entry points and `core/rule_engine.py` only evidence message-driven paths | Who owns trigger registration, consent, deduplication, wakeup and task-creation acceptance? |
| Multi-agent coordination | Target research/contract at `4ee7516`; current `core/mwv/*` | Where is coordination distinct from lifecycle ownership and private agent context? |
| Tool and environment interaction | `ToolGateway`, `ToolRegistry`, desktop/browser/GUI tools | Which semantic action contract spans API, filesystem, CLI, browser and GUI? |
| Verification and acceptance | Target verification contract `c92aef6`; current `VerifierRuntime` | Which evidence proves an outcome, who accepts it and how is uncertainty represented? |
| Policy, approval, identity and trust | `approval_policy.py`, `desktop_policy.py`, principal storage, deployment auth lanes; [selected authority audit](principal-policy-credential-acceptance-boundaries.md) | Which decisions are security authority, and how do they survive delegation/resume? |
| Credentials and human oversight | Owner-gated provider-key settings and application-level key store; UI decision packets and target acceptance contract | ADR-0007 fixes owner-only credential use by default, explicit delegation otherwise; exact runtime checks and distinct action/plan/trigger/result decision handoffs remain open. |
| Context, Memory and knowledge | Target Context `4b9a453`, Memory `f883893`; current `memory/*` | Separate short-lived task context, accepted long-term knowledge, retrieval and provenance. |
| Model access and routing | Current `llm/*`, model access research, ADR-0001 | Separate model family, access mode, protocol, auth, capability evidence and egress. A UI route can launch local Ollama; its host-process boundary needs audit. |
| Skills and reusable procedures | Current `core/skills/*` and `skills/*` | Are skills a governed capability with versioning and selection, or only context input? |
| Artifacts and external results | Current workspace/files/download tools and task packets | Who owns artifact identity, integrity, versions and retention independently of chat? |
| Persistence, events and recovery | Current UIHub/session history, databases, run state; lifecycle research | Which state is canonical, derived, replayable or ephemeral? |
| Resource governance | Current run/tool budgets and provider costs | How are time, tokens, external cost and child allocations enforced? |
| Observability, audit and evaluation | Current traces/logs/verifier output; target evidence and communication contracts | Keep operational telemetry, security audit, user progress and system quality eval distinct. |
| Deployment and extensibility | Current `deploy/*`, configuration, server and external integrations | Which lifecycle and trust concerns belong to deployment, extension/plugin or external-service boundaries? |

This table is deliberately broader than the existing placeholder directories.
Rows are not yet mutually exclusive domains. Security, identity, persistence,
provenance and evaluation may be cross-cutting concerns rather than standalone
owners; discovery must determine the stable ownership of each fact and action.

## External primary-source signals

- [Anthropic's agent architecture guide](https://www.anthropic.com/engineering/building-effective-agents)
  distinguishes fixed workflows from model-directed agents, and describes
  routing, parallelism, orchestrator-workers, evaluator loops, environmental
  feedback and stopping conditions. This supports checking both orchestration
  and execution/feedback, without prescribing a framework.
- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
  exposes tools, handoffs, guardrails, sessions, HITL and tracing as separable
  building blocks. Its SDK shape is an example, not a SlavikAI requirement.
- [LangGraph persistence docs](https://docs.langchain.com/oss/python/langgraph/persistence)
  and [Temporal workflow execution docs](https://docs.temporal.io/workflow-execution)
  show why durable execution/checkpointing deserves an explicit systems pass;
  neither source establishes that SlavikAI should adopt those runtimes.
- [LangGraph's agent design example](https://docs.langchain.com/oss/javascript/langgraph/thinking-in-langgraph)
  explains that recovery repeats work from the beginning of an interrupted
  node, so checkpoint granularity and application-level caching matter.
  This supports separating a durable checkpoint from exactly-once effects.
- [Anthropic's agent eval guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
  motivates an evaluation/quality feedback concern separate from per-task
  verification and acceptance.
- [OWASP's Excessive Agency analysis](https://genai.owasp.org/llmrisk/llm062025-excessive-agency/)
  identifies excessive functionality, permissions and autonomy as distinct
  causes of harmful agent actions. This reinforces explicit tool capability,
  principal and approval boundaries; it is threat guidance, not evidence that
  SlavikAI currently has a particular vulnerability.
- [NIST AI RMF Core](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
  treats governance as cross-cutting and distinguishes mapping, measurement
  and management. This supports keeping operational evaluation and risk
  governance visible in the map, without adopting NIST's categories as the
  product's domain decomposition.

A [separate boundary reference check](mature-system-boundary-check.md)
compares primary Temporal, MCP and LiveKit documents for work initiation,
external-tool trust and continuous media interruption. These examples support
asking the Gate 0 questions; they do not settle SlavikAI product scope.

## Questions tracked during promotion to the system map

1. Reconcile the [current-state flow/ownership audit](current-state-flow-ownership-matrix.md)
   with target contracts. It now traces entry points, restart/approval resume,
   memory paths and UI artifact retention; runtime recovery remains unproven.
2. Audit existing target contracts and unintegrated commits for owner,
   overlap and contradiction. ADR-0002 resolves terminal-cause/result
   disposition mapping; lifecycle snapshot remains unintegrated.
3. Study additional mature-agent references for long-running autonomy,
   security, memory, artifacts, multimodal interaction and evaluation, then
   distinguish observed design patterns from SlavikAI requirements.
4. Decide which rows are domains, which are shared contracts and which are
   projections. Identify missing areas, directed dependencies and
   change-amplifying foundations before defining implementation order.
   Work initiation is now an explicit candidate; it must not be hidden inside
   background execution, which begins after a task/run already exists.

The first reviewable [boundary proposal](capability-boundary-proposal.md)
compares candidate owners, cross-cutting contracts and directed dependencies.
It remains research rather than the normative map. Gate 0 criteria were checked
in the [closure matrix](gate-0-closure-matrix.md). Its proposed owner rows cover all 20 numbered plan sections;
media and extension Target scope is decided in ADR-0008/0009; detailed
contracts remain for their dependent sections.
**Next step:** reconcile the unintegrated Lifecycle snapshot under ADR-0002/0010
before section 4 contract integration. Detailed open questions stay with their
dependent sections.
