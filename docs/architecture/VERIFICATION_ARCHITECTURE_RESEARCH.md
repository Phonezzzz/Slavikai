# Verification Architecture Research

**Status:** audit/research and rationale for the normative target contract. Runtime,
UI, API and storage are unchanged.

Normative result: [`VERIFICATION_ARCHITECTURE_CONTRACT.md`](VERIFICATION_ARCHITECTURE_CONTRACT.md).

## 1. Current verification map

```text
worker/tool/model result
        ↓
local result checks / verifier / post-action observation
        ↓
current status or report
        ↓
workflow/Auto/MWV acceptance path
        ↓
assistant/UI response and lifecycle projection
```

The current flow is distributed rather than a single authoritative gate. `TaskPacket`
contains `acceptance_checks` and a `verifier` field. MWV has `WorkStatus.SUCCESS` and a
`VerifierRunner` that executes `scripts/check.sh`/fallback commands, returning
`passed`, `failed` or `error`. `VerifierRuntime` has deterministic desktop observation
rules for files, processes, browser and GUI actions. `Auto` stores verifier state and
terminal statuses. `ToolResult.ok` records tool execution outcome.

These are useful evidence/checking mechanisms, but the current runtime does not provide
one cross-mode contract for result revision, evidence identity, stale verification,
`inconclusive`, `blocked`, `not_applicable`, contradictory outcomes or authoritative
acceptance. A worker/model `done`, successful tool return, artifact existence or one
passing check is therefore not universally equivalent to completed lifecycle state.

## 2. Current-runtime verdicts

| Abstraction | Current evidence | Target mismatch | Verdict |
|---|---|---|---|
| MWV `WorkResult` | Worker reports success/failure, changes and diagnostics | Producer result is not independent acceptance; no revisioned evidence package | reuse with modification |
| `VerifierRunner` / `VerifierRuntime` | Deterministic command checks and desktop observations | Profile/outcome taxonomy and exact revision binding missing | reuse with modification |
| `TaskPacket.acceptance_checks` | Step-level string criteria, packet hash/revision | Criteria versioning/ambiguity handling and authoritative gate absent | reuse with modification |
| `RunContext` | Session/trace, retries, safe mode and attempt | Verification budget/evidence provenance not represented | reuse with modification |
| `ToolGateway` / `ToolResult` | Policy-gated dispatch and `ok/error/meta` | Execution success does not prove post-condition; unknown side effects weakly modeled | reuse with modification |
| `AgentToolLoop` final gate | Optional `final_gate` over executed calls | Not a general lifecycle acceptance decision | reuse with modification |
| Workspace checks | `make check`, scripts, tests and diff checks | Selection/provenance/pass-vs-unavailable not unified | reuse with modification |
| Browser/GUI verification | DOM/read/snapshot/wait and observation paths | Observation can be stale; screenshot is not automatically truth | reuse with modification |
| Approval flow | Policy/approval request and decision state | Approval answers permission, not correctness | reuse with modification |
| Auto verifier | Verifier/worker/internal terminal statuses | Runtime-specific; no shared acceptance contract | reuse with modification |
| UIHub/session history | Reports, activity, chat and bounded event replay | Presentation/history is not verification truth | reuse with modification |
| Background tasks | Auto/session state can outlive response | No general durable verification recovery gate | reuse with modification |

## 3. Verification taxonomy

- **Execution verification:** did the operation execute or return an attempt result? It
  may still have unknown external side effects.
- **Result verification:** does the produced result satisfy the intended semantic outcome?
- **Artifact verification:** does an artifact exist, remain intact, parse, match scope and
  have provenance?
- **State verification:** is the external system in the required post-condition state?
- **Policy verification:** was the operation permitted and within policy/scope? Approval
  is not verification.
- **Acceptance verification:** are current task/subtask criteria met? This gates lifecycle
  acceptance.

Epistemic verification qualifies evidence as verified, accepted, inferred, unresolved,
disputed or partial. It is cross-cutting and is not identical to workflow completion.

## 4. External research findings

1. OpenAI's official [Graders API](https://platform.openai.com/docs/api-reference/graders)
   distinguishes deterministic string/similarity graders from model-based score graders,
   supporting a layered deterministic-plus-semantic approach.
2. Anthropic's [agent-evaluation guidance](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
   describes combining grader types and notes that multi-turn tool use makes evaluation
   harder.
3. Anthropic's [trustworthy-agent research](https://www.anthropic.com/research/trustworthy-agents)
   emphasizes plan/act/observe/adjust, human control, transparency and privacy; this
   supports separating verification/acceptance from model prose.
4. LangGraph's official [re-execution guidance](https://langchain-ai.github.io/langgraph/how-tos/state-reducers/)
   states that resumed nodes may rerun and side effects need idempotency/read-before-write.
   Its [task guidance](https://langchain-ai.github.io/langgraph/how-tos/review-tool-calls-functional/)
   recommends idempotency keys or existing-state verification for side effects.
5. LangGraph's [interrupt guidance](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
   persists waiting state for resume, supporting durable approval/waiting semantics.

These sources inform principles, not copied framework APIs; repository contracts remain
authoritative.

## 5. Alternatives

| Alternative | Correctness | Cost/latency | Correlated failure | Auditability | Verdict |
|---|---|---|---|---|---|
| Worker self-declares completion | Weak | Lowest | Highest | Weak | reject |
| Always self-check | Better, still correlated | Low/medium | High | Medium | insufficient baseline |
| Always independent verifier | Stronger but wasteful | Highest | Lower, not zero | Strong | reject as universal rule |
| Deterministic checks + selective semantic verifier | Strong where checkable | Profile dependent | Controlled | Strong | select |
| Policy-driven verification profiles | Risk-adaptive | Bounded by policy | Explicit | Strong | selected control model |

## 6. Selected target architecture

```text
result / artifact / action outcome
        ↓
verification requirement + profile
        ↓
bounded versioned evidence package
        ↓
deterministic and/or semantic verification attempts
        ↓
verification outcome(s)
        ↓
authoritative acceptance decision
        ↓
lifecycle completion / rework / failure / escalation
```

The producer submits result/evidence; a verifier evaluates current criteria; the
coordinator/control plane resolves outcomes and alone accepts completion. A separate
verifier is required only when risk/profile/policy justifies it. Deterministic checks
are preferred where sufficient; semantic judgement covers meaning/quality. Exact models,
thresholds, runners and retry counts remain implementation-defined.

## 7. Gaps and open decisions

- No shared versioned evidence package or verification identity exists.
- Current `VerificationStatus` lacks `inconclusive`, `stale`, `blocked` and
  `not_applicable` semantics.
- No general result/artifact revision invalidation rule exists.
- Contradictory outcomes need attributed resolution, not last-write-wins.
- External side-effect timeout/exception needs unknown → reconcile/verify →
  accept/retry/compensate/STOP semantics.
- Human verification is valid only for subjective or unobservable criteria.
- Verification budgets and minimum required profile need policy design.
- Browser/UI screenshots are evidence, not truth by themselves.
