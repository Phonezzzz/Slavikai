# LIFECYCLE - SlavikAI request flow (MWV)

This describes the current runtime flow with explicit markers for LLM calls, tool invocations, and memory/index usage.

## Diagram (current)
```text
User input
  -> Agent.respond(...)
     -> Routing (classify_request)
        -> Chat lane
           -> Context build [Memory + VectorIndex + Workspace + Feedback]
           -> RuleEngine.apply
           -> Brain.generate [LLM]
           -> response -> logs
        -> MWV lane
           -> ManagerRuntime (build TaskPacket)
           -> WorkerRuntime (plan + execute via tools)
              -> Planner.build_plan [LLM or heuristic]
              -> Executor.run -> ToolGateway -> ToolRegistry.call [Tool]
           -> VerifierRuntime.run [scripts/check.sh]
           -> response -> logs
```

## Step-by-step (current behavior)
1) Input enters via `core/agent.py` as `Agent.respond([...])`.
2) Agent logs `user_input` to `logs/trace.log` via `core/tracer.py`.
3) Routing inside `core/agent.py`:
   - `core/mwv/routing.py` returns `chat` or `mwv` based on deterministic rules.
4) Chat lane:
   - Agent builds context from:
     - Memory notes/prefs (`memory/memory_manager.py`) [Memory]
     - Feedback hints (`memory/memory_companion_store.py`) [Memory]
     - Vector search (`memory/vector_index.py`) [Vector]
     - Workspace context (last file/selection) [Workspace]
   - RuleEngine adds approved policy instructions.
   - Primary brain is called via `Brain.generate(...)` [LLM].
5) MWV lane:
   - Manager builds a `TaskPacket` (goal + constraints + context).
   - Worker executes plan via `core/planner.py` and `core/executor.py`.
   - Tool steps go through `core/tool_gateway.py` -> `tools/tool_registry.py` [Tool].
   - Verifier runs `scripts/check.sh` and returns a structured `VerificationResult`.
   - Only a green verifier marks success.
6) Tool invocation path:
   - Tools are registered in `core/agent.py` and executed via `tools/tool_registry.py`.
   - Safe-mode blocks tools in the configured denylist and returns `ToolResult.failure`.
   - Tool results are logged to `logs/tool_calls.log` and to `memory/memory_companion.db` as tool interactions.
7) Memory and feedback:
   - Interaction logs (chat/tool) and feedback events are stored in `memory/memory_companion.db`.
   - Dialogue auto-save to `memory/memory.db` happens only if `config/memory.json` enables it.

## Notes
- Workspace tool calls use `Agent.call_tool(...)` directly (not `Agent.respond`), but still log tool interactions.
- Approval-required actions stop execution and return a structured approval prompt.
