# LIFECYCLE - SlavikAI request flow (Phase 1)

This describes the current runtime flow in the desktop app, with explicit markers for LLM calls, tool invocations, and memory/index usage.

## Diagram (current)
```text
User input
  -> UI (PySide6)
     -> Agent.respond(...)
        -> Command lane (/fs, /web, /sh, /project, /plan, /auto, /img*)
           -> ToolRegistry.call [Tool]
           -> ToolResult -> logs -> UI
        -> Auto lane ("авто" or /auto)
           -> AutoAgent -> Brain.generate [LLM] -> summary -> logs -> UI
        -> Plan lane (complex or "plan")
           -> Planner.build_plan [LLM or heuristic]
           -> Critic plan rewrite [LLM, dual mode]
           -> Executor.run
              -> (optional) step critic [LLM, dual mode]
              -> ToolGateway -> ToolRegistry.call [Tool] -> ToolResult
           -> Plan review [LLM, dual mode]
           -> response -> logs -> UI
        -> Simple lane (non-plan)
           -> Context build [Memory + VectorIndex + Workspace + Feedback]
           -> RuleEngine.apply (policy instructions)
           -> Brain.generate [LLM]
           -> Critic reply [LLM, dual mode]
           -> response -> logs -> UI
```

## Step-by-step (current behavior)
1) UI input enters via `ui/main_window.py`, which calls `core/agent.py` with `Agent.respond([...])` on a worker thread.
2) Agent logs `user_input` to `logs/trace.log` via `core/tracer.py`.
3) Routing inside `core/agent.py`:
   - Command lane (`/fs`, `/web`, `/sh`, `/project`, `/img*`, `/trace`, `/plan`, `/auto`): direct tool or special handler. Tool calls go through `ToolRegistry.call` [Tool]. Results are formatted and returned. Tool calls are logged to `logs/tool_calls.log` and stored in `memory/memory_companion.db`.
   - Auto lane (`"авто"` or `/auto`): `core/auto_agent.py` spawns parallel subagents and calls `Brain.generate` [LLM] for each subtask. No tools used in this lane.
   - Plan lane (complex input or "plan"): `core/planner.py` builds a `TaskPlan` using LLM if a brain is provided, otherwise heuristic. In DualBrain mode, the plan can be rewritten by the critic [LLM]. `core/executor.py` runs steps; in dual mode it calls a critic per step [LLM]. Tool steps go through `core/tool_gateway.py` -> `tools/tool_registry.py` [Tool]. After execution, plan review may call the critic [LLM]. The final response is a formatted plan summary.
   - Simple lane: Agent builds LLM context from:
     - Memory notes/prefs (`memory/memory_manager.py`) [Memory]
     - Feedback hints from `memory/memory_companion_store.py` [Memory]
     - Vector search (`memory/vector_index.py`) [Vector]
     - Workspace context from UI (`ui/workspace_panel.py`) [Workspace]
     The RuleEngine adds approved policy instructions, then the primary brain is called [LLM]. In dual mode, the critic is called [LLM] with the main reply.
4) Tool invocation path:
   - Tools are registered in `core/agent.py` and executed via `tools/tool_registry.py`.
   - Safe-mode blocks tools in the configured denylist and returns `ToolResult.failure`.
   - Tool results are logged to `logs/tool_calls.log` and to `memory/memory_companion.db` as tool interactions.
5) Memory and feedback:
   - Interaction logs (chat/tool) and feedback events are stored in `memory/memory_companion.db`.
   - Dialogue auto-save to `memory/memory.db` happens only if `config/memory.json` enables it.
6) Trace and UI:
   - `logs/trace.log` is used by `ui/trace_view.py` and `ui/reasoning_panel.py`.
   - `logs/tool_calls.log` is used by `ui/tool_logs_view.py`.

## Notes
- Workspace panel tool calls use `Agent.call_tool(...)` directly (not `Agent.respond`), but still log tool interactions.
- Critic output is stored in `Agent.last_critic_response` and rendered in the dual chat UI when enabled.
