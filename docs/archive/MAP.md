# MAP - SlavikAI codebase (Phase 1)

This map reflects the current repo structure and call graph for Slavik.

## Entry Points
- `server/__main__.py` - HTTP gateway entry point. Calls: `server/http_api.py`.

## Core Pipeline
- `core/agent.py` - Main router: command lane, planning/execution, tool registry/safe-mode, context assembly, memory/feedback/policy handling, tracing. Called by: `server/http_api.py`. Calls: `core/planner.py`, `core/executor.py`, `core/tool_gateway.py`, `core/tracer.py`, `core/auto_agent.py`, `core/rule_engine.py`, `core/batch_review.py`, `llm/brain_factory.py`, `llm/brain_manager.py`, `tools/tool_registry.py`, `memory/memory_manager.py`, `memory/vector_index.py`, `memory/memory_companion_store.py`.
- `core/planner.py` - Task complexity classification and plan generation (LLM when provided, heuristic fallback). Called by: `core/agent.py`. Calls: `llm/brain_base.py`, `core/tracer.py`, `shared/plan_models.py`.
- `core/executor.py` - Sequential plan execution with tool gateway or agent callback. Called by: `core/agent.py`. Calls: `core/tool_gateway.py`, `core/tracer.py`, `shared/models.py`.
- `core/tool_gateway.py` - Thin wrapper around ToolRegistry with pre/post hooks. Called by: `core/agent.py`, `core/executor.py`. Calls: `tools/tool_registry.py`.
- `core/tracer.py` - JSONL trace logger with rotation and sanitization. Called by: `core/agent.py`, `core/planner.py`, `core/executor.py`, `core/auto_agent.py`, `core/batch_review.py`.
- `core/auto_agent.py` - Parallel sub-agent runs (LLM-only). Called by: `core/agent.py`. Calls: `llm/brain_base.py`, `core/tracer.py`.
- `core/rule_engine.py` - Applies approved policy rules to user input. Called by: `core/agent.py`. Calls: `shared/policy_models.py`.
- `core/intent_paradox.py` - Deterministic intent/paradox analysis for BatchReview. Called by: `core/batch_review.py`. Calls: `shared/batch_review_models.py`.
- `core/batch_review.py` - Batch review + policy candidate generation from feedback. Called by: `core/agent.py`. Calls: `memory/memory_companion_store.py`, `core/intent_paradox.py`, `shared/batch_review_models.py`.

## MWV (Manager → Worker → Verifier)
- `core/mwv/models.py` - TaskPacket/WorkResult/VerificationResult/RunContext dataclasses.
- `core/mwv/verifier.py` - Verifier runner wrapper for `scripts/check.sh`.
- `core/mwv/routing.py` - Deterministic routing rules (`chat` vs `mwv`).
- `core/mwv/manager.py` - Manager contract (scaffold).
- `core/mwv/worker.py` - Worker contract (scaffold).
- `core/mwv/verifier_runtime.py` - Runtime verifier contract (scaffold).

## LLM Layer
- `llm/brain_base.py` - Brain interface. Called by: `core/agent.py`, `core/planner.py`, `core/auto_agent.py`.
- `llm/brain_factory.py` - Builds brain from ModelConfig. Called by: `core/agent.py`, `llm/brain_manager.py`.
- `llm/brain_manager.py` - Optional brain builder (main brain). Called by: `core/agent.py`.
- `llm/openrouter_brain.py` - OpenRouter client. Called by: `llm/brain_factory.py`.
- `llm/local_http_brain.py` - Local OpenAI-compatible client. Called by: `llm/brain_factory.py`.
- `llm/types.py` - ModelConfig/LLMResult types used across LLM and Agent.

## Tools Layer
- `tools/tool_registry.py` - Tool registration, enable/disable, safe-mode blocking, tool call logging. Called by: `core/agent.py`, `core/tool_gateway.py`. Calls: `tools/tool_logger.py`.
- `tools/tool_logger.py` - JSONL tool-call log writer with rotation/sanitization. Called by: `tools/tool_registry.py`.
- `tools/protocols.py` - Tool protocol definition. Called by: tool implementations and registry.
- `tools/filesystem_tool.py` - Sandbox file list/read/write under `sandbox/`. Called by: `tools/tool_registry.py`.
- `tools/workspace_tools.py` - Workspace list/read/write/patch/run under `sandbox/project/`. Called by: `tools/tool_registry.py`.
- `tools/shell_tool.py` - Shell with allowlist + sandbox root enforcement. Called by: `tools/tool_registry.py`.
- `tools/web_search_tool.py` - Serper search + HTTP fetch. Called by: `tools/tool_registry.py`.
- `tools/project_tool.py` - Index/search for `memory/vectors.db`. Called by: `tools/tool_registry.py`.
- `tools/image_analyze_tool.py` - Local image analysis (base64 or sandbox file). Called by: `tools/tool_registry.py`.
- `tools/image_generate_tool.py` - Local image generator (solid color PNG). Called by: `tools/tool_registry.py`.
- `tools/tts_tool.py` - TTS via HTTP; writes audio to `sandbox/audio/`. Called by: `tools/tool_registry.py`.
- `tools/stt_tool.py` - STT via HTTP; reads audio from `sandbox/audio/`. Called by: `tools/tool_registry.py`.
- `tools/http_client.py` - HTTP helper with size/time limits. Called by: `tools/web_search_tool.py`, `tools/tts_tool.py`, `tools/stt_tool.py`.

## Memory / Feedback / Index
- `memory/memory_manager.py` - SQLite memory store (notes/prefs/facts). Called by: `core/agent.py`.
- `memory/vector_index.py` - SQLite + embeddings index for code/docs. Called by: `core/agent.py`, `tools/project_tool.py`.
- `memory/memory_companion_store.py` - Interaction log, feedback, policy rules, batch review storage. Called by: `core/agent.py`, `core/batch_review.py`.

## Shared Models / Utilities
- `shared/models.py` - Core dataclasses (ToolRequest/Result, LLMMessage, TaskPlan, etc). Used across core/tools.
- `shared/plan_models.py` - Plan validation schema and allowed operations. Called by: `core/planner.py`.
- `shared/memory_companion_models.py` - Interaction and feedback models. Used by: `core/agent.py`, `memory/memory_companion_store.py`.
- `shared/batch_review_models.py` - Batch review and candidate models. Used by: `core/batch_review.py`, `core/intent_paradox.py`.
- `shared/policy_models.py` - Typed policy triggers/actions and JSON helpers. Used by: `core/rule_engine.py`, `memory/memory_companion_store.py`.
- `shared/sandbox.py` - Sandbox path normalization helpers. Used by: `tools/filesystem_tool.py`, `tools/workspace_tools.py`, `tools/project_tool.py`, `tools/shell_tool.py`.
- `shared/sanitize.py` - Log sanitization and safe JSON loads. Used by: `core/tracer.py`, `tools/tool_logger.py`.

## Config
- `config/model_store.py` - Load/save model configs and presets (OpenRouter fetch). Called by: `core/agent.py`.
- `config/tools_config.py` - Load/save tool enable flags + safe-mode. Called by: `core/agent.py`.
- `config/shell_config.py` - Shell allowlist and dev-mode handling. Called by: `tools/shell_tool.py`.
- `config/system_prompts.py` - System prompts for planner/thinking. Called by: `core/planner.py`, `core/agent.py`, `llm/*`.
- `config/memory_config.py` - Memory auto-save toggle. Called by: `core/agent.py`.
- `config/web_search_config.py` - Web search config. Called by: `tools/web_search_tool.py`.
- `config/tts_config.py` - TTS config. Called by: `tools/tts_tool.py`.
- `config/stt_config.py` - STT config. Called by: `tools/stt_tool.py`.
- `config/custom_tools_config.py` - Custom tool specs (JSON). Called by: `tools/tool_registry.py`.
- `config/settings.py` - AppSettings dataclass (not referenced at runtime).
