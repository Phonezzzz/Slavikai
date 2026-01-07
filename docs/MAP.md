# MAP - SlavikAI codebase (Phase 1)

This map reflects the current repo structure and call graph for Slavik.

## Entry Points
- `main.py` - Qt app bootstrap (dotenv, dev mode logging, fault handler). Called by: user runtime. Calls: `ui/main_window.py`.
- `ui/main_window.py` - Builds Agent + UI, dispatches chat requests in worker threads, updates panels. Called by: `main.py`. Calls: `core/agent.py` and all UI panels.

## Core Pipeline
- `core/agent.py` - Main router: command lane, planning/execution, tool registry/safe-mode, context assembly, memory/feedback/policy handling, DualBrain orchestration, tracing. Called by: `ui/main_window.py`, `ui/workspace_panel.py`, `ui/docs_panel.py`, `ui/feedback_panel.py`. Calls: `core/planner.py`, `core/executor.py`, `core/tool_gateway.py`, `core/tracer.py`, `core/auto_agent.py`, `core/rule_engine.py`, `core/batch_review.py`, `llm/brain_factory.py`, `llm/dual_brain.py`, `tools/tool_registry.py`, `memory/memory_manager.py`, `memory/vector_index.py`, `memory/memory_companion_store.py`.
- `core/planner.py` - Task complexity classification and plan generation (LLM when provided, heuristic fallback). Called by: `core/agent.py`. Calls: `llm/brain_base.py`, `core/tracer.py`, `shared/plan_models.py`.
- `core/executor.py` - Sequential plan execution with optional critic callback and tool gateway. Called by: `core/agent.py`. Calls: `core/tool_gateway.py`, `core/tracer.py`, `shared/models.py`.
- `core/tool_gateway.py` - Thin wrapper around ToolRegistry with pre/post hooks. Called by: `core/agent.py`, `core/executor.py`. Calls: `tools/tool_registry.py`.
- `core/tracer.py` - JSONL trace logger with rotation and sanitization. Called by: `core/agent.py`, `core/planner.py`, `core/executor.py`, `core/auto_agent.py`, `core/batch_review.py`, `ui/trace_view.py`, `ui/reasoning_panel.py`.
- `core/auto_agent.py` - Parallel sub-agent runs (LLM-only). Called by: `core/agent.py`. Calls: `llm/brain_base.py`, `core/tracer.py`.
- `core/rule_engine.py` - Applies approved policy rules to user input. Called by: `core/agent.py`. Calls: `shared/policy_models.py`.
- `core/intent_paradox.py` - Deterministic intent/paradox analysis for BatchReview. Called by: `core/batch_review.py`. Calls: `shared/batch_review_models.py`.
- `core/batch_review.py` - Batch review + policy candidate generation from feedback. Called by: `core/agent.py`, `ui/feedback_panel.py`. Calls: `memory/memory_companion_store.py`, `core/intent_paradox.py`, `shared/batch_review_models.py`.

## LLM Layer
- `llm/brain_base.py` - Brain interface. Called by: `core/agent.py`, `core/planner.py`, `core/auto_agent.py`.
- `llm/dual_brain.py` - DualBrain container (main + critic; modes single/dual/critic-only). Called by: `core/agent.py`, `llm/brain_manager.py`.
- `llm/brain_factory.py` - Builds brain from ModelConfig. Called by: `core/agent.py`, `llm/brain_manager.py`.
- `llm/brain_manager.py` - Optional brain builder (main + critic). Called by: `core/agent.py`.
- `llm/openrouter_brain.py` - OpenRouter client. Called by: `llm/brain_factory.py`.
- `llm/local_http_brain.py` - Local OpenAI-compatible client. Called by: `llm/brain_factory.py`.
- `llm/types.py` - ModelConfig/LLMResult types used across LLM and Agent.

## Tools Layer
- `tools/tool_registry.py` - Tool registration, enable/disable, safe-mode blocking, tool call logging. Called by: `core/agent.py`, `core/tool_gateway.py`. Calls: `tools/tool_logger.py`.
- `tools/tool_logger.py` - JSONL tool-call log writer with rotation/sanitization. Called by: `tools/tool_registry.py`.
- `tools/protocols.py` - Tool protocol definition. Called by: tool implementations and registry.
- `tools/filesystem_tool.py` - Sandbox file list/read/write under `sandbox/`. Called by: `tools/tool_registry.py`.
- `tools/workspace_tools.py` - Workspace list/read/write/patch/run under `sandbox/project/`. Called by: `tools/tool_registry.py`, `ui/workspace_panel.py`.
- `tools/shell_tool.py` - Shell with allowlist + sandbox root enforcement. Called by: `tools/tool_registry.py`.
- `tools/web_search_tool.py` - Serper search + HTTP fetch. Called by: `tools/tool_registry.py`.
- `tools/project_tool.py` - Index/search for `memory/vectors.db`. Called by: `tools/tool_registry.py`, `ui/docs_panel.py`.
- `tools/image_analyze_tool.py` - Local image analysis (base64 or sandbox file). Called by: `tools/tool_registry.py`.
- `tools/image_generate_tool.py` - Local image generator (solid color PNG). Called by: `tools/tool_registry.py`.
- `tools/tts_tool.py` - TTS via HTTP; writes audio to `sandbox/audio/`. Called by: `tools/tool_registry.py`.
- `tools/stt_tool.py` - STT via HTTP; reads audio from `sandbox/audio/`. Called by: `tools/tool_registry.py`.
- `tools/http_client.py` - HTTP helper with size/time limits. Called by: `tools/web_search_tool.py`, `tools/tts_tool.py`, `tools/stt_tool.py`.

## Memory / Feedback / Index
- `memory/memory_manager.py` - SQLite memory store (notes/prefs/facts). Called by: `core/agent.py`, `ui/memory_view.py`, `ui/docs_panel.py`.
- `memory/vector_index.py` - SQLite + embeddings index for code/docs. Called by: `core/agent.py`, `tools/project_tool.py`, `ui/docs_panel.py`.
- `memory/memory_companion_store.py` - Interaction log, feedback, policy rules, batch review storage. Called by: `core/agent.py`, `core/batch_review.py`, `ui/feedback_panel.py`.

## Shared Models / Utilities
- `shared/models.py` - Core dataclasses (ToolRequest/Result, LLMMessage, TaskPlan, etc). Used across core/tools/ui.
- `shared/plan_models.py` - Plan validation schema and allowed operations. Called by: `core/planner.py`.
- `shared/memory_companion_models.py` - Interaction and feedback models. Used by: `core/agent.py`, `memory/memory_companion_store.py`, UI panels.
- `shared/batch_review_models.py` - Batch review and candidate models. Used by: `core/batch_review.py`, `core/intent_paradox.py`, `ui/feedback_panel.py`.
- `shared/policy_models.py` - Typed policy triggers/actions and JSON helpers. Used by: `core/rule_engine.py`, `memory/memory_companion_store.py`, `ui/policy_candidate_dialog.py`.
- `shared/sandbox.py` - Sandbox path normalization helpers. Used by: `tools/filesystem_tool.py`, `tools/workspace_tools.py`, `tools/project_tool.py`, `tools/shell_tool.py`.
- `shared/sanitize.py` - Log sanitization and safe JSON loads. Used by: `core/tracer.py`, `tools/tool_logger.py`.

## UI (PySide6)
- `ui/main_window.py` - Main window + layout selection; holds Agent and panels. Called by: `main.py`.
- `ui/dual_chat_view.py` - Two-column chat (main + critic). Called by: `ui/main_window.py`.
- `ui/chat_view.py` - Chat UI with attachments, TTS/STT, plan/reasoning widgets, feedback buttons. Called by: `ui/dual_chat_view.py`, `ui/chat_page.py`.
- `ui/chat_message_widget.py` - Render a single chat message + action buttons. Called by: `ui/chat_view.py`.
- `ui/plan_widget.py` - Compact plan summary widget. Called by: `ui/chat_view.py`.
- `ui/reasoning_widget.py` - Collapsible reasoning widget. Called by: `ui/chat_view.py`.
- `ui/diff_summary_widget.py` - Collapsible diff summary for workspace changes. Called by: `ui/chat_view.py`.
- `ui/feedback_dialog.py` - Feedback modal (rating + labels + comment). Called by: `ui/chat_view.py`.
- `ui/policy_candidate_dialog.py` - Edit/approve policy candidates. Called by: `ui/feedback_panel.py`.
- `ui/tools_panel.py` - Tool toggle panel + safe-mode toggle. Called by: `ui/main_window.py`.
- `ui/mode_panel.py` - DualBrain mode selector. Called by: `ui/main_window.py`.
- `ui/model_selector.py` - Model preset selector + thinking toggle. Called by: `ui/main_window.py`.
- `ui/workspace_panel.py` - Workspace tree/editor/patch/run with tool calls. Called by: `ui/main_window.py`, `ui/ide_panel.py`.
- `ui/docs_panel.py` - Vector index view + reindex/search controls. Called by: `ui/main_window.py`, `ui/ide_panel.py`.
- `ui/memory_view.py` - Memory table + filtering. Called by: `ui/main_window.py`.
- `ui/trace_view.py` - Trace viewer for `logs/trace.log`. Called by: `ui/main_window.py`.
- `ui/tool_logs_view.py` - Tool call viewer for `logs/tool_calls.log`. Called by: `ui/main_window.py`.
- `ui/reasoning_panel.py` - Plan/trace viewer using Agent state + tracer. Called by: `ui/main_window.py`.
- `ui/feedback_panel.py` - Feedback + batch review UI. Called by: `ui/main_window.py`.
- `ui/logs_view.py` - Simple text log panel. Called by: `ui/main_window.py`.
- `ui/audio_player.py` - Qt Multimedia wrappers for playback/recording. Called by: `ui/main_window.py`, `ui/chat_view.py`.
- `ui/panels_window.py` - Secondary window for panels in NORMAL layout. Called by: `ui/main_window.py`.
- `ui/chat_page.py` - Chat page with sidebar (not wired in main window). Called by: not referenced.
- `ui/ide_panel.py` - Workspace + Docs splitter (not wired in main window). Called by: not referenced.
- `ui/settings_dialog.py` - Model + shell settings dialog. Called by: `ui/main_window.py`.
- `ui/prompts_dialog.py` - Edit main system prompt. Called by: `ui/main_window.py`.
- `ui/settings_panel.py` - Alternative settings panel with memory config (not wired). Called by: not referenced.
- `ui/tools_catalog.py` - Tool catalog + custom tool editor (not wired). Called by: not referenced.

## Config
- `config/model_store.py` - Load/save model configs and presets (OpenRouter fetch). Called by: `core/agent.py`, `ui/model_selector.py`.
- `config/mode_config.py` - Load/save DualBrain mode. Called by: `core/agent.py`.
- `config/tools_config.py` - Load/save tool enable flags + safe-mode. Called by: `core/agent.py`, `ui/tools_panel.py`.
- `config/shell_config.py` - Shell allowlist and dev-mode handling. Called by: `tools/shell_tool.py`, `ui/settings_dialog.py`, `ui/settings_panel.py`, `main.py`.
- `config/system_prompts.py` - System prompts for planner/critic/thinking. Called by: `core/planner.py`, `core/agent.py`, `llm/*`.
- `config/memory_config.py` - Memory auto-save toggle. Called by: `core/agent.py`, `ui/settings_panel.py`.
- `config/web_search_config.py` - Web search config. Called by: `tools/web_search_tool.py`.
- `config/tts_config.py` - TTS config. Called by: `tools/tts_tool.py`.
- `config/stt_config.py` - STT config. Called by: `tools/stt_tool.py`.
- `config/custom_tools_config.py` - Custom tool specs (JSON). Called by: `ui/tools_catalog.py`.
- `config/settings.py` - AppSettings dataclass (not referenced at runtime).
