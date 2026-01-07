# PHASE 0 FINDINGS - SlavikAI (current repo)

## Sources read
- `Architecture.md`
- `PROJECT_OVERVIEW.md`
- `DevRules.md`
- `CONTRIBUTING.md`
- `COMMANDS.md`
- `MEMORY_COMPANION_MINIPROMPTS.md`
- `my sheet/updated_roadmap.md` (inventory only, per clarification)
- `my sheet/codex_workplan.md`
- Code modules in `core/`, `llm/`, `tools/`, `memory/`, `shared/`, `ui/`, `config/`

## Findings / questions (contradictions or unclear points)
- Auto-save to memory exists: `config/memory.json` + `core/agent.py` can write dialogue to `memory/memory.db` without explicit user approval. This conflicts with the "controlled learning / explicit memory" invariant unless kept disabled. Confirm whether `auto_save_dialogue` must remain off.
- Critic failure path in `core/agent.py` (`_critic_step`) returns approve on exception and only logs to trace. This may violate "no silent fallback" because the user does not see the critic failure.
- Required UX contract (plan summary, execution summary, uncertainty notes) is not enforced in `core/agent.py` responses; outputs are raw LLM text or formatted plan only. Traces exist in logs, but no structured response payload is emitted for UI.
- Safe-mode is a hard blocklist in `tools/tool_registry.py` with no per-action approval flow or session approval state. This differs from the "ask only for dangerous actions" rule.
- DualBrain critic can rewrite the plan in `core/agent.py` (`_critic_plan`). The future roadmap says critic should only approve/block; current behavior differs.
- `tools/image_generate_tool.py` is explicitly a local stub (solid color image). If "no fake logic" is a hard invariant, confirm whether this tool is acceptable as-is.
- `core/executor.py` uses placeholder behavior for some operations (e.g., shell step runs `echo step`), which may be interpreted as fake logic for plan execution.
- Doc mismatch: `my sheet/updated_roadmap.md` references `memory/feedback_manager.py`, but there is no such file; feedback is handled by `memory/memory_companion_store.py`. `memory/feedback.db` exists as a legacy artifact but is not referenced in code.
- No `AGENTS.md` found in the repo root; no extra agent-specific instructions available.

## Assumptions captured
- The zip archive `slavikAI.zip` matches the current folder content (per user note); no diff was performed.
