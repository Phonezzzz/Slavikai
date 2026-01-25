# AGENTS — mandatory entry point

This repository is rules-first and policy-driven.
All agents must read and follow the documents listed below before doing any work.

## Initialization protocol (mandatory)

Before any work, the agent must:

1. Read all documents under "Canonical rules (must read)" — always.
2. Identify which contextual documents are relevant (architecture, flows, mini-prompts, other docs) and read them.
3. Create a Rules + Context Snapshot and treat it as hard constraints for all actions.

No task execution is allowed before completing this initialization.

## Canonical rules (must read)

- docs/DevRules.md — global invariants and project policies
- docs/dev_workflow.md — Git workflow (режим A)
- docs/merge_process.md — exact merge sequence (fast-forward only)
- docs/COMMAND_LANE_POLICY.md — command execution boundaries
- docs/ROUTING_POLICY.md — routing logic and escalation rules
- docs/STOP_RESPONSES.md — when to stop and ask for human input

## Contextual references (read if relevant)

- docs/Architecture.md
- docs/MWV_CANONICAL_FLOW.md
- docs/RUNTIME_MWV_FLOW.md
- docs/AGENT_MINIPROMPTS.md — task-specific mini-prompts

## Non-negotiable rules (summary)

- Work never happens on `main`
- Every task uses a PR branch
- Before any work: `make git-check`
- Before finalizing: `make check`
- Merge to `main` only via `git merge --ff-only`
- After merge always return to `main`
- If anything is unclear — stop and ask

### Язык (обязательно)
- Все ответы, планы, объяснения и комментарии — на русском.
- Английский допускается только внутри кода, команд, логов, названий файлов и точных цитат ошибок/документации.
- “Размышления” оформляй как: **Краткий план (RU)** + **Проверки/риски (RU)**. Без англ. текста.

## Workflow (short)

1. `git checkout main`
2. `git checkout -b pr-<id>-<name>`
3. `make git-check`
4. implement + commit + push
5. `make check`
6. rebase + fast-forward merge
7. back to `main`

No exceptions.
