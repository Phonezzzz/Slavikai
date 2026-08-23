---
id: handoff
version: 1.0.0
title: Session handoff
entrypoints: [workspace_list, workspace_read]
patterns: ["handoff", "передай следующему агенту", "сделай передачу контекста", "session handoff"]
requires: []
dependencies: []
supporting: false
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Handoff

Produce a compact continuation record: objective, accepted decisions, current repository and git
state, completed work with evidence, open work, blockers, verification results, and exact next
step. Reference existing artifacts instead of duplicating them and redact secrets or personal
data.

Return the handoff in the response unless the user explicitly asks for a file. Do not mutate the
repository or git state while preparing it.
