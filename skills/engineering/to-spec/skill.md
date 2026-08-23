---
id: to-spec
version: 1.0.0
title: Conversation to specification
entrypoints: [workspace_list, workspace_read, workspace_write]
patterns: ["to spec", "сделай spec", "подготовь спецификацию", "оформи как спецификацию"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: medium
tests: [tests/test_engineering_skills_runtime.py]
---

# To spec

Synthesize already accepted decisions into a specification; do not restart an interview. Verify
the current repository state and use its domain vocabulary. Cover the problem, user-visible
solution, numbered user stories, implementation decisions, public contracts, testing seams,
acceptance criteria, out-of-scope items, and unresolved blockers.

Do not publish to an issue tracker or write into the repository unless the user explicitly asked
for that destination. Do not silently create or update CONTEXT.md or ADRs.
