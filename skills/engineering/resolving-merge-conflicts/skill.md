---
id: resolving-merge-conflicts
version: 1.0.0
title: Resolve merge conflicts
entrypoints: [workspace_list, workspace_read, workspace_write, workspace_patch, terminal_exec]
patterns: ["resolve merge conflicts", "разреши конфликты", "rebase conflict", "merge conflict"]
requires: []
dependencies: []
supporting: false
risk: high
tests: [tests/test_engineering_skills_runtime.py]
---

# Resolving merge conflicts

Inspect the in-progress git operation and find the primary intent behind both sides of every
conflict. Resolve hunks without inventing unrelated behaviour, preserve both intents where they
are compatible, and run the repository's focused and canonical checks.

Do not abort, stage, commit, continue a merge/rebase, push, or rewrite history automatically. Stop
before git finalization and report the resolved files, remaining conflicts, checks, and the exact
human-approved next operation.
