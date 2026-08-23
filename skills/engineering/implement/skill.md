---
id: implement
version: 1.0.0
title: Implement approved work
entrypoints: [workspace_list, workspace_read, workspace_write, workspace_patch, terminal_exec]
patterns: ["implement spec", "реализуй spec", "выполни тикеты", "implement the ticket"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: high
tests: [tests/test_engineering_skills_runtime.py]
---

# Implement

Implement the requested specification or ticket as the smallest coherent change. Establish the
baseline, follow the real execution path and state ownership, use vertical test-driven slices at
public seams, and run focused checks plus the canonical project gate. Finish with a full diff
review for accidental edits, pseudo-runtime paths, weakened policy controls, and scope creep.

This workflow never stages, commits, pushes, creates or switches branches, merges, rebases, or
publishes a PR automatically. Those actions require explicit user authorization outside the skill
instructions. Do not weaken approvals, sandbox, ToolGateway, or verifier to make tests pass.
