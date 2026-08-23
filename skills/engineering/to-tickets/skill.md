---
id: to-tickets
version: 1.0.0
title: Specification to tickets
entrypoints: [workspace_list, workspace_read, workspace_write]
patterns: ["to tickets", "разбей на тикеты", "нарежь на задачи", "сделай tickets"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: medium
tests: [tests/test_engineering_skills_runtime.py]
---

# To tickets

Break the accepted plan or specification into narrow vertical slices. Each ticket must deliver a
complete, independently verifiable path and state its blocking edges and acceptance criteria.
Keep a slice small enough for one focused implementation run. Use expand-migrate-contract only
when a wide mechanical change cannot remain green as a vertical slice.

Present the proposed dependency graph before publishing. Do not create local files, GitHub issues,
labels, or tracker relationships unless the user explicitly requested publication.
