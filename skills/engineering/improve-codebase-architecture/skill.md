---
id: improve-codebase-architecture
version: 1.0.0
title: Improve codebase architecture
entrypoints: [workspace_list, workspace_read, terminal_exec]
patterns: ["improve codebase architecture", "улучши архитектуру", "architecture review", "архитектурный аудит"]
requires: []
dependencies: [codebase-design, grilling, domain-modeling]
supporting: false
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Improve codebase architecture

Perform a non-mutating scan focused on recently changing or explicitly named modules. Find shallow
interfaces, lost locality, duplicated seams, and places where deleting an abstraction merely moves
complexity to callers. Verify friction against current code and tests; do not propose speculative
layers.

Report a small ranked set of deepening opportunities with evidence, trade-offs, recommendation
strength, and the best first change. Do not automatically generate HTML, open applications, edit
CONTEXT.md, create ADRs, or implement a candidate. Those are separate user-authorized actions.
