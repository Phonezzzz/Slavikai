---
id: codebase-design
version: 1.0.0
title: Deep-module design vocabulary
entrypoints: []
patterns: []
requires: []
dependencies: []
supporting: true
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Codebase design

Use module, interface, implementation, seam, adapter, depth, leverage, and locality consistently.
A deep module gives callers substantial behaviour through a small interface. Put tests at the same
public seam callers use. Accept dependencies rather than constructing hidden globals; return
observable results rather than relying on side channels. Apply the deletion test: a useful module
concentrates complexity that would otherwise spread across callers. Introduce an adapter seam only
when behaviour actually varies across it.
