---
id: prototype
version: 1.0.0
title: Throwaway prototype
entrypoints: [workspace_list, workspace_read, workspace_write, workspace_patch]
patterns: ["prototype", "прототип", "sanity-check the design", "проверь идею прототипом"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: medium
tests: [tests/test_engineering_skills_runtime.py]
---

# Prototype

Build the smallest throwaway artifact that answers one stated design question. Make it obvious
that the artifact is a prototype, trivial to run, isolated from production persistence, and able
to expose the relevant state after each action. Skip production abstractions and polish.

Summarize the decision learned and identify what should be discarded or carried into production.
Do not create a branch, commit, or other git history automatically, and do not leave the prototype
reachable from production runtime unless the user explicitly chooses that outcome.
