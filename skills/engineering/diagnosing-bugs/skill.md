---
id: diagnosing-bugs
version: 1.0.0
title: Diagnosing bugs
entrypoints: [workspace_list, workspace_read, workspace_write, workspace_patch, terminal_exec]
patterns: ["diagnose", "debug this", "диагностируй", "разберись с багом", "почему падает"]
requires: []
dependencies: []
supporting: false
risk: medium
tests: [tests/test_engineering_skills_runtime.py]
---

# Diagnosing bugs

Build a tight, red-capable feedback loop before choosing a cause. Reproduce the user's exact
symptom, minimise the scenario, then rank falsifiable hypotheses and change one variable at a
time. Prefer a regression test at the real public seam before the fix. Re-run the original repro
and the regression test after the fix, remove temporary instrumentation, and report the actual
root cause.

Redact secrets from commands, output, traces, and artifacts. If no reliable loop can be built,
state exactly what evidence is missing instead of presenting a hypothesis as a diagnosis.

Do not stage, commit, push, switch branches, rebase, or merge unless the user explicitly requested
that repository action outside this skill run.
