---
id: grill-with-docs
version: 1.0.0
title: Decision grilling with documentation
entrypoints: [workspace_list, workspace_read]
patterns: ["grill with docs", "допроси по дизайну", "проверь решение вопросами"]
requires: []
dependencies: [grilling, domain-modeling]
supporting: false
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Grill with docs

Use only for a genuinely ambiguous, consequential design. Ask the current frontier of decision
blockers, include a recommended answer, and do not ask for facts available from repository or
tools. Stop questioning when no material blocker remains.

Summarize glossary or ADR candidates, but do not create or edit CONTEXT.md, ADRs, specs, or tickets
without separate explicit confirmation.
