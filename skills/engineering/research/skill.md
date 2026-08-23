---
id: research
version: 1.0.0
title: Primary-source research
entrypoints: [web, workspace_list, workspace_read]
patterns: ["research", "проведи исследование", "исследуй вопрос", "проверь по документации"]
requires: []
dependencies: []
supporting: false
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Research

Investigate the question against primary sources: official documentation, specifications, source
code, first-party APIs, or research papers. Trace material claims to the source that owns them,
separate verified facts from inference, and note meaningful conflicts between sources.

Return the findings in the current response unless the user explicitly asked for a repository
artifact. Do not create background agents, files, tickets, or external posts automatically.
