---
id: code-review
version: 1.0.0
title: Code review
entrypoints: [workspace_list, workspace_read, terminal_exec]
patterns: ["code review", "ревью кода", "проверь diff", "review the changes"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Code review

Review the diff against a verified fixed point and the current repository, not a prior summary.
Check three independent axes: repository standards, requested specification, and mechanism/runtime
integrity. For mechanism integrity, look for prose or regex pretending to be structured input,
fallbacks that bypass the native runtime, direct actions outside ToolGateway, duplicate reachable
runtimes, weakened approvals/sandbox/verifier, and tests that prove only an end effect.

Report only actionable findings, ordered by severity, with tight file/line evidence. Verify each
claim against current code. Do not mutate files, git state, issues, or PRs during a review unless
the user separately asked for fixes or repository actions.
