---
id: tdd
version: 1.0.0
title: Test-driven development
entrypoints: [workspace_list, workspace_read, workspace_write, workspace_patch, terminal_exec]
patterns: ["test-driven", "red-green-refactor", "через tdd", "сначала тест"]
requires: []
dependencies: [codebase-design]
supporting: false
risk: medium
tests: [tests/test_engineering_skills_runtime.py]
---

# Test-driven development

Work in vertical red-green slices. Identify the public seam from the request and repository
contracts, write one test that fails for the missing behaviour, run it red, implement only enough
to make it green, and repeat. Tests must observe public behaviour and use independent expected
values; do not assert private helpers or reproduce the implementation inside the assertion.

If the seam is materially ambiguous, stop on that decision rather than inventing a new public
contract. Run focused tests during the loop and the repository's canonical gate at the end.
