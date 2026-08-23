---
id: grilling
version: 1.0.0
title: Decision-blocker questioning
entrypoints: []
patterns: []
requires: []
dependencies: []
supporting: true
risk: low
tests: [tests/test_engineering_skills_runtime.py]
---

# Grilling

Model decisions as a dependency tree, but ask only questions whose answers materially change the
result and cannot be discovered from current files, tools, or accepted context. Ask the available
frontier together, provide one recommended answer for each blocker, and wait only when a user
decision is truly required. Do not turn routine implementation into an exhaustive interview.
