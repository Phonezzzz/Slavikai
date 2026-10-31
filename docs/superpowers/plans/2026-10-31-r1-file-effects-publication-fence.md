# R1 Staged File-Effects Publication Fence — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans and TDD. This plan describes the first safe prerequisite, not complete Auto enforcement.

**Goal:** Add a fail-closed, opt-in file-effect budget guard at the existing bounded workspace publication boundary.

**Architecture:** `ManagedWorkspace.publish_bounded` already captures a complete trusted output snapshot before atomic host-tree replacement. Compare it with the captured pre-execution snapshot, including file contents, executable mode, file additions/removals and directory changes; union with the caller's trusted prior touched-path set; refuse publication before materializing a host stage when the bound is exceeded. Preserve source revision, staged artifact and all original authorization/settlement checks on refusal.

**Tech stack:** Python 3.12, typed `WorkspaceSnapshot`, `pytest`.

**Spec:** `docs/SOURCE_OF_TRUTH.md`, `docs/architecture/ARCH_CANON.md`, `docs/architecture/roadmap/master-roadmap.md`.

## Global constraints
- No direct tool action outside `ToolGateway`; no separate runtime, adapter, fallback, store or authorization.
- Opt-in primitive only. Auto currently does not supply a trusted cumulative path ledger or invoke this guard; do not claim `max_files_touched` is enforced end-to-end.
- Count changed *paths* conservatively (including empty directories) to avoid underestimating effects. The final semantics for the user-facing `files_touched` metric need separate acceptance.
- Branch from `origin/main`; no merge. Tests and self-review before PR, GitHub Codex review when available.

## Review focus
- Output changed after validation: stage output must be materialized from the already verified snapshot, not recaptured later for the budget decision.
- Whole-tree rename/delete and empty directories cannot silently count as zero.
- A path touched in a previous tool operation is not counted twice.
- Invalid numeric limits, path sets, stale owner binding, or unsettled stages must fail without publication.
- Existing callers without a file budget retain their original behavior.

### Task 1: Mechanism tests
- [x] Add failing tests to `tests/test_managed_workspace.py` for threshold, cumulative unique paths, unchanged output, rename, directory effect, and no host exchange.
- [x] Confirm RED prior to implementation (missing typed budget exception).

### Task 2: Trusted stage guard
- [x] Add typed `WorkspaceFileBudgetExceeded` to `shared/workspace_binding.py`.
- [x] Add pure `changed_workspace_paths` comparison for validated snapshots.
- [x] Add optional `max_files_touched` and `already_touched` to `ManagedWorkspace.publish_bounded`, reject before host stage/publish.
- [x] Confirm GREEN for affected managed workspace tests.

### Task 3: Verification and review
- [x] Run `make check`, inspect exit code, regression stats, mypy and formatting. (exit 0; 2361 Python tests, 352 UI tests, coverage 83.11%).
- [x] Perform source-based self-review, correct Important findings with regression tests. (permission-only effects: failing regression then fix).
- [ ] Commit and push an unmerged PR; request Codex Review (or report quota blocker).

**Explicit next dependency:** Wire a run-owned, trusted cumulative file-effects ledger through the existing Gateway/ToolLoop and rooted stage process boundary; separately cover native direct writers. No runtime `implemented` claim from this PR alone.
