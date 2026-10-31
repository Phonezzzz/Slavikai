# R1 Trusted Staged File-Effects Transport — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans with TDD. This is a staged prerequisite, not a full Auto budget consumer.

**Goal:** Thread the trusted per-call file-effect budget through the qualified Python execution path and publish authoritative changed-path observations only after workspace publication.

**Architecture:** `ComputerBackend.execute_python` and `RootlessPythonExecutor` accept optional trusted `max_files_touched` and prior `already_touched` parameters, never model-supplied ToolRequest args. Rootless executor forwards them to the PR #124 atomic publication guard. Publication returns a typed `BoundedPublication` containing changed paths and the retired-tree cleanup warning. `WorkspacePythonOutcome` transports published paths, and `RunCodeTool` projects them only for `publication="published"`.

**Tech stack:** Python 3.12, RootlessPodmanProfile and typed WorkspaceSnapshot, pytest.

**Spec:** `docs/architecture/ARCH_CANON.md`, PR #124 staged publication prerequisite (base `0fc81fe`), `docs/SOURCE_OF_TRUTH.md`.

## Global constraints
- All tools remain mediated through the existing ToolGateway / ComputerBackend. No host command fallback, hidden publication, untrusted stdout parsing, new policy store or compatibility adapter.
- `changed_paths=None` means no authoritative published evidence; never coerce unknown to zero.
- `max_files_touched` and `already_touched` are Python-only trusted call parameters; do not read them from model-supplied `request.args`.
- Stacked PR, base PR #124; do not merge either PR without owner approval.
- Auto run-scoped ledger, native direct writer admission, approvals/continuation reconciliation remain subsequent work; no end-to-end enforcement claim.

## Review focus
- No changed_paths on rejected/retained/unknown output, even if malicious/fake backend supplies one.
- Over-limit stage returns typed unavailable outcome and preserves host tree/revision.
- Exactly-at-limit stage publishes and exposes the actual changed path.
- Cleanup warnings with already-published effects preserve evidence and demand review.
- Legacy unbounded callers continue using the unchanged default values.

### Task 1: Verify typed publication evidence
- [x] Write tests for `BoundedPublication.changed_paths` and no cleanup warning; confirm RED on old return type.
- [x] Return immutable verified paths from `ManagedWorkspace.publish_bounded` after atomic exchange.

### Task 2: Thread trusted budget parameters
- [x] Write failing tests for RootlessPythonExecutor and qualified ContainerComputerBackend argument forwarding.
- [x] Extend ComputerBackend Protocol, LocalComputerBackend and ContainerComputerBackend signatures.
- [x] Pass bound parameters into managed publish; never accept limits from model tool args.
- [x] Synthetic settled execution verifies refused vs accepted publication with path evidence.

### Task 3: Publish trustworthy tool observation
- [x] RED→GREEN projection test for successful changed_paths; reject uncommitted path claims.
- [x] Add published-only `changed_paths` projection to RunCodeTool.

### Task 4: Final gate
- [x] `make check` and `git diff --check`, inspect actual exit code (exit 0; 2367 Python / 352 UI tests, 82.94% coverage).
- [x] Self-review complete diff and fix Important issues with tests (published-only observation, rejection and cleanup-warning synthetic paths).
- [x] Commit and push stacked PR; request Codex Review (or record quota blocker). No merge.

**Next dependency:** trusted Auto run-owned cumulative set tracked across tool calls/approval continuations, admitted at ToolGateway and passed to this qualified backend. Native direct writers need independent pre-dispatch controls; neither is implemented by this PR.
