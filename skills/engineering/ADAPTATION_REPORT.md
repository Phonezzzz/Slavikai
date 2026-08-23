# Engineering skills adaptation report

Source: `mattpocock/skills` commit `5b15a47f2d7150f545fbcacbfe381787fc0230dc`
(MIT, copyright Matt Pocock). The originally referenced pre-integration archive was not available
in this workspace, so the adaptation was reconstructed from that pinned source and the accepted
SlavikAI constraints.

Included workflow skills: diagnosing-bugs, research, tdd, code-review, to-spec, to-tickets,
implement, prototype, improve-codebase-architecture, resolving-merge-conflicts, handoff, and
grill-with-docs. Included supporting skills: codebase-design, grilling, and domain-modeling.

Project-specific changes:

- removed automatic stage/commit/push, branch creation, merge/rebase continuation, and other git
  finalization;
- removed automatic sub-agent requirements and external issue publication;
- removed silent CONTEXT.md/ADR/spec/ticket mutation;
- made questioning conditional on real decision blockers;
- added mechanism/runtime-integrity review to code-review;
- made every body an immutable per-run instruction that grants no tools or permissions;
- added explicit supporting-skill dependency resolution and terminal run observability.
