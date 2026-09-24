# Blockers

- **Для чего:** отслеживать блокеры, мешающие продвижению.
- **Сюда:** описание блокера, влияние, что нужно для разблокировки, ответственный и статус.
- **Не сюда:** неблокирующие work items.
- **Обновлять:** при появлении или снятии блокера.

## B-2026-09-24-01 — Lifecycle contract cannot be integrated as-is

- **Status:** open; blocks Lifecycle contract integration and section 4 runtime
  specification, not the remaining read-only Capability Discovery research.
- **Evidence:** snapshot `6040640:docs/architecture/TASK_RUN_LIFECYCLE_CONTRACT.md`
  omits `aborted` from logical-task terminal states, while accepted ADR-0002
  requires a user-facing administratively terminated task revision to retain
  `aborted`. See
  `../research/lifecycle-communication-verification-reconciliation.md` and
  `../research/capability-boundary-proposal.md`. The snapshot also treats
  `superseded` as a terminal task-revision state; ADR-0010 now says that
  replacement is status/lineage without a separate old-revision final. Its
  sections 3 and 4 also differ on replacement within one logical task versus
  replacement by another task; section 13 jumps from run acceptance to final
  projection without an explicit logical-task terminal decision.
- **Resolution:** complete the Gate 0 boundary review, then produce a
  reconciled lifecycle contract before section 4 runtime work. Update
  its task state machine and transition/communication mapping, including
  same-task/cross-task supersession and run-versus-task final boundaries, then check the
  Communication and Verification contracts and claims registry together.
  Preserve the source snapshot; do not merge it mechanically.
- **Owner:** architecture/lifecycle workstream. The product rules are settled
  in ADR-0002 and ADR-0010; the remaining work is a reconciled lifecycle
  contract and cross-contract check, not another product preference.

## B-2026-09-24-02 — Memory snapshot promotion authority needs reconciliation

- **Status:** open; blocks integration of Memory snapshot `f883893` and any
  runtime path that automatically commits Memory. It does not block the
  remaining read-only Gate 0 owner review.
- **Evidence:** snapshot §5.3–5.4 permits automatic acceptance under explicit
  narrow policy. Current `docs/agent/DevRules.md` §11 forbids runtime Memory
  auto-updates without explicit approve, and `docs/SOURCE_OF_TRUTH.md` records
  per-request `confirm`/`edit_and_confirm` as implemented. The exact meaning
  of prior class-policy approval versus per-record approval is unresolved;
  see [OQ-MEM-01](../research/open-questions.md).
- **Resolution:** during section 3 reconcile the Target promotion authority
  with binding rules, consent and Current State. If the product decision
  changes the mandatory rule, record it explicitly in canonical docs/ADR;
  then design and test the corresponding runtime mechanism separately.
- **Owner:** Memory Architecture with Policy/Approval; product owner only if
  explicit policy scope is a new product choice.
