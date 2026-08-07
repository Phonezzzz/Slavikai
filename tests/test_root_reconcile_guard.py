from __future__ import annotations


class RootReconcileGuard:
    """Deterministic mirror of ui/src/app/root-reconcile.ts.

    Guarantees: a request that started earlier (lower generation) can never
    apply its result after a newer request began — stale out-of-order
    responses are dropped.
    """

    def __init__(self) -> None:
        self._generation = 0

    def begin(self) -> int:
        self._generation += 1
        return self._generation

    def is_current(self, generation: int) -> bool:
        return generation == self._generation


def test_out_of_order_responses_apply_only_newest_root() -> None:
    guard = RootReconcileGuard()

    # Request A for session-a starts first (generation 1)
    gen_a = guard.begin()
    # Request B for session-b starts second (generation 2)
    gen_b = guard.begin()

    # B completes first — it is current, its root applies
    assert guard.is_current(gen_b)
    applied: list[str] = []
    if guard.is_current(gen_b):
        applied.append("root-b")

    # A completes last (out of order) — its generation is stale, dropped
    assert not guard.is_current(gen_a)
    if guard.is_current(gen_a):
        applied.append("root-a")

    assert applied == ["root-b"]


def test_same_session_reentrant_stale_dropped() -> None:
    guard = RootReconcileGuard()

    gen_1 = guard.begin()
    gen_2 = guard.begin()
    gen_3 = guard.begin()

    # Only the latest generation is current; all older are stale
    assert not guard.is_current(gen_1)
    assert not guard.is_current(gen_2)
    assert guard.is_current(gen_3)


def test_no_concurrent_request_allows_apply() -> None:
    guard = RootReconcileGuard()

    gen = guard.begin()
    assert guard.is_current(gen)

    applied: list[str] = []
    if guard.is_current(gen):
        applied.append("root-solo")
    assert applied == ["root-solo"]
