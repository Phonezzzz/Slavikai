"""Minimal immutable run-level contract (E01-T02).

ExecutionMode conflates three independent axes: what the run is trying
to do (intent), where it executes (target), and what it may do (policy).
This module separates them into a frozen RunSpec that is the source of
truth for a single run:

- intent: "answer" | "plan" | "execute" — the run's purpose.
- target: ExecutionTarget ("sandbox" | "desktop") — where it executes.
- policy: RunPolicy — an immutable snapshot of the policy-relevant
  ToolRegistry state (mode, execution target, active plan/task, plan
  guard) taken at run start. Later global ToolRegistry mutations cannot
  change a running run's effective policy, and the snapshot's own nested
  data cannot be mutated through the policy either (recursively frozen).
- budget: RunBudget — the run's loop budgets.

The contract reuses existing types (ExecutionMode / ExecutionTarget from
tools.tool_registry) and the AgentToolLoop budget parameters. It is not
a framework: no hierarchy, no registry of run types, no new execution
engine. The Agent's runtime_mode / runtime_active_plan /
runtime_active_task fields remain only as the callers from which a spec
is derived (build_run_spec); the spec is the source of truth while the
run executes.

Safe mode is deliberately NOT part of the snapshot: it is a global user
setting (tools_enabled), not run state, and the registry enforces it
independently of the policy snapshot.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, SupportsIndex, TypedDict

from shared.models import JSONValue
from tools.tool_registry import ExecutionMode, ExecutionTarget

RunIntent = Literal["answer", "plan", "execute"]


class LoopKwargs(TypedDict):
    """Keyword arguments for AgentToolLoop(...), derived from a RunBudget."""

    max_iterations: int
    max_tool_calls: int | None
    max_model_tokens: int | None
    expires_at_monotonic: float | None


class FrozenDict(dict[str, JSONValue]):
    """A dict that refuses every mutation attempt.

    A MappingProxyType would break ToolRegistry._plan_guard_error, which
    requires isinstance(active_plan, dict); a dict subclass keeps the
    isinstance/== behaviour of a plain dict and only blocks writes.
    Construction still works: dict.__init__ fills the storage directly
    and does not go through the overridden __setitem__.
    """

    def __setitem__(self, key: str, value: JSONValue) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def __delitem__(self, key: str) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def clear(self) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def pop(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def popitem(self) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenDict is immutable")

    # NOTE: mypy requires a single-signature __ior__ to match the overloaded
    # dict.__or__ arg-for-arg, which is unsatisfiable; the override exists
    # to block |= (it bypasses __setitem__ at C level).
    def __ior__(self, other: object) -> NoReturn:  # type: ignore[misc]
        raise TypeError("FrozenDict is immutable")


class FrozenList(list[JSONValue]):
    """A list that refuses every mutation attempt.

    Same rationale as FrozenDict: _plan_guard_error requires
    isinstance(steps, list) and isinstance(allowed_tool_kinds, list),
    so the frozen form must stay a list subclass.
    """

    def __setitem__(self, index: Any, value: Any) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def __delitem__(self, index: Any) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def append(self, value: JSONValue) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def extend(self, values: Iterable[JSONValue]) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def insert(self, index: SupportsIndex, value: JSONValue) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def remove(self, value: JSONValue) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def pop(self, index: SupportsIndex = -1) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def clear(self) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def sort(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    def reverse(self) -> NoReturn:
        raise TypeError("FrozenList is immutable")

    # NOTE: see __ior__; a single-signature __iadd__ cannot match the
    # overloaded list.__add__ arg-for-arg; the override blocks +=.
    def __iadd__(self, values: object) -> NoReturn:  # type: ignore[misc]
        raise TypeError("FrozenList is immutable")

    # NOTE: see __ior__; the override blocks *=.
    def __imul__(self, count: object) -> NoReturn:
        raise TypeError("FrozenList is immutable")


def _freeze_json(value: JSONValue | None) -> JSONValue | None:
    """Recursively convert a JSON value into immutable containers.

    dict -> FrozenDict, list -> FrozenList; scalars pass through.
    Builds new containers, so the source is never aliased.
    """

    if isinstance(value, dict):
        return FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return FrozenList(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class RunBudget:
    """Per-run loop budgets. Mirrors the AgentToolLoop constructor."""

    max_iterations: int = 8
    max_tool_calls: int | None = None
    max_model_tokens: int | None = None
    max_runtime_seconds: float | None = None

    def loop_kwargs(self, *, started_monotonic: float | None = None) -> LoopKwargs:
        """Convert to AgentToolLoop kwargs.

        max_runtime_seconds is a duration; it becomes the absolute
        expires_at_monotonic only when set. started_monotonic is the
        run's start time (time.monotonic() when omitted).
        """
        expires_at: float | None = None
        if self.max_runtime_seconds is not None:
            start = started_monotonic if started_monotonic is not None else time.monotonic()
            expires_at = start + self.max_runtime_seconds
        return LoopKwargs(
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            max_model_tokens=self.max_model_tokens,
            expires_at_monotonic=expires_at,
        )


@dataclass(frozen=True)
class RunPolicy:
    """Immutable policy snapshot for one run.

    Captures exactly what ToolRegistry.set_execution_policy used to set
    globally (mode, execution target, active plan/task, plan guard).
    The active plan/task are recursively frozen at construction
    (dict -> FrozenDict, list -> FrozenList), so neither later mutation
    of the source dicts nor mutation through the snapshot itself can
    change this run's policy. A MappingProxyType/tuple freeze would
    break ToolRegistry._plan_guard_error, which requires
    isinstance(active_plan, dict) and isinstance(steps, list); dict/list
    subclasses keep those checks (and ==) working and only block writes.
    """

    mode: ExecutionMode
    execution_target: ExecutionTarget
    active_plan: dict[str, JSONValue] | None = None
    active_task: dict[str, JSONValue] | None = None
    enforce_plan_guard: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "active_plan", _freeze_json(self.active_plan))
        object.__setattr__(self, "active_task", _freeze_json(self.active_task))


@dataclass(frozen=True)
class RunSpec:
    """The run-level contract: intent / target / policy / budget."""

    intent: RunIntent
    target: ExecutionTarget
    policy: RunPolicy
    budget: RunBudget = field(default_factory=RunBudget)


def intent_for_mode(mode: ExecutionMode) -> RunIntent:
    """Derive the run intent from a legacy ExecutionMode."""
    if mode == "ask":
        return "answer"
    if mode == "plan":
        return "plan"
    return "execute"


def target_for_mode(mode: ExecutionMode) -> ExecutionTarget:
    """Derive the run target from a legacy ExecutionMode."""
    return "desktop" if mode == "desktop" else "sandbox"


def build_run_spec(
    *,
    mode: ExecutionMode,
    active_plan: dict[str, JSONValue] | None = None,
    active_task: dict[str, JSONValue] | None = None,
    enforce_plan_guard: bool = False,
    budget: RunBudget | None = None,
) -> RunSpec:
    """Build a RunSpec from the legacy mode fields (the callers).

    This is the single place that derives a run contract from an
    Agent's runtime state. Callers pass the mode/plan/task/guard they
    already hold; the returned spec is the source of truth for the run.
    """
    target = target_for_mode(mode)
    return RunSpec(
        intent=intent_for_mode(mode),
        target=target,
        policy=RunPolicy(
            mode=mode,
            execution_target=target,
            active_plan=active_plan,
            active_task=active_task,
            enforce_plan_guard=enforce_plan_guard,
        ),
        budget=budget if budget is not None else RunBudget(),
    )
