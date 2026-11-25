"""Typed postconditions (E05-T01): effect checks, not tool-return claims.

A postcondition says *what state must hold* after an action — a file
exists, a process exited 0, a unit is active — and it is evaluated by
reading real state through a :class:`PostconditionReader`, never by
trusting a tool's ``ok`` flag or a ``verified: true`` field. A
``ToolResult`` that claims success proves nothing here: the typed
channel only believes what the reader (or the captured browser facts)
says.

Design notes:

- The model is pure data + pure checkers. It never does I/O itself:
  every read goes through the injected reader, so the checkers are
  unit-testable with fake readers and the production readers live in
  the tools layer (``tools.postcondition_readers``).
- ``criterion_revision`` binds the check to its criterion at
  declaration time: ``revision()`` is the sha256 of the canonical
  JSON of kind + params. A check whose params changed after binding
  evaluates to ``STALE`` without touching the reader — a stale
  criterion can never pass on evidence taken for a different one.
- Check outcomes are six-valued (``CheckOutcome``). Only ``PASSED``
  is a pass. ``INCONCLUSIVE`` means "could not tell" (missing file,
  denied path, unknown process) and ``BLOCKED`` means "the check
  could not run at all" (no reader). Both map to ``UNKNOWN`` in the
  canonical gate — never to a pass, never to a fail.
- File checks read through the desktop security layer (the reader
  canonicalizes the path first): a symlink escaping the allowed roots
  is denied, not followed. Reads are size-capped; a file over the cap
  is ``INCONCLUSIVE`` (``file_too_large``), not a silent truncation.
- ``process_running`` with a bare pid is a construction error: a pid
  without its ``create_time`` cannot distinguish the process from a
  later pid reuse. The registry form (``process_id``) is the
  bridge-backed identity; the pid form is the desktop identity.
- ``custom_command`` v1 is exactly the canonical check
  (``make -B -r -f Makefile check``). Any other argv evaluates to
  ``INCONCLUSIVE`` and never spawns a process.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final, Literal, Protocol

from core.mwv.models import VerificationResult, VerificationStatus
from shared.models import JSONValue, ToolResult

MAX_DETAIL_CHARS: Final[int] = 500
"""Human-readable evaluation details are truncated to this length."""

SERVICE_ACTIVE_STATES: Final[frozenset[str]] = frozenset(
    {"active", "inactive", "failed", "activating", "deactivating"}
)
"""``ActiveState`` values a ``service_state`` check may expect."""

BRIDGE_TERMINAL_STATES: Final[frozenset[str]] = frozenset(
    {"exited", "killed", "timeout", "spawn_failed"}
)
"""Bridge process states that are final (see ``host_bridge/process.py``)."""

BRIDGE_TRANSITIONAL_STATES: Final[frozenset[str]] = frozenset({"launching", "killing"})
"""Bridge process states that are not final yet."""

PROCESS_CREATE_TIME_TOLERANCE_S: Final[float] = 0.01
"""Pid-form staleness tolerance (seconds).

Same 10 ms window the legacy ``DesktopVerifyTool`` used: a
``create_time`` read through JSON (or from another observer) can
differ from the live value by float round-trip noise. A difference
within the window still counts as the same process.
"""


class CheckOutcome(StrEnum):
    """One evaluation outcome. Only ``PASSED`` is a pass."""

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    STALE = "stale"
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"


class PostconditionKind(StrEnum):
    """The seven typed postcondition kinds (E05-T01)."""

    FILE_EXISTS = "file_exists"
    FILE_CONTENT_MATCHES = "file_content_matches"
    PROCESS_RUNNING = "process_running"
    PROCESS_EXIT = "process_exit"
    SERVICE_STATE = "service_state"
    BROWSER_STATE = "browser_state"
    CUSTOM_COMMAND = "custom_command"


def _require_str(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    return float(value)


def _reject_unknown_keys(data: dict[str, object], *, allowed: set[str], where: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{where} has unknown fields: {', '.join(unknown)}")


def _canonical_json(value: dict[str, JSONValue]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def browser_expectation_digest(params_json: dict[str, JSONValue]) -> str:
    """sha256 binding browser facts to the expectation they prove.

    The tool stamps this digest into the ``browser_state`` facts it
    captures; the typed checker requires it to equal the
    postcondition's own revision. The payload is exactly what
    :meth:`Postcondition.revision` hashes for a ``browser_state``
    postcondition with these params, so facts taken for one
    expectation (``after="Saved"``) can never confirm a postcondition
    declaring another (``after="Deleted"``). The digest carries no
    page text: observed strings never leave the tool, only this hash
    of the declared criterion does.
    """
    payload: dict[str, JSONValue] = {
        "kind": PostconditionKind.BROWSER_STATE.value,
        "params": params_json,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Parameters (one frozen dataclass per kind; construction validates).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileExistsParams:
    """``file_exists``: the path must be an existing regular file."""

    path: str

    def __post_init__(self) -> None:
        _require_str(self.path, field_name="path")

    def to_json(self) -> dict[str, JSONValue]:
        return {"path": self.path}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> FileExistsParams:
        _reject_unknown_keys(data, allowed={"path"}, where="file_exists params")
        if "path" not in data:
            raise ValueError("file_exists params require 'path'")
        return cls(path=_require_str(data["path"], field_name="path"))


@dataclass(frozen=True)
class FileContentMatchesParams:
    """``file_content_matches``: file content must match ``expected``.

    ``match="substring"`` is the desktop-verify semantics; ``"exact"``
    compares the whole file byte-for-byte after UTF-8 decoding (a
    trailing newline fails). An empty ``expected`` with ``"substring"``
    is vacuous — it matches everything — and is rejected.
    """

    path: str
    expected: str
    match: Literal["substring", "exact"] = "substring"

    def __post_init__(self) -> None:
        _require_str(self.path, field_name="path")
        if not isinstance(self.expected, str):
            raise ValueError("expected must be a string")
        if self.match not in ("substring", "exact"):
            raise ValueError("match must be 'substring' or 'exact'")
        if self.match == "substring" and not self.expected:
            raise ValueError("substring expected must not be empty")

    def to_json(self) -> dict[str, JSONValue]:
        return {"path": self.path, "expected": self.expected, "match": self.match}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> FileContentMatchesParams:
        _reject_unknown_keys(
            data, allowed={"path", "expected", "match"}, where="file_content_matches params"
        )
        if "path" not in data or "expected" not in data:
            raise ValueError("file_content_matches params require 'path' and 'expected'")
        match = data.get("match", "substring")
        if match not in ("substring", "exact"):
            raise ValueError("match must be 'substring' or 'exact'")
        expected = data["expected"]
        if not isinstance(expected, str):
            raise ValueError("expected must be a string")
        return cls(
            path=_require_str(data["path"], field_name="path"),
            expected=expected,
            match=match,
        )


@dataclass(frozen=True)
class ProcessRunningParams:
    """``process_running``: exactly one identity form.

    - ``process_id``: bridge/registry identity (E03-T06 record).
    - ``pid`` + ``create_time``: desktop identity; ``create_time`` is
      mandatory — a bare pid cannot rule out pid reuse.
    """

    pid: int | None = None
    create_time: float | None = None
    process_id: str | None = None

    def __post_init__(self) -> None:
        if self.pid is not None and self.process_id is not None:
            raise ValueError("process_running takes either pid or process_id, not both")
        if self.pid is None and self.process_id is None:
            raise ValueError("process_running requires pid or process_id")
        if self.pid is not None:
            _require_int(self.pid, field_name="pid")
            if self.pid <= 0:
                raise ValueError("pid must be positive")
            if self.create_time is None:
                raise ValueError("pid form requires create_time (bare pid cannot rule out reuse)")
            _require_number(self.create_time, field_name="create_time")
            if self.create_time < 0:
                raise ValueError("create_time must not be negative")
            # Normalize to float here, not only in from_json: an int
            # create_time (10) and its round-tripped float (10.0)
            # compare equal as dataclass values but hash different
            # canonical JSON, so a bound criterion would go STALE
            # right after its own round-trip.
            object.__setattr__(self, "create_time", float(self.create_time))
        if self.process_id is not None:
            _require_str(self.process_id, field_name="process_id")
            if self.create_time is not None:
                raise ValueError("process_id form takes no create_time")

    def to_json(self) -> dict[str, JSONValue]:
        return {"pid": self.pid, "create_time": self.create_time, "process_id": self.process_id}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> ProcessRunningParams:
        _reject_unknown_keys(
            data, allowed={"pid", "create_time", "process_id"}, where="process_running params"
        )
        pid = data.get("pid")
        if pid is not None:
            pid = _require_int(pid, field_name="pid")
        create_time = data.get("create_time")
        if create_time is not None:
            create_time = _require_number(create_time, field_name="create_time")
        process_id = data.get("process_id")
        if process_id is not None:
            process_id = _require_str(process_id, field_name="process_id")
        return cls(pid=pid, create_time=create_time, process_id=process_id)


@dataclass(frozen=True)
class ProcessExitParams:
    """``process_exit``: the bridge record must be final with this exit code."""

    process_id: str
    expected_exit_code: int

    def __post_init__(self) -> None:
        _require_str(self.process_id, field_name="process_id")
        _require_int(self.expected_exit_code, field_name="expected_exit_code")

    def to_json(self) -> dict[str, JSONValue]:
        return {"process_id": self.process_id, "expected_exit_code": self.expected_exit_code}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> ProcessExitParams:
        _reject_unknown_keys(
            data, allowed={"process_id", "expected_exit_code"}, where="process_exit params"
        )
        if "process_id" not in data or "expected_exit_code" not in data:
            raise ValueError("process_exit params require 'process_id' and 'expected_exit_code'")
        return cls(
            process_id=_require_str(data["process_id"], field_name="process_id"),
            expected_exit_code=_require_int(
                data["expected_exit_code"], field_name="expected_exit_code"
            ),
        )


@dataclass(frozen=True)
class ServiceStateParams:
    """``service_state``: a systemd unit must be in the expected ActiveState."""

    unit: str
    scope: str
    expected_active_state: str

    def __post_init__(self) -> None:
        _require_str(self.unit, field_name="unit")
        if self.scope not in ("user", "system"):
            raise ValueError("scope must be 'user' or 'system'")
        if self.expected_active_state not in SERVICE_ACTIVE_STATES:
            raise ValueError(
                f"expected_active_state must be one of {sorted(SERVICE_ACTIVE_STATES)}"
            )

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "unit": self.unit,
            "scope": self.scope,
            "expected_active_state": self.expected_active_state,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> ServiceStateParams:
        _reject_unknown_keys(
            data,
            allowed={"unit", "scope", "expected_active_state"},
            where="service_state params",
        )
        if "unit" not in data or "scope" not in data or "expected_active_state" not in data:
            raise ValueError(
                "service_state params require 'unit', 'scope' and 'expected_active_state'"
            )
        return cls(
            unit=_require_str(data["unit"], field_name="unit"),
            scope=_require_str(data["scope"], field_name="scope"),
            expected_active_state=_require_str(
                data["expected_active_state"], field_name="expected_active_state"
            ),
        )


@dataclass(frozen=True)
class BrowserStateParams:
    """``browser_state``: the E02-T10 contract B expectation, as data.

    ``kind="text"``: ``after`` is the exact expected text; ``kind="count"``:
    ``after`` is the expected element count (the contract only supports
    expecting exactly one element — see ``evaluate_browser_state``).
    ``has_before`` records whether a ``before`` expectation was declared:
    a declared ``before`` must also match the captured facts.
    """

    entity_id: str
    kind: str
    after: str | int
    before: str | int | None = None
    has_before: bool = False

    def __post_init__(self) -> None:
        _require_str(self.entity_id, field_name="entity_id")
        if self.kind not in ("text", "count"):
            raise ValueError("kind must be 'text' or 'count'")
        if self.kind == "text":
            if not isinstance(self.after, str):
                raise ValueError("after must be a string for kind 'text'")
            if self.has_before and not isinstance(self.before, str):
                raise ValueError("before must be a string for kind 'text'")
        else:
            if isinstance(self.after, bool) or not isinstance(self.after, int):
                raise ValueError("after must be an integer for kind 'count'")
            if self.has_before and (
                isinstance(self.before, bool) or not isinstance(self.before, int)
            ):
                raise ValueError("before must be an integer for kind 'count'")
        if not self.has_before and self.before is not None:
            raise ValueError("before requires has_before=true")
        if self.has_before and self.before is None:
            raise ValueError("has_before=true requires before")

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "after": self.after,
            "before": self.before,
            "has_before": self.has_before,
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> BrowserStateParams:
        _reject_unknown_keys(
            data,
            allowed={"entity_id", "kind", "after", "before", "has_before"},
            where="browser_state params",
        )
        if "entity_id" not in data or "kind" not in data or "after" not in data:
            raise ValueError("browser_state params require 'entity_id', 'kind' and 'after'")
        kind = data["kind"]
        if kind not in ("text", "count"):
            raise ValueError("kind must be 'text' or 'count'")
        after = data["after"]
        if kind == "text":
            # Text values may be empty (appearance of text from
            # nothing, or clearing to ""); only the type is enforced
            # here, exactly as BrowserStateParams.__post_init__ does.
            if not isinstance(after, str):
                raise ValueError("after must be a string for kind 'text'")
        else:
            after = _require_int(after, field_name="after")
        has_before = data.get("has_before", False)
        if not isinstance(has_before, bool):
            raise ValueError("has_before must be a boolean")
        before = data.get("before")
        if before is not None:
            if kind == "text":
                if not isinstance(before, str):
                    raise ValueError("before must be a string for kind 'text'")
            else:
                before = _require_int(before, field_name="before")
        return cls(
            entity_id=_require_str(data["entity_id"], field_name="entity_id"),
            kind=kind,
            after=after,
            before=before,
            has_before=has_before,
        )


@dataclass(frozen=True)
class CustomCommandParams:
    """``custom_command`` v1: exactly the canonical check.

    ``argv`` must equal the canonical check command
    (``make -B -r -f Makefile check``); anything else evaluates to
    ``INCONCLUSIVE`` and never spawns. ``criterion_sha256`` is the
    Makefile hash fixed before the loop; the checker passes it to the
    reader so the executor enforces it at execution (the checker
    itself does not read the Makefile).
    """

    argv: tuple[str, ...]
    criterion_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("argv must not be empty")
        for item in self.argv:
            if not isinstance(item, str) or not item:
                raise ValueError("argv items must be non-empty strings")
        if self.criterion_sha256 is not None:
            _require_str(self.criterion_sha256, field_name="criterion_sha256")

    def to_json(self) -> dict[str, JSONValue]:
        return {"argv": list(self.argv), "criterion_sha256": self.criterion_sha256}

    @classmethod
    def from_json(cls, data: dict[str, object]) -> CustomCommandParams:
        _reject_unknown_keys(
            data, allowed={"argv", "criterion_sha256"}, where="custom_command params"
        )
        if "argv" not in data:
            raise ValueError("custom_command params require 'argv'")
        argv = data["argv"]
        if not isinstance(argv, list) or not argv:
            raise ValueError("argv must be a non-empty list of strings")
        for item in argv:
            if not isinstance(item, str) or not item:
                raise ValueError("argv items must be non-empty strings")
        criterion_sha256 = data.get("criterion_sha256")
        if criterion_sha256 is not None:
            criterion_sha256 = _require_str(criterion_sha256, field_name="criterion_sha256")
        return cls(argv=tuple(argv), criterion_sha256=criterion_sha256)


PostconditionParams = (
    FileExistsParams
    | FileContentMatchesParams
    | ProcessRunningParams
    | ProcessExitParams
    | ServiceStateParams
    | BrowserStateParams
    | CustomCommandParams
)

_KIND_PARAMS: Final[dict[PostconditionKind, type[PostconditionParams]]] = {
    PostconditionKind.FILE_EXISTS: FileExistsParams,
    PostconditionKind.FILE_CONTENT_MATCHES: FileContentMatchesParams,
    PostconditionKind.PROCESS_RUNNING: ProcessRunningParams,
    PostconditionKind.PROCESS_EXIT: ProcessExitParams,
    PostconditionKind.SERVICE_STATE: ServiceStateParams,
    PostconditionKind.BROWSER_STATE: BrowserStateParams,
    PostconditionKind.CUSTOM_COMMAND: CustomCommandParams,
}

# ---------------------------------------------------------------------------
# Postcondition (the check itself) + revision binding.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Postcondition:
    """One typed postcondition: kind + params + provenance + revision.

    ``criterion_revision`` binds the check to its criterion at
    declaration time. It must be set with :meth:`with_revision` by
    the declarer (the trusted side, e.g. T02 binding); a check whose
    ``criterion_revision`` does not match :meth:`revision` evaluates
    to ``STALE`` without any read. An unbound check
    (``criterion_revision == ""``) is always ``STALE``.
    """

    kind: PostconditionKind
    params: PostconditionParams
    source: str = ""
    task_id: str | None = None
    step_id: str | None = None
    criterion_revision: str = ""

    def __post_init__(self) -> None:
        expected = _KIND_PARAMS.get(self.kind)
        if expected is None:
            raise ValueError(f"unknown postcondition kind: {self.kind!r}")
        if not isinstance(self.params, expected):
            raise ValueError(
                f"kind {self.kind.value} requires {expected.__name__}, "
                f"got {type(self.params).__name__}"
            )
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")
        if self.task_id is not None and not isinstance(self.task_id, str):
            raise ValueError("task_id must be a string or null")
        if self.step_id is not None and not isinstance(self.step_id, str):
            raise ValueError("step_id must be a string or null")
        if not isinstance(self.criterion_revision, str):
            raise ValueError("criterion_revision must be a string")

    def revision(self) -> str:
        """sha256 of the canonical JSON of kind + params (the criterion).

        Provenance (source/task/step) is deliberately excluded: the
        revision binds *what* is checked, not where it was declared.
        """
        payload: dict[str, JSONValue] = {
            "kind": self.kind.value,
            "params": self.params.to_json(),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    def with_revision(self) -> Postcondition:
        """Return a copy bound to its current criterion."""
        return replace(self, criterion_revision=self.revision())

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "kind": self.kind.value,
            "params": self.params.to_json(),
            "source": self.source,
            "task_id": self.task_id,
            "step_id": self.step_id,
            "criterion_revision": self.criterion_revision,
        }

    @classmethod
    def from_json(cls, data: object) -> Postcondition:
        if not isinstance(data, dict):
            raise ValueError("postcondition must be an object")
        _reject_unknown_keys(
            data,
            allowed={"kind", "params", "source", "task_id", "step_id", "criterion_revision"},
            where="postcondition",
        )
        if "kind" not in data or "params" not in data:
            raise ValueError("postcondition requires 'kind' and 'params'")
        raw_kind = data["kind"]
        if not isinstance(raw_kind, str):
            raise ValueError("kind must be a string")
        try:
            kind = PostconditionKind(raw_kind)
        except ValueError:
            raise ValueError(f"unknown postcondition kind: {raw_kind!r}") from None
        params_data = data["params"]
        if not isinstance(params_data, dict):
            raise ValueError("params must be an object")
        params = _KIND_PARAMS[kind].from_json(params_data)
        source = data.get("source", "")
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        task_id = data.get("task_id")
        if task_id is not None and not isinstance(task_id, str):
            raise ValueError("task_id must be a string or null")
        step_id = data.get("step_id")
        if step_id is not None and not isinstance(step_id, str):
            raise ValueError("step_id must be a string or null")
        criterion_revision = data.get("criterion_revision", "")
        if not isinstance(criterion_revision, str):
            raise ValueError("criterion_revision must be a string")
        return cls(
            kind=kind,
            params=params,
            source=source,
            task_id=task_id,
            step_id=step_id,
            criterion_revision=criterion_revision,
        )


# ---------------------------------------------------------------------------
# Reader protocol + errors.
# ---------------------------------------------------------------------------


class PathDeniedError(ValueError):
    """The reader refused a path (outside allowed roots, denied file)."""


class FileTooLargeError(Exception):
    """The reader refused a file over its read cap."""


@dataclass(frozen=True)
class FileInfo:
    """What the reader knows about a path without reading it."""

    exists: bool
    is_regular_file: bool


class PostconditionReader(Protocol):
    """State reads for the checkers. Implementations never trust claims.

    ``describe_file``/``read_file`` canonicalize through the desktop
    security layer first (a symlink escaping the allowed roots raises
    :class:`PathDeniedError`). ``read_file`` returns ``None`` for an
    absent file and raises :class:`FileTooLargeError` over the cap.
    ``process_status`` returns the bridge record (``{"state": ...,
    "exit_code": ...}``) or ``None`` when no record exists.
    ``os_process_running`` returns ``(alive, create_time)``; a zombie
    counts as not alive; an access denial raises ``PermissionError``.
    ``service_status`` returns the systemd unit dict or ``None`` when
    the unit is not found. ``canonical_check`` runs the canonical
    check through the executor bound to the given criterion hash:
    the executor enforces the hash at execution, so a PASSED for
    another Makefile never confirms this postcondition.
    """

    def describe_file(self, path: str) -> FileInfo: ...

    def read_file(self, path: str) -> bytes | None: ...

    def process_status(self, process_id: str) -> dict[str, JSONValue] | None: ...

    def os_process_running(self, pid: int) -> tuple[bool, float | None]: ...

    def service_status(self, unit: str, scope: str) -> dict[str, JSONValue] | None: ...

    def canonical_check(self, criterion_sha256: str | None = None) -> VerificationResult: ...


# ---------------------------------------------------------------------------
# Evaluation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PostconditionEvaluation:
    """One evaluated postcondition.

    ``evidence`` always carries a machine-readable ``reason`` code and
    kind-specific facts (paths, exit codes, unit states) — never file
    contents and never secrets.
    """

    postcondition: Postcondition
    outcome: CheckOutcome
    detail: str
    checker: str
    checked_at: float
    evidence: dict[str, JSONValue] = field(default_factory=dict)


def check_identity(postcondition: Postcondition) -> str:
    """Stable identity of the checked subject (used in check names)."""
    params = postcondition.params
    if isinstance(params, (FileExistsParams, FileContentMatchesParams)):
        return params.path
    if isinstance(params, ProcessRunningParams):
        return params.process_id if params.process_id is not None else f"pid:{params.pid}"
    if isinstance(params, ProcessExitParams):
        return params.process_id
    if isinstance(params, ServiceStateParams):
        return f"{params.scope}:{params.unit}"
    if isinstance(params, BrowserStateParams):
        return params.entity_id
    if isinstance(params, CustomCommandParams):
        return " ".join(params.argv)
    raise ValueError(f"unknown params type: {type(params).__name__}")


def check_name(postcondition: Postcondition) -> str:
    """Check-row name: ``<kind>:<identity>`` (e.g. ``file_exists:/x/y``)."""
    return f"{postcondition.kind.value}:{check_identity(postcondition)}"


def _detail(text: str) -> str:
    if len(text) > MAX_DETAIL_CHARS:
        return text[: MAX_DETAIL_CHARS - 3] + "..."
    return text


def _evaluation(
    postcondition: Postcondition,
    outcome: CheckOutcome,
    *,
    detail: str,
    checker: str,
    now: float,
    reason: str,
    evidence: dict[str, JSONValue] | None = None,
) -> PostconditionEvaluation:
    merged: dict[str, JSONValue] = {"reason": reason}
    if evidence:
        merged.update(evidence)
    return PostconditionEvaluation(
        postcondition=postcondition,
        outcome=outcome,
        detail=_detail(detail),
        checker=checker,
        checked_at=now,
        evidence=merged,
    )


def evaluate_postcondition(
    postcondition: Postcondition,
    reader: PostconditionReader | None,
    *,
    browser_facts: Sequence[dict[str, JSONValue]] | None = None,
    now: float | None = None,
) -> PostconditionEvaluation:
    """Evaluate one postcondition. Stale criteria never read anything."""
    at = time.monotonic() if now is None else now
    if not postcondition.criterion_revision:
        return _evaluation(
            postcondition,
            CheckOutcome.STALE,
            detail="criterion_revision is missing (unbound check)",
            checker="",
            now=at,
            reason="unbound",
        )
    if postcondition.criterion_revision != postcondition.revision():
        return _evaluation(
            postcondition,
            CheckOutcome.STALE,
            detail="criterion_revision does not match the declared criterion (stale)",
            checker="",
            now=at,
            reason="revision_mismatch",
        )
    params = postcondition.params
    if isinstance(params, FileExistsParams):
        return evaluate_file_exists(params, reader, postcondition=postcondition, now=at)
    if isinstance(params, FileContentMatchesParams):
        return evaluate_file_content_matches(params, reader, postcondition=postcondition, now=at)
    if isinstance(params, ProcessRunningParams):
        return evaluate_process_running(params, reader, postcondition=postcondition, now=at)
    if isinstance(params, ProcessExitParams):
        return evaluate_process_exit(params, reader, postcondition=postcondition, now=at)
    if isinstance(params, ServiceStateParams):
        return evaluate_service_state(params, reader, postcondition=postcondition, now=at)
    if isinstance(params, BrowserStateParams):
        return evaluate_browser_state(
            params, browser_facts or (), postcondition=postcondition, now=at
        )
    if isinstance(params, CustomCommandParams):
        return evaluate_custom_command(params, reader, postcondition=postcondition, now=at)
    raise ValueError(f"unknown params type: {type(params).__name__}")


def _blocked(postcondition: Postcondition, *, now: float, checker: str) -> PostconditionEvaluation:
    return _evaluation(
        postcondition,
        CheckOutcome.BLOCKED,
        detail="no reader available for this check",
        checker=checker,
        now=now,
        reason="no_reader",
    )


def evaluate_file_exists(
    params: FileExistsParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """The path must be an existing regular file (directories fail)."""
    checker = "desktop_verify"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    try:
        info = reader.describe_file(params.path)
    except PathDeniedError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"path denied: {exc}",
            checker=checker,
            now=now,
            reason="path_denied",
            evidence={"path": params.path},
        )
    except OSError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not stat file: {exc}",
            checker=checker,
            now=now,
            reason="read_error",
            evidence={"path": params.path},
        )
    if not info.exists:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"file does not exist: {params.path}",
            checker=checker,
            now=now,
            reason="absent",
            evidence={"path": params.path},
        )
    if not info.is_regular_file:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"path is not a regular file: {params.path}",
            checker=checker,
            now=now,
            reason="not_regular_file",
            evidence={"path": params.path},
        )
    return _evaluation(
        postcondition,
        CheckOutcome.PASSED,
        detail=f"file exists: {params.path}",
        checker=checker,
        now=now,
        reason="ok",
        evidence={"path": params.path},
    )


def evaluate_file_content_matches(
    params: FileContentMatchesParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """File content must match ``expected`` (substring or exact)."""
    checker = "desktop_verify"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    try:
        info = reader.describe_file(params.path)
    except PathDeniedError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"path denied: {exc}",
            checker=checker,
            now=now,
            reason="path_denied",
            evidence={"path": params.path},
        )
    except OSError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not stat file: {exc}",
            checker=checker,
            now=now,
            reason="read_error",
            evidence={"path": params.path},
        )
    if not info.exists:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"file does not exist: {params.path}",
            checker=checker,
            now=now,
            reason="file_missing",
            evidence={"path": params.path},
        )
    if not info.is_regular_file:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"path is not a regular file: {params.path}",
            checker=checker,
            now=now,
            reason="not_regular_file",
            evidence={"path": params.path},
        )
    try:
        content = reader.read_file(params.path)
    except FileTooLargeError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"file exceeds the read limit: {params.path} ({exc})",
            checker=checker,
            now=now,
            reason="file_too_large",
            evidence={"path": params.path},
        )
    except PathDeniedError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"path denied: {exc}",
            checker=checker,
            now=now,
            reason="path_denied",
            evidence={"path": params.path},
        )
    except FileNotFoundError:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"file disappeared before the read: {params.path}",
            checker=checker,
            now=now,
            reason="file_missing",
            evidence={"path": params.path},
        )
    except OSError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not read file: {exc}",
            checker=checker,
            now=now,
            reason="read_error",
            evidence={"path": params.path},
        )
    if content is None:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"file does not exist: {params.path}",
            checker=checker,
            now=now,
            reason="file_missing",
            evidence={"path": params.path},
        )
    text = content.decode("utf-8", errors="replace")
    if params.match == "substring":
        # Legacy file_contains read with Path.read_text(): universal
        # newlines translate CRLF/CR to LF. Keep that normalization
        # for substring; exact keeps byte-exact semantics.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        matched = params.expected in text
    else:
        matched = text == params.expected
    if matched:
        return _evaluation(
            postcondition,
            CheckOutcome.PASSED,
            detail=f"content matches ({params.match}): {params.path}",
            checker=checker,
            now=now,
            reason="ok",
            evidence={"path": params.path, "match": params.match},
        )
    return _evaluation(
        postcondition,
        CheckOutcome.FAILED,
        detail=f"content does not match ({params.match}): {params.path}",
        checker=checker,
        now=now,
        reason="content_mismatch",
        evidence={"path": params.path, "match": params.match},
    )


def evaluate_process_running(
    params: ProcessRunningParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """A process is running: bridge record or pid + create_time."""
    if params.process_id is not None:
        checker = "bridge_poll"
        if reader is None:
            return _blocked(postcondition, now=now, checker=checker)
        try:
            status = reader.process_status(params.process_id)
        except (OSError, ValueError) as exc:
            return _evaluation(
                postcondition,
                CheckOutcome.INCONCLUSIVE,
                detail=f"could not poll process record: {exc}",
                checker=checker,
                now=now,
                reason="read_error",
                evidence={"process_id": params.process_id},
            )
        if status is None:
            return _evaluation(
                postcondition,
                CheckOutcome.INCONCLUSIVE,
                detail=f"no process record: {params.process_id}",
                checker=checker,
                now=now,
                reason="no_record",
                evidence={"process_id": params.process_id},
            )
        state = status.get("state")
        if not isinstance(state, str):
            return _evaluation(
                postcondition,
                CheckOutcome.INCONCLUSIVE,
                detail=f"process record has no state: {params.process_id}",
                checker=checker,
                now=now,
                reason="unknown_state",
                evidence={"process_id": params.process_id},
            )
        if state == "running":
            return _evaluation(
                postcondition,
                CheckOutcome.PASSED,
                detail=f"process is running: {params.process_id}",
                checker=checker,
                now=now,
                reason="ok",
                evidence={"process_id": params.process_id, "state": state},
            )
        if state in BRIDGE_TERMINAL_STATES:
            return _evaluation(
                postcondition,
                CheckOutcome.FAILED,
                detail=f"process is final ({state}): {params.process_id}",
                checker=checker,
                now=now,
                reason="process_final",
                evidence={"process_id": params.process_id, "state": state},
            )
        if state in BRIDGE_TRANSITIONAL_STATES:
            return _evaluation(
                postcondition,
                CheckOutcome.INCONCLUSIVE,
                detail=f"process is not final yet ({state}): {params.process_id}",
                checker=checker,
                now=now,
                reason="not_final",
                evidence={"process_id": params.process_id, "state": state},
            )
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"unknown process state ({state}): {params.process_id}",
            checker=checker,
            now=now,
            reason="unknown_state",
            evidence={"process_id": params.process_id, "state": state},
        )
    checker = "desktop_verify"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    assert params.pid is not None and params.create_time is not None
    try:
        alive, create_time = reader.os_process_running(params.pid)
    except PermissionError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"cannot inspect process {params.pid}: {exc}",
            checker=checker,
            now=now,
            reason="access_denied",
            evidence={"pid": params.pid},
        )
    except OSError as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not inspect process {params.pid}: {exc}",
            checker=checker,
            now=now,
            reason="read_error",
            evidence={"pid": params.pid},
        )
    if not alive:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"process is not running: pid {params.pid}",
            checker=checker,
            now=now,
            reason="pid_dead",
            evidence={"pid": params.pid},
        )
    if create_time is None or (
        abs(create_time - params.create_time) > PROCESS_CREATE_TIME_TOLERANCE_S
    ):
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"stale pid {params.pid}: create_time mismatch",
            checker=checker,
            now=now,
            reason="stale_pid",
            evidence={"pid": params.pid, "create_time": create_time},
        )
    return _evaluation(
        postcondition,
        CheckOutcome.PASSED,
        detail=f"process is running: pid {params.pid}",
        checker=checker,
        now=now,
        reason="ok",
        evidence={"pid": params.pid, "create_time": create_time},
    )


def evaluate_process_exit(
    params: ProcessExitParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """The bridge record must be final with the expected exit code."""
    checker = "bridge_poll"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    try:
        status = reader.process_status(params.process_id)
    except (OSError, ValueError) as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not poll process record: {exc}",
            checker=checker,
            now=now,
            reason="read_error",
            evidence={"process_id": params.process_id},
        )
    if status is None:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"no process record: {params.process_id}",
            checker=checker,
            now=now,
            reason="no_record",
            evidence={"process_id": params.process_id},
        )
    state = status.get("state")
    if not isinstance(state, str):
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"process record has no state: {params.process_id}",
            checker=checker,
            now=now,
            reason="unknown_state",
            evidence={"process_id": params.process_id},
        )
    if state not in BRIDGE_TERMINAL_STATES:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"process is not final yet ({state}): {params.process_id}",
            checker=checker,
            now=now,
            reason="not_final",
            evidence={"process_id": params.process_id, "state": state},
        )
    exit_code = status.get("exit_code")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        exit_code = None
    if exit_code == params.expected_exit_code:
        return _evaluation(
            postcondition,
            CheckOutcome.PASSED,
            detail=f"process exited {exit_code}: {params.process_id}",
            checker=checker,
            now=now,
            reason="ok",
            evidence={"process_id": params.process_id, "state": state, "exit_code": exit_code},
        )
    return _evaluation(
        postcondition,
        CheckOutcome.FAILED,
        detail=f"process {state} with exit_code={exit_code}, expected {params.expected_exit_code}",
        checker=checker,
        now=now,
        reason="exit_code_mismatch",
        evidence={"process_id": params.process_id, "state": state, "exit_code": exit_code},
    )


def evaluate_service_state(
    params: ServiceStateParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """A systemd unit must be in the expected ActiveState."""
    checker = "systemd_status"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    try:
        status = reader.service_status(params.unit, params.scope)
    except (OSError, ValueError) as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"could not read unit status: {exc}",
            checker=checker,
            now=now,
            reason="reader_error",
            evidence={"unit": params.unit, "scope": params.scope},
        )
    if status is None:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"unit not found: {params.scope}:{params.unit}",
            checker=checker,
            now=now,
            reason="unit_not_found",
            evidence={"unit": params.unit, "scope": params.scope},
        )
    active_state = status.get("active_state")
    if not isinstance(active_state, str) or not active_state:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"unit status has no ActiveState: {params.scope}:{params.unit}",
            checker=checker,
            now=now,
            reason="reader_error",
            evidence={"unit": params.unit, "scope": params.scope},
        )
    if active_state == params.expected_active_state:
        return _evaluation(
            postcondition,
            CheckOutcome.PASSED,
            detail=f"unit is {active_state}: {params.scope}:{params.unit}",
            checker=checker,
            now=now,
            reason="ok",
            evidence={
                "unit": params.unit,
                "scope": params.scope,
                "active_state": active_state,
            },
        )
    return _evaluation(
        postcondition,
        CheckOutcome.FAILED,
        detail=(
            f"unit is {active_state}, expected {params.expected_active_state}: "
            f"{params.scope}:{params.unit}"
        ),
        checker=checker,
        now=now,
        reason="state_mismatch",
        evidence={
            "unit": params.unit,
            "scope": params.scope,
            "active_state": active_state,
        },
    )


def browser_state_params_from_declaration(
    declaration: dict[str, JSONValue],
) -> BrowserStateParams | None:
    """Build typed ``browser_state`` params from a legacy declaration.

    Returns ``None`` when the declaration is malformed (missing
    entity/expect, wrong types, ``before`` of the wrong type) — the
    caller treats that as "not confirmed", exactly like the legacy
    ``_browser_state_confirms``.
    """
    entity_id = declaration.get("entity_id")
    expect = declaration.get("expect")
    if not isinstance(entity_id, str) or not entity_id or not isinstance(expect, dict):
        return None
    kind = expect.get("kind")
    if kind not in ("text", "count"):
        return None
    after = expect.get("after")
    has_before = "before" in expect
    before = expect.get("before")
    try:
        return BrowserStateParams(
            entity_id=entity_id,
            kind=kind,
            after=after,  # type: ignore[arg-type]
            before=before,
            has_before=has_before,
        )
    except ValueError:
        return None


def evaluate_browser_state(
    params: BrowserStateParams,
    facts: Sequence[dict[str, JSONValue]],
    *,
    postcondition: Postcondition | None = None,
    now: float | None = None,
) -> PostconditionEvaluation:
    """E02-T10 contract B, as a typed checker.

    The tool is the witness (it captures ``browser_state`` facts
    before/after the action); the verifier is the judge (exact match
    between the declared expectation and the facts, plus proven
    change: ``before`` did not already match ``expect.after``).
    Absence of facts is a fail
    (``browser_postcondition_unconfirmed``), exactly as before.
    Text facts carry ``expectation_sha256`` — the digest of the
    expectation the tool compared against; the checker requires it to
    equal the postcondition's own revision, so facts taken for one
    expectation cannot confirm a postcondition declaring another.
    The ``count`` kind only supports expecting exactly one element
    (``after == 1``), as in the legacy contract.
    """
    at = time.monotonic() if now is None else now
    checker = "browser_facts"
    if postcondition is None:
        postcondition = Postcondition(
            kind=PostconditionKind.BROWSER_STATE, params=params
        ).with_revision()
    evidence: dict[str, JSONValue] = {"entity_id": params.entity_id, "kind": params.kind}

    def failed(detail: str) -> PostconditionEvaluation:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=detail,
            checker=checker,
            now=at,
            reason="browser_postcondition_unconfirmed",
            evidence=evidence,
        )

    matching = [
        fact
        for fact in facts
        if isinstance(fact, dict)
        and fact.get("entity_id") == params.entity_id
        and fact.get("kind") == params.kind
    ]
    if not matching:
        return failed(f"no browser_state facts for entity {params.entity_id}")
    if params.kind == "text":
        # One run may capture the same entity twice (Pending -> Saved
        # -> Published): the first fact is not necessarily the one
        # proving this expectation. Select the first fact whose digest
        # binds the expectation this postcondition declares; facts for
        # another expectation (or unbound facts) prove nothing.
        bound = [
            fact for fact in matching if fact.get("expectation_sha256") == postcondition.revision()
        ]
        if not bound:
            return failed(
                f"browser facts were captured for a different expectation: {params.entity_id}"
            )
        fact = bound[0]
        # The booleans below were computed by the tool against the
        # expectation in its own declaration. A missing digest fails
        # closed: unbound facts prove nothing.
        if fact.get("after_matches_after") is not True:
            return failed(f"browser text does not match the expected state: {params.entity_id}")
        if fact.get("before_matches_after") is not False:
            return failed(f"browser text already matched before the action: {params.entity_id}")
        if params.has_before and fact.get("before_matches_expected_before") is not True:
            return failed(
                f"browser text before does not match the declared before: {params.entity_id}"
            )
        return _evaluation(
            postcondition,
            CheckOutcome.PASSED,
            detail=f"browser state confirmed for entity {params.entity_id}",
            checker=checker,
            now=at,
            reason="ok",
            evidence=evidence,
        )
    # kind == "count": the contract expects exactly one element.
    # Count facts carry no expectation digest, so the first capture
    # is the transition proof (0 -> 1); a later capture would be 1
    # -> 1 and cannot prove the change.
    fact = matching[0]
    if isinstance(params.after, bool) or not isinstance(params.after, int) or params.after != 1:
        return failed(f"browser count expects exactly one element: {params.entity_id}")
    before_count = fact.get("before_count")
    after_count = fact.get("after_count")
    if (
        isinstance(before_count, bool)
        or not isinstance(before_count, int)
        or isinstance(after_count, bool)
        or not isinstance(after_count, int)
    ):
        return failed(f"browser count facts are not integers: {params.entity_id}")
    if before_count != 0 or after_count != params.after:
        return failed(
            f"browser count is {after_count}, expected {params.after}: {params.entity_id}"
        )
    if params.has_before and before_count != params.before:
        return failed(
            f"browser count before is {before_count}, expected {params.before}: {params.entity_id}"
        )
    return _evaluation(
        postcondition,
        CheckOutcome.PASSED,
        detail=f"browser state confirmed for entity {params.entity_id}",
        checker=checker,
        now=at,
        reason="ok",
        evidence=evidence,
    )


def evaluate_custom_command(
    params: CustomCommandParams,
    reader: PostconditionReader | None,
    *,
    postcondition: Postcondition,
    now: float,
) -> PostconditionEvaluation:
    """``custom_command`` v1: only the canonical check, via the executor."""
    checker = "canonical_check"
    if reader is None:
        return _blocked(postcondition, now=now, checker=checker)
    # Local import: verifier_runtime imports this module for
    # verify_postconditions; the canonical command stays its single
    # source of truth.
    from core.mwv.verifier_runtime import CANONICAL_CHECK_COMMAND

    if tuple(params.argv) != tuple(CANONICAL_CHECK_COMMAND):
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail="custom_command must be the canonical check (make -B -r -f Makefile check)",
            checker=checker,
            now=now,
            reason="non_canonical_command",
            evidence={"argv": list(params.argv)},
        )
    try:
        # The declared criterion hash travels to the reader, so the
        # executor enforces it at execution: without it, one PASSED
        # would confirm two postconditions bound to different
        # Makefile hashes.
        result = reader.canonical_check(criterion_sha256=params.criterion_sha256)
    except (OSError, ValueError) as exc:
        return _evaluation(
            postcondition,
            CheckOutcome.INCONCLUSIVE,
            detail=f"canonical check could not run: {exc}",
            checker=checker,
            now=now,
            reason="canonical_check_error",
            evidence={"argv": list(params.argv)},
        )
    evidence: dict[str, JSONValue] = {
        "argv": list(params.argv),
        "exit_code": result.exit_code,
    }
    if result.status is VerificationStatus.PASSED:
        return _evaluation(
            postcondition,
            CheckOutcome.PASSED,
            detail="canonical check passed",
            checker=checker,
            now=now,
            reason="ok",
            evidence=evidence,
        )
    if result.status is VerificationStatus.FAILED:
        return _evaluation(
            postcondition,
            CheckOutcome.FAILED,
            detail=f"canonical check failed (exit_code={result.exit_code})",
            checker=checker,
            now=now,
            reason="check_failed",
            evidence=evidence,
        )
    return _evaluation(
        postcondition,
        CheckOutcome.INCONCLUSIVE,
        detail=f"canonical check errored: {result.error or result.fail_type or 'unknown'}",
        checker=checker,
        now=now,
        reason="canonical_check_error",
        evidence=evidence,
    )


def collect_browser_state_facts(
    results: Sequence[ToolResult],
) -> list[dict[str, JSONValue]]:
    """Collect ``browser_state`` fact dicts from tool results.

    Only the facts travel to the typed channel — never the tool's
    ``ok`` flag or its ``verified`` field.
    """
    facts: list[dict[str, JSONValue]] = []
    for result in results:
        data = result.data
        if not isinstance(data, dict):
            continue
        fact = data.get("browser_state")
        if isinstance(fact, dict):
            facts.append(fact)
    return facts
