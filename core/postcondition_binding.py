"""Effect→postcondition binding (E05-T02).

Derives typed postconditions from what state-changing tools actually
did — the check side of the tool→effect→check mapping (T01 Q7).
Derivation lives in one place, in trusted code: the binder re-derives
a suggestion from ``(tool name, call args, result)`` and never reads
model text, a tool's ``ok`` flag, or a ``verified: true`` field. A
suggestion is never proof: it is evaluated by the typed checker
against real state, exactly like any declared check.

Two mechanisms (schema ``slavikai-e05-t02``, §2):

- **Suggested** — :func:`suggest_postconditions` offers a candidate
  check for one successful state-changing call, per the derivation
  table below.
- **Expected** — :func:`bind_step_postconditions` /
  :func:`bind_run_postconditions` union the step/run caller's declared
  conditions with the suggestions. Expected wins per subject
  (``(kind, check_identity)``); a suggestion for a subject the caller
  already declared is dropped and recorded as superseded, never
  verified, never failed.

Fail-closed rules enforced here (schema §2.5, Q13):

- no suggestion from a failed call or a cancelled loop;
- no suggestion from a read tool (``capability == "read"``), even if
  the tool name is in the table;
- no ``file_content_matches`` suggestion when the expected content is
  empty or exceeds the reader cap — a check known to be
  INCONCLUSIVE is never suggested;
- minted identities are validated, never trusted blindly (a pid must
  be a positive int; a dead process at bind time suggests nothing);
- every returned postcondition is bound via ``with_revision()`` by
  this module — the binder is the trusted declarer.

Deviations from the schema's §1.3/§2.2 table, verified against the
code on ``origin/main`` (the schema was written on an older base):

- ``workspace_rename`` args are ``old_path``/``new_path`` and
  ``workspace_move`` args are ``from_path``/``to_path`` (the schema
  says ``path``/``source`` + ``destination``); the binder reads the
  result data first (same keys), then the call args.
- ``desktop_process`` launch result data *does* carry ``create_time``
  (the schema says it does not), but the binder deliberately ignores
  it and re-reads the live process through the reader at bind time:
  the read proves the process is alive at the acceptance moment, and
  the derivation does not depend on which wrapper returned the pid.
- ``workspace_rename``/``workspace_move``/``desktop_file_transfer``
  may move a *directory*; ``file_exists`` is regular-file-only, so a
  suggested ``file_exists(destination)`` would FAIL a good directory
  move. When a reader is available the binder stats the destination
  first: a directory destination suppresses the ``file_exists``
  suggestion (move/rename keep ``file_absent(source)``; copy keeps
  nothing). A path the reader cannot describe suppresses that check
  too — a check known to be INCONCLUSIVE is never suggested.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from core.mwv.verifier_runtime import VerifierRuntime
from core.postconditions import (
    FileAbsentParams,
    FileContentMatchesParams,
    FileExistsParams,
    PathDeniedError,
    Postcondition,
    PostconditionKind,
    PostconditionReader,
    ProcessRunningParams,
    ServiceStateParams,
    browser_state_params_from_declaration,
    check_identity,
    check_name,
    collect_browser_state_facts,
)
from core.run_outcome import Verification
from core.tool_loop import AgentToolLoopResult
from shared.models import JSONValue, ToolResult
from tools.desktop_system_tools import UNIT_RE
from tools.desktop_tools import MAX_DESKTOP_READ_BYTES
from tools.tool_registry import ToolCapability

GetCapability = Callable[[str], ToolCapability | None]
"""How the binder learns a tool's capability (``ToolRegistry.get_capability``)."""


@dataclass(frozen=True)
class StepPostconditionBinding:
    """Bound postconditions for one step, with the superseded record.

    ``bound`` is what the step is verified against (expected first,
    then browser-declared, then suggested). ``superseded`` are
    suggestions dropped because an expected condition covers the same
    subject; ``notes`` names each drop in human-readable form for the
    step result text.
    """

    bound: tuple[Postcondition, ...]
    superseded: tuple[Postcondition, ...]
    notes: tuple[str, ...] = ()


def _str_arg(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _path_from(result: ToolResult, args: dict[str, JSONValue], *keys: str) -> str | None:
    """Canonical path for a check: result data first, then call args."""
    data = result.data if isinstance(result.data, dict) else {}
    for key in keys:
        value = _str_arg(data.get(key))
        if value is not None:
            return value
    for key in keys:
        value = _str_arg(args.get(key))
        if value is not None:
            return value
    return None


def _describe_safely(reader: PostconditionReader | None, path: str) -> tuple[bool, bool] | None:
    """``(exists, is_regular_file)`` for ``path`` or ``None`` if unknowable.

    ``None`` covers a denied/unreadable path: the checker would call
    it INCONCLUSIVE, so the binder must not suggest a check on it (Q13).
    Only the reader's own failure modes are swallowed; anything else
    is a bug and must surface.
    """
    if reader is None:
        return None
    try:
        info = reader.describe_file(path)
    except (PathDeniedError, OSError, ValueError):
        return None
    return (info.exists, info.is_regular_file)


def _suggest_file_write(
    *, tool_name: str, args: dict[str, JSONValue], result: ToolResult
) -> list[Postcondition]:
    path = _path_from(result, args, "path")
    content = args.get("content")
    if path is None or not isinstance(content, str):
        return []
    checks = [
        Postcondition(
            kind=PostconditionKind.FILE_EXISTS,
            params=FileExistsParams(path=path),
            source=f"suggested:{tool_name}",
        ).with_revision()
    ]
    if len(content.encode("utf-8")) <= MAX_DESKTOP_READ_BYTES:
        checks.append(
            Postcondition(
                kind=PostconditionKind.FILE_CONTENT_MATCHES,
                params=FileContentMatchesParams(path=path, expected=content, match="exact"),
                source=f"suggested:{tool_name}",
            ).with_revision()
        )
    return checks


def _suggest_patch(*, args: dict[str, JSONValue], result: ToolResult) -> list[Postcondition]:
    if args.get("dry_run") is True or result.data.get("dry_run") is True:
        return []
    path = _path_from(result, args, "path")
    if path is None:
        return []
    checks = [
        Postcondition(
            kind=PostconditionKind.FILE_EXISTS,
            params=FileExistsParams(path=path),
            source="suggested:workspace_patch",
        ).with_revision()
    ]
    patched = result.data.get("content")
    # Empty patched content is a real bar (the file must be empty);
    # only the read cap suppresses the content check (Q13).
    if isinstance(patched, str) and len(patched.encode("utf-8")) <= MAX_DESKTOP_READ_BYTES:
        checks.append(
            Postcondition(
                kind=PostconditionKind.FILE_CONTENT_MATCHES,
                params=FileContentMatchesParams(path=path, expected=patched, match="exact"),
                source="suggested:workspace_patch",
            ).with_revision()
        )
    return checks


def _suggest_transfer(
    *,
    tool_name: str,
    args: dict[str, JSONValue],
    result: ToolResult,
    reader: PostconditionReader | None,
) -> list[Postcondition]:
    if tool_name == "workspace_rename":
        source = _path_from(result, args, "old_path")
        destination = _path_from(result, args, "new_path")
        operation = "move"
    elif tool_name == "workspace_move":
        source = _path_from(result, args, "from_path")
        destination = _path_from(result, args, "to_path")
        operation = "move"
    else:  # desktop_file_transfer
        source = _path_from(result, args, "source")
        destination = _path_from(result, args, "destination")
        raw_operation = result.data.get("operation", args.get("operation", "move"))
        operation = raw_operation if isinstance(raw_operation, str) else "move"
        operation = operation.lower()
    if source is None or destination is None or operation not in {"copy", "move", "rename"}:
        return []
    checks: list[Postcondition] = []
    described = _describe_safely(reader, destination) if reader is not None else None
    if reader is None:
        # No guard possible; evaluation without a reader is BLOCKED,
        # never a wrong FAILED — the table rule stands.
        dest_checkable = True
    elif described is None:
        # The reader cannot describe the destination: the check would
        # be INCONCLUSIVE, and a known-INCONCLUSIVE check is never
        # suggested (Q13) — it would block a DONE the other evidence
        # supports.
        dest_checkable = False
    elif described == (True, False):
        # A directory destination: file_exists is regular-file-only
        # and would FAIL a good move.
        dest_checkable = False
    else:
        dest_checkable = True
    if dest_checkable:
        checks.append(
            Postcondition(
                kind=PostconditionKind.FILE_EXISTS,
                params=FileExistsParams(path=destination),
                source=f"suggested:{tool_name}",
            ).with_revision()
        )
    if operation in {"move", "rename"}:
        source_described = _describe_safely(reader, source) if reader is not None else None
        if reader is None or source_described is not None:
            checks.append(
                Postcondition(
                    kind=PostconditionKind.FILE_ABSENT,
                    params=FileAbsentParams(path=source),
                    source=f"suggested:{tool_name}",
                ).with_revision()
            )
    return checks


def _suggest_delete(
    *, tool_name: str, args: dict[str, JSONValue], result: ToolResult
) -> list[Postcondition]:
    path = _path_from(result, args, "path")
    if path is None:
        return []
    return [
        Postcondition(
            kind=PostconditionKind.FILE_ABSENT,
            params=FileAbsentParams(path=path),
            source=f"suggested:{tool_name}",
        ).with_revision()
    ]


def _suggest_launch(
    *, args: dict[str, JSONValue], result: ToolResult, reader: PostconditionReader | None
) -> list[Postcondition]:
    operation = _str_arg(args.get("operation")) or _str_arg(result.data.get("operation"))
    if operation != "launch":
        return []
    pid_raw = result.data.get("pid")
    if isinstance(pid_raw, bool) or not isinstance(pid_raw, int) or pid_raw <= 0:
        return []
    if reader is None:
        return []
    try:
        alive, create_time = reader.os_process_running(pid_raw)
    except (PermissionError, OSError):
        # Cannot tell whether the process is alive: no suggestion (Q13).
        return []
    if not alive or create_time is None:
        return []
    return [
        Postcondition(
            kind=PostconditionKind.PROCESS_RUNNING,
            params=ProcessRunningParams(pid=pid_raw, create_time=create_time),
            source="suggested:desktop_process",
        ).with_revision()
    ]


def _suggest_systemd(*, args: dict[str, JSONValue]) -> list[Postcondition]:
    operation = args.get("operation")
    if operation not in {"start", "stop", "restart"}:
        return []
    unit = _str_arg(args.get("unit"))
    scope = args.get("scope")
    if unit is None or UNIT_RE.fullmatch(unit) is None:
        return []
    # Scope must be explicit in the call args — no default (the tool
    # defaults to "system", the binder does not).
    if scope not in {"user", "system"}:
        return []
    expected = "inactive" if operation == "stop" else "active"
    return [
        Postcondition(
            kind=PostconditionKind.SERVICE_STATE,
            params=ServiceStateParams(unit=unit, scope=scope, expected_active_state=expected),
            source="suggested:desktop_systemd",
        ).with_revision()
    ]


def suggest_postconditions(
    *,
    tool_name: str,
    args: dict[str, JSONValue],
    result: ToolResult,
    get_capability: GetCapability | None = None,
    reader: PostconditionReader | None = None,
) -> list[Postcondition]:
    """Derive candidate checks from one successful state-changing call.

    Returns ``[]`` unless ``result.ok`` and the ``(tool, operation)``
    is in the derivation table. Every returned postcondition is bound
    via ``with_revision()``. ``desktop_browser`` returns ``[]`` here:
    a click/submit declaration is an *expected* condition, collected
    by the binder from the call args (see :func:`_browser_expected`).
    With ``reader=None`` returns ``[]`` as well: no reader can
    evaluate any suggestion, so suggesting one would only ever
    evaluate BLOCKED and veto a run the caller never asked to verify
    (Q13 — never suggest a check known to be un-evaluable).
    """
    if not result.ok:
        return []
    if get_capability is not None and get_capability(tool_name) == "read":
        return []
    if reader is None:
        # No reader can evaluate any suggestion: every suggested
        # check reads real state (file/process/service), so with no
        # reader it could only ever evaluate BLOCKED — and a BLOCKED
        # typed check vetoes a run the caller never asked to verify
        # (regression caught by
        # tests/ui_api/test_desktop_mode.py::test_desktop_chat_approval_resumes_same_pipeline:
        # the UI desktop path runs without a reader, and the
        # suggested file_absent turned a verified delete into
        # UNVERIFIED). Expected conditions are unaffected — the
        # caller asked for those explicitly, and T01 evaluates them
        # BLOCKED with no reader. This is also Q13: never suggest a
        # check known to be un-evaluable.
        return []
    if tool_name in {"workspace_write", "workspace_create", "desktop_file_write"}:
        return _suggest_file_write(tool_name=tool_name, args=args, result=result)
    if tool_name == "workspace_patch":
        return _suggest_patch(args=args, result=result)
    if tool_name in {"workspace_rename", "workspace_move", "desktop_file_transfer"}:
        return _suggest_transfer(tool_name=tool_name, args=args, result=result, reader=reader)
    if tool_name in {"workspace_delete", "desktop_file_delete"}:
        return _suggest_delete(tool_name=tool_name, args=args, result=result)
    if tool_name == "desktop_process":
        return _suggest_launch(args=args, result=result, reader=reader)
    if tool_name == "desktop_systemd":
        return _suggest_systemd(args=args)
    return []


def _browser_expected(args: dict[str, JSONValue]) -> Postcondition | None:
    """The ``browser_state`` declaration as an expected condition.

    Declared before the effect (E02-T10 contract B), validated
    fail-closed by :func:`browser_state_params_from_declaration`;
    a malformed declaration is not a check at all.
    """
    declaration = args.get("browser_state")
    if not isinstance(declaration, dict):
        return None
    params = browser_state_params_from_declaration(declaration)
    if params is None:
        return None
    return Postcondition(
        kind=PostconditionKind.BROWSER_STATE,
        params=params,
        source="expected:browser_state",
    ).with_revision()


def _subject_key(postcondition: Postcondition) -> tuple[str, str]:
    return (postcondition.kind.value, check_identity(postcondition))


def _bind(
    *,
    expected: Sequence[Postcondition],
    browser_expected: Sequence[Postcondition],
    suggested: Sequence[Postcondition],
) -> StepPostconditionBinding:
    """Expected wins per subject; suggested fills undeclared subjects."""
    bound: list[Postcondition] = []
    covering: dict[tuple[str, str], Postcondition] = {}
    for pc in expected:
        key = _subject_key(pc)
        if key not in covering:
            covering[key] = pc
            bound.append(pc)
    for pc in browser_expected:
        key = _subject_key(pc)
        if key not in covering:
            covering[key] = pc
            bound.append(pc)
    superseded: list[Postcondition] = []
    notes: list[str] = []
    for pc in suggested:
        key = _subject_key(pc)
        if key in covering:
            superseded.append(pc)
            notes.append(
                "suggestion superseded by expected: "
                f"{check_name(pc)} (expected {check_name(covering[key])})"
            )
        else:
            covering[key] = pc
            bound.append(pc)
    return StepPostconditionBinding(
        bound=tuple(bound), superseded=tuple(superseded), notes=tuple(notes)
    )


def _collect_calls(
    loop_result: AgentToolLoopResult,
    *,
    get_capability: GetCapability | None,
    reader: PostconditionReader | None,
) -> tuple[list[Postcondition], list[Postcondition]]:
    """Suggestions + browser-declared expectations from one loop result.

    Reads ``loop_result.tool_calls`` only — never model text, never a
    tool's ``verified`` flag.
    """
    suggested: list[Postcondition] = []
    browser_expected: list[Postcondition] = []
    for executed in loop_result.tool_calls:
        call = executed.call
        args = call.arguments if isinstance(call.arguments, dict) else {}
        suggested.extend(
            suggest_postconditions(
                tool_name=call.name,
                args=args,
                result=executed.result,
                get_capability=get_capability,
                reader=reader,
            )
        )
        if call.name == "desktop_browser" and executed.result.ok:
            operation = args.get("operation")
            if operation in {"click", "submit"}:
                expected = _browser_expected(args)
                if expected is not None:
                    browser_expected.append(expected)
    return suggested, browser_expected


def _bind_step_detailed(
    *,
    step: dict[str, JSONValue],
    loop_result: AgentToolLoopResult,
    get_capability: GetCapability | None,
    reader: PostconditionReader | None,
) -> StepPostconditionBinding:
    raw = step.get("expected_postconditions")
    expected: list[Postcondition] = []
    if raw is not None:
        if not isinstance(raw, list):
            raise ValueError("expected_postconditions must be a list")
        for entry in raw:
            # Fail-closed: a malformed entry is a broken acceptance
            # bar, never a silently dropped one. Plans that passed
            # validate_approved_plan cannot reach this (it parses the
            # same field first).
            expected.append(Postcondition.from_json(entry).with_revision())
    suggested, browser_expected = _collect_calls(
        loop_result, get_capability=get_capability, reader=reader
    )
    return _bind(expected=expected, browser_expected=browser_expected, suggested=suggested)


def bind_step_postconditions(
    *,
    step: dict[str, JSONValue],
    loop_result: AgentToolLoopResult,
    get_capability: GetCapability | None = None,
    reader: PostconditionReader | None = None,
) -> list[Postcondition]:
    """Expected (from the step) + suggested (from the step's calls).

    Precedence (§2.4): expected wins per subject; suggested fills
    subjects the step did not declare. Returns ``[]`` when the step
    has neither.
    """
    binding = _bind_step_detailed(
        step=step, loop_result=loop_result, get_capability=get_capability, reader=reader
    )
    return list(binding.bound)


def bind_run_postconditions(
    *,
    expected: Sequence[Postcondition] | None = None,
    loop_result: AgentToolLoopResult,
    get_capability: GetCapability | None = None,
    reader: PostconditionReader | None = None,
) -> list[Postcondition]:
    """Expected (caller-passed, trusted server code) + suggestions.

    The direct-run twin of :func:`bind_step_postconditions`: the
    caller-passed conditions are the acceptance bar (Q9 — the model
    never declares them for its own direct run); suggestions fill the
    undeclared subjects. Re-binds every condition (idempotent for
    already-bound ones).
    """
    bound_expected = [pc.with_revision() for pc in expected] if expected else []
    suggested, browser_expected = _collect_calls(
        loop_result, get_capability=get_capability, reader=reader
    )
    binding = _bind(expected=bound_expected, browser_expected=browser_expected, suggested=suggested)
    return list(binding.bound)


class _StepVerificationHook:
    """The ``StepVerificationHook`` (E05-T01 seam) for plan steps.

    Binds the step's postconditions, evaluates them through
    ``VerifierRuntime.verify_postconditions``, returns the typed
    ``Verification`` — or ``None`` when the step has no checks
    (legacy UNVERIFIED behaviour kept). ``last_binding_notes`` carries
    the superseded-suggestion record for the step result text; the
    executor reads it after the call.
    """

    def __init__(
        self,
        *,
        reader: PostconditionReader | None,
        get_capability: GetCapability | None = None,
    ) -> None:
        self._reader = reader
        self._get_capability = get_capability
        self._verifier = VerifierRuntime()
        self._last_notes: tuple[str, ...] = ()

    @property
    def last_binding_notes(self) -> tuple[str, ...]:
        return self._last_notes

    def __call__(
        self, step: dict[str, JSONValue], loop_result: AgentToolLoopResult
    ) -> Verification | None:
        binding = _bind_step_detailed(
            step=step,
            loop_result=loop_result,
            get_capability=self._get_capability,
            reader=self._reader,
        )
        self._last_notes = binding.notes
        if not binding.bound:
            return None
        return self._verifier.verify_postconditions(
            binding.bound,
            self._reader,
            browser_facts=collect_browser_state_facts(
                [executed.result for executed in loop_result.tool_calls]
            ),
        )


def make_step_verification_hook(
    *,
    reader: PostconditionReader | None,
    get_capability: GetCapability | None = None,
) -> Callable[[dict[str, JSONValue], AgentToolLoopResult], Verification | None]:
    """Build the plan-step hook: bind + evaluate, ``None`` when no checks."""
    return _StepVerificationHook(reader=reader, get_capability=get_capability)
