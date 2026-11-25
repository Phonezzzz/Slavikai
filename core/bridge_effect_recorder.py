"""Record bridge call outcomes into the effect journal (E03-T08).

The recorder classifies one bridge call — success, typed error,
or transport failure — into the journal's outcome trichotomy
(rev. 2 §1) and persists it through :class:`BridgeEffectStore`.

Outcome trichotomy (owner condition 2 is about the *resolution*
of the third case):

- ``effect``: the answer carried a valid effect block — a real
  mutation, identified by its daemon emission identity
  ``(boot_id, daemon_seq)``.
- ``no_effect``: the call provably changed nothing (a refusal
  issued before any mutation, a chunk/abort/dry-run, a kill of an
  already-final process, ...).
- ``unknown``: the call may or may not have mutated — the answer
  was lost, the effect block is missing where one is mandatory,
  or the error does not prove the call was refused pre-mutation.
  Fail-closed: anything not proven ``no_effect`` is ``unknown``.

Classification rules are checked handler-by-handler against
``host_bridge/fs.py`` and ``host_bridge/process.py`` on main
``67c4e81`` — a code is ``no_effect`` only when its handler
raises it before any publish syscall / spawn / signal. The table
is exhaustive by construction: an unlisted code is ``unknown``.

Two retry rules (rev. 2 §1) need the message, not just the code:
the daemon raises ``invalid_params`` with the message
``"unknown write_id"`` for an unknown chunk-session id (verified
in ``host_bridge/fs.py`` — there is no distinct wire code), and
``process_not_found`` for an unknown process. On a first call
both are ``no_effect`` (nothing existed to mutate); on a retry
they are ``unknown`` (the first attempt may have mutated).

Metadata-only (owner condition 6): the persisted ``params``
projection is a fixed per-capability field whitelist — file
content, patch text, chunk data, process args, environment and
stdout/stderr can never reach the journal, because they are not
in the list. ``sanitize_record`` runs on top as defense-in-depth
only (with an explicit ``max_bytes`` above the projection's
worst case, so its 4 KiB default truncation can never destroy
the projection). The claim guarantees the exclusion of those
categories — NOT the absolute absence of any sensitive value in
string metadata (a path or run id may itself contain one).

Observations attach generically (rev. 2 §6): the recorder
verifies the observation's subject belongs to the effect (the
effect's own subject, or one of its processes for a stop) and
that the payload is metadata-only. Sources ``poll-loop`` and
``manual-recheck`` are fully implemented. Source
``e05-t02-binder`` is accepted as a tag, but the
binder-outcome -> observation field mapping is NOT implemented
here — checkpoint C-V1 is open (the E05-T02 schema is still
awaiting owner approval), so the mapping arrives with the
binder; the seam is marked, not silently stubbed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Final, cast

from core.bridge_effect_store import BridgeEffectStore
from host_bridge.effect import EFFECT_KINDS, validate_effect_shape
from shared.models import JSONValue
from shared.sanitize import sanitize_record

# Outcome literals (also the store's CHECK constraint).
OUTCOME_EFFECT: Final[str] = "effect"
OUTCOME_NO_EFFECT: Final[str] = "no_effect"
OUTCOME_UNKNOWN: Final[str] = "unknown"

# Unknown-reason literals (free-form, but kept to this set so the
# journal stays greppable).
REASON_TIMEOUT: Final[str] = "timeout"
REASON_TRANSPORT: Final[str] = "transport_error"
REASON_MISSING_EFFECT: Final[str] = "missing_effect_block"
REASON_INTERNAL_ERROR: Final[str] = "internal_error"
REASON_SPAWN_FAILED: Final[str] = "spawn_failed"
REASON_IO_ERROR: Final[str] = "io_error"
REASON_PROCESS_NOT_FOUND_RETRY: Final[str] = "process_not_found_on_retry"
REASON_UNKNOWN_WRITE_ID_RETRY: Final[str] = "unknown_write_id_on_retry"
REASON_MALFORMED_RESPONSE: Final[str] = "malformed_response"
REASON_UNCLASSIFIED: Final[str] = "unclassified_error"

# Error codes whose refusal provably precedes any mutation
# (handler-by-handler check on main 67c4e81). Anything not in
# this set (and not in the retry-unknown rules below) is unknown.
_NO_EFFECT_CODES: Final[frozenset[str]] = frozenset(
    {
        # Pure validation, raised before any state change.
        "invalid_params",
        "unknown_profile",
        "invalid_args",
        "binding_mismatch",
        "invalid_patch",
        "invalid_move",
        "is_a_directory",
        "not_a_file",
        "path_outside_root",
        "file_too_large",
        "binary_file",
        "invalid_utf8",
        "content_too_large",
        "subtree_too_large",
        # Root/path validation in host_bridge/fs.py (_resolve /
        # _parse_rel_path / _open_parent): every raise site runs
        # before any mutation, so the refusal proves no effect
        # (Codex round 5, finding 2).
        "invalid_path",
        "unknown_root",
        "root_read_only",
        # Pre-commit refusals: the check runs before the publish
        # syscall / spawn / signal.
        "run_stopped",
        "patch_mismatch",
        "already_exists",
        "trash_unavailable",
        "process_unavailable",
        "too_many_processes",
        "write_incomplete",
        # The effect could not be built before the mutation
        # (owner condition 5): the publish never ran.
        "effect_build_failed",
        # The request never reached a handler: nothing mutated.
        "unknown_capability",
        "unsupported_version",
        "malformed_request",
        "frame_too_large",
        "invalid_request_id",
    }
)

# Codes whose first call is no_effect but whose RETRY is unknown.
_RETRY_UNKNOWN_CODES: Final[frozenset[str]] = frozenset({"process_not_found"})
# The daemon raises invalid_params with this message for an
# unknown chunk-session id (no distinct wire code exists).
_UNKNOWN_WRITE_ID_MESSAGE: Final[str] = "unknown write_id"

# Final states of a process record (host_bridge/process.py
# _finalize): a kill answer in one of these states, without an
# effect block, proves this call requested nothing.
_FINAL_STATES: Final[frozenset[str]] = frozenset({"exited", "killed", "timeout", "spawn_failed"})

# Metadata-only params projection per capability (rev. 2 §4).
# Deliberately NOT in the list: content, patch, args, env,
# stdout/stderr, chunk data.
_PARAM_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "host.fs.write": (
        "root_id",
        "path",
        "mode",
        "offset",
        "write_id",
        "commit",
        "abort",
        "create_parents",
    ),
    "host.fs.patch": ("root_id", "path", "dry_run"),
    "host.fs.move": ("root_id", "path", "dest_root_id", "dest_path", "overwrite"),
    "host.fs.delete": ("root_id", "path", "recursive"),
    "host.process.run": ("profile_id", "timeout_s"),
    "host.process.kill": ("process_id",),
    "host.process.stop": ("run_id",),
}
_MAX_PARAM_STRING_CHARS: Final[int] = 4096
# sanitize_record's default max_bytes (4 KiB) would truncate a
# projection with two long paths; the whitelist already bounds
# the projection (~8.5 KiB worst case), so the regex pass runs
# with a ceiling above it and can never trigger truncation.
_SANITIZE_MAX_BYTES: Final[int] = 65536

# Observation sources. "poll-loop" and "manual-recheck" are fully
# implemented; "e05-t02-binder" is a tag-only seam (C-V1 open).
OBSERVATION_SOURCE_POLL_LOOP: Final[str] = "poll-loop"
OBSERVATION_SOURCE_MANUAL: Final[str] = "manual-recheck"
OBSERVATION_SOURCE_BINDER: Final[str] = "e05-t02-binder"
_MAX_OBSERVATION_DETAIL_CHARS: Final[int] = 1024


@dataclass(frozen=True)
class RecordOutcome:
    """What the journal did with one bridge call."""

    seq: int | None  # None when persistence failed (audit_gap)
    outcome: str  # 'effect' | 'no_effect' | 'unknown'
    audit_gap: bool = False
    duplicate: bool = False  # an effect with this identity was already recorded
    resolved_unknown_seqs: list[int] = field(default_factory=list)


class BridgeEffectRecorder:
    """Classify bridge call outcomes and persist them."""

    def __init__(self, store: BridgeEffectStore) -> None:
        self._store = store

    # -- success -----------------------------------------------------------

    def record_success(
        self,
        *,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params: dict[str, object] | None,
        result: dict[str, object],
    ) -> RecordOutcome:
        effect = result.get("effect")
        if isinstance(effect, dict) and validate_effect_shape(effect) is None:
            expected_kind = EFFECT_KINDS.get(capability)
            if expected_kind is not None and effect.get("kind") == expected_kind:
                return self._persist(
                    capability=capability,
                    request_id=request_id,
                    principal_id=principal_id,
                    run_id=run_id,
                    params=params,
                    effect_json=json.dumps(effect, ensure_ascii=False, sort_keys=True),
                    daemon_boot_id=effect["boot_id"],
                    daemon_seq=effect["seq"],
                    outcome=OUTCOME_EFFECT,
                    error_code=None,
                    unknown_reason=None,
                )
        # No (valid) effect block: the result must prove no
        # mutation happened, or the outcome is unknown.
        outcome, unknown_reason = self._classify_success_without_effect(capability, result)
        return self._persist(
            capability=capability,
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            effect_json=None,
            daemon_boot_id=None,
            daemon_seq=None,
            outcome=outcome,
            error_code=None,
            unknown_reason=unknown_reason,
        )

    def _classify_success_without_effect(
        self, capability: str, result: dict[str, object]
    ) -> tuple[str, str | None]:
        """(outcome, unknown_reason) for a success without an
        effect block. ``no_effect`` only where the result itself
        proves this call changed nothing."""
        if capability == "host.fs.write":
            if result.get("aborted") is True:
                return OUTCOME_NO_EFFECT, None
            if result.get("committed") is True:
                # A commit MUST carry an effect; its absence is a
                # daemon bug — unknown, never assumed applied.
                return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
            # Chunk response (write_id, no commit): nothing published.
            return OUTCOME_NO_EFFECT, None
        if capability == "host.fs.patch":
            if result.get("dry_run") is True:
                return OUTCOME_NO_EFFECT, None
            # applied=True without an effect is a missing block.
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        if capability == "host.fs.move":
            if result.get("moved") is True:
                return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        if capability == "host.fs.delete":
            if result.get("recoverable") is True:
                return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        if capability == "host.process.run":
            # A run result without an effect: either a retry of a
            # failed spawn (state spawn_failed) or a daemon bug.
            # spawn_failed does NOT prove no mutation — the
            # spawn_destroyed path means the command may have
            # executed — so fail-closed: unknown.
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        if capability == "host.process.kill":
            state = result.get("state")
            if isinstance(state, str) and state in _FINAL_STATES:
                # Final state, no effect: this call requested
                # nothing (the process died on its own, or an
                # earlier kill already finished it).
                return OUTCOME_NO_EFFECT, None
            # A non-final state without an effect is impossible by
            # construction (a kill request always carries its
            # effect) — unknown, never assumed.
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        if capability == "host.process.stop":
            # STOP always emits an effect, even for an unknown run
            # (processes: [], mark_set: true). Its absence is a
            # missing block.
            return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT
        return OUTCOME_UNKNOWN, REASON_MISSING_EFFECT

    # -- errors and transport failures --------------------------------------

    def record_error(
        self,
        *,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params: dict[str, object] | None,
        code: str,
        message: str,
    ) -> RecordOutcome:
        try:
            is_retry = self._store.has_request_id(request_id)
        except Exception:
            # The journal cannot be read (locked, schema
            # unavailable): the history is unverifiable, so fail
            # closed — classify as a retry. A potential retry must
            # never be classified as a proven no_effect. _persist
            # below still converts a persist failure to audit_gap,
            # so the typed bridge error is never replaced by a
            # journal error (Codex round 1, finding 3).
            is_retry = True
        outcome, unknown_reason = self._classify_error(code, message, is_retry=is_retry)
        return self._persist(
            capability=capability,
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            effect_json=None,
            daemon_boot_id=None,
            daemon_seq=None,
            outcome=outcome,
            error_code=code,
            unknown_reason=unknown_reason,
        )

    def record_transport_failure(
        self,
        *,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params: dict[str, object] | None,
        reason: str,
    ) -> RecordOutcome:
        """A call whose answer was lost (timeout, closed
        connection, malformed response): the outcome is unknown by
        definition — the publish may already have run."""
        return self._persist(
            capability=capability,
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            effect_json=None,
            daemon_boot_id=None,
            daemon_seq=None,
            outcome=OUTCOME_UNKNOWN,
            error_code=None,
            unknown_reason=reason,
        )

    @staticmethod
    def _classify_error(code: str, message: str, *, is_retry: bool) -> tuple[str, str | None]:
        if code in _RETRY_UNKNOWN_CODES:
            if is_retry:
                return OUTCOME_UNKNOWN, REASON_PROCESS_NOT_FOUND_RETRY
            return OUTCOME_NO_EFFECT, None
        if code == "invalid_params" and _UNKNOWN_WRITE_ID_MESSAGE in message and is_retry:
            return OUTCOME_UNKNOWN, REASON_UNKNOWN_WRITE_ID_RETRY
        if code in _NO_EFFECT_CODES:
            return OUTCOME_NO_EFFECT, None
        reason = {
            "request_timeout": REASON_TIMEOUT,
            "connection_closed": REASON_TRANSPORT,
            "malformed_response": REASON_MALFORMED_RESPONSE,
            "internal_error": REASON_INTERNAL_ERROR,
            "spawn_failed": REASON_SPAWN_FAILED,
            "io_error": REASON_IO_ERROR,
        }.get(code, REASON_UNCLASSIFIED)
        return OUTCOME_UNKNOWN, reason

    # -- observations -------------------------------------------------------

    def attach_observation(
        self,
        *,
        effect_seq: int,
        source: str,
        source_ref: str | None,
        payload: dict[str, object],
    ) -> int:
        """Attach an observation to an effect (generic mechanism).

        The payload is ``{subject, observed_at, state, detail?}``
        — metadata only, bounded. The subject must belong to the
        effect: its own subject, or one of its processes for a
        stop effect. Returns the observation id (idempotent: the
        same (effect, source, payload) returns the existing id).
        Raises ``ValueError`` on a bad payload, an unknown effect,
        or a subject that does not belong to the effect.
        """
        subject = payload.get("subject")
        observed_at = payload.get("observed_at")
        state = payload.get("state")
        detail = payload.get("detail")
        if not isinstance(subject, str) or not subject:
            raise ValueError("observation subject must be a non-empty string")
        if not isinstance(observed_at, str) or not observed_at:
            raise ValueError("observation observed_at must be a non-empty string")
        if not isinstance(state, str) or not state:
            raise ValueError("observation state must be a non-empty string")
        if detail is not None and not isinstance(detail, str):
            raise ValueError("observation detail must be a string")
        if isinstance(detail, str) and len(detail) > _MAX_OBSERVATION_DETAIL_CHARS:
            raise ValueError("observation detail exceeds the length limit")
        unknown_keys = set(payload) - {"subject", "observed_at", "state", "detail"}
        if unknown_keys:
            raise ValueError(f"observation has unknown fields: {sorted(unknown_keys)}")
        record = self._store.get_record(effect_seq)
        if record is None:
            raise ValueError(f"unknown effect_seq {effect_seq}")
        if subject not in self._effect_subjects(record):
            raise ValueError(
                f"observation subject {subject!r} does not belong to effect {effect_seq}"
            )
        observation_json = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        observation_sha256 = hashlib.sha256(observation_json.encode("utf-8")).hexdigest()
        return self._store.attach_observation(
            effect_seq=effect_seq,
            source=source,
            source_ref=source_ref,
            observation_json=observation_json,
            observation_sha256=observation_sha256,
        )

    @staticmethod
    def _effect_subjects(record: dict[str, object]) -> set[str]:
        """Subjects an observation may be about for this effect:
        the effect's own subject, plus — for a stop effect — every
        process it lists."""
        subjects: set[str] = set()
        effect = record.get("effect")
        if not isinstance(effect, dict):
            return subjects
        kind = effect.get("kind")
        if kind in ("fs.write", "fs.patch", "fs.move", "fs.delete"):
            path = effect.get("path")
            if isinstance(path, str):
                subjects.add(path)
        elif kind in ("process.run", "process.kill"):
            process_id = effect.get("process_id")
            if isinstance(process_id, str):
                subjects.add(process_id)
        elif kind == "process.stop":
            run_id = effect.get("run_id")
            if isinstance(run_id, str):
                subjects.add(run_id)
            processes = effect.get("processes")
            if isinstance(processes, list):
                for entry in processes:
                    if isinstance(entry, dict):
                        process_id = entry.get("process_id")
                        if isinstance(process_id, str):
                            subjects.add(process_id)
        return subjects

    # -- persistence ---------------------------------------------------------

    def _project_params(
        self, capability: str, params: dict[str, object] | None
    ) -> dict[str, object]:
        """Metadata-only projection of the call params: only the
        whitelisted fields, only str/int/bool values, strings
        truncated. Content, patch text, args and env can never
        reach the journal — they are not in the list."""
        allowed = _PARAM_FIELDS.get(capability, ())
        projected: dict[str, JSONValue] = {}
        if isinstance(params, dict):
            for key in allowed:
                value = params.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    projected[key] = value[:_MAX_PARAM_STRING_CHARS]
                elif isinstance(value, bool) or isinstance(value, int):
                    projected[key] = value
        # Defense-in-depth only (rev. 2 §4): the structural
        # guarantee is the whitelist above, never this regex pass.
        sanitized = sanitize_record(projected, max_bytes=_SANITIZE_MAX_BYTES)
        kept: dict[str, object] = {}
        for key, value in sanitized.items():
            if isinstance(value, (str, int, bool)):
                kept[key] = value
            elif isinstance(value, dict):
                # A whitelisted metadata string longer than 256
                # bytes was truncated by sanitize_record into a
                # preview dict (Codex round 4, finding 4). Dropping
                # the field entirely loses the only pointer to the
                # file on timeout/transport-error paths, where no
                # effect block exists. Keep the bounded preview
                # string instead -- it is still metadata only.
                preview = value.get("preview")
                if isinstance(preview, str):
                    kept[key] = preview
        return kept

    def _persist(
        self,
        *,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params: dict[str, object] | None,
        effect_json: str | None,
        daemon_boot_id: object,
        daemon_seq: object,
        outcome: str,
        error_code: str | None,
        unknown_reason: str | None,
    ) -> RecordOutcome:
        params_json = json.dumps(
            self._project_params(capability, params), ensure_ascii=False, sort_keys=True
        )
        try:
            stored = self._store.record_call(
                capability=capability,
                request_id=request_id,
                principal_id=principal_id,
                run_id=run_id,
                params_json=params_json,
                effect_json=effect_json,
                daemon_boot_id=daemon_boot_id if isinstance(daemon_boot_id, str) else None,
                daemon_seq=daemon_seq if isinstance(daemon_seq, int) else None,
                outcome=outcome,
                error_code=error_code,
                unknown_reason=unknown_reason,
            )
        except Exception:
            # Bounded persistence failed (owner condition 4): the
            # journal has a gap, but the bridge call itself must
            # not fail because of it — audit_gap, not an exception.
            return RecordOutcome(seq=None, outcome=outcome, audit_gap=True)
        return RecordOutcome(
            seq=cast(int, stored["seq"]),
            outcome=str(stored["outcome"]),
            duplicate=bool(stored["duplicate"]),
            resolved_unknown_seqs=[
                int(seq) for seq in cast(list[int], stored["resolved_unknown_seqs"])
            ],
        )
