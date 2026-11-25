"""Effect emission for the Owner Host Bridge (E03-T08).

Every mutation answer carries an ``effect`` block describing what
changed on the host — the audit trail the server journal (T08) is
built from. The block is metadata only: kind, sequence, boot id,
time, attribution and the kind's fixed fields (paths, ids, sizes,
states, ``content_sha256``). It never carries file content, patch
text, chunk data, process args, environment or stdout/stderr.

Identity (owner condition 1): one effect = one ``(boot_id, seq)``
pair, assigned by a single :class:`EffectEmitter` per daemon boot.
``request_id`` identifies the *logical request* (a retry re-uses it);
the *subject* (``process_id`` / ``run_id`` / path) is correlation
only and never a dedupe key — the recorder must not cancel a real
emitted effect because its subject matches an earlier one.

Frame budget (owner condition 5): the serialized result + effect
is checked BEFORE the publish syscall. Handlers build the effect
as a *projection* — placeholder ``seq``/``at`` at their maximum
serialized length, so a projection that fits the frame guarantees
the finalized effect fits too — run the check, and only then
mutate. :meth:`EffectEmitter.finalize` assigns the real ``seq``
and ``at`` once the mutation has succeeded; a projection discarded
by a failed publish consumes no sequence number, so a failed
mutation can never leave a hole in the daemon sequence.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime
from typing import Final

from host_bridge.protocol import MAX_FRAME_BYTES, encode_success

# Wire error code raised when the result + effect cannot fit the
# response frame. Raised BEFORE any mutation (condition 5); the
# recorder classifies it as ``no_effect`` — the publish never ran.
EFFECT_BUILD_FAILED: Final[str] = "effect_build_failed"

# Mutating capability -> effect ``kind``. No new capability is
# introduced in T08: the effect rides inside the mutation result,
# so the protocol version does not change.
EFFECT_KINDS: Final[dict[str, str]] = {
    "host.fs.write": "fs.write",
    "host.fs.patch": "fs.patch",
    "host.fs.move": "fs.move",
    "host.fs.delete": "fs.delete",
    "host.process.run": "process.run",
    "host.process.kill": "process.kill",
    "host.process.stop": "process.stop",
}

# Worst-case placeholders for the pre-mutation frame check. The
# serialized projection must be >= any real effect/result, so a
# check that passes can never be followed by a frame overflow.
_MAX_SEQ: Final[int] = 2**63 - 1
# Longest possible ISO timestamp with microseconds and a UTC
# offset (32 chars); also the placeholder for values not yet
# known before the mutation (e.g. a process's started_at).
AT_PLACEHOLDER: Final[str] = "9999-12-31T23:59:59.999999+00:00"
# Longest process state string ("spawn_failed", 12 chars); the
# run/kill/stop projections use it for the state they cannot know
# before the mutation.
LONGEST_STATE: Final[str] = "spawn_failed"
# Longest serialized exit_code (int32 minimum, 11 chars) and signal
# (systemd scopes use signal numbers up to 64, 2 chars).
MAX_INT_PLACEHOLDER: Final[int] = -2147483648
MAX_SIGNAL_PLACEHOLDER: Final[int] = 64
# Entries in the process.stop projection: the daemon keeps at most
# MAX_FINISHED_RECORDS (32) finished records plus MAX_RUNNING_PROCESSES
# (4) live ones, so one run can never have more than ~36 targets;
# 64 leaves a wide margin while staying far inside the frame budget.
STOP_PROJECTION_ENTRIES: Final[int] = 64


def stop_process_placeholder() -> dict[str, object]:
    """One worst-case process entry for the stop projection."""
    return {
        "process_id": "0" * 32,
        "state": LONGEST_STATE,
        "exit_code": MAX_INT_PLACEHOLDER,
        "signal": MAX_SIGNAL_PLACEHOLDER,
    }


def validate_effect_shape(effect: object) -> str | None:
    """Check an effect block's shape; ``None`` when valid, else why.

    Shared by the client (fail-closed: a malformed effect is a
    malformed response) and the recorder (a malformed effect is
    treated as a missing effect block — ``unknown``, never trusted).
    """
    if not isinstance(effect, dict):
        return "effect must be an object"
    kind = effect.get("kind")
    if not isinstance(kind, str) or not kind:
        return "effect kind must be a non-empty string"
    seq = effect.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        return "effect seq must be a positive integer"
    boot_id = effect.get("boot_id")
    if not isinstance(boot_id, str) or not boot_id:
        return "effect boot_id must be a non-empty string"
    at = effect.get("at")
    if not isinstance(at, str) or not at:
        return "effect at must be a non-empty string"
    return None


class EffectEmitter:
    """Assigns ``(boot_id, seq)`` to effects, one per daemon boot.

    Thread-safe: ``seq`` is monotonic per boot and never re-used.
    ``note_started`` rotates the boot identity on (re)start, so a
    restarted daemon never re-uses a ``(boot_id, seq)`` pair.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._boot_id = uuid.uuid4().hex
        self._seq = 0

    @property
    def boot_id(self) -> str:
        return self._boot_id

    def note_started(self) -> None:
        """Rotate the boot identity: a new daemon boot starts a new
        effect sequence (the daemon calls this from ``start()``)."""
        with self._lock:
            self._boot_id = uuid.uuid4().hex
            self._seq = 0

    def build_projection(
        self,
        *,
        kind: str,
        principal_id: str | None,
        run_id: str | None,
        fields: dict[str, object],
    ) -> dict[str, object]:
        """Effect dict with placeholder ``seq``/``at`` at maximum
        serialized length, for the pre-mutation frame check.

        The projection is not an emitted effect yet: it consumes no
        sequence number. Only :meth:`finalize`, called after the
        mutation succeeded, assigns the real ``seq``/``at``.
        """
        return {
            "seq": _MAX_SEQ,
            "boot_id": self._boot_id,
            "kind": kind,
            "at": AT_PLACEHOLDER,
            "principal_id": principal_id,
            "run_id": run_id,
            **fields,
        }

    def finalize(
        self, projection: dict[str, object], *, fields: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Assign the real ``seq``/``at`` to a projection whose
        mutation has succeeded (and override ``fields`` with the
        values known only after the mutation, e.g. final states)."""
        with self._lock:
            self._seq += 1
            seq = self._seq
        projection["seq"] = seq
        projection["at"] = datetime.now(UTC).isoformat()
        if fields is not None:
            projection.update(fields)
        return projection

    @staticmethod
    def frame_fits(result: dict[str, object], request_id: str) -> bool:
        """True when the serialized result fits the response frame.

        The same bound the daemon's backstop enforces
        (``MAX_FRAME_BYTES``); a projection that fits here is
        guaranteed to fit after finalization, because the real
        ``seq`` (<= 19 digits) and ``at`` (<= 32 chars) are never
        longer than their placeholders.
        """
        return len(encode_success(result, request_id)) - 1 <= MAX_FRAME_BYTES
