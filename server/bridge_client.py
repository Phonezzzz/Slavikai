"""Server-side bridge client with effect recording (E03-T08).

Thin wrapper over ``host_bridge.client.call``: every mutating
call is recorded into the effect journal through
:class:`core.bridge_effect_recorder.BridgeEffectRecorder`, and
every lost answer is recorded as ``unknown`` with its transport
reason. Read-only calls are not recorded (only observations may
attach to effects later, through the recorder).

This is the recording seam for T11: the ToolGateway will call
``BridgeClient.acall`` instead of ``host_bridge.client.call``
directly. T08 ships the client + journal + API only — no live
ToolGateway -> bridge wiring (that is T11's adoption).

Recording never fails the bridge call (owner condition 4): a
persist failure surfaces as ``audit_gap=True`` on the result,
and a recording failure on an error path is attached to the
raised error instead of replacing it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from core.bridge_effect_recorder import (
    REASON_MALFORMED_RESPONSE,
    REASON_TIMEOUT,
    REASON_TRANSPORT,
    BridgeEffectRecorder,
    RecordOutcome,
)
from host_bridge import client
from host_bridge.client import BridgeClientError
from host_bridge.effect import EFFECT_KINDS

# Outcome for calls the journal does not record (read-only).
OUTCOME_NOT_RECORDED: Final[str] = "not_recorded"


@dataclass(frozen=True)
class BridgeCallResult:
    """A bridge call's answer plus its journal outcome."""

    result: dict[str, object]
    record_seq: int | None  # None for read-only calls and audit gaps
    outcome: str  # 'effect' | 'no_effect' | 'unknown' | 'not_recorded'
    audit_gap: bool = False


class RecordedBridgeError(BridgeClientError):
    """A bridge error with its journal outcome attached.

    The caller gets the typed error AND the record: the journal
    row exists (or ``audit_gap`` says it does not), so the caller
    can decide whether to verify the outcome instead of retrying
    blindly.
    """

    def __init__(self, error: BridgeClientError, outcome: RecordOutcome) -> None:
        super().__init__(error.code, str(error))
        self.record_seq = outcome.seq
        self.outcome = outcome.outcome
        self.audit_gap = outcome.audit_gap


class BridgeClient:
    """Bridge client that records mutating calls (E03-T08)."""

    def __init__(self, socket_path: Path, recorder: BridgeEffectRecorder) -> None:
        self._socket_path = socket_path
        self._recorder = recorder

    def call(
        self,
        capability: str,
        *,
        request_id: str | None = None,
        principal_id: str | None = None,
        run_id: str | None = None,
        params: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> BridgeCallResult:
        rid = request_id if request_id is not None else uuid.uuid4().hex
        try:
            result = client.call(
                self._socket_path,
                capability,
                timeout=timeout,
                request_id=rid,
                principal_id=principal_id,
                run_id=run_id,
                params=params,
            )
        except BridgeClientError as exc:
            outcome = self._record_error(capability, rid, principal_id, run_id, params, exc)
            raise RecordedBridgeError(exc, outcome) from exc
        if capability not in EFFECT_KINDS:
            return BridgeCallResult(result=result, record_seq=None, outcome=OUTCOME_NOT_RECORDED)
        outcome = self._recorder.record_success(
            capability=capability,
            request_id=rid,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            result=result,
        )
        return BridgeCallResult(
            result=result,
            record_seq=outcome.seq,
            outcome=outcome.outcome,
            audit_gap=outcome.audit_gap,
        )

    def _record_error(
        self,
        capability: str,
        request_id: str,
        principal_id: str | None,
        run_id: str | None,
        params: dict[str, object] | None,
        exc: BridgeClientError,
    ) -> RecordOutcome:
        if capability not in EFFECT_KINDS:
            # Read-only calls are not recorded, even on error.
            return RecordOutcome(seq=None, outcome=OUTCOME_NOT_RECORDED)
        if exc.code == "request_timeout":
            return self._recorder.record_transport_failure(
                capability=capability,
                request_id=request_id,
                principal_id=principal_id,
                run_id=run_id,
                params=params,
                reason=REASON_TIMEOUT,
            )
        if exc.code in ("connection_closed", "malformed_response"):
            return self._recorder.record_transport_failure(
                capability=capability,
                request_id=request_id,
                principal_id=principal_id,
                run_id=run_id,
                params=params,
                reason=(
                    REASON_TRANSPORT
                    if exc.code == "connection_closed"
                    else REASON_MALFORMED_RESPONSE
                ),
            )
        return self._recorder.record_error(
            capability=capability,
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            code=exc.code,
            message=str(exc),
        )

    async def acall(
        self,
        capability: str,
        *,
        request_id: str | None = None,
        principal_id: str | None = None,
        run_id: str | None = None,
        params: dict[str, object] | None = None,
        timeout: float | None = None,
    ) -> BridgeCallResult:
        """Async entry: the blocking socket call AND the SQLite
        write both run off the event loop (owner condition 4)."""
        return await asyncio.to_thread(
            self.call,
            capability,
            request_id=request_id,
            principal_id=principal_id,
            run_id=run_id,
            params=params,
            timeout=timeout,
        )

    async def attach_observation(
        self,
        *,
        effect_seq: int,
        source: str,
        source_ref: str | None,
        payload: dict[str, object],
    ) -> int:
        """Attach an observation to an effect, off the event loop."""
        return await asyncio.to_thread(
            self._recorder.attach_observation,
            effect_seq=effect_seq,
            source=source,
            source_ref=source_ref,
            payload=payload,
        )
