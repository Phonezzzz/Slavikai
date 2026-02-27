from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from shared.models import JSONValue

DEFAULT_IDEMPOTENCY_WINDOW_SECONDS = 90
MAX_IDEMPOTENCY_KEY_CHARS = 128


@dataclass(frozen=True)
class IdempotencyReplay:
    status: int
    payload: dict[str, JSONValue]


@dataclass
class _IdempotencyEntry:
    fingerprint: str
    expires_at: datetime
    replay: IdempotencyReplay | None = None


class IdempotencyStore:
    def __init__(self, *, window_seconds: int = DEFAULT_IDEMPOTENCY_WINDOW_SECONDS) -> None:
        self._window_seconds = (
            window_seconds if window_seconds > 0 else DEFAULT_IDEMPOTENCY_WINDOW_SECONDS
        )
        self._lock = asyncio.Lock()
        self._entries: dict[tuple[str, str, str], _IdempotencyEntry] = {}

    async def begin(
        self,
        *,
        endpoint: str,
        session_id: str,
        key: str,
        fingerprint: str,
    ) -> tuple[str, IdempotencyReplay | None]:
        storage_key = (endpoint, session_id, key)
        async with self._lock:
            self._prune_locked()
            existing = self._entries.get(storage_key)
            if existing is None:
                self._entries[storage_key] = _IdempotencyEntry(
                    fingerprint=fingerprint,
                    expires_at=datetime.now(timezone.utc)  # noqa: UP017
                    + timedelta(seconds=self._window_seconds),
                    replay=None,
                )
                return "execute", None
            if existing.fingerprint != fingerprint:
                return "conflict", None
            if existing.replay is not None:
                return "replay", existing.replay
            return "in_progress", None

    async def complete(
        self,
        *,
        endpoint: str,
        session_id: str,
        key: str,
        fingerprint: str,
        status: int,
        payload: dict[str, JSONValue],
    ) -> None:
        storage_key = (endpoint, session_id, key)
        async with self._lock:
            self._prune_locked()
            existing = self._entries.get(storage_key)
            if existing is None or existing.fingerprint != fingerprint:
                return
            existing.replay = IdempotencyReplay(status=status, payload=dict(payload))
            existing.expires_at = (
                datetime.now(timezone.utc)  # noqa: UP017
                + timedelta(seconds=self._window_seconds)
            )

    async def abort(
        self,
        *,
        endpoint: str,
        session_id: str,
        key: str,
        fingerprint: str,
    ) -> None:
        storage_key = (endpoint, session_id, key)
        async with self._lock:
            existing = self._entries.get(storage_key)
            if existing is None or existing.fingerprint != fingerprint:
                return
            self._entries.pop(storage_key, None)

    def _prune_locked(self) -> None:
        now = datetime.now(timezone.utc)  # noqa: UP017
        stale_keys = [key for key, entry in self._entries.items() if entry.expires_at < now]
        for key in stale_keys:
            self._entries.pop(key, None)


def normalize_idempotency_key(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized:
        return None
    return normalized[:MAX_IDEMPOTENCY_KEY_CHARS]


def fingerprint_json_payload(value: dict[str, JSONValue]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
