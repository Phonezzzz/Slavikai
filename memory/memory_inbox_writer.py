from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from config.memory_config import MemoryConfig
from memory.categorized_memory_store import CategorizedMemoryStore
from shared.memory_category_models import MemoryCategory, MemoryItem, MemorySource
from shared.models import JSONValue


class MemoryInboxWriter:
    def __init__(
        self,
        store: CategorizedMemoryStore,
        config: MemoryConfig,
        *,
        now: Callable[[], datetime] | None = None,
        max_items: int | None = None,
        ttl_days: int | None = None,
        writes_per_minute: int | None = None,
    ) -> None:
        self._store = store
        self._now = now or _utc_now
        self._max_items = max_items or config.inbox_max_items
        self._ttl_days = ttl_days or config.inbox_ttl_days
        self._writes_per_minute = writes_per_minute or config.inbox_writes_per_minute
        self._recent_writes: deque[datetime] = deque()

    def write_once(
        self,
        content: str,
        *,
        source: MemorySource = "agent",
        meta: dict[str, JSONValue] | None = None,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem | None:
        normalized = _normalize_content(content)
        if not normalized:
            return None
        fingerprint = _fingerprint(normalized, source)
        existing = self._store.get_by_fingerprint(MemoryCategory.INBOX, fingerprint)
        if existing is not None:
            return existing
        now = self._now()
        if not self._allow_write(now):
            return None
        item = self._store.add_item(
            MemoryCategory.INBOX,
            content,
            source,
            meta or {},
            title=title,
            tags=tags,
            created_at=now,
        )
        self._recent_writes.append(now)
        self._store.cleanup_inbox(
            max_items=self._max_items,
            ttl_days=self._ttl_days,
            now=now,
        )
        return item

    def _allow_write(self, now: datetime) -> bool:
        cutoff = now - timedelta(minutes=1)
        while self._recent_writes and self._recent_writes[0] < cutoff:
            self._recent_writes.popleft()
        return len(self._recent_writes) < self._writes_per_minute


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _normalize_content(content: str) -> str:
    return " ".join(content.split())


def _fingerprint(content: str, source: MemorySource) -> str:
    payload = f"{source}:{content}".encode()
    return hashlib.sha256(payload).hexdigest()
