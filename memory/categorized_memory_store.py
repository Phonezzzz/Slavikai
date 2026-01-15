from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from shared.memory_category_models import MemoryCategory, MemoryItem, MemorySource
from shared.models import JSONValue

DEFAULT_DB_PATH: Final[Path] = Path("memory/memory_categories.db")
_CURSOR_SEP: Final[str] = "|"


@dataclass(frozen=True)
class ListPage:
    items: list[MemoryItem]
    next_cursor: str | None


class CategorizedMemoryStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_tables()

    def close(self) -> None:
        self.conn.close()

    def add_item(
        self,
        category: MemoryCategory,
        content: str,
        source: MemorySource,
        meta: dict[str, JSONValue] | None = None,
        *,
        title: str | None = None,
        tags: list[str] | None = None,
        created_at: datetime | None = None,
    ) -> MemoryItem:
        normalized = _normalize_content(content)
        fingerprint = _fingerprint(normalized, source)
        existing = self._get_by_fingerprint(category, fingerprint)
        if existing is not None:
            return existing
        now = _ensure_utc(created_at) if created_at else _utc_now()
        record = MemoryItem(
            id=_new_id(now),
            category=category,
            created_at=now,
            updated_at=None,
            title=title,
            content=content,
            tags=tags or [],
            source=source,
            fingerprint=fingerprint,
            meta=meta or {},
            triaged_from=None,
            triaged_at=None,
        )
        self._insert(record)
        return record

    def list_items(
        self,
        category: MemoryCategory,
        *,
        limit: int = 50,
        cursor: str | None = None,
        source: MemorySource | None = None,
    ) -> ListPage:
        if limit <= 0:
            raise ValueError("limit должен быть положительным")
        params: list[object] = [category.value]
        where = ["category = ?"]
        if source is not None:
            where.append("source = ?")
            params.append(source)
        if cursor:
            cursor_created, cursor_id = _decode_cursor(cursor)
            where.append("(created_at < ? OR (created_at = ? AND id < ?))")
            params.extend([cursor_created, cursor_created, cursor_id])
        query = (
            "SELECT * FROM memory_item "
            f"WHERE {' AND '.join(where)} "
            "ORDER BY created_at DESC, id DESC "
            "LIMIT ?"
        )
        params.append(limit + 1)
        rows = self.conn.execute(query, tuple(params)).fetchall()
        items = [self._row_to_item(row) for row in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            last = items[-1]
            next_cursor = _encode_cursor(last.created_at, last.id)
        return ListPage(items=items, next_cursor=next_cursor)

    def delete_item(self, item_id: str) -> bool:
        with self.conn:
            cur = self.conn.execute("DELETE FROM memory_item WHERE id = ?", (item_id,))
        return cur.rowcount > 0

    def update_item(
        self,
        item_id: str,
        *,
        category: MemoryCategory | None = None,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
        meta: dict[str, JSONValue] | None = None,
        source: MemorySource | None = None,
        triaged_from: str | None = None,
        triaged_at: datetime | None = None,
    ) -> MemoryItem | None:
        current = self.get_item(item_id)
        if current is None:
            return None
        new_category = category or current.category
        new_source = source or current.source
        new_content = content if content is not None else current.content
        fingerprint = _fingerprint(_normalize_content(new_content), new_source)
        updated = MemoryItem(
            id=current.id,
            category=new_category,
            created_at=current.created_at,
            updated_at=_utc_now(),
            title=title if title is not None else current.title,
            content=new_content,
            tags=tags if tags is not None else current.tags,
            source=new_source,
            fingerprint=fingerprint,
            meta=meta if meta is not None else current.meta,
            triaged_from=triaged_from if triaged_from is not None else current.triaged_from,
            triaged_at=_ensure_utc(triaged_at) if triaged_at else current.triaged_at,
        )
        self._replace(updated)
        return updated

    def get_item(self, item_id: str) -> MemoryItem | None:
        row = self.conn.execute("SELECT * FROM memory_item WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)

    def cleanup_inbox(self, *, max_items: int, ttl_days: int, now: datetime | None = None) -> int:
        if max_items <= 0:
            raise ValueError("max_items должен быть положительным")
        if ttl_days <= 0:
            raise ValueError("ttl_days должен быть положительным")
        removed = 0
        reference = _ensure_utc(now) if now else _utc_now()
        cutoff = reference - timedelta(days=ttl_days)
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM memory_item WHERE category = ? AND created_at < ?",
                (MemoryCategory.INBOX.value, cutoff.isoformat()),
            )
            removed += cur.rowcount
        rows = self.conn.execute(
            "SELECT id FROM memory_item WHERE category = ? ORDER BY created_at DESC, id DESC",
            (MemoryCategory.INBOX.value,),
        ).fetchall()
        if len(rows) <= max_items:
            return removed
        overflow = rows[max_items:]
        with self.conn:
            for row in overflow:
                cur = self.conn.execute("DELETE FROM memory_item WHERE id = ?", (row["id"],))
                removed += cur.rowcount
        return removed

    def _init_tables(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_item (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                title TEXT,
                content TEXT NOT NULL,
                tags TEXT NOT NULL,
                source TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                meta TEXT NOT NULL,
                triaged_from TEXT,
                triaged_at TEXT
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_item_category_created "
            "ON memory_item(category, created_at DESC, id DESC)"
        )
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_item_category_fingerprint "
            "ON memory_item(category, fingerprint)"
        )
        self.conn.commit()

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        category = MemoryCategory(str(row["category"]))
        return MemoryItem(
            id=str(row["id"]),
            category=category,
            created_at=_parse_dt_required(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            title=row["title"],
            content=str(row["content"]),
            tags=_loads_str_list(str(row["tags"])),
            source=_parse_source(row["source"]),
            fingerprint=str(row["fingerprint"]),
            meta=_loads_dict(str(row["meta"])),
            triaged_from=row["triaged_from"],
            triaged_at=_parse_dt(row["triaged_at"]),
        )

    def _insert(self, item: MemoryItem) -> None:
        payload = _serialize_item(item)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO memory_item (
                    id, category, created_at, updated_at, title, content, tags,
                    source, fingerprint, meta, triaged_from, triaged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def _replace(self, item: MemoryItem) -> None:
        payload = _serialize_item(item)
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO memory_item (
                    id, category, created_at, updated_at, title, content, tags,
                    source, fingerprint, meta, triaged_from, triaged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )

    def _get_by_fingerprint(self, category: MemoryCategory, fingerprint: str) -> MemoryItem | None:
        row = self.conn.execute(
            "SELECT * FROM memory_item WHERE category = ? AND fingerprint = ?",
            (category.value, fingerprint),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_item(row)


def _serialize_item(item: MemoryItem) -> tuple[object, ...]:
    return (
        item.id,
        item.category.value,
        item.created_at.isoformat(),
        item.updated_at.isoformat() if item.updated_at else None,
        item.title,
        item.content,
        _dumps(item.tags),
        item.source,
        item.fingerprint,
        _dumps(item.meta),
        item.triaged_from,
        item.triaged_at.isoformat() if item.triaged_at else None,
    )


def _normalize_content(content: str) -> str:
    return " ".join(content.split())


def _fingerprint(content: str, source: MemorySource) -> str:
    payload = f"{source}:{content}".encode()
    return hashlib.sha256(payload).hexdigest()


def _new_id(created_at: datetime) -> str:
    return f"mem-{created_at.strftime('%Y%m%d%H%M%S%f')}"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    return _ensure_utc(datetime.fromisoformat(str(value)))


def _parse_dt_required(value: object) -> datetime:
    parsed = _parse_dt(value)
    if parsed is None:
        raise ValueError("created_at отсутствует в записи памяти")
    return parsed


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_str_list(text: str) -> list[str]:
    raw = json.loads(text)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError("Expected JSON list[str].")
    return raw


def _loads_dict(text: str) -> dict[str, JSONValue]:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("Expected JSON object.")
    return raw


def _parse_source(value: object) -> MemorySource:
    raw = str(value)
    if raw == "agent":
        return "agent"
    if raw == "user":
        return "user"
    if raw == "triage":
        return "triage"
    raise ValueError(f"Unknown MemorySource: {raw}")


def _encode_cursor(created_at: datetime, item_id: str) -> str:
    return f"{created_at.isoformat()}{_CURSOR_SEP}{item_id}"


def _decode_cursor(cursor: str) -> tuple[str, str]:
    parts = cursor.split(_CURSOR_SEP, 1)
    if len(parts) != 2:
        raise ValueError("Invalid cursor format")
    return parts[0], parts[1]
