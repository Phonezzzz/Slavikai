from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from memory.categorized_memory_store import CategorizedMemoryStore
from shared.memory_category_models import MemoryCategory


def _make_store(tmp_path: Path) -> CategorizedMemoryStore:
    return CategorizedMemoryStore(tmp_path / "mem.db")


def test_add_item_dedup_by_fingerprint(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    first = store.add_item(
        MemoryCategory.INBOX,
        "hello world",
        "agent",
        meta={"trace_id": "t1"},
    )
    second = store.add_item(
        MemoryCategory.INBOX,
        "hello world",
        "agent",
        meta={"trace_id": "t2"},
    )
    assert first.id == second.id
    page = store.list_items(MemoryCategory.INBOX, limit=10)
    assert len(page.items) == 1


def test_cleanup_inbox_respects_max_items(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    base = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    for offset in range(5):
        store.add_item(
            MemoryCategory.INBOX,
            f"item-{offset}",
            "agent",
            meta={},
            created_at=base - timedelta(minutes=offset),
        )
    removed = store.cleanup_inbox(max_items=3, ttl_days=30, now=base)
    assert removed == 2
    page = store.list_items(MemoryCategory.INBOX, limit=10)
    assert [item.content for item in page.items] == ["item-0", "item-1", "item-2"]


def test_cleanup_inbox_respects_ttl(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    base = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    store.add_item(
        MemoryCategory.INBOX,
        "old",
        "agent",
        meta={},
        created_at=base - timedelta(days=20),
    )
    store.add_item(
        MemoryCategory.INBOX,
        "new",
        "agent",
        meta={},
        created_at=base - timedelta(days=1),
    )
    removed = store.cleanup_inbox(max_items=200, ttl_days=7, now=base)
    assert removed == 1
    page = store.list_items(MemoryCategory.INBOX, limit=10)
    assert [item.content for item in page.items] == ["new"]


def test_list_items_cursor_paginates(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    base = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)
    for offset in range(3):
        store.add_item(
            MemoryCategory.INBOX,
            f"item-{offset}",
            "agent",
            meta={},
            created_at=base - timedelta(minutes=offset),
        )
    first = store.list_items(MemoryCategory.INBOX, limit=2)
    assert len(first.items) == 2
    assert first.next_cursor is not None
    second = store.list_items(MemoryCategory.INBOX, limit=2, cursor=first.next_cursor)
    assert len(second.items) == 1
