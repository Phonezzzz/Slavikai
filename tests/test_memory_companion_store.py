from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory.memory_companion_store import (
    MIN_SCHEMA_VERSION,
    InvalidMemoryCompanionDbError,
    MemoryCompanionStore,
    SchemaVersionMismatchError,
)
from shared.memory_companion_models import (
    BlockedReason,
    ChatInteractionLog,
    InteractionKind,
    InteractionMode,
    ToolInteractionLog,
    ToolStatus,
)

_CMD = "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)"


def _make_v_db(path: Path, version: int) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(_CMD, (str(version),))
    conn.execute(
        """
        CREATE TABLE interaction_log (
            interaction_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            interaction_kind TEXT NOT NULL,
            raw_input TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT NOT NULL,
            retrieved_memory_ids TEXT NOT NULL DEFAULT '[]',
            applied_policy_ids TEXT NOT NULL DEFAULT '[]',
            response_text TEXT,
            tool_name TEXT,
            tool_args TEXT,
            tool_status TEXT,
            blocked_reason TEXT,
            tool_output_preview TEXT,
            tool_error TEXT,
            tool_meta TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO interaction_log (interaction_id, user_id, interaction_kind, "
        "raw_input, mode, created_at, response_text) "
        "VALUES ('chat-old', 'local', 'chat', 'hello', 'standard', "
        "'2025-01-01', 'ok')"
    )
    conn.execute("CREATE TABLE policy_rule (rule_id TEXT PRIMARY KEY, user_id TEXT)")
    conn.execute("CREATE TABLE feedback_event (feedback_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE batch_review_run (batch_review_run_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE policy_rule_candidate (candidate_id TEXT PRIMARY KEY)")
    conn.commit()
    conn.close()


def test_store_creates_and_logs_chat_and_tool(tmp_path: Path) -> None:
    db_path = tmp_path / "memory_companion.db"
    store = MemoryCompanionStore(db_path)

    store.log_interaction(
        ChatInteractionLog(
            interaction_id="chat-1",
            user_id="local",
            interaction_kind=InteractionKind.CHAT,
            raw_input="hello",
            mode=InteractionMode.STANDARD,
            created_at="2025-01-01 00:00:01",
            response_text="ok",
            retrieved_memory_ids=["m1"],
            applied_policy_ids=["p1"],
        )
    )

    long_output = "x" * 3000
    store.log_interaction(
        ToolInteractionLog(
            interaction_id="tool-1",
            user_id="local",
            interaction_kind=InteractionKind.TOOL,
            raw_input="/fs list",
            mode=InteractionMode.STANDARD,
            created_at="2025-01-01 00:00:02",
            tool_name="fs",
            tool_args={"op": "list"},
            tool_status=ToolStatus.BLOCKED,
            blocked_reason=BlockedReason.TOOL_DISABLED,
            tool_output_preview=long_output,
            tool_error="disabled",
            tool_meta={"k": "v"},
        )
    )

    recent = store.get_recent(10)
    kinds = {item.interaction_kind for item in recent}
    assert InteractionKind.CHAT in kinds
    assert InteractionKind.TOOL in kinds

    tool_items = [item for item in recent if item.interaction_kind == InteractionKind.TOOL]
    assert tool_items
    tool_item = tool_items[0]
    assert isinstance(tool_item, ToolInteractionLog)
    assert tool_item.tool_name == "fs"
    assert tool_item.tool_status == ToolStatus.BLOCKED
    assert tool_item.blocked_reason == BlockedReason.TOOL_DISABLED
    assert tool_item.tool_output_preview and tool_item.tool_output_preview.endswith("…[truncated]")


def test_store_fails_fast_on_schema_mismatch(tmp_path: Path) -> None:
    db_path = tmp_path / "bad.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta (key, value) VALUES ('schema_version', '999')")
    conn.commit()
    conn.close()

    with pytest.raises(SchemaVersionMismatchError):
        MemoryCompanionStore(db_path)


def test_store_fails_fast_without_schema_meta(tmp_path: Path) -> None:
    db_path = tmp_path / "invalid.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE something (id INTEGER)")
    conn.commit()
    conn.close()

    with pytest.raises(InvalidMemoryCompanionDbError):
        MemoryCompanionStore(db_path)


def test_store_unsupported_old_version(tmp_path: Path) -> None:
    db_path = tmp_path / "old.db"
    _make_v_db(db_path, version=3)
    original_mtime = db_path.stat().st_mtime

    with pytest.raises(SchemaVersionMismatchError):
        MemoryCompanionStore(db_path)

    assert db_path.stat().st_mtime == original_mtime


def test_store_future_version_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "future.db"
    _make_v_db(db_path, version=999)
    original_mtime = db_path.stat().st_mtime

    with pytest.raises(SchemaVersionMismatchError):
        MemoryCompanionStore(db_path)

    assert db_path.stat().st_mtime == original_mtime


def test_migration_registry_is_empty_for_baseline_v4() -> None:
    from memory.memory_companion_store import _MIGRATIONS, SCHEMA_VERSION

    assert _MIGRATIONS == {}, (
        "Registry должен быть пустым для baseline v4: "
        f"MIN_SCHEMA_VERSION={MIN_SCHEMA_VERSION}, SCHEMA_VERSION={SCHEMA_VERSION}"
    )


def test_migration_error_rolls_back_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from memory import memory_companion_store as mcs

    db_path = tmp_path / "rollback.db"
    _make_v_db(db_path, version=4)

    assert mcs.MIN_SCHEMA_VERSION == 4

    def _failing_migration(self: mcs.MemoryCompanionStore) -> None:
        self.conn.execute(
            "INSERT INTO interaction_log "
            "(interaction_id, user_id, interaction_kind, raw_input, mode, created_at) "
            "VALUES ('partial', 'local', 'chat', 'partial', 'standard', '2026-01-01')"
        )
        self.conn.commit()
        raise RuntimeError("simulated migration failure")

    monkeypatch.setattr(mcs, "SCHEMA_VERSION", 5)
    monkeypatch.setitem(mcs._MIGRATIONS, 4, "_test_failing_v4_to_v5")
    monkeypatch.setattr(
        mcs.MemoryCompanionStore,
        "_test_failing_v4_to_v5",
        _failing_migration,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="simulated migration failure"):
        mcs.MemoryCompanionStore(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cur = conn.execute("SELECT * FROM interaction_log")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0]["interaction_id"] == "chat-old"

    cur = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'")
    version_row = cur.fetchone()
    assert version_row is not None
    assert int(version_row[0]) == 4
    conn.close()

    assert mcs.MIN_SCHEMA_VERSION == 4
