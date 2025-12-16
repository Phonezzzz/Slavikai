from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memory.memory_companion_store import (
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
