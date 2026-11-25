"""Versioned schema for UI session storage (DB-06, DB-01, DB-02, DB-04, DB-05).

Acceptance tests from section 3 of the storage-group schema proposal:
v1 adoption, fail-closed validation, legacy reset gate, v2 upgrade with a
global message sequence, order stability, batched loads (DB-03),
idempotent re-open, and crash-safe upgrade rollback.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from server.ui_session_storage import SQLiteUISessionStorage

# The current schema version the migration chain must stamp. Written as a
# literal on purpose: the test asserts the number, it does not parrot the
# production constant (which would make the check tautological).
EXPECTED_VERSION = 2

_V1_SESSION_DDL = """
    CREATE TABLE ui_sessions (
        session_id TEXT PRIMARY KEY,
        principal_id TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        status TEXT NOT NULL,
        decision_json TEXT,
        model_provider TEXT,
        model_id TEXT,
        title_override TEXT,
        folder_id TEXT,
        output_text TEXT,
        output_updated_at TEXT,
        files_json TEXT,
        artifacts_json TEXT,
        workspace_root TEXT,
        policy_profile TEXT,
        yolo_armed INTEGER NOT NULL DEFAULT 0,
        yolo_armed_at TEXT,
        tools_state_json TEXT,
        security_revision TEXT,
        mode TEXT NOT NULL DEFAULT 'ask',
        active_plan_json TEXT,
        active_task_json TEXT,
        auto_state_json TEXT
    )
"""

_V1_FOLDER_DDL = """
    CREATE TABLE ui_folders (
        folder_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        principal_id TEXT
    )
"""

_V1_MESSAGE_DDL = """
    CREATE TABLE {table_name} (
        message_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        trace_id TEXT,
        parent_user_message_id TEXT,
        attachments_json TEXT NOT NULL DEFAULT '[]',
        FOREIGN KEY(session_id) REFERENCES ui_sessions(session_id) ON DELETE CASCADE
    )
"""


def _create_v1_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_V1_SESSION_DDL)
    conn.execute(_V1_FOLDER_DDL)
    conn.execute("CREATE TABLE ui_storage_migrations (migration_id TEXT PRIMARY KEY)")
    for table_name in ("chat_messages", "workspace_messages"):
        conn.execute(_V1_MESSAGE_DDL.format(table_name=table_name))
        conn.execute(
            f"CREATE INDEX idx_{table_name}_session_created "
            f"ON {table_name} (session_id, created_at)"
        )


def _insert_session(
    conn: sqlite3.Connection,
    session_id: str = "s1",
    *,
    security_revision: str = "a" * 32,
) -> None:
    conn.execute(
        "INSERT INTO ui_sessions (session_id, created_at, updated_at, status, "
        "security_revision) VALUES (?, '2026-11-25T00:00:00', '2026-11-25T00:00:00', "
        "'idle', ?)",
        (session_id, security_revision),
    )


def _insert_message(
    conn: sqlite3.Connection,
    table_name: str,
    message_id: str,
    session_id: str,
    content: str,
    created_at: str,
) -> None:
    conn.execute(
        f"INSERT INTO {table_name} (message_id, session_id, role, content, "
        "created_at) VALUES (?, ?, 'user', ?, ?)",
        (message_id, session_id, content, created_at),
    )


def _make_v1_db(
    path: Path,
    *,
    user_version: int = 0,
    with_messages: bool = False,
) -> Path:
    """Hand-build a canonical unversioned (v1) database with plain sqlite3."""
    with sqlite3.connect(path) as conn:
        _create_v1_schema(conn)
        _insert_session(conn)
        if with_messages:
            # Interleaved lanes sharing created_at values: the backfill must
            # order them by insertion (rowid) within each timestamp.
            _insert_message(conn, "chat_messages", "c1", "s1", "chat 1", "2026-11-25T00:00:01")
            _insert_message(conn, "workspace_messages", "w1", "s1", "ws 1", "2026-11-25T00:00:01")
            _insert_message(conn, "chat_messages", "c2", "s1", "chat 2", "2026-11-25T00:00:02")
            _insert_message(conn, "workspace_messages", "w2", "s1", "ws 2", "2026-11-25T00:00:02")
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit()
    return path


def _user_version(path: Path) -> int:
    with sqlite3.connect(path) as conn:
        row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0])


def _table_columns(path: Path, table_name: str) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table_name})")}


# --- Test 1: v1 adoption -------------------------------------------------


def test_v1_adoption_stamps_current_and_keeps_data(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", with_messages=True)
    store = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    sessions = store.load_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == "s1"
    assert [message["message_id"] for message in sessions[0].messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]
    assert [message["seq"] for message in sessions[0].messages] == [1, 2, 3, 4]


# --- Test 2: extra column fails closed (DB-02) ----------------------------


def test_extra_column_fails_closed_and_resets_only_with_flag(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", with_messages=True)
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE ui_sessions ADD COLUMN extra TEXT")
        conn.execute("UPDATE ui_sessions SET extra = 'keep-me'")
        conn.commit()
    with pytest.raises(RuntimeError, match="unexpected columns"):
        SQLiteUISessionStorage(path)
    # The failed open did not mutate anything.
    assert _user_version(path) == 0
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT extra FROM ui_sessions").fetchone()[0] == "keep-me"
    store = SQLiteUISessionStorage(path, allow_destructive_reset=True)
    assert _user_version(path) == EXPECTED_VERSION
    assert store.load_sessions() == []


# --- Test 3: legacy ui_messages layout ------------------------------------


def test_legacy_ui_messages_fails_closed_and_resets_only_with_flag(
    tmp_path: Path,
) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", with_messages=True)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE ui_messages (message_id TEXT PRIMARY KEY)")
        conn.commit()
    with pytest.raises(RuntimeError, match="Legacy ui_messages"):
        SQLiteUISessionStorage(path)
    assert _user_version(path) == 0
    store = SQLiteUISessionStorage(path, allow_destructive_reset=True)
    assert _user_version(path) == EXPECTED_VERSION
    assert store.load_sessions() == []
    with sqlite3.connect(path) as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert "ui_messages" not in tables


# --- Test 4: v2 upgrade assigns the global sequence -----------------------


def test_v2_upgrade_assigns_global_seq_in_created_at_rowid_order(
    tmp_path: Path,
) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", user_version=1, with_messages=True)
    store = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    sessions = store.load_sessions()
    assert [message["message_id"] for message in sessions[0].messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]
    assert [message["seq"] for message in sessions[0].messages] == [1, 2, 3, 4]
    with sqlite3.connect(path) as conn:
        meta = conn.execute(
            "SELECT value FROM ui_storage_meta WHERE key = 'message_seq'"
        ).fetchone()
    assert meta is not None and int(meta[0]) == 4


# --- Test 5: order survives restart and a save round-trip -----------------


def test_interleaved_order_survives_restart_and_save_round_trip(
    tmp_path: Path,
) -> None:
    from server.ui_session_storage import PersistedSession

    path = tmp_path / "sessions.db"
    messages = [
        {
            "message_id": "c1",
            "lane": "chat",
            "role": "user",
            "content": "chat 1",
            "created_at": "2026-11-25T00:00:01",
            "trace_id": None,
            "parent_user_message_id": None,
            "attachments": [],
        },
        {
            "message_id": "w1",
            "lane": "workspace",
            "role": "user",
            "content": "ws 1",
            "created_at": "2026-11-25T00:00:01",
            "trace_id": None,
            "parent_user_message_id": None,
            "attachments": [],
        },
        {
            "message_id": "c2",
            "lane": "chat",
            "role": "user",
            "content": "chat 2",
            "created_at": "2026-11-25T00:00:01",
            "trace_id": None,
            "parent_user_message_id": None,
            "attachments": [],
        },
        {
            "message_id": "w2",
            "lane": "workspace",
            "role": "user",
            "content": "ws 2",
            "created_at": "2026-11-25T00:00:01",
            "trace_id": None,
            "parent_user_message_id": None,
            "attachments": [],
        },
    ]
    session = PersistedSession(
        session_id="s1",
        created_at="2026-11-25T00:00:00",
        updated_at="2026-11-25T00:00:00",
        status="idle",
        decision=None,
        messages=messages,
    )
    SQLiteUISessionStorage(path).save_session(session)
    reopened = SQLiteUISessionStorage(path)
    loaded = reopened.load_sessions()[0]
    assert [message["message_id"] for message in loaded.messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]
    # A save of the loaded (seq-carrying) session keeps the same order.
    reopened.save_session(loaded)
    reloaded = SQLiteUISessionStorage(path).load_sessions()[0]
    assert [message["message_id"] for message in reloaded.messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]
    assert [message["seq"] for message in reloaded.messages] == [1, 2, 3, 4]


# --- Test 6: batched loads (DB-03) -----------------------------------------


def test_load_sessions_uses_two_message_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from server.ui_session_storage import PersistedSession

    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    for index in range(5):
        store.save_session(
            PersistedSession(
                session_id=f"s{index}",
                created_at="2026-11-25T00:00:00",
                updated_at="2026-11-25T00:00:00",
                status="idle",
                decision=None,
                messages=[
                    {
                        "message_id": f"c{index}",
                        "lane": "chat",
                        "role": "user",
                        "content": "hi",
                        "created_at": "2026-11-25T00:00:01",
                        "trace_id": None,
                        "parent_user_message_id": None,
                        "attachments": [],
                    },
                    {
                        "message_id": f"w{index}",
                        "lane": "workspace",
                        "role": "user",
                        "content": "hi",
                        "created_at": "2026-11-25T00:00:01",
                        "trace_id": None,
                        "parent_user_message_id": None,
                        "attachments": [],
                    },
                ],
            )
        )
    seen: list[str] = []
    original_connect = SQLiteUISessionStorage._connect

    def counting_connect(self: SQLiteUISessionStorage) -> sqlite3.Connection:
        conn = original_connect(self)
        conn.set_trace_callback(seen.append)
        return conn

    monkeypatch.setattr(SQLiteUISessionStorage, "_connect", counting_connect)
    seen.clear()
    loaded = store.load_sessions()
    assert len(loaded) == 5
    message_selects = [
        statement
        for statement in seen
        if "FROM chat_messages" in statement or "FROM workspace_messages" in statement
    ]
    assert len(message_selects) == 2


# --- Test 7: real-DB column order variance ---------------------------------


def test_v1_column_order_variance_is_adopted(tmp_path: Path) -> None:
    """The owner's real DB has security_revision last (added by
    _ensure_columns); another DB has it in CREATE TABLE order. Both must
    adopt: validation compares column SETS, not order."""
    path = tmp_path / "sessions.db"
    session_ddl_without_revision = _V1_SESSION_DDL.replace("        security_revision TEXT,\n", "")
    assert "security_revision" not in session_ddl_without_revision
    with sqlite3.connect(path) as conn:
        conn.execute(session_ddl_without_revision)
        conn.execute(_V1_FOLDER_DDL)
        conn.execute("CREATE TABLE ui_storage_migrations (migration_id TEXT PRIMARY KEY)")
        for table_name in ("chat_messages", "workspace_messages"):
            conn.execute(_V1_MESSAGE_DDL.format(table_name=table_name))
        conn.execute(
            "INSERT INTO ui_sessions (session_id, created_at, updated_at, status) "
            "VALUES ('s1', '2026-11-25T00:00:00', '2026-11-25T00:00:00', 'idle')"
        )
        # security_revision lands LAST, as in the owner's real DB.
        conn.execute("ALTER TABLE ui_sessions ADD COLUMN security_revision TEXT")
        conn.execute("UPDATE ui_sessions SET security_revision = 'b' * 32")
        conn.commit()
    with sqlite3.connect(path) as conn:
        ordered = [row[1] for row in conn.execute("PRAGMA table_info(ui_sessions)")]
    assert ordered[-1] == "security_revision"
    store = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    assert [session.session_id for session in store.load_sessions()] == ["s1"]


# --- Extra: unknown version / extra table ----------------------------------


def test_unknown_version_fails_closed(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", user_version=99)
    with pytest.raises(RuntimeError, match="schema version"):
        SQLiteUISessionStorage(path)
    # With the opt-in the unknown schema resets instead of failing.
    store = SQLiteUISessionStorage(path, allow_destructive_reset=True)
    assert _user_version(path) == EXPECTED_VERSION
    assert store.load_sessions() == []


def test_extra_table_fails_closed(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db")
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE rogue (id TEXT PRIMARY KEY)")
        conn.commit()
    with pytest.raises(RuntimeError, match="unexpected tables"):
        SQLiteUISessionStorage(path)


# --- Test 11: idempotent re-open --------------------------------------------


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", with_messages=True)
    first = SQLiteUISessionStorage(path)
    before = first.load_sessions()
    second = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    assert second.load_sessions() == before
    third = SQLiteUISessionStorage(path)
    assert third.load_sessions() == before


# --- Test 12: crash-safe upgrade rollback -----------------------------------


def test_upgrade_failure_rolls_back_and_retry_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", user_version=1, with_messages=True)
    calls = {"n": 0}
    original_backfill = SQLiteUISessionStorage._backfill_message_seq

    def flaky_backfill(self: SQLiteUISessionStorage, conn: sqlite3.Connection) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("injected upgrade failure")
        original_backfill(self, conn)

    monkeypatch.setattr(SQLiteUISessionStorage, "_backfill_message_seq", flaky_backfill)
    with pytest.raises(RuntimeError, match="injected upgrade failure"):
        SQLiteUISessionStorage(path)
    # Nothing from the failed upgrade survived: version still v1, no seq
    # column, all message rows intact.
    assert _user_version(path) == 1
    assert "seq" not in _table_columns(path, "chat_messages")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM workspace_messages").fetchone()[0] == 2
    # The next start retries the upgrade and completes it.
    store = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    loaded = store.load_sessions()[0]
    assert [message["message_id"] for message in loaded.messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]


# --- Extra: v2 healing ------------------------------------------------------


def test_v2_heals_dropped_seq_column(tmp_path: Path) -> None:
    path = _make_v1_db(tmp_path / "sessions.db", with_messages=True)
    SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    with sqlite3.connect(path) as conn:
        # SQLite refuses to drop a column an index references, so the
        # damage is index + column together.
        conn.execute("DROP INDEX IF EXISTS idx_chat_messages_session_seq")
        conn.execute("ALTER TABLE chat_messages DROP COLUMN seq")
        conn.commit()
    assert "seq" not in _table_columns(path, "chat_messages")
    store = SQLiteUISessionStorage(path)
    assert _user_version(path) == EXPECTED_VERSION
    loaded = store.load_sessions()[0]
    assert [message["message_id"] for message in loaded.messages] == [
        "c1",
        "w1",
        "c2",
        "w2",
    ]
    assert [message["seq"] for message in loaded.messages] == [1, 2, 3, 4]
