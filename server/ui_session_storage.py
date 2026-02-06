from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from shared.models import JSONValue
from shared.sanitize import safe_json_loads


@dataclass(frozen=True)
class PersistedSession:
    session_id: str
    created_at: str
    updated_at: str
    status: str
    decision: dict[str, JSONValue] | None
    messages: list[dict[str, str]]
    model_provider: str | None = None
    model_id: str | None = None
    canvas_output: dict[str, JSONValue] | None = None


class UISessionStorage(Protocol):
    def load_sessions(self) -> list[PersistedSession]: ...

    def save_session(self, session: PersistedSession) -> None: ...

    def delete_sessions(self, session_ids: Sequence[str]) -> None: ...


class InMemoryUISessionStorage:
    def __init__(self) -> None:
        self._sessions: dict[str, PersistedSession] = {}

    def load_sessions(self) -> list[PersistedSession]:
        return [self._clone_session(item) for item in self._sessions.values()]

    def save_session(self, session: PersistedSession) -> None:
        self._sessions[session.session_id] = self._clone_session(session)

    def delete_sessions(self, session_ids: Sequence[str]) -> None:
        for session_id in session_ids:
            self._sessions.pop(session_id, None)

    def _clone_session(self, session: PersistedSession) -> PersistedSession:
        decision = dict(session.decision) if session.decision is not None else None
        messages = [dict(item) for item in session.messages]
        return PersistedSession(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            status=session.status,
            decision=decision,
            messages=messages,
            model_provider=session.model_provider,
            model_id=session.model_id,
            canvas_output=(
                dict(session.canvas_output) if session.canvas_output is not None else None
            ),
        )


class SQLiteUISessionStorage:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._initialize_schema()

    def load_sessions(self) -> list[PersistedSession]:
        sessions: list[PersistedSession] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, created_at, updated_at, status, decision_json
                    , model_provider, model_id, canvas_json
                FROM ui_sessions
                """,
            ).fetchall()
            for row in rows:
                session_id = str(row["session_id"])
                messages = self._load_messages(conn, session_id)
                sessions.append(
                    PersistedSession(
                        session_id=session_id,
                        created_at=str(row["created_at"]),
                        updated_at=str(row["updated_at"]),
                        status=str(row["status"]),
                        decision=self._decode_decision(row["decision_json"]),
                        messages=messages,
                        model_provider=_optional_str(row["model_provider"]),
                        model_id=_optional_str(row["model_id"]),
                        canvas_output=self._decode_canvas(row["canvas_json"]),
                    ),
                )
        return sessions

    def save_session(self, session: PersistedSession) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_sessions (
                    session_id,
                    created_at,
                    updated_at,
                    status,
                    decision_json,
                    model_provider,
                    model_id,
                    canvas_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    decision_json=excluded.decision_json,
                    model_provider=excluded.model_provider,
                    model_id=excluded.model_id,
                    canvas_json=excluded.canvas_json
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.updated_at,
                    session.status,
                    self._encode_decision(session.decision),
                    session.model_provider,
                    session.model_id,
                    self._encode_canvas(session.canvas_output),
                ),
            )
            conn.execute("DELETE FROM ui_messages WHERE session_id = ?", (session.session_id,))
            for index, message in enumerate(session.messages):
                conn.execute(
                    """
                    INSERT INTO ui_messages (session_id, message_index, role, content)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        session.session_id,
                        index,
                        message["role"],
                        message["content"],
                    ),
                )
            conn.commit()

    def delete_sessions(self, session_ids: Sequence[str]) -> None:
        if not session_ids:
            return
        placeholders = ", ".join("?" for _ in session_ids)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM ui_messages WHERE session_id IN ({placeholders})",
                tuple(session_ids),
            )
            conn.execute(
                f"DELETE FROM ui_sessions WHERE session_id IN ({placeholders})",
                tuple(session_ids),
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _initialize_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision_json TEXT,
                    model_provider TEXT,
                    model_id TEXT,
                    canvas_json TEXT
                )
                """,
            )
            self._ensure_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES ui_sessions(session_id) ON DELETE CASCADE
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ui_messages_session_idx
                ON ui_messages (session_id, message_index)
                """,
            )
            conn.commit()

    def _load_messages(self, conn: sqlite3.Connection, session_id: str) -> list[dict[str, str]]:
        rows = conn.execute(
            """
            SELECT role, content
            FROM ui_messages
            WHERE session_id = ?
            ORDER BY message_index ASC
            """,
            (session_id,),
        ).fetchall()
        return [{"role": str(row["role"]), "content": str(row["content"])} for row in rows]

    def _decode_decision(self, value: object) -> dict[str, JSONValue] | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        parsed = safe_json_loads(value)
        if not isinstance(parsed, dict):
            return None
        decoded: dict[str, JSONValue] = {}
        for key, item in parsed.items():
            decoded[str(key)] = item
        return decoded

    def _encode_decision(self, value: dict[str, JSONValue] | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def _decode_canvas(self, value: object) -> dict[str, JSONValue] | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        parsed = safe_json_loads(value)
        if not isinstance(parsed, dict):
            return None
        decoded: dict[str, JSONValue] = {}
        for key, item in parsed.items():
            decoded[str(key)] = item
        return decoded

    def _encode_canvas(self, value: dict[str, JSONValue] | None) -> str | None:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False)

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(ui_sessions)").fetchall()
            if isinstance(row, sqlite3.Row)
        }
        if "model_provider" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN model_provider TEXT")
        if "model_id" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN model_id TEXT")
        if "canvas_json" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN canvas_json TEXT")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
