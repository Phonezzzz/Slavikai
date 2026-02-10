from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
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
    title_override: str | None = None
    folder_id: str | None = None
    output_text: str | None = None
    output_updated_at: str | None = None
    files: list[str] = field(default_factory=list)
    artifacts: list[dict[str, JSONValue]] = field(default_factory=list)


@dataclass(frozen=True)
class PersistedFolder:
    folder_id: str
    name: str
    created_at: str
    updated_at: str


class UISessionStorage(Protocol):
    def load_sessions(self) -> list[PersistedSession]: ...

    def save_session(self, session: PersistedSession) -> None: ...

    def delete_sessions(self, session_ids: Sequence[str]) -> None: ...

    def load_folders(self) -> list[PersistedFolder]: ...

    def save_folder(self, folder: PersistedFolder) -> None: ...

    def delete_folders(self, folder_ids: Sequence[str]) -> None: ...


class InMemoryUISessionStorage:
    def __init__(self) -> None:
        self._sessions: dict[str, PersistedSession] = {}
        self._folders: dict[str, PersistedFolder] = {}

    def load_sessions(self) -> list[PersistedSession]:
        return [self._clone_session(item) for item in self._sessions.values()]

    def save_session(self, session: PersistedSession) -> None:
        self._sessions[session.session_id] = self._clone_session(session)

    def delete_sessions(self, session_ids: Sequence[str]) -> None:
        for session_id in session_ids:
            self._sessions.pop(session_id, None)

    def load_folders(self) -> list[PersistedFolder]:
        return [self._clone_folder(item) for item in self._folders.values()]

    def save_folder(self, folder: PersistedFolder) -> None:
        self._folders[folder.folder_id] = self._clone_folder(folder)

    def delete_folders(self, folder_ids: Sequence[str]) -> None:
        for folder_id in folder_ids:
            self._folders.pop(folder_id, None)

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
            title_override=session.title_override,
            folder_id=session.folder_id,
            output_text=session.output_text,
            output_updated_at=session.output_updated_at,
            files=list(session.files),
            artifacts=[dict(item) for item in session.artifacts],
        )

    def _clone_folder(self, folder: PersistedFolder) -> PersistedFolder:
        return PersistedFolder(
            folder_id=folder.folder_id,
            name=folder.name,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
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
                    , model_provider, model_id, title_override, folder_id
                    , output_text, output_updated_at, files_json, artifacts_json
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
                        title_override=_optional_str(row["title_override"]),
                        folder_id=_optional_str(row["folder_id"]),
                        output_text=_optional_str(row["output_text"]),
                        output_updated_at=_optional_str(row["output_updated_at"]),
                        files=self._decode_files(row["files_json"]),
                        artifacts=self._decode_artifacts(row["artifacts_json"]),
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
                    title_override,
                    folder_id,
                    output_text,
                    output_updated_at,
                    files_json,
                    artifacts_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id)
                DO UPDATE SET
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    status=excluded.status,
                    decision_json=excluded.decision_json,
                    model_provider=excluded.model_provider,
                    model_id=excluded.model_id,
                    title_override=excluded.title_override,
                    folder_id=excluded.folder_id,
                    output_text=excluded.output_text,
                    output_updated_at=excluded.output_updated_at,
                    files_json=excluded.files_json,
                    artifacts_json=excluded.artifacts_json
                """,
                (
                    session.session_id,
                    session.created_at,
                    session.updated_at,
                    session.status,
                    self._encode_decision(session.decision),
                    session.model_provider,
                    session.model_id,
                    session.title_override,
                    session.folder_id,
                    session.output_text,
                    session.output_updated_at,
                    self._encode_files(session.files),
                    self._encode_artifacts(session.artifacts),
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

    def load_folders(self) -> list[PersistedFolder]:
        folders: list[PersistedFolder] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT folder_id, name, created_at, updated_at
                FROM ui_folders
                """,
            ).fetchall()
            for row in rows:
                folder_id = str(row["folder_id"])
                folders.append(
                    PersistedFolder(
                        folder_id=folder_id,
                        name=str(row["name"]),
                        created_at=str(row["created_at"]),
                        updated_at=str(row["updated_at"]),
                    ),
                )
        return folders

    def save_folder(self, folder: PersistedFolder) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_folders (folder_id, name, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(folder_id)
                DO UPDATE SET
                    name=excluded.name,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    folder.folder_id,
                    folder.name,
                    folder.created_at,
                    folder.updated_at,
                ),
            )
            conn.commit()

    def delete_folders(self, folder_ids: Sequence[str]) -> None:
        if not folder_ids:
            return
        placeholders = ", ".join("?" for _ in folder_ids)
        with self._connect() as conn:
            conn.execute(
                f"DELETE FROM ui_folders WHERE folder_id IN ({placeholders})",
                tuple(folder_ids),
            )
            conn.execute(
                f"UPDATE ui_sessions SET folder_id = NULL WHERE folder_id IN ({placeholders})",
                tuple(folder_ids),
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
                    title_override TEXT,
                    folder_id TEXT,
                    output_text TEXT,
                    output_updated_at TEXT,
                    files_json TEXT,
                    artifacts_json TEXT
                )
                """,
            )
            self._ensure_columns(conn)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_folders (
                    folder_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """,
            )
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

    def _decode_files(self, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, str):
            return []
        parsed = safe_json_loads(value)
        if not isinstance(parsed, list):
            return []
        files: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                files.append(item.strip())
        return files

    def _encode_files(self, value: list[str]) -> str:
        return json.dumps(value, ensure_ascii=False)

    def _decode_artifacts(self, value: object) -> list[dict[str, JSONValue]]:
        if value is None:
            return []
        if not isinstance(value, str):
            return []
        parsed = safe_json_loads(value)
        if not isinstance(parsed, list):
            return []
        artifacts: list[dict[str, JSONValue]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            normalized: dict[str, JSONValue] = {}
            for key, entry in item.items():
                normalized[str(key)] = entry
            artifacts.append(normalized)
        return artifacts

    def _encode_artifacts(self, value: list[dict[str, JSONValue]]) -> str:
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
        if "title_override" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN title_override TEXT")
        if "folder_id" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN folder_id TEXT")
        if "output_text" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN output_text TEXT")
        if "output_updated_at" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN output_updated_at TEXT")
        if "files_json" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN files_json TEXT")
        if "artifacts_json" not in existing:
            conn.execute("ALTER TABLE ui_sessions ADD COLUMN artifacts_json TEXT")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
