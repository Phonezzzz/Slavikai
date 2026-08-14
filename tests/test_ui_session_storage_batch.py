from __future__ import annotations

import sqlite3
from pathlib import Path

from server.ui_session_storage import PersistedSession, SQLiteUISessionStorage


class _CountingConnection:
    def __init__(self, real: sqlite3.Connection, owner: CountingStorage) -> None:
        self._real = real
        self._owner = owner

    def execute(self, sql: str, *args: object) -> sqlite3.Cursor:
        normalized = str(sql).strip().upper()
        if normalized.startswith("SELECT") and "FROM CHAT_MESSAGES" in normalized:
            self._owner.chat_queries += 1
        return self._real.execute(sql, *args)

    def __enter__(self) -> _CountingConnection:
        self._real.__enter__()
        return self

    def __exit__(self, *exc: object) -> object:
        return self._real.__exit__(*exc)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


class CountingStorage(SQLiteUISessionStorage):
    def __init__(self, db_path: Path) -> None:
        super().__init__(db_path)
        self.chat_queries = 0

    def _connect(self) -> sqlite3.Connection:
        real = super()._connect()
        return _CountingConnection(real, self)  # type: ignore[return-value]


def _session(
    session_id: str,
    messages: list[dict[str, object]],
    *,
    principal_id: str = "p1",
) -> PersistedSession:
    return PersistedSession(
        session_id=session_id,
        principal_id=principal_id,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
        status="ok",
        decision=None,
        messages=messages,
    )


def _message(message_id: str, lane: str, content: str, created_at: str) -> dict[str, object]:
    return {
        "message_id": message_id,
        "role": "user",
        "lane": lane,
        "content": content,
        "created_at": created_at,
    }


def test_load_sessions_batch_queries_messages_once(tmp_path: Path) -> None:
    db_path = tmp_path / "ui_sessions.db"
    storage = CountingStorage(db_path)
    for idx in range(3):
        storage.save_session(
            _session(
                f"session-{idx}",
                [
                    _message(f"s{idx}-m1", "chat", f"chat-{idx}", f"2026-01-01T00:00:0{idx}+00:00"),
                    _message(
                        f"s{idx}-m2",
                        "workspace",
                        f"ws-{idx}",
                        f"2026-01-01T00:00:0{idx + 1}+00:00",
                    ),
                ],
            )
        )
    storage.save_session(_session("empty-session", []))

    loaded = storage.load_sessions()

    assert storage.chat_queries == 1, "ожидается один батч-запрос вместо N+1"

    by_id = {item.session_id: item for item in loaded}
    assert set(by_id) == {"session-0", "session-1", "session-2", "empty-session"}
    assert by_id["empty-session"].messages == []
    for idx in range(3):
        messages = by_id[f"session-{idx}"].messages
        assert [item["content"] for item in messages] == [f"chat-{idx}", f"ws-{idx}"]
        assert [item["lane"] for item in messages] == ["chat", "workspace"]


def test_load_sessions_batch_preserves_order_semantics_and_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "ui_sessions.db"
    storage = CountingStorage(db_path)
    storage.save_session(
        _session(
            "a",
            [
                _message("a-1", "chat", "first", "2026-01-01T00:00:01+00:00"),
                _message("a-2", "chat", "second", "2026-01-01T00:00:02+00:00"),
            ],
        )
    )
    storage.save_session(
        _session(
            "b",
            [
                _message("b-1", "chat", "only", "2026-01-01T00:00:03+00:00"),
            ],
        )
    )

    loaded = {item.session_id: item.messages for item in storage.load_sessions()}

    assert [item["content"] for item in loaded["a"]] == ["first", "second"]
    assert [item["content"] for item in loaded["b"]] == ["only"]
    assert all("message_id" in item and "created_at" in item for item in loaded["a"])
    assert storage.chat_queries == 1
