from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import TracebackType


class _ThreadConnectionState(threading.local):
    connection: sqlite3.Connection | None = None


class ThreadLocalSQLiteConnection:
    """SQLite facade that owns one real connection per calling thread."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        row_factory: type[sqlite3.Row] | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._row_factory = row_factory
        self._state = _ThreadConnectionState()
        self._connections: list[sqlite3.Connection] = []
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    def cursor(self) -> sqlite3.Cursor:
        return self._connection().cursor()

    def execute(
        self,
        sql: str,
        parameters: tuple[object, ...] = (),
    ) -> sqlite3.Cursor:
        return self._connection().execute(sql, parameters)

    def commit(self) -> None:
        self._connection().commit()

    def rollback(self) -> None:
        self._connection().rollback()

    def close(self) -> None:
        with self._lifecycle_lock:
            connections = tuple(self._connections)
            self._connections.clear()
            self._closed = True
        self._state.connection = None
        for connection in connections:
            connection.close()

    def __enter__(self) -> ThreadLocalSQLiteConnection:
        self._connection().__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        return self._connection().__exit__(exc_type, exc_value, traceback)

    def _connection(self) -> sqlite3.Connection:
        connection = self._state.connection
        if connection is not None:
            return connection
        with self._lifecycle_lock:
            if self._closed:
                raise sqlite3.ProgrammingError("SQLite connection lifecycle is closed")
            connection = sqlite3.connect(self._db_path, check_same_thread=False)
            if self._row_factory is not None:
                connection.row_factory = self._row_factory
            self._connections.append(connection)
            self._state.connection = connection
        return connection
