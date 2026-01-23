from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from memory.categorized_memory_store import CategorizedMemoryStore
from memory.feedback_manager import FeedbackManager
from memory.memory_companion_store import MemoryCompanionStore
from memory.memory_manager import MemoryManager
from memory.vector_index import VectorIndex

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolate_skill_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    candidates_dir = tmp_path / "skills" / "_candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SKILLS_CANDIDATES_DIR", str(candidates_dir))


@pytest.fixture(autouse=True)
def _close_sqlite_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    tracked: list[object] = []
    connections: list[sqlite3.Connection] = []

    original_connect = sqlite3.connect

    def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", _connect)

    def _wrap_init(cls: type[object]) -> None:
        original_init = cls.__init__  # type: ignore[assignment]

        def _init(self: object, *args: object, **kwargs: object) -> None:
            original_init(self, *args, **kwargs)  # type: ignore[misc]
            tracked.append(self)

        monkeypatch.setattr(cls, "__init__", _init)

    for store_cls in (
        CategorizedMemoryStore,
        FeedbackManager,
        MemoryCompanionStore,
        MemoryManager,
        VectorIndex,
    ):
        _wrap_init(store_cls)

    yield

    for item in tracked:
        close = getattr(item, "close", None)
        if callable(close):
            close()

    for conn in connections:
        conn.close()
