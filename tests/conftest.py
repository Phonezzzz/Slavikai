from __future__ import annotations

import sqlite3
import sys
import threading
from pathlib import Path

import pytest

from memory.canonical_atom_store import CanonicalAtomStore
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
    connections: list[tuple[int, bool, sqlite3.Connection]] = []

    original_connect = sqlite3.connect

    def _connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        conn = original_connect(*args, **kwargs)
        same_thread = kwargs.get("check_same_thread", args[4] if len(args) > 4 else True)
        connections.append((threading.get_ident(), bool(same_thread), conn))
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
        CanonicalAtomStore,
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

    for owner_thread, same_thread, conn in connections:
        # Thread-affine соединения worker закрываются в контексте владеющего Store.
        if not same_thread or owner_thread == threading.get_ident():
            conn.close()


@pytest.fixture(autouse=True)
def _isolate_http_task_run_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from core.task_run_storage import TaskRunStore
    from server.http import app as http_app

    counter = 0

    def isolated_store(_path: Path) -> TaskRunStore:
        nonlocal counter
        counter += 1
        return TaskRunStore(tmp_path / "http-runs" / f"app-{counter}.db")

    # Default bootstrap dependencies are test-owned; explicit injected stores retain their path.
    monkeypatch.setattr(http_app, "TaskRunStore", isolated_store)


@pytest.fixture(autouse=True)
def _isolate_http_preferences_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from server.http import app as http_app
    from server.preferences_store import PreferencesStore

    monkeypatch.setattr(
        http_app, "PreferencesStore", lambda _path: PreferencesStore(tmp_path / "preferences.db")
    )
