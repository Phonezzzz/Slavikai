"""Delta saves and off-loop persistence for UI session storage (MEM-01).

Acceptance tests from section 3 of the storage-group schema proposal:
delta writes keep rowids and seqs, the event loop stays responsive during
a slow save (to_thread), seq survives the lane split/rejoin, and seq never
leaks into the wire format.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path

import pytest

from server.ui_hub import UIHub
from server.ui_session_storage import (
    PersistedSession,
    SQLiteUISessionStorage,
    legacy_session_from_domain_records,
    split_persisted_session_domains,
)


def _message(
    message_id: str,
    lane: str,
    content: str,
    created_at: str = "2026-11-25T00:00:01",
) -> dict:
    return {
        "message_id": message_id,
        "lane": lane,
        "role": "user",
        "content": content,
        "created_at": created_at,
        "trace_id": None,
        "parent_user_message_id": None,
        "attachments": [],
    }


def _session(
    session_id: str,
    messages: list[dict],
    *,
    principal_id: str | None = None,
) -> PersistedSession:
    return PersistedSession(
        session_id=session_id,
        principal_id=principal_id,
        created_at="2026-11-25T00:00:00",
        updated_at="2026-11-25T00:00:00",
        status="idle",
        decision=None,
        messages=messages,
        # One fixed revision: every _session() rebuild is the same logical
        # writer; a fresh uuid4 per call would trip the CAS check.
        security_revision="c" * 32,
    )


def _rowids_and_seqs(path: Path, session_id: str) -> dict[str, tuple[int, int]]:
    """message_id -> (rowid, seq) across both message tables."""
    found: dict[str, tuple[int, int]] = {}
    with sqlite3.connect(path) as conn:
        for table_name in ("chat_messages", "workspace_messages"):
            for rowid, message_id, seq in conn.execute(
                f"SELECT rowid, message_id, seq FROM {table_name} WHERE session_id = ?",
                (session_id,),
            ).fetchall():
                found[str(message_id)] = (int(rowid), int(seq))
    return found


# --- Test 7: delta saves (MEM-01) ------------------------------------------


def test_delta_save_keeps_rowids_and_seqs(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(
        _session(
            "s1",
            [
                _message("c1", "chat", "chat 1"),
                _message("w1", "workspace", "ws 1"),
                _message("c2", "chat", "chat 2"),
            ],
        )
    )
    before = _rowids_and_seqs(path, "s1")
    assert {message_id: seq for message_id, (_, seq) in before.items()} == {
        "c1": 1,
        "w1": 2,
        "c2": 3,
    }

    # Append one message: existing rows keep rowid and seq, the new one
    # takes the next seq.
    store.save_session(
        _session(
            "s1",
            [
                _message("c1", "chat", "chat 1"),
                _message("w1", "workspace", "ws 1"),
                _message("c2", "chat", "chat 2"),
                _message("w2", "workspace", "ws 2"),
            ],
        )
    )
    after_append = _rowids_and_seqs(path, "s1")
    for message_id in ("c1", "w1", "c2"):
        assert after_append[message_id] == before[message_id]
    assert after_append["w2"][1] == 4

    # Delete the middle message: the survivors keep rowid and seq.
    store.save_session(
        _session(
            "s1",
            [
                _message("c1", "chat", "chat 1"),
                _message("c2", "chat", "chat 2"),
                _message("w2", "workspace", "ws 2"),
            ],
        )
    )
    after_delete = _rowids_and_seqs(path, "s1")
    assert "w1" not in after_delete
    assert after_delete["c1"] == before["c1"]
    assert after_delete["c2"] == before["c2"]
    assert after_delete["w2"] == after_append["w2"]

    # Edit content in place: same rowid, same seq, new content.
    store.save_session(
        _session(
            "s1",
            [
                _message("c1", "chat", "chat 1 EDITED"),
                _message("c2", "chat", "chat 2"),
                _message("w2", "workspace", "ws 2"),
            ],
        )
    )
    after_edit = _rowids_and_seqs(path, "s1")
    assert after_edit["c1"] == before["c1"]
    assert after_edit["c1"][1] == 1
    loaded = store.load_sessions()[0]
    edited = next(item for item in loaded.messages if item["message_id"] == "c1")
    assert edited["content"] == "chat 1 EDITED"


def test_delta_save_lane_change_moves_message(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(_session("s1", [_message("m1", "chat", "hello")]))
    before = _rowids_and_seqs(path, "s1")
    assert before["m1"][1] == 1
    # The same message_id moves to the other lane: the old row is deleted,
    # a new row takes a fresh seq.
    store.save_session(_session("s1", [_message("m1", "workspace", "hello")]))
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0] == 0
        row = conn.execute("SELECT message_id, seq FROM workspace_messages").fetchone()
    assert row is not None and row[0] == "m1" and int(row[1]) == 2
    loaded = store.load_sessions()[0]
    assert loaded.messages[0]["lane"] == "workspace"


def test_save_assigns_seq_in_input_order_not_lane_major(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(
        _session(
            "s1",
            [
                _message("w1", "workspace", "ws 1"),
                _message("c1", "chat", "chat 1"),
                _message("w2", "workspace", "ws 2"),
            ],
        )
    )
    loaded = store.load_sessions()[0]
    assert [item["message_id"] for item in loaded.messages] == ["w1", "c1", "w2"]
    assert [item["seq"] for item in loaded.messages] == [1, 2, 3]


# --- Rework round 1: merge-import must keep the incoming order -----------


def test_delta_save_reorders_messages_on_merge_import(tmp_path: Path) -> None:
    """A merge-import of a fuller history places new messages before
    already-stored ones ([m1, m2] over stored [m2]). The delta save must
    reassign seq to the incoming order instead of keeping m2's old
    position, or the DB order contradicts the in-memory history after a
    restart."""
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(_session("s1", [_message("m2", "chat", "m2")]))
    assert [item["message_id"] for item in store.load_sessions()[0].messages] == ["m2"]
    store.save_session(_session("s1", [_message("m1", "chat", "m1"), _message("m2", "chat", "m2")]))
    loaded = store.load_sessions()[0]
    assert [item["message_id"] for item in loaded.messages] == ["m1", "m2"]
    # Both messages take fresh seq values in incoming order.
    assert [item["seq"] for item in loaded.messages] == [2, 3]


def test_merge_import_reorders_messages_on_reload(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(_session("s1", [_message("m2", "chat", "m2")], principal_id="owner"))
    hub = UIHub(storage=store)

    async def run() -> None:
        imported = await hub.import_sessions(
            [
                _session(
                    "s1",
                    [_message("m1", "chat", "m1"), _message("m2", "chat", "m2")],
                    principal_id="owner",
                )
            ],
            principal_id="owner",
            mode="merge",
        )
        assert imported == 1
        session = await hub.get_session("s1")
        assert session is not None
        assert [item["message_id"] for item in session["messages"]] == ["m1", "m2"]
        reloaded = SQLiteUISessionStorage(path).load_sessions()[0]
        assert [item["message_id"] for item in reloaded.messages] == ["m1", "m2"]

    asyncio.run(run())


def test_cancelled_policy_commit_keeps_memory_and_db_in_sync(tmp_path: Path) -> None:
    """Cancelling set_session_policy mid-write must not leave the new
    policy/revision in SQLite and the old one in memory: the write runs to
    completion, the in-memory state is installed, and only then the
    cancellation propagates."""
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    hub = UIHub(storage=store)

    async def run() -> None:
        sid = await hub.create_session("owner")
        original_save = store.save_session

        def slow_save(*args, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.5)
            return original_save(*args, **kwargs)

        store.save_session = slow_save  # type: ignore[method-assign]
        task = asyncio.create_task(hub.set_session_policy(sid, profile="index"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The off-loop write ran to completion despite the cancellation.
        await asyncio.sleep(0.6)
        assert (await hub.get_session_policy(sid))["profile"] == "index"
        snapshot = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert snapshot is not None
        loaded = store.load_sessions()[0]
        assert loaded.security_revision == snapshot.revision
        # A follow-up write no longer trips the revision CAS.
        await hub.set_session_title(sid, "after cancel")

    asyncio.run(run())


def test_cancelled_import_commit_keeps_session_in_memory(tmp_path: Path) -> None:
    """Same hazard as the policy commit, on the import path: a cancelled
    import must still install the committed session in memory."""
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    hub = UIHub(storage=store)

    async def run() -> None:
        original_save = store.save_session

        def slow_save(*args, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.5)
            return original_save(*args, **kwargs)

        store.save_session = slow_save  # type: ignore[method-assign]
        task = asyncio.create_task(
            hub.import_sessions(
                [
                    _session(
                        "s1",
                        [_message("m1", "chat", "m1")],
                        principal_id="owner",
                    )
                ],
                principal_id="owner",
                mode="replace",
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.6)
        session = await hub.get_session("s1")
        assert session is not None
        assert [item["message_id"] for item in session["messages"]] == ["m1"]

    asyncio.run(run())


# --- Test 8: to_thread (MEM-01) ---------------------------------------------


def test_persist_runs_off_event_loop(tmp_path: Path) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    hub = UIHub(storage=store)

    async def run() -> None:
        sid = await hub.create_session("owner")
        original_save = store.save_session

        def slow_save(*args, **kwargs):  # type: ignore[no-untyped-def]
            time.sleep(0.5)
            return original_save(*args, **kwargs)

        store.save_session = slow_save  # type: ignore[method-assign]
        done = asyncio.Event()
        ticks: list[float] = []

        async def ticker() -> None:
            while not done.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.05)

        ticker_task = asyncio.create_task(ticker())
        try:
            await hub.set_session_title(sid, "slow persist")
        finally:
            done.set()
            await ticker_task
        # A sync 0.5 s save would block the loop: ~0 ticks. Off-loop, the
        # ticker fires about every 50 ms: at least 5 ticks.
        assert len(ticks) >= 5

    asyncio.run(run())


# --- seq through split/rejoin and the wire format ---------------------------


def test_split_rejoin_merges_by_seq_when_present() -> None:
    session = _session(
        "s1",
        [
            {**_message("c1", "chat", "chat 1"), "seq": 1},
            {**_message("w1", "workspace", "ws 1"), "seq": 2},
            {**_message("c2", "chat", "chat 2"), "seq": 3},
            {**_message("w2", "workspace", "ws 2"), "seq": 4},
        ],
    )
    restored = legacy_session_from_domain_records(split_persisted_session_domains(session))
    assert [item["message_id"] for item in restored.messages] == ["c1", "w1", "c2", "w2"]


def test_split_rejoin_without_seq_stays_lane_major() -> None:
    session = _session(
        "s1",
        [
            _message("c1", "chat", "chat 1"),
            _message("w1", "workspace", "ws 1"),
            _message("c2", "chat", "chat 2"),
        ],
    )
    restored = legacy_session_from_domain_records(split_persisted_session_domains(session))
    assert [item["message_id"] for item in restored.messages] == ["c1", "c2", "w1"]


def test_seq_never_leaks_into_wire_format_and_history_check_works(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.db"
    store = SQLiteUISessionStorage(path)
    store.save_session(
        _session(
            "s1",
            [
                _message("c1", "chat", "chat 1"),
                _message("w1", "workspace", "ws 1"),
            ],
        )
    )
    hub = UIHub(storage=store)

    async def run() -> None:
        history = await hub.get_messages("s1", lane="chat")
        assert history, "expected the chat lane to be restored"
        assert all("seq" not in item for item in history)
        session = await hub.get_session("s1")
        assert session is not None
        assert all("seq" not in item for item in session["messages"])
        # The expected-history comparison in _append_message uses the same
        # stripped view: appending with the wire history must not raise a
        # history conflict just because stored messages carry seq.
        await hub.append_message_with_history(
            "s1",
            hub.create_message(role="user", content="chat 2", lane="chat"),
            lane="chat",
            expected_history=history,
        )

    asyncio.run(run())
