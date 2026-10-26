import asyncio

import pytest

from server.ui_hub import UIHub
from server.ui_session_storage import SQLiteUISessionStorage


def test_session_security_cas_is_owned_persisted_and_not_updated_by_unrelated_writes(tmp_path):
    async def run():
        path = tmp_path / "sessions.db"
        hub = UIHub(storage=SQLiteUISessionStorage(path))
        sid = await hub.create_session("owner")
        a = await hub.get_session_security_snapshot(sid, principal_id="owner")
        b = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert a == b and a is not None
        assert await hub.get_session_security_snapshot(sid, principal_id="member") is None
        await hub.set_session_model(sid, "local", "fixture")
        assert await hub.get_session_security_snapshot(sid, principal_id="owner") == a
        updated = await hub.compare_and_set_session_security(
            sid,
            principal_id="owner",
            expected_revision=a.revision,
            profile="index",
            tools_state={"fs": False},
        )
        assert updated.profile == "index" and updated.tools_state == {"fs": False}
        assert updated.revision != a.revision
        with pytest.raises(ValueError, match="conflict"):
            await hub.compare_and_set_session_security(
                sid,
                principal_id="owner",
                expected_revision=b.revision,
                profile="sandbox",
                tools_state={"fs": True},
            )
        with pytest.raises(PermissionError):
            await hub.compare_and_set_session_security(
                sid,
                principal_id="member",
                expected_revision=updated.revision,
                profile="sandbox",
                tools_state={"fs": True},
            )
        restored = UIHub(storage=SQLiteUISessionStorage(path))
        assert await restored.get_session_security_snapshot(sid, principal_id="owner") == updated

    asyncio.run(run())


@pytest.mark.parametrize("writer", ["cas", "policy", "tools"])
def test_security_sql_failure_does_not_publish_configuration_or_revision(tmp_path, writer):
    import sqlite3

    async def run():
        path = tmp_path / "sessions.db"
        hub = UIHub(storage=SQLiteUISessionStorage(path))
        sid = await hub.create_session("owner")
        before = await hub.get_session_security_snapshot(sid, principal_id="owner")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TRIGGER reject_security BEFORE UPDATE ON ui_sessions "
                "BEGIN SELECT RAISE(ABORT, 'fixture storage failure'); END"
            )
        with pytest.raises(sqlite3.DatabaseError, match="fixture storage failure"):
            if writer == "cas":
                await hub.compare_and_set_session_security(
                    sid,
                    principal_id="owner",
                    expected_revision=before.revision,
                    profile="index",
                    tools_state={"web": True},
                )
            elif writer == "policy":
                await hub.set_session_policy(sid, profile="index")
            else:
                await hub.set_session_tools_state(sid, tools_state={"web": True})
        assert await hub.get_session_security_snapshot(sid, principal_id="owner") == before
        restored = UIHub(storage=SQLiteUISessionStorage(path))
        assert await restored.get_session_security_snapshot(sid, principal_id="owner") == before

    asyncio.run(run())


def test_import_invalidates_prior_binding_and_trusted_writers_advance_revision(tmp_path):
    async def run():
        path = tmp_path / "sessions.db"
        hub = UIHub(storage=SQLiteUISessionStorage(path))
        sid = await hub.create_session("owner")
        old = await hub.get_session_security_snapshot(sid, principal_id="owner")
        exported = await hub.export_sessions("owner")
        await hub.set_session_policy(sid, profile="index")
        middle = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert middle.revision != old.revision
        await hub.set_session_policy(sid, profile="index")
        assert await hub.get_session_security_snapshot(sid, principal_id="owner") == middle
        await hub.set_session_tools_state(sid, tools_state={"fs": False})
        edited = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert edited.revision != middle.revision
        assert await hub.import_sessions(exported, principal_id="owner", mode="merge") == 1
        imported = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert imported.profile == "sandbox" and imported.tools_state is None
        assert imported.revision not in {old.revision, middle.revision, edited.revision}
        with pytest.raises(ValueError, match="conflict"):
            await hub.compare_and_set_session_security(
                sid, principal_id="owner", expected_revision=old.revision, profile="index"
            )
        restored = UIHub(storage=SQLiteUISessionStorage(path))
        assert await restored.get_session_security_snapshot(sid, principal_id="owner") == imported

    asyncio.run(run())


def test_revision_initialization_preserves_existing_sql_values_and_messages(tmp_path):
    import sqlite3

    async def run():
        path = tmp_path / "sessions.db"
        hub = UIHub(storage=SQLiteUISessionStorage(path))
        sid = await hub.create_session("owner")
        await hub.set_session_policy(sid, profile="index")
        await hub.set_session_tools_state(sid, tools_state={"web": False})
        await hub.append_message(
            sid,
            hub.create_message(role="user", content="PRIVATE_FIXTURE", lane="chat"),
            lane="chat",
        )
        with sqlite3.connect(path) as conn:
            conn.execute("ALTER TABLE ui_sessions DROP COLUMN security_revision")
            original = conn.execute(
                "SELECT principal_id, policy_profile, tools_state_json, updated_at FROM ui_sessions"
            ).fetchall()
            messages = conn.execute("SELECT * FROM chat_messages").fetchall()
        restored = UIHub(storage=SQLiteUISessionStorage(path))
        snapshot = await restored.get_session_security_snapshot(sid, principal_id="owner")
        assert snapshot.profile == "index" and snapshot.tools_state == {"web": False}
        assert len(snapshot.revision) == 32
        with sqlite3.connect(path) as conn:
            assert (
                conn.execute(
                    "SELECT principal_id, policy_profile, tools_state_json, updated_at "
                    "FROM ui_sessions"
                ).fetchall()
                == original
            )
            assert conn.execute("SELECT * FROM chat_messages").fetchall() == messages
        again = UIHub(storage=SQLiteUISessionStorage(path))
        assert await again.get_session_security_snapshot(sid, principal_id="owner") == snapshot

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
@pytest.mark.parametrize("invalid", ["unchanged_revision", "empty_expected", "stale_snapshot"])
def test_storage_rejects_unversioned_security_updates(tmp_path, backend, invalid):
    import uuid
    from dataclasses import replace

    from server.ui_session_storage import InMemoryUISessionStorage, PersistedSession

    store = (
        SQLiteUISessionStorage(tmp_path / "sessions.db")
        if backend == "sqlite"
        else InMemoryUISessionStorage()
    )
    initial = PersistedSession(
        session_id="fixture",
        principal_id="owner",
        created_at="now",
        updated_at="now",
        status="idle",
        decision=None,
        messages=[],
        policy_profile="sandbox",
    )
    store.save_session(initial)
    if invalid == "unchanged_revision":
        candidate = replace(initial, policy_profile="index")
        expected = initial.security_revision
    elif invalid == "empty_expected":
        candidate = initial
        expected = ""
    else:
        updated = replace(initial, policy_profile="index", security_revision=uuid.uuid4().hex)
        store.save_session(updated, expected_security_revision=initial.security_revision)
        candidate = initial
        expected = None
    before = store.load_sessions()
    with pytest.raises(ValueError, match="revision"):
        store.save_session(candidate, expected_security_revision=expected)
    assert store.load_sessions() == before


def test_competing_cas_is_session_bound_and_snapshots_are_detached(tmp_path):
    async def run():
        hub = UIHub(storage=SQLiteUISessionStorage(tmp_path / "sessions.db"))
        sid = await hub.create_session("owner")
        other = await hub.create_session("owner")
        await hub.set_session_tools_state(sid, tools_state={"web": False})
        before = await hub.get_session_security_snapshot(sid, principal_id="owner")
        before.tools_state["web"] = True
        captured = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert captured.tools_state == {"web": False}
        with pytest.raises(ValueError, match="conflict"):
            await hub.compare_and_set_session_security(
                other, principal_id="owner", expected_revision=captured.revision, profile="index"
            )
        results = await asyncio.gather(
            hub.compare_and_set_session_security(
                sid, principal_id="owner", expected_revision=captured.revision, profile="index"
            ),
            hub.compare_and_set_session_security(
                sid,
                principal_id="owner",
                expected_revision=captured.revision,
                tools_state={"web": True},
            ),
            return_exceptions=True,
        )
        assert sum(isinstance(item, ValueError) for item in results) == 1
        saved = await hub.get_session_security_snapshot(sid, principal_id="owner")
        assert saved.profile == "index" and saved.tools_state == {"web": False}
        restored = UIHub(storage=SQLiteUISessionStorage(tmp_path / "sessions.db"))
        assert await restored.get_session_security_snapshot(sid, principal_id="owner") == saved

    asyncio.run(run())


def test_import_storage_failure_preserves_existing_published_configuration(tmp_path):
    import sqlite3

    async def run():
        path = tmp_path / "sessions.db"
        hub = UIHub(storage=SQLiteUISessionStorage(path))
        sid = await hub.create_session("owner")
        exported = await hub.export_sessions("owner")
        await hub.set_session_policy(sid, profile="index")
        before = await hub.get_session_security_snapshot(sid, principal_id="owner")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TRIGGER reject_import BEFORE UPDATE ON ui_sessions "
                "BEGIN SELECT RAISE(ABORT, 'fixture import failure'); END"
            )
        with pytest.raises(sqlite3.DatabaseError, match="fixture import failure"):
            await hub.import_sessions(exported, principal_id="owner", mode="merge")
        assert await hub.get_session_security_snapshot(sid, principal_id="owner") == before
        restored = UIHub(storage=SQLiteUISessionStorage(path))
        assert await restored.get_session_security_snapshot(sid, principal_id="owner") == before

    asyncio.run(run())


def test_existing_null_revision_cannot_be_overwritten_as_a_new_record(tmp_path):
    import sqlite3

    async def run():
        path = tmp_path / "sessions.db"
        store = SQLiteUISessionStorage(path)
        hub = UIHub(storage=store)
        await hub.create_session("owner")
        record = store.load_sessions()[0]
        with sqlite3.connect(path) as conn:
            conn.execute("UPDATE ui_sessions SET security_revision = NULL")
        with pytest.raises(ValueError, match="conflict"):
            store.save_session(record)
        with sqlite3.connect(path) as conn:
            assert conn.execute("SELECT security_revision FROM ui_sessions").fetchone() == (None,)

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
@pytest.mark.parametrize("stored_principal", [None, "", "   ", " member ", " legacy "])
def test_startup_principal_normalization_preserves_existing_restore_contract(
    tmp_path, backend, stored_principal
):
    from dataclasses import replace

    from server.ui_session_storage import InMemoryUISessionStorage, PersistedSession

    store = (
        SQLiteUISessionStorage(tmp_path / "sessions.db")
        if backend == "sqlite"
        else InMemoryUISessionStorage()
    )
    initial = PersistedSession(
        session_id="fixture",
        principal_id=stored_principal,
        created_at="now",
        updated_at="now",
        status="idle",
        decision=None,
        messages=[],
        policy_profile="index",
        tools_state={"fs": False},
    )
    store.save_session(initial)
    # Ordinary writers cannot use the restore exception to transfer ownership.
    with pytest.raises(PermissionError, match="principal"):
        store.save_session(replace(initial, principal_id="attacker"))
    hub = UIHub(storage=store, legacy_principal_id="owner")
    expected_principal = "member" if stored_principal == " member " else "owner"

    async def read():
        snapshot = await hub.get_session_security_snapshot(
            "fixture", principal_id=expected_principal
        )
        assert snapshot.profile == "index" and snapshot.tools_state == {"fs": False}
        assert snapshot.revision == initial.security_revision
        assert await hub.get_session_security_snapshot("fixture", principal_id="attacker") is None

    asyncio.run(read())
    saved = store.load_sessions()[0]
    assert saved.principal_id == expected_principal
    assert saved.security_revision == initial.security_revision
    with pytest.raises(PermissionError, match="principal"):
        store.save_session(replace(saved, principal_id="attacker"))


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
def test_startup_normalization_cannot_reassign_named_principal_or_ignore_revision(
    tmp_path, backend
):
    import uuid
    from dataclasses import replace

    from server.ui_session_storage import InMemoryUISessionStorage, PersistedSession

    store = (
        SQLiteUISessionStorage(tmp_path / "sessions.db")
        if backend == "sqlite"
        else InMemoryUISessionStorage()
    )
    record = PersistedSession(
        session_id="fixture",
        principal_id="member",
        created_at="now",
        updated_at="now",
        status="idle",
        decision=None,
        messages=[],
    )
    store.save_session(record)
    with pytest.raises(PermissionError, match="principal"):
        store.save_session(
            replace(record, principal_id="owner"), restore_owner_principal_id="owner"
        )
    newer = replace(record, security_revision=uuid.uuid4().hex)
    store.save_session(newer, expected_security_revision=record.security_revision)
    with pytest.raises(ValueError, match="revision"):
        store.save_session(record, restore_owner_principal_id="owner")
    assert store.load_sessions() == [newer]
