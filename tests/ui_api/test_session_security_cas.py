import asyncio

import pytest

from server.ui_hub import UIHub
from server.ui_session_storage import SQLiteUISessionStorage
from shared.identity import principal_id_for_email
from tests.ui_api.test_memory_management import _isolate, _start


async def _client_with_storage(path):
    client = await _start()
    client.app["ui_hub"] = UIHub(
        storage=SQLiteUISessionStorage(path),
        legacy_principal_id=principal_id_for_email("owner@example.com"),
    )
    return client


@pytest.mark.parametrize("who", ["owner", "member"])
def test_missing_security_precondition_never_changes_saved_configuration(
    monkeypatch, tmp_path, who
):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _client_with_storage(tmp_path / "sessions.db")
        try:
            hub = client.app["ui_hub"]
            sid = await hub.create_session(principal_id_for_email(who + "@example.com"))
            await hub.set_workspace_root(sid, str(tmp_path))
            before = hub._storage.load_sessions()
            response = await client.post(
                "/ui/api/session/security",
                headers={"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": sid},
                json={"policy": {"profile": "index"}, "tools": {"state": {"fs": False}}},
            )
            assert response.status == 428
            assert hub._storage.load_sessions() == before
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("who", ["owner", "member"])
def test_security_cas_keeps_first_writer_and_survives_restart(monkeypatch, tmp_path, who):
    _isolate(monkeypatch, tmp_path)

    async def run():
        path = tmp_path / "sessions.db"
        client = await _client_with_storage(path)
        try:
            hub = client.app["ui_hub"]
            principal = principal_id_for_email(who + "@example.com")
            sid = await hub.create_session(principal)
            other = await hub.create_session(principal)
            await hub.set_workspace_root(sid, str(tmp_path))
            headers = {"Cf-Access-Jwt-Assertion": who, "X-Slavik-Session": sid}
            a = await client.get("/ui/api/session/security", headers=headers)
            b = await client.get("/ui/api/session/security", headers=headers)
            captured = await a.json()
            assert (
                a.headers["ETag"] == b.headers["ETag"] == '"' + captured["security_revision"] + '"'
            )
            assert a.headers["Cache-Control"] == "no-store"
            matching = {**headers, "If-Match": a.headers["ETag"]}
            saved = await client.post(
                "/ui/api/session/security",
                headers=matching,
                json={"policy": {"profile": "index"}, "tools": {"state": {"fs": False}}},
            )
            assert saved.status == 200
            result = await saved.json()
            assert saved.headers["ETag"] != a.headers["ETag"]
            assert result["policy"] == {"profile": "index"}
            assert result["tools_state"]["fs"] is False
            assert result["tool_sources"]["fields"]["fs"] == "session_override"
            before = hub._storage.load_sessions()
            for attempted_headers, expected_status in [
                (matching, 409),
                ({**matching, "X-Slavik-Session": other}, 409),
                (
                    {
                        **matching,
                        "Cf-Access-Jwt-Assertion": "member" if who == "owner" else "owner",
                    },
                    403,
                ),
            ]:
                stale = await client.post(
                    "/ui/api/session/security",
                    headers=attempted_headers,
                    json={"policy": {"profile": "sandbox"}, "tools": {"state": {"fs": True}}},
                )
                assert stale.status == expected_status
                assert hub._storage.load_sessions() == before
            restored = UIHub(storage=SQLiteUISessionStorage(path))
            snapshot = await restored.get_session_security_snapshot(sid, principal_id=principal)
            assert snapshot.revision == result["security_revision"]
            assert snapshot.profile == "index" and snapshot.tools_state == {"fs": False}
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("etag", ["", "*", 'W/"' + "a" * 32 + '"', '"wrong"'])
def test_malformed_precondition_cannot_write(monkeypatch, tmp_path, etag):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _client_with_storage(tmp_path / "sessions.db")
        try:
            hub = client.app["ui_hub"]
            sid = await hub.create_session(principal_id_for_email("owner@example.com"))
            before = hub._storage.load_sessions()
            response = await client.post(
                "/ui/api/session/security",
                headers={
                    "Cf-Access-Jwt-Assertion": "owner",
                    "X-Slavik-Session": sid,
                    "If-Match": etag,
                },
                json={"tools": {"state": {"fs": False}}},
            )
            assert response.status == 400
            assert hub._storage.load_sessions() == before
        finally:
            await client.close()

    asyncio.run(run())


def test_http_combined_storage_failure_never_publishes_configuration(monkeypatch, tmp_path):
    import sqlite3

    _isolate(monkeypatch, tmp_path)

    async def run():
        path = tmp_path / "sessions.db"
        client = await _client_with_storage(path)
        try:
            hub = client.app["ui_hub"]
            principal = principal_id_for_email("owner@example.com")
            sid = await hub.create_session(principal)
            await hub.set_workspace_root(sid, str(tmp_path))
            headers = {"Cf-Access-Jwt-Assertion": "owner", "X-Slavik-Session": sid}
            read = await client.get("/ui/api/session/security", headers=headers)
            before = await read.json()
            with sqlite3.connect(path) as conn:
                conn.execute(
                    "CREATE TRIGGER reject_config BEFORE UPDATE ON ui_sessions "
                    "BEGIN SELECT RAISE(ABORT, 'fixture storage fault'); END"
                )
            response = await client.post(
                "/ui/api/session/security",
                headers={**headers, "If-Match": read.headers["ETag"]},
                json={"policy": {"profile": "index"}, "tools": {"state": {"fs": False}}},
            )
            assert response.status == 500
            after = await client.get("/ui/api/session/security", headers=headers)
            assert await after.json() == before
            assert after.headers["ETag"] == read.headers["ETag"]
            restored = UIHub(storage=SQLiteUISessionStorage(path))
            snapshot = await restored.get_session_security_snapshot(sid, principal_id=principal)
            assert (
                snapshot.revision == before["security_revision"] and snapshot.profile == "sandbox"
            )
            assert snapshot.tools_state is None
        finally:
            await client.close()

    asyncio.run(run())


def test_competing_http_requests_with_one_binding_have_one_winner(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)

    async def run():
        client = await _client_with_storage(tmp_path / "sessions.db")
        try:
            hub = client.app["ui_hub"]
            sid = await hub.create_session(principal_id_for_email("owner@example.com"))
            await hub.set_workspace_root(sid, str(tmp_path))
            headers = {"Cf-Access-Jwt-Assertion": "owner", "X-Slavik-Session": sid}
            read = await client.get("/ui/api/session/security", headers=headers)
            matching = {**headers, "If-Match": read.headers["ETag"]}
            responses = await asyncio.gather(
                *[
                    client.post(
                        "/ui/api/session/security",
                        headers=matching,
                        json={"tools": {"state": {"fs": enabled}}},
                    )
                    for enabled in [False, True]
                ]
            )
            assert sorted(response.status for response in responses) == [200, 409]
            winner = next(response for response in responses if response.status == 200)
            saved = await winner.json()
            current = await client.get("/ui/api/session/security", headers=headers)
            assert await current.json() == saved
        finally:
            await client.close()

    asyncio.run(run())
