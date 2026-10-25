import asyncio
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from memory.canonical_atom_store import CanonicalAtomStore
from server.http.app import create_app
from server.http.common.request_identity import VerifiedAccessClaims
from server.principal_storage import principal_storage_paths
from server.ui_session_storage import InMemoryUISessionStorage
from shared.canonical_atom_models import ClaimType
from shared.identity import principal_id_for_email
from tests.test_cloudflare_access import StubAccessVerifier


def _paths(root, who):
    return principal_storage_paths(
        principal_id=principal_id_for_email(who + "@example.com"),
        owner_principal_id=principal_id_for_email("owner@example.com"),
        memory_root=root / "memory",
    )


def _seed(root, who, key="fact:fixture"):
    store = CanonicalAtomStore(str(_paths(root, who).canonical_atoms_db))
    atom = store.create_atom(
        atom_id=who + "-atom",
        stable_key=key,
        claim_type=ClaimType.FACT,
        value_json={"text": who + "_PRIVATE"},
        confidence=0.9,
        summary_text=who + "_PRIVATE",
    )
    store.close()
    return atom


async def _start():
    app = create_app(
        ui_storage=InMemoryUISessionStorage(),
        auth_config=HttpAuthConfig(
            api_token="automation",
            browser_auth_mode="cloudflare",
            cloudflare_access_issuer="https://example.cloudflareaccess.com",
            cloudflare_access_aud="test",
            owner_email="owner@example.com",
        ),
        cloudflare_access_verifier=StubAccessVerifier(
            {who: VerifiedAccessClaims(email=who + "@example.com") for who in ["owner", "member"]}
        ),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


def _isolate(monkeypatch, root):
    monkeypatch.setattr("server.http_api.PROJECT_ROOT", root)
    monkeypatch.setattr("server.http.app.load_model_configs", lambda: None)


def test_owned_list_preview_confirm_without_model_or_session(monkeypatch, tmp_path: Path):
    _isolate(monkeypatch, tmp_path)

    def no_encoding(*args, **kwargs):
        raise AssertionError("Memory management must not initialize/use embeddings")

    monkeypatch.setattr("memory.vector_index.VectorIndex._encode_texts", no_encoding)
    owner, member = _seed(tmp_path, "owner"), _seed(tmp_path, "member")

    async def run():
        client = await _start()
        try:
            h = {"Cf-Access-Jwt-Assertion": "member"}
            response = await client.get("/ui/api/memory/items", headers=h)
            assert response.status == 200
            data = await response.json()
            assert response.headers["Cache-Control"] == "no-store"
            assert [i["value_json"] for i in data["items"]] == [{"text": "member_PRIVATE"}]
            assert data["next_cursor"] is None and data["subject"]
            forbidden = await client.post(
                "/ui/api/memory/delete/preview",
                headers=h,
                json={"atom_id": owner.atom_id, "revision": owner.revision},
            )
            assert forbidden.status == 404
            preview = await client.post(
                "/ui/api/memory/delete/preview",
                headers=h,
                json={"atom_id": member.atom_id, "revision": member.revision},
            )
            assert preview.status == 200
            binding = await preview.json()
            op = binding["operation_id"]
            no = await client.post(
                "/ui/api/memory/delete/confirm",
                headers=h,
                json={"operation_id": op, "confirm": False},
            )
            assert no.status == 400
            yes = await client.post(
                "/ui/api/memory/delete/confirm",
                headers=h,
                json={"operation_id": op, "confirm": True},
            )
            assert yes.status == 200 and (await yes.json())["status"] == "deleted"
            again = await client.post(
                "/ui/api/memory/delete/confirm",
                headers=h,
                json={"operation_id": op, "confirm": True},
            )
            assert again.status == 200 and (await again.json())["status"] == "deleted"
            empty = await client.get("/ui/api/memory/items", headers=h)
            assert (await empty.json())["items"] == []
            other = await client.get(
                "/ui/api/memory/items", headers={"Cf-Access-Jwt-Assertion": "owner"}
            )
            assert len((await other.json())["items"]) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_pagination_auth_cursor_and_no_secret_logs(monkeypatch, tmp_path):
    from shared.canonical_atom_models import AtomStatus, CanonicalAtom, utc_now_iso

    _isolate(monkeypatch, tmp_path)
    store = CanonicalAtomStore(str(_paths(tmp_path, "member").canonical_atoms_db))
    store.upsert_many(
        [
            CanonicalAtom(
                atom_id=f"atom-{i:04d}",
                stable_key=f"fact:{i}",
                claim_type=ClaimType.FACT,
                value_json={"text": f"PRIVATE_{i}"},
                confidence=0.9,
                support_count=1,
                contradict_count=0,
                last_seen_at=utc_now_iso(),
                status=AtomStatus.ACTIVE,
                summary_text=f"PRIVATE_{i}",
            )
            for i in range(405)
        ]
    )
    store.close()

    async def run():
        client = await _start()
        try:
            h = {
                "Cf-Access-Jwt-Assertion": "member",
                "Cf-Access-Authenticated-User-Email": "owner@example.com",
            }
            for headers in [
                {},
                {"Authorization": "Bearer automation"},
                {"Cf-Access-Jwt-Assertion": "forged"},
            ]:
                assert (await client.get("/ui/api/memory/items", headers=headers)).status == 401
            seen = []
            cursor = None
            first_cursor = None
            while True:
                params = {"limit": "100"}
                if cursor:
                    params["cursor"] = cursor
                response = await client.get("/ui/api/memory/items", headers=h, params=params)
                assert response.status == 200
                page = await response.json()
                seen.extend(x["atom_id"] for x in page["items"])
                cursor = page["next_cursor"]
                if first_cursor is None:
                    first_cursor = cursor
                if cursor is None:
                    break
            assert len(seen) == len(set(seen)) == 405
            foreign = await client.get(
                "/ui/api/memory/items",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
                params={"cursor": first_cursor},
            )
            assert foreign.status == 400 and "PRIVATE_" not in await foreign.text()
            for query in [{"limit": "0"}, {"limit": "101"}, {"limit": "bad"}, {"cursor": "%%%"}]:
                assert (
                    await client.get("/ui/api/memory/items", headers=h, params=query)
                ).status == 400
        finally:
            await client.close()

    asyncio.run(run())
    logs = list((tmp_path / "memory").rglob("memory_tool_calls.jsonl"))
    assert all("PRIVATE_" not in p.read_text() for p in logs)


def test_preview_restart_foreign_operation_stale_revision_and_expiry(monkeypatch, tmp_path):
    import time
    from dataclasses import replace

    _isolate(monkeypatch, tmp_path)
    atom = _seed(tmp_path, "member")

    async def preview(client):
        response = await client.post(
            "/ui/api/memory/delete/preview",
            headers={"Cf-Access-Jwt-Assertion": "member"},
            json={"atom_id": atom.atom_id, "revision": atom.revision},
        )
        assert response.status == 200
        return (await response.json())["operation_id"]

    async def run():
        client = await _start()
        op = await preview(client)
        await client.close()
        client = await _start()
        try:
            route = "/ui/api/memory/delete/confirm"
            payload = {"operation_id": op, "confirm": True}
            other = await client.post(
                route, headers={"Cf-Access-Jwt-Assertion": "owner"}, json=payload
            )
            assert other.status == 404
            h = {"Cf-Access-Jwt-Assertion": "member"}
            forged = await client.post(route, headers=h, json={**payload, "principal_id": "owner"})
            assert forged.status == 400
            store = CanonicalAtomStore(str(_paths(tmp_path, "member").canonical_atoms_db))
            changed = store.upsert(replace(atom, summary_text="new"))
            stale = await client.post(route, headers=h, json=payload)
            assert stale.status == 409
            fresh = await client.post(
                "/ui/api/memory/delete/preview",
                headers=h,
                json={"atom_id": changed.atom_id, "revision": changed.revision},
            )
            newop = (await fresh.json())["operation_id"]
            store.conn.execute(
                "UPDATE atom_delete_preview SET expires_at = ? WHERE operation_id = ?",
                (time.time() - 1, newop),
            )
            store.conn.commit()
            expired = await client.post(
                route, headers=h, json={"operation_id": newop, "confirm": True}
            )
            assert expired.status == 410
            assert store.get_by_id(atom.atom_id).summary_text == "new"
            store.close()
        finally:
            await client.close()

    asyncio.run(run())


def test_confirmation_replays_after_restart_and_logs_only_opaque_binding(monkeypatch, tmp_path):
    from core.tool_gateway import ToolGateway

    _isolate(monkeypatch, tmp_path)
    atom = _seed(tmp_path, "member")
    calls = []
    original = ToolGateway.call

    def observe(self, request):
        descriptor = self.registry.get_descriptor(request.name)
        calls.append(
            (
                self.confirmed_decision,
                descriptor.model_exposed,
                descriptor.confirmed_decision_only,
                request,
            )
        )
        return original(self, request)

    monkeypatch.setattr(ToolGateway, "call", observe)

    async def run():
        client = await _start()
        h = {"Cf-Access-Jwt-Assertion": "member"}
        response = await client.post(
            "/ui/api/memory/delete/preview",
            headers=h,
            json={"atom_id": atom.atom_id, "revision": atom.revision},
        )
        op = (await response.json())["operation_id"]
        route = "/ui/api/memory/delete/confirm"
        body = {"operation_id": op, "confirm": True}
        done = await client.post(route, headers=h, json=body)
        assert done.status == 200
        await client.close()
        client = await _start()
        try:
            replay = await client.post(route, headers=h, json=body)
            assert replay.status == 200 and (await replay.json())["status"] == "deleted"
        finally:
            await client.close()

    asyncio.run(run())
    assert len(calls) == 2
    assert all(
        confirmed and not exposed and required and set(req.args) == {"operation_id"}
        for confirmed, exposed, required, req in calls
    )
    log = (
        _paths(tmp_path, "member")
        .canonical_atoms_db.with_name("memory_tool_calls.jsonl")
        .read_text()
    )
    assert "member_PRIVATE" not in log and "fact:fixture" not in log


def test_concurrent_confirm_and_storage_failure_are_idempotent_and_fail_closed(
    monkeypatch, tmp_path
):
    from memory.vector_index import VectorIndex

    _isolate(monkeypatch, tmp_path)
    atom = _seed(tmp_path, "member")
    paths = _paths(tmp_path, "member")
    vectors = VectorIndex(str(paths.vectors_db))

    async def run():
        client = await _start()
        h = {"Cf-Access-Jwt-Assertion": "member"}
        try:
            p = await client.post(
                "/ui/api/memory/delete/preview",
                headers=h,
                json={"atom_id": atom.atom_id, "revision": atom.revision},
            )
            op = (await p.json())["operation_id"]
            body = {"operation_id": op, "confirm": True}
            vectors.conn.execute("PRAGMA journal_mode=WAL")
            failed = await client.post("/ui/api/memory/delete/confirm", headers=h, json=body)
            assert (
                failed.status == 503
                and (await failed.json())["error"]["code"] == "memory_storage_unavailable"
            )
            store = CanonicalAtomStore(str(paths.canonical_atoms_db))
            assert store.get_by_id(atom.atom_id) is not None
            store.close()
            vectors.conn.execute("PRAGMA journal_mode=DELETE")
            responses = await asyncio.gather(
                *[
                    client.post("/ui/api/memory/delete/confirm", headers=h, json=body)
                    for _ in range(2)
                ]
            )
            assert all(r.status == 200 for r in responses)
            assert [(await r.json())["status"] for r in responses] == ["deleted", "deleted"]
        finally:
            await client.close()
            vectors.close()

    asyncio.run(run())


def test_server_delete_tool_is_not_exposed_or_callable_without_confirmation(tmp_path):
    from memory.management import MemoryManagement
    from shared.models import ToolRequest

    service = MemoryManagement(
        canonical_path=tmp_path / "atoms.db",
        vectors_path=tmp_path / "vectors.db",
        principal_id="fixture",
    )
    try:
        assert service.registry.list_tool_specs() == []
        result = service.registry.call(
            ToolRequest(name="memory_delete_confirmed", args={"operation_id": "unknown"})
        )
        assert not result.ok and "CONFIRMED_DECISION_REQUIRED" in result.error
        assert not (tmp_path / "vectors.db").exists()
    finally:
        service.close()


def test_simple_cross_site_posts_do_not_create_or_confirm_deletion(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    atom = _seed(tmp_path, "member")

    async def run():
        client = await _start()
        h = {"Cf-Access-Jwt-Assertion": "member"}
        try:
            import json

            payload = {"atom_id": atom.atom_id, "revision": atom.revision}
            simple = await client.post(
                "/ui/api/memory/delete/preview",
                headers={**h, "Content-Type": "text/plain"},
                data=json.dumps(payload),
            )
            assert simple.status == 415
            cross = await client.post(
                "/ui/api/memory/delete/preview",
                headers={**h, "Sec-Fetch-Site": "cross-site"},
                json=payload,
            )
            assert cross.status == 403
            normal = await client.post("/ui/api/memory/delete/preview", headers=h, json=payload)
            op = (await normal.json())["operation_id"]
            denied = await client.post(
                "/ui/api/memory/delete/confirm",
                headers={**h, "Sec-Fetch-Site": "cross-site"},
                json={"operation_id": op, "confirm": True},
            )
            assert denied.status == 403
            left = await client.get("/ui/api/memory/items", headers=h)
            assert len((await left.json())["items"]) == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_action_log_records_rejected_and_successful_delete_outcomes(monkeypatch, tmp_path):
    import json
    import time
    from dataclasses import replace

    _isolate(monkeypatch, tmp_path)
    atom = _seed(tmp_path, "member")

    async def run():
        client = await _start()
        h = {"Cf-Access-Jwt-Assertion": "member"}
        route = "/ui/api/memory/delete/confirm"

        async def preview(current):
            response = await client.post(
                "/ui/api/memory/delete/preview",
                headers=h,
                json={"atom_id": current.atom_id, "revision": current.revision},
            )
            assert response.status == 200
            return (await response.json())["operation_id"]

        try:
            unknown = await client.post(
                route, headers=h, json={"operation_id": "unknown", "confirm": True}
            )
            assert unknown.status == 404
            stale_op = await preview(atom)
            store = CanonicalAtomStore(str(_paths(tmp_path, "member").canonical_atoms_db))
            current = store.upsert(replace(atom, summary_text="changed"))
            stale = await client.post(
                route, headers=h, json={"operation_id": stale_op, "confirm": True}
            )
            assert stale.status == 409
            expired_op = await preview(current)
            store.conn.execute(
                "UPDATE atom_delete_preview SET expires_at = ? WHERE operation_id = ?",
                (time.time() - 1, expired_op),
            )
            store.conn.commit()
            expired = await client.post(
                route, headers=h, json={"operation_id": expired_op, "confirm": True}
            )
            assert expired.status == 410
            valid_op = await preview(current)
            done = await client.post(
                route, headers=h, json={"operation_id": valid_op, "confirm": True}
            )
            assert done.status == 200
            store.close()
        finally:
            await client.close()

    asyncio.run(run())
    path = _paths(tmp_path, "member").canonical_atoms_db.with_name("memory_tool_calls.jsonl")
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(e["ok"], (e.get("meta") or {}).get("status")) for e in entries] == [
        (False, "not_found"),
        (False, "conflict"),
        (False, "expired"),
        (True, "deleted"),
    ]
    assert all(set(e["args"]) == {"operation_id"} for e in entries)
    assert "member_PRIVATE" not in path.read_text()
