import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config.http_server_config import HttpAuthConfig
from config.ui_embeddings_settings import UIEmbeddingsSettings
from server.embedding_download import EmbeddingDownloadManager
from server.http.app import create_app
from server.http.common.request_identity import VerifiedAccessClaims
from server.ui_session_storage import InMemoryUISessionStorage
from tests.test_cloudflare_access import StubAccessVerifier
from tests.ui_api.fakes import DummyAgent


def test_shared_embedding_download_requires_owner_and_explicit_confirm(monkeypatch):
    monkeypatch.setattr(
        "server.http.handlers.embeddings._load_embeddings_settings", UIEmbeddingsSettings
    )

    async def run():
        downloads = []
        app = create_app(
            agent=DummyAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(
                browser_auth_mode="cloudflare",
                api_token="synthetic",
                cloudflare_access_issuer="https://example.cloudflareaccess.com",
                cloudflare_access_aud="test",
                owner_email="owner@example.com",
            ),
            cloudflare_access_verifier=StubAccessVerifier(
                {
                    who: VerifiedAccessClaims(email=who + "@example.com")
                    for who in ["owner", "member"]
                }
            ),
        )
        manager = EmbeddingDownloadManager(
            package_available=lambda: True,
            model_cached=lambda _: False,
            download_model=downloads.append,
        )
        app["embedding_download_manager"] = manager
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            for who, allowed in [("owner", True), ("member", False)]:
                response = await client.get(
                    "/ui/api/embeddings/status", headers={"Cf-Access-Jwt-Assertion": who}
                )
                assert response.status == 200
                assert (await response.json())["can_download"] is allowed
            assert downloads == []
            denied = await client.post(
                "/ui/api/embeddings/download",
                headers={"Cf-Access-Jwt-Assertion": "member"},
                json={"confirm": True},
            )
            assert denied.status == 403
            await denied.read()
            assert downloads == []
            status = await manager.status("all-MiniLM-L6-v2")
            assert status.state == "missing"
            unconfirmed = await client.post(
                "/ui/api/embeddings/download", headers={"Cf-Access-Jwt-Assertion": "owner"}, json={}
            )
            assert unconfirmed.status == 400
            await unconfirmed.read()
            assert downloads == []
            confirmed = await client.post(
                "/ui/api/embeddings/download",
                headers={"Cf-Access-Jwt-Assertion": "owner"},
                json={"confirm": True},
            )
            assert confirmed.status == 202
            await confirmed.read()
            await manager.shutdown()
            assert downloads == ["all-MiniLM-L6-v2"]
        finally:
            await client.close()

    asyncio.run(run())


def test_cached_activation_error_returns_runtime_snapshot():
    async def run():
        def fail(_model):
            raise RuntimeError("cached files invalid")

        app = create_app(
            agent=DummyAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token=None, allow_unauth_local=True),
        )
        app["embedding_download_manager"] = EmbeddingDownloadManager(
            package_available=lambda: True,
            model_cached=lambda _: True,
            load_cached_model=fail,
        )
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/ui/api/embeddings/download", json={"confirm": True})
            assert response.status == 503
            payload = await response.json()
            assert payload["error"]["details"]["state"] == "error"
            assert payload["error"]["details"]["error"] == "cached files invalid"

    asyncio.run(run())


@pytest.mark.parametrize(
    "package,cached,provider",
    [(False, False, "local"), (True, True, "local"), (True, False, "openai")],
)
def test_status_does_not_offer_download_without_a_missing_local_model(
    monkeypatch, package, cached, provider
):
    monkeypatch.setattr(
        "server.http.handlers.embeddings._load_embeddings_settings",
        lambda: UIEmbeddingsSettings(provider=provider),
    )

    async def run():
        app = create_app(
            agent=DummyAgent(),
            ui_storage=InMemoryUISessionStorage(),
            auth_config=HttpAuthConfig(api_token=None, allow_unauth_local=True),
        )
        app["embedding_download_manager"] = EmbeddingDownloadManager(
            package_available=lambda: package, model_cached=lambda _: cached
        )
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/ui/api/embeddings/status")
            assert response.status == 200
            assert (await response.json())["can_download"] is False

    asyncio.run(run())
