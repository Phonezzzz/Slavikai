from __future__ import annotations

# ruff: noqa: F403,F405
import asyncio
from pathlib import Path

import pytest

from server.embedding_download import EmbeddingDownloadManager

from .fakes import *


def test_health_endpoint_reports_version_and_ui_state() -> None:
    async def run() -> None:
        client = await _create_client(DummyAgent())
        try:
            response = await client.get("/healthz")
            assert response.status == 200
            payload = await response.json()
            assert payload.get("status") == "ok"
            assert isinstance(payload.get("version"), str)
            assert isinstance(payload.get("ui_built"), bool)
        finally:
            await client.close()

    asyncio.run(run())


def test_embeddings_status_and_download_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("server.http_api.UI_SETTINGS_PATH", tmp_path / "ui_settings.json")
    downloaded: list[str] = []

    async def run() -> None:
        client = await _create_client(DummyAgent())
        client.server.app["embedding_download_manager"] = EmbeddingDownloadManager(
            package_available=lambda: True,
            model_cached=lambda model: False,
            download_model=downloaded.append,
        )
        try:
            status_response = await client.get("/ui/api/embeddings/status")
            assert status_response.status == 200
            assert (await status_response.json()).get("state") == "missing"

            download_response = await client.post(
                "/ui/api/embeddings/download",
                json={"confirm": True, "model": "all-MiniLM-L6-v2"},
            )
            assert download_response.status == 202
            assert (await download_response.json()).get("state") == "downloading"

            for _ in range(100):
                status_response = await client.get("/ui/api/embeddings/status")
                payload = await status_response.json()
                if payload.get("state") == "ready":
                    break
                await asyncio.sleep(0)
            assert payload.get("state") == "ready"
            assert downloaded == ["all-MiniLM-L6-v2"]
        finally:
            await client.close()

    asyncio.run(run())
