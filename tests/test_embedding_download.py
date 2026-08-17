from __future__ import annotations

import asyncio

from server.embedding_download import EmbeddingDownloadManager


def test_embedding_status_reports_missing_package() -> None:
    async def run() -> None:
        manager = EmbeddingDownloadManager(package_available=lambda: False)
        snapshot = await manager.status("all-MiniLM-L6-v2")
        assert snapshot.state == "package_missing"
        assert snapshot.error is not None

    asyncio.run(run())


def test_embedding_download_reaches_ready_state() -> None:
    downloaded: list[str] = []

    async def run() -> None:
        manager = EmbeddingDownloadManager(
            package_available=lambda: True,
            model_cached=lambda model: False,
            download_model=downloaded.append,
        )
        snapshot, started = await manager.start("all-MiniLM-L6-v2")
        assert started is True
        assert snapshot.state == "downloading"
        for _ in range(100):
            snapshot = await manager.status("all-MiniLM-L6-v2")
            if snapshot.state == "ready":
                break
            await asyncio.sleep(0)
        assert snapshot.state == "ready"
        assert downloaded == ["all-MiniLM-L6-v2"]

    asyncio.run(run())


def test_cached_embedding_model_is_ready_without_download() -> None:
    async def run() -> None:
        manager = EmbeddingDownloadManager(
            package_available=lambda: True,
            model_cached=lambda model: model == "cached-model",
            download_model=lambda model: (_ for _ in ()).throw(AssertionError(model)),
        )
        snapshot, started = await manager.start("cached-model")
        assert started is False
        assert snapshot.state == "ready"

    asyncio.run(run())
