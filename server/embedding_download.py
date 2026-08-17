from __future__ import annotations

import asyncio
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from shared.models import JSONValue

EmbeddingDownloadState = Literal[
    "missing",
    "package_missing",
    "downloading",
    "ready",
    "error",
]


@dataclass(frozen=True)
class EmbeddingDownloadSnapshot:
    model: str
    state: EmbeddingDownloadState
    error: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "model": self.model,
            "state": self.state,
            "error": self.error,
        }


def _package_available() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def _repository_id(model: str) -> str:
    normalized = model.strip()
    if "/" in normalized:
        return normalized
    return f"sentence-transformers/{normalized}"


def _model_cached(model: str) -> bool:
    if importlib.util.find_spec("huggingface_hub") is None:
        return False
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo_id=_repository_id(model), local_files_only=True)
    except Exception:  # noqa: BLE001
        return False
    return True


def _download_model(model: str) -> None:
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model)


class EmbeddingDownloadManager:
    def __init__(
        self,
        *,
        package_available: Callable[[], bool] = _package_available,
        model_cached: Callable[[str], bool] = _model_cached,
        download_model: Callable[[str], None] = _download_model,
    ) -> None:
        self._package_available = package_available
        self._model_cached = model_cached
        self._download_model = download_model
        self._lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._snapshot: EmbeddingDownloadSnapshot | None = None

    async def status(self, model: str) -> EmbeddingDownloadSnapshot:
        normalized = model.strip()
        if not normalized:
            return EmbeddingDownloadSnapshot(model=model, state="error", error="Model is empty.")
        async with self._lock:
            current = self._snapshot
            if current is not None and current.model == normalized:
                if current.state in {"downloading", "ready", "error"}:
                    return current
            if not self._package_available():
                snapshot = EmbeddingDownloadSnapshot(
                    model=normalized,
                    state="package_missing",
                    error="Local embeddings dependencies are not installed.",
                )
            elif self._model_cached(normalized):
                snapshot = EmbeddingDownloadSnapshot(model=normalized, state="ready")
            else:
                snapshot = EmbeddingDownloadSnapshot(model=normalized, state="missing")
            self._snapshot = snapshot
            return snapshot

    async def start(self, model: str) -> tuple[EmbeddingDownloadSnapshot, bool]:
        normalized = model.strip()
        if not normalized:
            return (
                EmbeddingDownloadSnapshot(model=model, state="error", error="Model is empty."),
                False,
            )
        async with self._lock:
            if self._task is not None and not self._task.done():
                current = self._snapshot or EmbeddingDownloadSnapshot(
                    model=normalized,
                    state="downloading",
                )
                return current, False
            if not self._package_available():
                snapshot = EmbeddingDownloadSnapshot(
                    model=normalized,
                    state="package_missing",
                    error="Local embeddings dependencies are not installed.",
                )
                self._snapshot = snapshot
                return snapshot, False
            if self._model_cached(normalized):
                snapshot = EmbeddingDownloadSnapshot(model=normalized, state="ready")
                self._snapshot = snapshot
                return snapshot, False
            snapshot = EmbeddingDownloadSnapshot(model=normalized, state="downloading")
            self._snapshot = snapshot
            self._task = asyncio.create_task(self._run_download(normalized))
            return snapshot, True

    async def _run_download(self, model: str) -> None:
        try:
            await asyncio.to_thread(self._download_model, model)
        except Exception as exc:  # noqa: BLE001
            snapshot = EmbeddingDownloadSnapshot(model=model, state="error", error=str(exc))
        else:
            snapshot = EmbeddingDownloadSnapshot(model=model, state="ready")
        async with self._lock:
            self._snapshot = snapshot

    async def shutdown(self) -> None:
        async with self._lock:
            task = self._task
        if task is None or task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
