from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, cast

import numpy as np
import requests

from shared.models import JSONValue, VectorSearchResult

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

EmbeddingsProvider = Literal["local", "openai"]

DEFAULT_LOCAL_EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OPENAI_EMBEDDINGS_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"


class VectorIndex:
    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        db_path: str = "memory/vectors.db",
        model_name: str | None = None,
        max_records: int = 5000,
        max_total_records: int = 20000,
        batch_size: int = 32,
        provider: EmbeddingsProvider = "local",
        local_model: str = DEFAULT_LOCAL_EMBEDDINGS_MODEL,
        openai_model: str = DEFAULT_OPENAI_EMBEDDINGS_MODEL,
        openai_api_key: str | None = None,
        openai_base_url: str = DEFAULT_OPENAI_BASE_URL,
        openai_timeout_seconds: int = 30,
    ):
        normalized_provider = provider.strip().lower() if isinstance(provider, str) else "local"
        if normalized_provider not in {"local", "openai"}:
            raise ValueError(f"Неизвестный embeddings provider: {provider}")

        selected_local_model = local_model.strip() if isinstance(local_model, str) else ""
        if model_name is not None and isinstance(model_name, str) and model_name.strip():
            selected_local_model = model_name.strip()
        if not selected_local_model:
            selected_local_model = DEFAULT_LOCAL_EMBEDDINGS_MODEL

        selected_openai_model = openai_model.strip() if isinstance(openai_model, str) else ""
        if not selected_openai_model:
            selected_openai_model = DEFAULT_OPENAI_EMBEDDINGS_MODEL

        self.db_path = db_path
        self.provider = cast(EmbeddingsProvider, normalized_provider)
        self.local_model = selected_local_model
        self.openai_model = selected_openai_model
        self.model_name = self.local_model if self.provider == "local" else self.openai_model
        self.openai_api_key = openai_api_key.strip() if isinstance(openai_api_key, str) else None
        base = openai_base_url.strip() if isinstance(openai_base_url, str) else ""
        self.openai_base_url = base or DEFAULT_OPENAI_BASE_URL
        self.openai_timeout_seconds = openai_timeout_seconds
        self.max_records = max_records
        self.max_total_records = max_total_records
        self.batch_size = batch_size
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.model: SentenceTransformer | None = None
        self._load_error: Exception | None = None
        self._init_db()

    @classmethod
    def _get_model(cls, model_name: str) -> SentenceTransformer:
        model = cls._model_cache.get(model_name)
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Local embeddings provider requires sentence-transformers package"
                ) from exc
            model = SentenceTransformer(model_name)
            cls._model_cache[model_name] = model
        return model

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT DEFAULT 'default',
                path TEXT,
                content TEXT,
                embedding BLOB,
                meta TEXT
            )
            """
        )
        # миграция старых схем
        columns = {row[1] for row in cur.execute("PRAGMA table_info(vectors)").fetchall()}
        if "namespace" not in columns:
            cur.execute("ALTER TABLE vectors ADD COLUMN namespace TEXT DEFAULT 'default'")
        if "meta" not in columns:
            cur.execute("ALTER TABLE vectors ADD COLUMN meta TEXT")
        self.conn.commit()

    def index_text(
        self,
        path: str,
        content: str,
        namespace: str = "default",
        meta: dict[str, JSONValue] | None = None,
    ) -> None:
        if not content.strip():
            return
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("meta для VectorIndex должно быть словарём")
        embedding = self._encode_texts([content])[0].astype(np.float32).tobytes()
        with self.conn:
            self.conn.execute(
                "INSERT INTO vectors (namespace, path, content, embedding, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    namespace,
                    path,
                    content[:5000],
                    embedding,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            self._prune_namespace(namespace)
            self._prune_total()

    def upsert_text(
        self,
        path: str,
        content: str,
        namespace: str = "default",
        meta: dict[str, JSONValue] | None = None,
    ) -> None:
        self.delete_path(path, namespace=namespace)
        self.index_text(path, content, namespace=namespace, meta=meta)

    def index_batch(
        self,
        items: list[tuple[str, str]],
        namespace: str = "default",
        meta: dict[str, JSONValue] | None = None,
    ) -> None:
        if not items:
            return
        if meta is not None and not isinstance(meta, dict):
            raise ValueError("meta для VectorIndex должно быть словарём")
        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start : batch_start + self.batch_size]
            paths, texts = zip(*batch, strict=False)
            embeddings = self._encode_texts(list(texts))
            with self.conn:
                for path, content, embedding in zip(paths, texts, embeddings, strict=False):
                    if not content.strip():
                        continue
                    self.conn.execute(
                        "INSERT INTO vectors (namespace, path, content, embedding, meta) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            namespace,
                            path,
                            content[:5000],
                            embedding.astype(np.float32).tobytes(),
                            json.dumps(meta or {}, ensure_ascii=False),
                        ),
                    )
                self._prune_namespace(namespace)
                self._prune_total()

    def _prune_namespace(self, namespace: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id FROM vectors WHERE namespace = ? ORDER BY id DESC",
            (namespace,),
        )
        ids = [row[0] for row in cur.fetchall()]
        if len(ids) <= self.max_records:
            return
        to_delete = ids[self.max_records :]
        if to_delete:
            cur.execute(
                f"DELETE FROM vectors WHERE id IN ({','.join('?' for _ in to_delete)})",
                to_delete,
            )
            self.conn.commit()

    def _prune_total(self) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM vectors ORDER BY id DESC")
        ids = [row[0] for row in cur.fetchall()]
        if len(ids) <= self.max_total_records:
            return
        to_delete = ids[self.max_total_records :]
        if to_delete:
            cur.execute(
                f"DELETE FROM vectors WHERE id IN ({','.join('?' for _ in to_delete)})",
                to_delete,
            )
            self.conn.commit()

    def delete_path(self, path: str, namespace: str = "default") -> int:
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM vectors WHERE namespace = ? AND path = ?",
                (namespace, path),
            )
        return int(cur.rowcount)

    def clear_namespace(self, namespace: str = "default") -> int:
        with self.conn:
            cur = self.conn.execute("DELETE FROM vectors WHERE namespace = ?", (namespace,))
        return int(cur.rowcount)

    def search(
        self, query: str, namespace: str = "default", top_k: int = 5
    ) -> list[VectorSearchResult]:
        if not query.strip():
            return []
        query_embedding = self._encode_texts([query])[0]
        cur = self.conn.cursor()
        cur.execute(
            "SELECT path, content, embedding, meta FROM vectors WHERE namespace = ?",
            (namespace,),
        )
        results: list[VectorSearchResult] = []
        for path, content, emb_blob, meta_json in cur.fetchall():
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
            if query_embedding.shape != embedding.shape:
                continue
            denominator = np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            if denominator == 0:
                continue
            similarity = float(np.dot(query_embedding, embedding) / denominator)
            meta = json.loads(meta_json) if meta_json else {}
            results.append(
                VectorSearchResult(path=path, snippet=content[:200], score=similarity, meta=meta)
            )
        results.sort(key=lambda item: item.score, reverse=True)
        return results[:top_k]

    def ensure_runtime_ready(self) -> None:
        if self.provider == "local":
            self._ensure_local_model()
            return
        if self.provider == "openai":
            if not self.openai_api_key:
                raise RuntimeError(
                    "OpenAI embeddings provider требует API key "
                    "(providers.openai.api_key или OPENAI_API_KEY)."
                )
            return
        raise RuntimeError(f"Неизвестный embeddings provider: {self.provider}")

    def _ensure_local_model(self) -> SentenceTransformer:
        if self.model is not None:
            return self.model
        if self._load_error is not None:
            raise RuntimeError(
                f"VectorIndex модель недоступна: {self._load_error}"
            ) from self._load_error
        try:
            self.model = self._get_model(self.local_model)
            return self.model
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            raise RuntimeError(
                f"VectorIndex не смог загрузить локальную модель {self.local_model}: {exc}"
            ) from exc

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        self.ensure_runtime_ready()
        if self.provider == "local":
            model = self._ensure_local_model()
            raw = model.encode(texts)
            return np.asarray(raw, dtype=np.float32)
        return self._encode_openai(texts)

    def _encode_openai(self, texts: list[str]) -> np.ndarray:
        if not self.openai_api_key:
            raise RuntimeError(
                "OpenAI embeddings provider требует API key "
                "(providers.openai.api_key или OPENAI_API_KEY)."
            )
        endpoint = self._openai_embeddings_endpoint()
        try:
            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.openai_model, "input": texts},
                timeout=self.openai_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OpenAI embeddings request failed: {exc}") from exc

        if response.status_code >= 400:
            message = response.text.strip() or f"HTTP {response.status_code}"
            try:
                payload = response.json()
            except Exception:  # noqa: BLE001
                payload = None
            if isinstance(payload, dict):
                error_raw = payload.get("error")
                if isinstance(error_raw, dict):
                    details = error_raw.get("message")
                    if isinstance(details, str) and details.strip():
                        message = details.strip()
            raise RuntimeError(
                f"OpenAI embeddings request failed ({response.status_code}): {message}"
            )

        try:
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenAI embeddings JSON parse failed: {exc}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI embeddings returned unexpected payload type")
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError("OpenAI embeddings payload missing data[]")

        vectors: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("OpenAI embeddings payload has invalid item type")
            emb_raw = item.get("embedding")
            if not isinstance(emb_raw, list):
                raise RuntimeError("OpenAI embeddings payload item missing embedding[]")
            try:
                vectors.append([float(value) for value in emb_raw])
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("OpenAI embeddings payload contains non-numeric vector") from exc

        if len(vectors) != len(texts):
            raise RuntimeError(
                "OpenAI embeddings returned unexpected vector count: "
                f"expected {len(texts)}, got {len(vectors)}"
            )
        return np.asarray(vectors, dtype=np.float32)

    def _openai_embeddings_endpoint(self) -> str:
        base = self.openai_base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/embeddings"
        return f"{base}/v1/embeddings"

    def close(self) -> None:
        self.conn.close()
