from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer

from shared.models import JSONValue, VectorSearchResult


class VectorIndex:
    _model_cache: ClassVar[dict[str, SentenceTransformer]] = {}

    def __init__(
        self,
        db_path: str = "memory/vectors.db",
        model_name: str = "all-MiniLM-L6-v2",
        max_records: int = 5000,
        max_total_records: int = 20000,
        batch_size: int = 32,
    ):
        self.db_path = db_path
        self.model_name = model_name
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
        model = self._ensure_model()
        embedding = model.encode([content])[0].astype(np.float32).tobytes()
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
        model = self._ensure_model()
        for batch_start in range(0, len(items), self.batch_size):
            batch = items[batch_start : batch_start + self.batch_size]
            paths, texts = zip(*batch, strict=False)
            embeddings = model.encode(list(texts))
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
        model = self._ensure_model()
        query_embedding = model.encode([query])[0]
        cur = self.conn.cursor()
        cur.execute(
            "SELECT path, content, embedding, meta FROM vectors WHERE namespace = ?",
            (namespace,),
        )
        results: list[VectorSearchResult] = []
        for path, content, emb_blob, meta_json in cur.fetchall():
            embedding = np.frombuffer(emb_blob, dtype=np.float32)
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

    def _ensure_model(self) -> SentenceTransformer:
        if self.model is not None:
            return self.model
        if self._load_error is not None:
            raise RuntimeError(
                f"VectorIndex модель недоступна: {self._load_error}"
            ) from self._load_error
        try:
            self.model = self._get_model(self.model_name)
            return self.model
        except Exception as exc:  # noqa: BLE001
            self._load_error = exc
            raise RuntimeError(
                f"VectorIndex не смог загрузить модель {self.model_name}: {exc}"
            ) from exc

    def close(self) -> None:
        self.conn.close()
