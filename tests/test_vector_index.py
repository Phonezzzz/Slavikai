from __future__ import annotations

import hashlib
import sqlite3
import time

import numpy as np
import pytest

from memory.vector_index import VectorIndex
from shared.models import VectorSearchResult


class DummyModel:
    def encode(self, texts):
        # deterministic small vectors
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def test_vector_index_namespace_and_limit(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda self, name: DummyModel()
    )
    index = VectorIndex(str(db_path), max_records=2)

    index.index_text("file1", "content1", namespace="projA")
    index.index_text("file2", "content2", namespace="projA")
    # third insert should prune oldest in namespace
    index.index_text("file3", "content3", namespace="projA")

    results = index.search("query", namespace="projA", top_k=5)
    assert len(results) == 2
    paths = [res.path for res in results]
    assert "file1" not in paths  # pruned

    # other namespace independent
    index.index_text("other", "data", namespace="projB")
    res_other = index.search("x", namespace="projB", top_k=5)
    assert len(res_other) == 1
    assert res_other[0].path == "other"


def test_vector_index_batch_and_total_limit(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda self, name: DummyModel()
    )
    index = VectorIndex(str(db_path), max_records=10, max_total_records=3, batch_size=2)
    items = [(f"path{i}", f"content{i}") for i in range(5)]
    index.index_batch(items, namespace="code")
    results = index.search("query", namespace="code", top_k=10)
    assert len(results) == 3  # total pruned to max_total_records


def test_vector_index_rejects_bad_meta(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda self, name: DummyModel()
    )
    index = VectorIndex(str(db_path))
    with pytest.raises(ValueError):
        index.index_text("p", "content", namespace="code", meta="not a dict")


def test_vector_index_does_not_load_model_on_startup(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    calls = 0

    def _get_model(_self: object, _name: str) -> DummyModel:
        nonlocal calls
        calls += 1
        return DummyModel()

    monkeypatch.setattr("memory.vector_index.VectorIndex._get_model", _get_model)
    _ = VectorIndex(str(db_path))
    assert calls == 0


def test_vector_index_first_search_loads_then_reuses_model(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    calls = 0

    def _slow_get_model(_self: object, _name: str) -> DummyModel:
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        return DummyModel()

    monkeypatch.setattr("memory.vector_index.VectorIndex._get_model", _slow_get_model)
    index = VectorIndex(str(db_path))
    embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
    with index.conn:
        index.conn.execute(
            "INSERT INTO vectors (namespace, path, content, embedding, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            ("default", "file", "content", embedding, "{}"),
        )

    started = time.perf_counter()
    first = index.search("query", namespace="default", top_k=1)
    first_duration = time.perf_counter() - started

    started = time.perf_counter()
    second = index.search("query", namespace="default", top_k=1)
    second_duration = time.perf_counter() - started

    assert len(first) == 1
    assert len(second) == 1
    assert calls == 1
    assert first_duration >= 0.045
    assert second_duration < first_duration


def test_vector_index_upsert_delete_and_clear_namespace(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda _self, _name: DummyModel()
    )
    index = VectorIndex(str(db_path))

    index.upsert_text("a", "one", namespace="atoms")
    index.upsert_text("a", "two", namespace="atoms")
    results = index.search("query", namespace="atoms", top_k=5)
    assert len(results) == 1
    assert results[0].path == "a"
    assert results[0].snippet.startswith("two")

    deleted = index.delete_path("a", namespace="atoms")
    assert deleted == 1
    assert index.search("query", namespace="atoms", top_k=5) == []

    index.index_text("x", "1", namespace="atoms")
    index.index_text("y", "2", namespace="atoms")
    cleared = index.clear_namespace("atoms")
    assert cleared == 2
    assert index.search("query", namespace="atoms", top_k=5) == []


def test_vector_index_openai_requires_key(tmp_path) -> None:
    db_path = tmp_path / "vec.db"
    index = VectorIndex(str(db_path), provider="openai", openai_model="text-embedding-3-small")
    with pytest.raises(RuntimeError, match="requires API key|требует API key"):
        index.search("query", namespace="default", top_k=1)


def test_vector_index_local_provider_errors_without_sentence_transformers(
    tmp_path, monkeypatch
) -> None:
    db_path = tmp_path / "vec.db"

    def _raise(*_args: object, **_kwargs: object) -> DummyModel:
        raise RuntimeError("sentence-transformers not available")

    monkeypatch.setattr("memory.vector_index.VectorIndex._get_model", _raise)
    index = VectorIndex(str(db_path), provider="local", local_model="all-MiniLM-L6-v2")
    with pytest.raises(RuntimeError, match="не смог загрузить локальную модель"):
        index.search("query", namespace="default", top_k=1)


class RandomModel:
    def encode(self, texts):
        vectors = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
            rng = np.random.default_rng(seed)
            vector = rng.normal(size=8).astype(np.float32)
            vector /= np.linalg.norm(vector)
            vectors.append(vector)
        return np.stack(vectors)


def test_vector_index_batched_search_matches_bruteforce(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda _self, _name: RandomModel()
    )
    index = VectorIndex(str(db_path))
    for idx in range(20):
        index.index_text(f"path{idx}", f"content{idx}", namespace="proj")

    query = "find me"
    batched = index.search(query, namespace="proj", top_k=5)

    cur = index.conn.cursor()
    cur.execute(
        "SELECT path, content, embedding, meta FROM vectors WHERE namespace = ?",
        ("proj",),
    )
    query_embedding = RandomModel().encode([query])[0]
    brute: list[VectorSearchResult] = []
    for row in cur.fetchall():
        path, content, emb_blob, meta_json = row
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        if query_embedding.shape != embedding.shape:
            continue
        denominator = np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        if denominator == 0:
            continue
        similarity = float(np.dot(query_embedding, embedding) / denominator)
        brute.append(
            VectorSearchResult(path=path, snippet=content[:200], score=similarity, meta={})
        )
    brute.sort(key=lambda item: item.score, reverse=True)
    brute = brute[:5]

    assert [item.path for item in batched] == [item.path for item in brute]
    for actual, expected in zip(batched, brute, strict=False):
        assert actual.score == pytest.approx(expected.score, abs=1e-5)


def test_vector_index_batched_search_skips_mismatched_and_zero_norm(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda _self, _name: DummyModel()
    )
    index = VectorIndex(str(db_path))
    index.index_text("ok", "content", namespace="proj")
    with index.conn:
        index.conn.execute(
            "INSERT INTO vectors (namespace, path, content, embedding, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            ("proj", "mismatch", "m", np.zeros(5, dtype=np.float32).tobytes(), "{}"),
        )
        index.conn.execute(
            "INSERT INTO vectors (namespace, path, content, embedding, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            ("proj", "zero", "z", np.zeros(3, dtype=np.float32).tobytes(), "{}"),
        )

    results = index.search("query", namespace="proj", top_k=10)
    assert [item.path for item in results] == ["ok"]


def test_vector_index_context_manager_closes_connection(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model", lambda _self, _name: DummyModel()
    )
    with VectorIndex(str(db_path)) as index:
        index.index_text("a", "content", namespace="proj")
        assert index.conn is not None
    with pytest.raises(sqlite3.ProgrammingError):
        index.conn.execute("SELECT 1")
