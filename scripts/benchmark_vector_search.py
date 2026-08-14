from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import time
from pathlib import Path

import numpy as np

DIMS = 384
N_RECORDS = 5000
N_QUERIES = 30


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def make_vector(text: str) -> np.ndarray:
    rng = np.random.default_rng(_seed(text))
    v = rng.normal(size=DIMS).astype(np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def build_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS vectors ("
        "namespace TEXT NOT NULL, path TEXT NOT NULL, content TEXT, "
        "embedding BLOB, meta TEXT)"
    )
    conn.execute("DELETE FROM vectors")
    for idx in range(N_RECORDS):
        text = f"content-{idx}"
        conn.execute(
            "INSERT INTO vectors (namespace, path, content, embedding, meta) VALUES (?,?,?,?,?)",
            ("default", f"file{idx}.md", text, make_vector(text).tobytes(), "{}"),
        )
    conn.commit()
    conn.close()


def search_loop(
    conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int = 5
) -> list[tuple[str, float, dict[str, object]]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT path, content, embedding, meta FROM vectors WHERE namespace = ?", ("default",)
    )
    results: list[tuple[str, float, dict[str, object]]] = []
    for path, _content, emb_blob, meta_json in cur.fetchall():
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        if query_embedding.shape != embedding.shape:
            continue
        denominator = np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
        if denominator == 0:
            continue
        similarity = float(np.dot(query_embedding, embedding) / denominator)
        meta = json.loads(meta_json) if meta_json else {}
        results.append((path, similarity, meta))
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_k]


def search_vector(
    conn: sqlite3.Connection, query_embedding: np.ndarray, top_k: int = 5
) -> list[tuple[str, float, dict[str, object]]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT path, content, embedding, meta FROM vectors WHERE namespace = ?", ("default",)
    )
    rows = cur.fetchall()
    paths = [row[0] for row in rows]
    metas = [row[3] for row in rows]
    embeddings: list[np.ndarray] = []
    for row in rows:
        emb = np.frombuffer(row[2], dtype=np.float32)
        if query_embedding.shape == emb.shape and np.linalg.norm(emb) != 0:
            embeddings.append(emb)
    matrix = np.stack(embeddings)
    scores = matrix @ query_embedding / np.linalg.norm(query_embedding)
    order = np.argsort(-scores)[:top_k]
    return [(paths[i], float(scores[i]), json.loads(metas[i]) if metas[i] else {}) for i in order]


def main() -> None:
    db_path = Path("/tmp/opencode/bench_vectors.db")
    build_db(db_path)
    conn = sqlite3.connect(db_path)
    queries = [f"query {i}" for i in range(N_QUERIES)]
    query_vectors = [make_vector(q) for q in queries]

    loop_times = []
    vec_times = []
    for qv in query_vectors:
        t0 = time.perf_counter()
        search_loop(conn, qv)
        loop_times.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        search_vector(conn, qv)
        vec_times.append((time.perf_counter() - t0) * 1000)

    print(f"records={N_RECORDS} dims={DIMS} queries={N_QUERIES}")
    print(
        f"loop    median={statistics.median(loop_times):.2f} ms  "
        f"mean={statistics.mean(loop_times):.2f} ms"
    )
    print(
        f"vector  median={statistics.median(vec_times):.2f} ms  "
        f"mean={statistics.mean(vec_times):.2f} ms"
    )
    speedup = statistics.median(loop_times) / statistics.median(vec_times)
    print(f"speedup (median)={speedup:.2f}x")

    r0 = search_loop(conn, query_vectors[0])
    r1 = search_vector(conn, query_vectors[0])
    same = [a[0] for a in r0] == [b[0] for b in r1]
    print(f"top-5 paths identical: {same}")
    conn.close()


if __name__ == "__main__":
    main()
