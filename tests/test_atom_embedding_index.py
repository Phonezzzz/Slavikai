from __future__ import annotations

import numpy as np

from memory.atom_embedding_index import AtomEmbeddingIndex
from memory.vector_index import VectorIndex
from shared.canonical_atom_models import AtomStatus, CanonicalAtom, ClaimType


class DummyModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def _atom(*, status: AtomStatus) -> CanonicalAtom:
    return CanonicalAtom(
        atom_id="atom-1",
        stable_key="preference:response_length",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "short"},
        confidence=0.8,
        support_count=3,
        contradict_count=0,
        last_seen_at="2026-01-01T00:00:00+00:00",
        status=status,
        summary_text="preference:response_length=short",
    )


def test_atom_embedding_index_sync_upsert_and_delete(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model",
        lambda _self, _name: DummyModel(),
    )
    vectors = VectorIndex(str(db_path))
    index = AtomEmbeddingIndex(vectors)

    index.sync_atom(_atom(status=AtomStatus.ACTIVE))
    results = vectors.search("response", namespace="atoms", top_k=5)
    assert len(results) == 1
    assert results[0].path == "preference:response_length"

    index.sync_atom(_atom(status=AtomStatus.DEPRECATED))
    after = vectors.search("response", namespace="atoms", top_k=5)
    assert after == []
