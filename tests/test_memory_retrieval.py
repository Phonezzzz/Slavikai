from __future__ import annotations

import numpy as np

from memory.atom_embedding_index import AtomEmbeddingIndex
from memory.canonical_atom_store import CanonicalAtomStore
from memory.memory_retrieval import RetrievalConfig, build_memory_capsule, filter_atoms
from memory.vector_index import VectorIndex
from shared.canonical_atom_models import AtomStatus, ClaimType


class DummyModel:
    def encode(self, texts):
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


def test_retrieval_filter_rank_pack_excludes_conflicts(tmp_path, monkeypatch) -> None:
    atoms_db = tmp_path / "atoms.db"
    vectors_db = tmp_path / "vectors.db"

    monkeypatch.setattr(
        "memory.vector_index.VectorIndex._get_model",
        lambda _self, _name: DummyModel(),
    )

    store = CanonicalAtomStore(str(atoms_db))
    vectors = VectorIndex(str(vectors_db))
    atom_index = AtomEmbeddingIndex(vectors)

    active = store.create_atom(
        atom_id="a1",
        stable_key="preference:response_length",
        claim_type=ClaimType.PREFERENCE,
        value_json={"value": "short"},
        confidence=0.9,
        support_count=3,
        contradict_count=0,
        last_seen_at="2026-01-02T00:00:00+00:00",
        status=AtomStatus.ACTIVE,
        summary_text="prefer short responses",
    )
    conflict = store.create_atom(
        atom_id="a2",
        stable_key="policy:avoid_emoji",
        claim_type=ClaimType.POLICY,
        value_json={"rule": "avoid emoji"},
        confidence=0.8,
        support_count=1,
        contradict_count=2,
        last_seen_at="2026-01-02T00:00:00+00:00",
        status=AtomStatus.CONFLICT,
        summary_text="avoid emoji",
    )
    atom_index.sync_atom(active)
    atom_index.sync_atom(conflict)

    filtered = filter_atoms(
        [active, conflict],
        min_confidence=0.5,
        allowed_types={ClaimType.PREFERENCE, ClaimType.POLICY},
        include_conflicts=False,
        recency_days=365,
    )
    assert [item.stable_key for item in filtered] == ["preference:response_length"]

    capsule = build_memory_capsule(
        query="коротко",
        store=store,
        vector_index=vectors,
        for_mwv=False,
        config=RetrievalConfig(
            min_confidence=0.5,
            top_k=5,
            max_context_chars=500,
            recency_days=365,
        ),
        include_conflicts=False,
    )

    items_raw = capsule.get("items")
    assert isinstance(items_raw, list)
    assert len(items_raw) == 1
    first = items_raw[0]
    assert isinstance(first, dict)
    assert first.get("stable_key") == "preference:response_length"
    text_raw = capsule.get("text")
    assert isinstance(text_raw, str)
    assert "preference:response_length" in text_raw
