from __future__ import annotations

import json

from memory.canonical_atom_store import CanonicalAtomStore
from memory.vector_index import VectorIndex
from shared.canonical_atom_models import AtomStatus, CanonicalAtom


class AtomEmbeddingIndex:
    def __init__(self, vector_index: VectorIndex, store: CanonicalAtomStore) -> None:
        self._vector_index = vector_index
        self._store = store
        store.bind_vectors(vector_index.db_path)

    @property
    def namespace(self) -> str:
        return "atoms"

    def sync_atom(self, atom: CanonicalAtom) -> bool:
        # Encoding never holds a database transaction; publication revalidates revision.
        embedding = (
            b""
            if atom.status is AtomStatus.DEPRECATED
            else self._vector_index.encode_text(self._embedding_text(atom))
        )
        return self._store.publish_vector(
            atom,
            embedding,
            content=self._embedding_text(atom),
            namespace="atoms",
            max_records=self._vector_index.max_records,
            max_total_records=self._vector_index.max_total_records,
        )

    def rebuild(self, atoms: list[CanonicalAtom]) -> None:
        for atom in atoms:
            self.sync_atom(atom)

    def _embedding_text(self, atom: CanonicalAtom) -> str:
        value_preview = json.dumps(atom.value_json, ensure_ascii=False, sort_keys=True)
        if len(value_preview) > 400:
            value_preview = value_preview[:400]
        parts = [
            f"key: {atom.stable_key}",
            f"type: {atom.claim_type.value}",
            f"summary: {atom.summary_text}",
            f"value: {value_preview}",
        ]
        return "\n".join(parts)
