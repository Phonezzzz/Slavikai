from __future__ import annotations

import json

from memory.vector_index import VectorIndex
from shared.canonical_atom_models import AtomStatus, CanonicalAtom


class AtomEmbeddingIndex:
    def __init__(self, vector_index: VectorIndex, *, namespace: str = "atoms") -> None:
        self._vector_index = vector_index
        self._namespace = namespace

    @property
    def namespace(self) -> str:
        return self._namespace

    def sync_atom(self, atom: CanonicalAtom) -> None:
        if atom.status is AtomStatus.DEPRECATED:
            self._vector_index.delete_path(atom.stable_key, namespace=self._namespace)
            return
        text = self._embedding_text(atom)
        self._vector_index.upsert_text(
            atom.stable_key,
            text,
            namespace=self._namespace,
            meta={
                "stable_key": atom.stable_key,
                "claim_type": atom.claim_type.value,
                "status": atom.status.value,
                "confidence": atom.confidence,
                "last_seen_at": atom.last_seen_at,
            },
        )

    def rebuild(self, atoms: list[CanonicalAtom]) -> None:
        self._vector_index.clear_namespace(self._namespace)
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
