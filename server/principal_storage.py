from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PrincipalStoragePaths:
    memory_db: Path
    vectors_db: Path
    memory_companion_db: Path
    memory_categories_db: Path
    canonical_atoms_db: Path


def principal_storage_paths(
    *,
    principal_id: str,
    owner_principal_id: str,
    memory_root: Path,
) -> PrincipalStoragePaths:
    normalized_principal = principal_id.strip()
    normalized_owner = owner_principal_id.strip()
    if not normalized_principal:
        raise ValueError("principal_id must be non-empty")
    if not normalized_owner:
        raise ValueError("owner_principal_id must be non-empty")

    root = memory_root
    if normalized_principal != normalized_owner:
        principal_hash = hashlib.sha256(normalized_principal.encode("utf-8")).hexdigest()
        root = memory_root / "principals" / principal_hash

    return PrincipalStoragePaths(
        memory_db=root / "memory.db",
        vectors_db=root / "vectors.db",
        memory_companion_db=root / "memory_companion.db",
        memory_categories_db=root / "memory_categories.db",
        canonical_atoms_db=root / "canonical_atoms.db",
    )
