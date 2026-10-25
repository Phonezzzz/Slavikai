from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Literal, cast

from shared.canonical_atom_models import AtomStatus, CanonicalAtom, ClaimType, utc_now_iso
from shared.models import JSONValue
from shared.sanitize import safe_json_loads
from shared.sqlite import ThreadLocalSQLiteConnection


class CanonicalAtomStore:
    def __init__(self, db_path: str = "memory/canonical_atoms.db") -> None:
        self.db_path = db_path
        self._vectors_path: str | None = None
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = ThreadLocalSQLiteConnection(db_path, row_factory=sqlite3.Row)
        self._init_db()

    def _init_db(self) -> None:
        with self.conn.transaction(immediate=True):
            self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS canonical_atom (
                atom_id TEXT PRIMARY KEY,
                stable_key TEXT NOT NULL UNIQUE,
                claim_type TEXT NOT NULL,
                value_json TEXT NOT NULL,
                confidence REAL NOT NULL,
                support_count INTEGER NOT NULL,
                contradict_count INTEGER NOT NULL,
                last_seen_at TEXT NOT NULL,
                status TEXT NOT NULL,
                summary_text TEXT NOT NULL
            )
            """
        )
        self._ensure_pinned_column()
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canonical_atom_claim_status "
            "ON canonical_atom (claim_type, status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canonical_atom_pinned "
            "ON canonical_atom (pinned, last_seen_at)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canonical_atom_last_seen "
            "ON canonical_atom (last_seen_at)"
        )
        columns = {
            str(row["name"]) for row in self.conn.execute("PRAGMA table_info(canonical_atom)")
        }
        if "revision" not in columns:
            self.conn.execute(
                "ALTER TABLE canonical_atom ADD COLUMN revision TEXT NOT NULL DEFAULT ''"
            )
        for row in self.conn.execute(
            "SELECT stable_key FROM canonical_atom WHERE revision = ''"
        ).fetchall():
            self.conn.execute(
                "UPDATE canonical_atom SET revision = ? WHERE stable_key = ?",
                (uuid.uuid4().hex, row["stable_key"]),
            )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS atom_delete_receipt (operation_id TEXT PRIMARY KEY, "
            "item_hash TEXT NOT NULL, revision TEXT NOT NULL, expires_at REAL NOT NULL)"
        )

        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS atom_delete_preview (operation_id TEXT PRIMARY KEY, "
            "atom_id TEXT NOT NULL, item_hash TEXT NOT NULL, revision TEXT NOT NULL, "
            "expires_at REAL NOT NULL)"
        )

    def _ensure_pinned_column(self) -> None:
        columns = {
            str(row["name"])
            for row in self.conn.execute("PRAGMA table_info(canonical_atom)").fetchall()
        }
        if "pinned" in columns:
            return
        self.conn.execute("ALTER TABLE canonical_atom ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")

    def close(self) -> None:
        self.conn.close()

    def get_by_stable_key(self, stable_key: str) -> CanonicalAtom | None:
        row = self.conn.execute(
            "SELECT * FROM canonical_atom WHERE stable_key = ?",
            (stable_key.strip(),),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_atom(row)

    def get_by_id(self, atom_id: str) -> CanonicalAtom | None:
        row = self.conn.execute(
            "SELECT * FROM canonical_atom WHERE atom_id = ?",
            (atom_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_atom(row)

    def list_atoms(
        self,
        *,
        statuses: set[AtomStatus] | None = None,
        claim_types: set[ClaimType] | None = None,
        stable_key_prefix: str | None = None,
        limit: int = 200,
    ) -> list[CanonicalAtom]:
        if limit <= 0:
            raise ValueError("limit должен быть > 0")
        query = (
            "SELECT * FROM canonical_atom WHERE 1=1 "
            "{status_clause} {type_clause} {stable_key_clause} "
            "ORDER BY last_seen_at DESC, stable_key ASC LIMIT ?"
        )
        params: list[object] = []
        status_clause = ""
        if statuses:
            status_values = sorted(item.value for item in statuses)
            placeholders = ",".join("?" for _ in status_values)
            status_clause = f"AND status IN ({placeholders})"
            params.extend(status_values)
        type_clause = ""
        if claim_types:
            type_values = sorted(item.value for item in claim_types)
            placeholders = ",".join("?" for _ in type_values)
            type_clause = f"AND claim_type IN ({placeholders})"
            params.extend(type_values)
        stable_key_clause = ""
        if stable_key_prefix is not None:
            normalized_prefix = stable_key_prefix.strip()
            if normalized_prefix:
                stable_key_clause = "AND stable_key LIKE ?"
                params.append(f"{normalized_prefix}%")
        params.append(limit)
        rows = self.conn.execute(
            query.format(
                status_clause=status_clause,
                type_clause=type_clause,
                stable_key_clause=stable_key_clause,
            ),
            tuple(params),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def list_conflicts(self, *, limit: int = 100) -> list[CanonicalAtom]:
        return self.list_atoms(statuses={AtomStatus.CONFLICT}, limit=limit)

    def list_pinned(self, *, limit: int = 20) -> list[CanonicalAtom]:
        if limit <= 0:
            raise ValueError("limit должен быть > 0")
        rows = self.conn.execute(
            "SELECT * FROM canonical_atom "
            "WHERE pinned = 1 AND status != 'deprecated' "
            "ORDER BY last_seen_at DESC, stable_key ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def set_pinned(self, stable_key: str, pinned: bool) -> bool:
        normalized = stable_key.strip()
        if not normalized:
            raise ValueError("stable_key должен быть непустым")
        with self.transaction():
            cursor = self.conn.execute(
                "UPDATE canonical_atom SET pinned = ?, revision = ? WHERE stable_key = ?",
                (1 if pinned else 0, uuid.uuid4().hex, normalized),
            )
        return cursor.rowcount > 0

    def upsert(self, atom: CanonicalAtom) -> CanonicalAtom:
        return self.upsert_many([atom])[0]

    def upsert_many(self, atoms: list[CanonicalAtom]) -> list[CanonicalAtom]:
        if not atoms:
            return []
        with self.transaction():
            for atom in atoms:
                current = self.get_by_stable_key(atom.stable_key)
                if (current is None and atom.revision) or (
                    current is not None and current.revision != atom.revision
                ):
                    raise ValueError("canonical_revision_conflict")
                atom = replace(atom, revision=uuid.uuid4().hex)
                if atom.status is AtomStatus.DEPRECATED:
                    self._clear_vector(atom.stable_key)
                self.conn.execute(
                    """
                    INSERT INTO canonical_atom (
                        atom_id,
                        stable_key,
                        claim_type,
                        value_json,
                        confidence,
                        support_count,
                        contradict_count,
                        last_seen_at,
                        status,
                        summary_text,
                        pinned, revision
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(stable_key) DO UPDATE SET
                        atom_id=excluded.atom_id,
                        claim_type=excluded.claim_type,
                        value_json=excluded.value_json,
                        confidence=excluded.confidence,
                        support_count=excluded.support_count,
                        contradict_count=excluded.contradict_count,
                        last_seen_at=excluded.last_seen_at,
                        status=excluded.status,
                        summary_text=excluded.summary_text,
                        revision=excluded.revision
                    """,
                    self._atom_for_write(atom),
                )
            persisted: list[CanonicalAtom] = []
            for atom in atoms:
                stored = self.get_by_stable_key(atom.stable_key)
                if stored is None:
                    raise RuntimeError("atom missing after batch upsert")
                persisted.append(stored)
        return persisted

    def create_atom(
        self,
        *,
        atom_id: str,
        stable_key: str,
        claim_type: ClaimType,
        value_json: JSONValue,
        confidence: float,
        support_count: int = 1,
        contradict_count: int = 0,
        last_seen_at: str | None = None,
        status: AtomStatus = AtomStatus.ACTIVE,
        summary_text: str,
        pinned: bool = False,
    ) -> CanonicalAtom:
        with self.transaction():
            current = self.get_by_stable_key(stable_key)
            atom = CanonicalAtom(
                atom_id=atom_id,
                stable_key=stable_key,
                claim_type=claim_type,
                value_json=value_json,
                confidence=confidence,
                support_count=support_count,
                contradict_count=contradict_count,
                last_seen_at=last_seen_at or utc_now_iso(),
                status=status,
                summary_text=summary_text,
                pinned=pinned,
                revision=current.revision if current is not None else "",
            )
            return self.upsert(atom)

    def resolve_conflict(
        self,
        *,
        stable_key: str,
        resolution: AtomStatus,
        value_json: JSONValue | None = None,
        summary_text: str | None = None,
    ) -> CanonicalAtom | None:
        with self.transaction():
            current = self.get_by_stable_key(stable_key)
            if current is None:
                return None
            next_value = current.value_json if value_json is None else value_json
            next_summary = current.summary_text if summary_text is None else summary_text
            resolved = CanonicalAtom(
                atom_id=current.atom_id,
                stable_key=current.stable_key,
                claim_type=current.claim_type,
                value_json=next_value,
                confidence=current.confidence,
                support_count=current.support_count,
                contradict_count=current.contradict_count,
                last_seen_at=utc_now_iso(),
                status=resolution,
                summary_text=next_summary,
                pinned=current.pinned,
                revision=current.revision,
            )
            return self.upsert(resolved)

    def delete_atom(self, stable_key: str) -> CanonicalAtom | None:
        """Soft-delete: устанавливает статус DEPRECATED.

        Для retrieval требуется явный status filter; pinned исключает deprecated.
        """
        return self.resolve_conflict(
            stable_key=stable_key,
            resolution=AtomStatus.DEPRECATED,
        )

    def _row_to_atom(self, row: sqlite3.Row) -> CanonicalAtom:
        value_raw = row["value_json"]
        value_parsed = safe_json_loads(value_raw) if isinstance(value_raw, str) else None
        if value_parsed is None and value_raw not in {"null", None}:
            value_json: JSONValue = str(value_raw)
        else:
            value_json = cast(JSONValue, value_parsed)
        return CanonicalAtom(
            atom_id=str(row["atom_id"]),
            stable_key=str(row["stable_key"]),
            claim_type=ClaimType(str(row["claim_type"])),
            value_json=value_json,
            confidence=float(row["confidence"]),
            support_count=int(row["support_count"]),
            contradict_count=int(row["contradict_count"]),
            last_seen_at=str(row["last_seen_at"]),
            status=AtomStatus(str(row["status"])),
            summary_text=str(row["summary_text"]),
            pinned=bool(int(row["pinned"])),
            revision=str(row["revision"]),
        )

    def _atom_for_write(self, atom: CanonicalAtom) -> tuple[object, ...]:
        return (
            atom.atom_id,
            atom.stable_key,
            atom.claim_type.value,
            json.dumps(atom.value_json, ensure_ascii=False, sort_keys=True),
            atom.confidence,
            atom.support_count,
            atom.contradict_count,
            atom.last_seen_at,
            atom.status.value,
            atom.summary_text,
            1 if atom.pinned else 0,
            atom.revision,
        )

    def bind_vectors(self, db_path: str) -> None:
        """Bind the existing principal vector store, never a client-supplied path."""
        path = str(Path(db_path).resolve())
        if self._vectors_path is not None and self._vectors_path != path:
            raise ValueError("canonical_vector_binding_changed")
        if (
            db_path == ":memory:"
            or self.db_path == ":memory:"
            or path == str(Path(self.db_path).resolve())
        ):
            raise ValueError("canonical_vector_requires_separate_files")
        self._vectors_path = path

    @contextmanager
    def transaction(self) -> Iterator[None]:
        if self._vectors_path is not None:
            databases = {
                str(row[1]): str(row[2]) for row in self.conn.execute("PRAGMA database_list")
            }
            if "atom_vectors" not in databases:
                self.conn.execute("ATTACH DATABASE ? AS atom_vectors", (self._vectors_path,))
            elif str(Path(databases["atom_vectors"]).resolve()) != self._vectors_path:
                raise ValueError("canonical_vector_binding_changed")
        with self.conn.transaction(immediate=True):
            if self._vectors_path is not None:
                for schema in ("main", "atom_vectors"):
                    journal = self.conn.execute(f"PRAGMA {schema}.journal_mode").fetchone()[0]
                    synchronous = self.conn.execute(f"PRAGMA {schema}.synchronous").fetchone()[0]
                    if journal not in {"delete", "persist", "truncate"} or synchronous < 2:
                        raise ValueError("canonical_vector_atomicity_unavailable")
            yield

    def _clear_vector(self, stable_key: str) -> None:
        if self._vectors_path is not None:
            self.conn.execute(
                "DELETE FROM atom_vectors.vectors WHERE namespace = 'atoms' AND path = ?",
                (stable_key,),
            )

    def publish_vector(
        self,
        atom: CanonicalAtom,
        embedding: bytes,
        *,
        content: str,
        namespace: str,
        max_records: int,
        max_total_records: int,
    ) -> bool:
        if self._vectors_path is None or namespace != "atoms":
            raise ValueError("canonical_vector_binding_required")
        with self.transaction():
            current = self.get_by_stable_key(atom.stable_key)
            if not atom.revision or current is None or current.revision != atom.revision:
                return False
            self._clear_vector(atom.stable_key)
            if current.status is AtomStatus.DEPRECATED:
                return True
            self.conn.execute(
                "INSERT INTO atom_vectors.vectors(namespace, path, content, embedding, meta) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    namespace,
                    atom.stable_key,
                    content[:5000],
                    embedding,
                    json.dumps(
                        {
                            "stable_key": atom.stable_key,
                            "claim_type": atom.claim_type.value,
                            "status": atom.status.value,
                            "confidence": atom.confidence,
                            "last_seen_at": atom.last_seen_at,
                        }
                    ),
                ),
            )
            self.conn.execute(
                "DELETE FROM atom_vectors.vectors WHERE id IN (SELECT id FROM "
                "atom_vectors.vectors WHERE namespace = ? ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (namespace, max_records),
            )
            self.conn.execute(
                "DELETE FROM atom_vectors.vectors WHERE id IN (SELECT id FROM "
                "atom_vectors.vectors ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (max_total_records,),
            )
            return True

    def delete_confirmed(
        self,
        stable_key: str,
        expected_revision: str,
        operation_id: str,
        *,
        expires_at: float,
    ) -> Literal["deleted", "conflict", "not_found", "expired"]:
        """Storage prerequisite only: callers must supply an exact confirmed binding."""
        if self._vectors_path is None:
            raise ValueError("canonical_vector_binding_required")
        if (
            not operation_id
            or len(operation_id) > 128
            or not expected_revision
            or len(expected_revision) > 128
            or not math.isfinite(expires_at)
        ):
            raise ValueError("invalid_memory_delete_binding")
        item_hash = hashlib.sha256(stable_key.encode()).hexdigest()
        with self.transaction():
            now = time.time()
            self.conn.execute("DELETE FROM atom_delete_receipt WHERE expires_at <= ?", (now,))
            if expires_at <= now or expires_at > now + 86400:
                return "expired"
            receipt = self.conn.execute(
                "SELECT * FROM atom_delete_receipt WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if receipt is not None:
                return (
                    "deleted"
                    if receipt["item_hash"] == item_hash
                    and receipt["revision"] == expected_revision
                    else "conflict"
                )
            atom = self.get_by_stable_key(stable_key)
            if atom is None:
                return "not_found"
            if atom.revision != expected_revision:
                return "conflict"
            self._clear_vector(stable_key)
            self.conn.execute(
                "DELETE FROM canonical_atom WHERE stable_key = ? AND revision = ?",
                (stable_key, expected_revision),
            )
            self.conn.execute(
                "INSERT INTO atom_delete_receipt VALUES (?, ?, ?, ?)",
                (operation_id, item_hash, expected_revision, expires_at),
            )
            self.conn.execute(
                "DELETE FROM atom_delete_receipt WHERE operation_id IN (SELECT operation_id "
                "FROM atom_delete_receipt ORDER BY expires_at DESC, operation_id LIMIT -1 "
                "OFFSET 1024)"
            )
            return "deleted"

    def list_page(self, *, after: str | None, limit: int) -> tuple[list[CanonicalAtom], str | None]:
        if not 1 <= limit <= 100:
            raise ValueError("memory_page_limit_invalid")
        rows = self.conn.execute(
            "SELECT * FROM canonical_atom WHERE atom_id > ? ORDER BY atom_id LIMIT ?",
            (after or "", limit + 1),
        ).fetchall()
        atoms = [self._row_to_atom(row) for row in rows[:limit]]
        return atoms, atoms[-1].atom_id if len(rows) > limit else None

    def preview_delete(self, atom_id: str, revision: str) -> tuple[str, CanonicalAtom, float]:
        with self.transaction():
            atom = self.get_by_id(atom_id)
            if atom is None:
                raise ValueError("memory_item_not_found")
            if atom.revision != revision:
                raise ValueError("memory_revision_conflict")
            now = time.time()
            self.conn.execute("DELETE FROM atom_delete_preview WHERE expires_at <= ?", (now,))
            operation_id = uuid.uuid4().hex
            expires_at = now + 300
            self.conn.execute(
                "INSERT INTO atom_delete_preview VALUES (?, ?, ?, ?, ?)",
                (
                    operation_id,
                    atom.atom_id,
                    hashlib.sha256(atom.stable_key.encode()).hexdigest(),
                    revision,
                    expires_at,
                ),
            )
            self.conn.execute(
                "DELETE FROM atom_delete_preview WHERE operation_id IN (SELECT operation_id "
                "FROM atom_delete_preview ORDER BY expires_at DESC, operation_id LIMIT -1 "
                "OFFSET 1024)"
            )
            return operation_id, atom, expires_at

    def confirm_delete_preview(
        self, operation_id: str
    ) -> Literal["deleted", "conflict", "not_found", "expired"]:
        with self.transaction():
            preview = self.conn.execute(
                "SELECT * FROM atom_delete_preview WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if preview is None:
                return "not_found"
            if preview["expires_at"] <= time.time():
                return "expired"
            receipt = self.conn.execute(
                "SELECT * FROM atom_delete_receipt WHERE operation_id = ?", (operation_id,)
            ).fetchone()
            if receipt is not None:
                return (
                    "deleted"
                    if receipt["item_hash"] == preview["item_hash"]
                    and receipt["revision"] == preview["revision"]
                    else "conflict"
                )
            atom = self.get_by_id(str(preview["atom_id"]))
            if atom is None:
                return "not_found"
            if (
                atom.revision != preview["revision"]
                or hashlib.sha256(atom.stable_key.encode()).hexdigest() != preview["item_hash"]
            ):
                return "conflict"
            return self.delete_confirmed(
                atom.stable_key,
                atom.revision,
                operation_id,
                expires_at=float(preview["expires_at"]),
            )
