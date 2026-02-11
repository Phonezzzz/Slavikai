from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

from shared.canonical_atom_models import AtomStatus, CanonicalAtom, ClaimType, utc_now_iso
from shared.models import JSONValue
from shared.sanitize import safe_json_loads


class CanonicalAtomStore:
    def __init__(self, db_path: str = "memory/canonical_atoms.db") -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
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
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canonical_atom_claim_status "
            "ON canonical_atom (claim_type, status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_canonical_atom_last_seen "
            "ON canonical_atom (last_seen_at)"
        )
        self.conn.commit()

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
        limit: int = 200,
    ) -> list[CanonicalAtom]:
        if limit <= 0:
            raise ValueError("limit должен быть > 0")
        query = (
            "SELECT * FROM canonical_atom WHERE 1=1 "
            "{status_clause} {type_clause} "
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
        params.append(limit)
        rows = self.conn.execute(
            query.format(status_clause=status_clause, type_clause=type_clause),
            tuple(params),
        ).fetchall()
        return [self._row_to_atom(row) for row in rows]

    def list_conflicts(self, *, limit: int = 100) -> list[CanonicalAtom]:
        return self.list_atoms(statuses={AtomStatus.CONFLICT}, limit=limit)

    def upsert(self, atom: CanonicalAtom) -> CanonicalAtom:
        payload = self._atom_for_write(atom)
        with self.conn:
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
                    summary_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stable_key) DO UPDATE SET
                    atom_id=excluded.atom_id,
                    claim_type=excluded.claim_type,
                    value_json=excluded.value_json,
                    confidence=excluded.confidence,
                    support_count=excluded.support_count,
                    contradict_count=excluded.contradict_count,
                    last_seen_at=excluded.last_seen_at,
                    status=excluded.status,
                    summary_text=excluded.summary_text
                """,
                payload,
            )
        persisted = self.get_by_stable_key(atom.stable_key)
        if persisted is None:
            raise RuntimeError("atom missing after upsert")
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
    ) -> CanonicalAtom:
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
        )
        return self.upsert(resolved)

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
        )
