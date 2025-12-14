from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from shared.models import MemoryItem, MemoryKind, MemoryRecord, ProjectFact, UserPreference


class MemoryManager:
    def __init__(self, db_path: str = "memory/memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                id TEXT PRIMARY KEY,
                kind TEXT,
                content TEXT,
                tags TEXT,
                meta TEXT,
                timestamp TEXT
            )
            """
        )
        columns = {row[1] for row in cur.execute("PRAGMA table_info(memory)").fetchall()}
        if "kind" not in columns:
            cur.execute("ALTER TABLE memory ADD COLUMN kind TEXT DEFAULT 'note'")
        if "meta" not in columns:
            cur.execute("ALTER TABLE memory ADD COLUMN meta TEXT")
        self.conn.commit()

    def save(self, item: MemoryRecord | MemoryItem) -> None:
        record = self._validate_record(self._to_record(item))
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO memory (id, kind, content, tags, meta, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.kind.value,
                    record.content,
                    ",".join(record.tags),
                    json.dumps(record.meta or {}, ensure_ascii=False),
                    record.timestamp,
                ),
            )

    def search(self, query: str, kind: MemoryKind | None = None) -> list[MemoryRecord]:
        cur = self.conn.cursor()
        if kind:
            cur.execute(
                "SELECT id, kind, content, tags, meta, timestamp "
                "FROM memory WHERE content LIKE ? AND kind = ?",
                (f"%{query}%", kind.value),
            )
        else:
            cur.execute(
                "SELECT id, kind, content, tags, meta, timestamp FROM memory WHERE content LIKE ?",
                (f"%{query}%",),
            )
        return [self._row_to_record(row) for row in cur.fetchall()]

    def get_recent(self, limit: int = 5, kind: MemoryKind | None = None) -> list[MemoryRecord]:
        cur = self.conn.cursor()
        if kind:
            cur.execute(
                "SELECT id, kind, content, tags, meta, timestamp "
                "FROM memory WHERE kind = ? ORDER BY timestamp DESC LIMIT ?",
                (kind.value, limit),
            )
        else:
            cur.execute(
                "SELECT id, kind, content, tags, meta, timestamp "
                "FROM memory ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
        return [self._row_to_record(row) for row in cur.fetchall()]

    def get_user_prefs(self, key: str | None = None) -> list[MemoryRecord]:
        records = self.search("", kind=MemoryKind.USER_PREF)
        if key is None:
            return records
        return [rec for rec in records if (rec.meta or {}).get("key") == key]

    def get_project_facts(self, project: str) -> list[MemoryRecord]:
        records = self.search("", kind=MemoryKind.PROJECT_FACT)
        return [
            rec
            for rec in records
            if (rec.meta or {}).get("project") == project or project in rec.tags
        ]

    def _row_to_record(self, row: Iterable[str]) -> MemoryRecord:
        id_val, kind_val, content, tags, meta_json, timestamp = row
        meta = json.loads(meta_json) if meta_json else {}
        tags_list = tags.split(",") if tags else []
        return MemoryRecord(
            id=id_val,
            kind=MemoryKind(kind_val),
            content=content,
            tags=tags_list,
            timestamp=timestamp,
            meta=meta,
        )

    def _to_record(self, item: MemoryRecord | MemoryItem) -> MemoryRecord:
        if isinstance(item, MemoryRecord):
            return item
        return MemoryRecord(
            id=item.id,
            kind=MemoryKind.NOTE,
            content=item.content,
            tags=item.tags,
            timestamp=item.timestamp,
            meta={},
        )

    def save_user_pref(self, pref: UserPreference) -> None:
        validated = self._validate_user_pref(pref)
        if isinstance(validated.value, (bytes, bytearray)):
            value_str = validated.value.decode("utf-8", errors="replace")
        else:
            value_str = str(validated.value)
        record = MemoryRecord(
            id=validated.id,
            kind=MemoryKind.USER_PREF,
            content=f"{validated.key}={value_str}",
            tags=validated.tags,
            timestamp=validated.timestamp,
            meta={"key": validated.key, "value": validated.value, "source": validated.source},
        )
        self.save(record)

    def save_project_fact(self, fact: ProjectFact) -> None:
        validated = self._validate_project_fact(fact)
        record = MemoryRecord(
            id=validated.id,
            kind=MemoryKind.PROJECT_FACT,
            content=validated.content,
            tags=[*validated.tags, validated.project],
            timestamp=validated.timestamp,
            meta={"project": validated.project, **(validated.meta or {})},
        )
        self.save(record)

    def _validate_record(self, record: MemoryRecord) -> MemoryRecord:
        if not record.id or not record.content or not record.timestamp:
            raise ValueError("Невалидная запись памяти: пустые обязательные поля")
        if not isinstance(record.tags, list) or not all(isinstance(t, str) for t in record.tags):
            raise ValueError("Невалидные теги в записи памяти")
        if record.meta is not None and not isinstance(record.meta, dict):
            raise ValueError("Невалидное meta в записи памяти")
        return record

    def _validate_user_pref(self, pref: UserPreference) -> UserPreference:
        if not pref.id or not pref.key or pref.value is None or not pref.timestamp:
            raise ValueError("Невалидная пользовательская настройка: обязательные поля пусты")
        if not isinstance(pref.tags, list) or not all(isinstance(t, str) for t in pref.tags):
            raise ValueError("Невалидные теги в пользовательской настройке")
        if pref.meta is not None and not isinstance(pref.meta, dict):
            raise ValueError("Невалидное meta в пользовательской настройке")
        return pref

    def _validate_project_fact(self, fact: ProjectFact) -> ProjectFact:
        if not fact.id or not fact.project or not fact.content or not fact.timestamp:
            raise ValueError("Невалидный факт проекта: обязательные поля пусты")
        if not isinstance(fact.tags, list) or not all(isinstance(t, str) for t in fact.tags):
            raise ValueError("Невалидные теги в факте проекта")
        if fact.meta is not None and not isinstance(fact.meta, dict):
            raise ValueError("Невалидное meta в факте проекта")
        return fact
