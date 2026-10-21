from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from shared.models import JSONValue

PERSONAL_FIELDS = frozenset({"personalization", "appearance", "composer", "reset_preferences"})


class PreferencesConflictError(Exception):
    pass


@dataclass(frozen=True)
class Preferences:
    tone: str = "balanced"
    system_prompt: str = ""
    theme: str = "default"
    long_paste_to_file_enabled: bool = True
    long_paste_threshold_chars: int = 12_000

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "personalization": {"tone": self.tone, "system_prompt": self.system_prompt},
            "appearance": {"theme": self.theme},
            "composer": {
                "long_paste_to_file_enabled": self.long_paste_to_file_enabled,
                "long_paste_threshold_chars": self.long_paste_threshold_chars,
            },
        }

    def patched(self, payload: dict[str, object]) -> Preferences:
        if not payload or set(payload) - PERSONAL_FIELDS:
            raise ValueError("Нужны только личные настройки.")
        if "reset_preferences" in payload:
            if set(payload) != {"reset_preferences"} or payload["reset_preferences"] is not True:
                raise ValueError("Сброс личных настроек выполняется отдельным запросом.")
            return Preferences()
        result = self
        for group, value in payload.items():
            if not isinstance(value, dict):
                raise ValueError(f"{group} должен быть объектом.")
            fields = {
                "personalization": {"tone", "system_prompt"},
                "appearance": {"theme"},
                "composer": {"long_paste_to_file_enabled", "long_paste_threshold_chars"},
            }[group]
            if not value or set(value) - fields:
                raise ValueError(f"Неизвестные или отсутствующие поля {group}.")
            for name, item in value.items():
                if name == "tone":
                    if not isinstance(item, str) or not item.strip() or len(item) > 64:
                        raise ValueError("tone должен быть непустой строкой до 64 символов.")
                    result = replace(result, tone=item.strip())
                elif name == "system_prompt":
                    if not isinstance(item, str) or len(item) > 32_000:
                        raise ValueError("system_prompt должен быть строкой до 32000 символов.")
                    result = replace(result, system_prompt=item)
                elif name == "theme":
                    if item not in ("default", "oled"):
                        raise ValueError("theme должен быть default или oled.")
                    result = replace(result, theme=cast(str, item))
                elif name == "long_paste_to_file_enabled":
                    if not isinstance(item, bool):
                        raise ValueError("long_paste_to_file_enabled должен быть bool.")
                    result = replace(result, long_paste_to_file_enabled=item)
                else:
                    if (
                        isinstance(item, bool)
                        or not isinstance(item, int)
                        or not 1000 <= item <= 80000
                    ):
                        raise ValueError(
                            "long_paste_threshold_chars должен быть int в 1000..80000."
                        )
                    result = replace(result, long_paste_threshold_chars=item)
        return result


@dataclass(frozen=True)
class PreferencesSnapshot:
    value: Preferences
    revision: int
    principal_id: str

    @property
    def subject(self) -> str:
        return hashlib.sha256(self.principal_id.encode()).hexdigest()

    @property
    def etag(self) -> str:
        return f'"preferences-{self.subject}-{self.revision}"'


class PreferencesStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600, exist_ok=True)
        path.chmod(0o600)
        with closing(sqlite3.connect(path, check_same_thread=False)) as conn, conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS preferences "
                "(principal_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, payload TEXT NOT NULL)"
            )

    @staticmethod
    def _read(conn: sqlite3.Connection, principal_id: str) -> PreferencesSnapshot:
        if not principal_id.strip():
            raise ValueError("principal_id обязателен.")
        row = conn.execute(
            "SELECT revision, payload FROM preferences WHERE principal_id = ?", (principal_id,)
        ).fetchone()
        if row is None:
            return PreferencesSnapshot(Preferences(), 0, principal_id)
        raw = json.loads(row[1])
        if not isinstance(raw, dict):
            raise ValueError("Повреждены сохранённые личные настройки.")
        return PreferencesSnapshot(Preferences().patched(raw), row[0], principal_id)

    def get(self, principal_id: str) -> PreferencesSnapshot:
        with closing(sqlite3.connect(self._path, check_same_thread=False)) as conn:
            return self._read(conn, principal_id)

    def update(
        self, principal_id: str, *, expected_etag: str, payload: dict[str, object]
    ) -> PreferencesSnapshot:
        with closing(sqlite3.connect(self._path, check_same_thread=False)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            current = self._read(conn, principal_id)
            if current.etag != expected_etag:
                raise PreferencesConflictError("Настройки изменились или сменился пользователь.")
            updated = PreferencesSnapshot(
                current.value.patched(payload), current.revision + 1, principal_id
            )
            conn.execute(
                "INSERT INTO preferences VALUES (?, ?, ?) ON CONFLICT(principal_id) DO UPDATE "
                "SET revision = excluded.revision, payload = excluded.payload",
                (
                    principal_id,
                    updated.revision,
                    json.dumps(updated.value.to_dict(), ensure_ascii=False),
                ),
            )
            return updated
