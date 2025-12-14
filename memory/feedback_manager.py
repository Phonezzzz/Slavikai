from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class FeedbackManager:
    def __init__(self, db_path: str = "memory/feedback.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                answer TEXT,
                rating TEXT,
                severity TEXT,
                hint TEXT,
                timestamp TEXT
            )
            """
        )
        # миграция старых таблиц
        columns = {row[1] for row in cur.execute("PRAGMA table_info(feedback)").fetchall()}
        if "severity" not in columns:
            cur.execute("ALTER TABLE feedback ADD COLUMN severity TEXT DEFAULT 'minor'")
        if "hint" not in columns:
            cur.execute("ALTER TABLE feedback ADD COLUMN hint TEXT")
        self.conn.commit()

    def save_feedback(
        self,
        prompt: str,
        answer: str,
        rating: str,
        severity: str = "minor",
        hint: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO feedback (prompt, answer, rating, severity, hint, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (prompt, answer, rating, severity, hint, time.strftime("%Y-%m-%d %H:%M:%S")),
            )

    def analyze_trends(self) -> dict[str, float]:
        cur = self.conn.cursor()
        cur.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
        data = {rating: count for rating, count in cur.fetchall()}
        total = sum(data.values()) or 1
        return {rating: count / total for rating, count in data.items()}

    def stats(self) -> dict[str, object]:
        """Возвращает агрегаты по рейтингу/серьёзности и топ-подсказки."""
        cur = self.conn.cursor()
        cur.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating")
        ratings = {str(r): int(c) for r, c in cur.fetchall()}

        cur.execute("SELECT severity, COUNT(*) FROM feedback GROUP BY severity")
        severity = {str(r): int(c) for r, c in cur.fetchall()}

        cur.execute(
            "SELECT hint, COUNT(*) as cnt FROM feedback "
            "WHERE hint IS NOT NULL AND hint != '' "
            "GROUP BY hint ORDER BY cnt DESC LIMIT 5"
        )
        top_hints = [{"hint": str(h), "count": int(c)} for h, c in cur.fetchall()]

        return {"ratings": ratings, "severity": severity, "top_hints": top_hints}

    def get_recent(self, limit: int = 10) -> list[tuple[str, str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt, answer, rating FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [(prompt, answer, rating) for prompt, answer, rating in cur.fetchall()]

    def get_recent_records(self, limit: int = 10) -> list[dict[str, str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt, answer, rating, severity, hint, timestamp "
            "FROM feedback ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "prompt": str(r[0]),
                "answer": str(r[1]),
                "rating": str(r[2]),
                "severity": str(r[3]),
                "hint": str(r[4]) if r[4] else "",
                "timestamp": str(r[5]),
            }
            for r in rows
        ]

    def get_recent_bad(self, limit: int = 10) -> list[dict[str, str]]:
        """Последние bad/offtopic с хинтами."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT prompt, answer, rating, severity, hint, timestamp "
            "FROM feedback WHERE rating IN ('bad','offtopic') "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            {
                "prompt": str(r[0]),
                "answer": str(r[1]),
                "rating": str(r[2]),
                "severity": str(r[3]),
                "hint": str(r[4]) if r[4] else "",
                "timestamp": str(r[5]),
            }
            for r in rows
        ]

    def get_recent_hints(
        self, limit: int = 3, severity_filter: list[str] | None = None
    ) -> list[str]:
        cur = self.conn.cursor()
        if severity_filter:
            placeholders = ",".join("?" for _ in severity_filter)
            cur.execute(
                f"SELECT hint FROM feedback WHERE hint IS NOT NULL AND hint != '' "
                f"AND severity IN ({placeholders}) "
                "ORDER BY id DESC LIMIT ?",
                (*severity_filter, limit),
            )
        else:
            cur.execute(
                "SELECT hint FROM feedback WHERE hint IS NOT NULL AND hint != '' "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        return [str(row[0]) for row in cur.fetchall()]

    def get_recent_hints_meta(
        self, limit: int = 3, severity_filter: list[str] | None = None
    ) -> list[dict[str, str]]:
        cur = self.conn.cursor()
        if severity_filter:
            placeholders = ",".join("?" for _ in severity_filter)
            cur.execute(
                f"SELECT hint, severity, timestamp FROM feedback "
                f"WHERE hint IS NOT NULL AND hint != '' AND severity IN ({placeholders}) "
                "ORDER BY id DESC LIMIT ?",
                (*severity_filter, limit),
            )
        else:
            cur.execute(
                "SELECT hint, severity, timestamp FROM feedback "
                "WHERE hint IS NOT NULL AND hint != '' "
                "ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        rows = cur.fetchall()
        return [
            {
                "hint": str(row[0]),
                "severity": str(row[1]),
                "timestamp": str(row[2]),
            }
            for row in rows
        ]
