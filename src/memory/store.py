"""SQLite-backed memory: conversations, learned personality facts, mood journal.

Everything Mahsa "remembers" lives here so it survives restarts:

* ``messages``       — rolling per-user chat history (context for replies)
* ``facts``          — learned personality traits / things people taught her
* ``journal``        — her daily emotional diary (source for channel posts)
* ``user_notes``     — a short remembered summary per user she talks to
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER NOT NULL,
    role      TEXT    NOT NULL,          -- 'user' or 'assistant'
    content   TEXT    NOT NULL,
    created_at TEXT   NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, id);

CREATE TABLE IF NOT EXISTS facts (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    category  TEXT    NOT NULL DEFAULT 'general',
    content   TEXT    NOT NULL UNIQUE,
    source    TEXT,                        -- who/what taught this
    created_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS journal (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    day       TEXT    NOT NULL,            -- ISO date
    mood      TEXT,
    entry     TEXT    NOT NULL,
    posted    INTEGER NOT NULL DEFAULT 0,
    created_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS user_notes (
    user_id   INTEGER PRIMARY KEY,
    display   TEXT,
    note      TEXT,
    updated_at TEXT NOT NULL
);
"""


@dataclass
class Turn:
    role: str
    content: str


class MemoryStore:
    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False + a lock: safe for our single-process async app.
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- conversation history -------------------------------------------
    def add_message(self, user_id: int, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(user_id, role, content, created_at) VALUES (?,?,?,?)",
                (user_id, role, content, datetime.utcnow().isoformat()),
            )
            self._conn.commit()

    def recent_turns(self, user_id: int, limit: int) -> list[Turn]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [Turn(r["role"], r["content"]) for r in reversed(rows)]

    def clear_user(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
            self._conn.commit()

    # ---- learned personality facts --------------------------------------
    def add_fact(self, content: str, category: str = "general", source: str | None = None) -> bool:
        content = content.strip()
        if not content:
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO facts(category, content, source, created_at) VALUES (?,?,?,?)",
                    (category, content, source, datetime.utcnow().isoformat()),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # duplicate

    def get_facts(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM facts ORDER BY id"
            ).fetchall()
        return [r["content"] for r in rows]

    def delete_fact(self, fact_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def list_facts_with_ids(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content FROM facts ORDER BY id"
            ).fetchall()
        return [(r["id"], r["content"]) for r in rows]

    # ---- mood / diary journal -------------------------------------------
    def add_journal(self, entry: str, mood: str | None = None, day: str | None = None) -> int:
        day = day or date.today().isoformat()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO journal(day, mood, entry, posted, created_at) VALUES (?,?,?,0,?)",
                (day, mood, entry, datetime.utcnow().isoformat()),
            )
            self._conn.commit()
            return cur.lastrowid

    def mark_journal_posted(self, journal_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE journal SET posted=1 WHERE id=?", (journal_id,))
            self._conn.commit()

    def recent_journal(self, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT day, mood, entry FROM journal ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def latest_mood(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT mood FROM journal WHERE mood IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row["mood"] if row else None

    def has_journal_today(self) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM journal WHERE day=? LIMIT 1", (date.today().isoformat(),)
            ).fetchone()
        return row is not None

    # ---- per-user notes --------------------------------------------------
    def set_user_note(self, user_id: int, note: str, display: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO user_notes(user_id, display, note, updated_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       note=excluded.note,
                       display=COALESCE(excluded.display, user_notes.display),
                       updated_at=excluded.updated_at""",
                (user_id, display, note, datetime.utcnow().isoformat()),
            )
            self._conn.commit()

    def get_user_note(self, user_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT note FROM user_notes WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["note"] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
