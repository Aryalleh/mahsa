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

CREATE TABLE IF NOT EXISTS style_samples (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source    TEXT,                        -- which channel it came from
    content   TEXT    NOT NULL UNIQUE,
    created_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS contacts (
    user_id   INTEGER PRIMARY KEY,
    display   TEXT,
    status    TEXT    NOT NULL DEFAULT 'pending',  -- pending | approved | blocked
    created_at TEXT   NOT NULL,
    updated_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS lovers (
    user_id   INTEGER PRIMARY KEY,   -- consenting adults who get the intimate mode
    created_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS spouses (
    user_id   INTEGER PRIMARY KEY,   -- people Mahsa is married to (polygamy is fine)
    name      TEXT,
    married_at TEXT   NOT NULL
);

CREATE TABLE IF NOT EXISTS ships (
    user_a    INTEGER NOT NULL,      -- two *other* users married to each other
    user_b    INTEGER NOT NULL,      -- stored with user_a < user_b
    created_at TEXT   NOT NULL,
    PRIMARY KEY (user_a, user_b)
);

CREATE TABLE IF NOT EXISTS pending_asks (
    target_id INTEGER NOT NULL,      -- the friend Mahsa was told to ask
    asker_id  INTEGER NOT NULL,      -- who wants the answer reported back
    question  TEXT,
    created_at TEXT   NOT NULL,
    PRIMARY KEY (target_id, asker_id)
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

    # ---- style samples (learned texting vibe from public channels) ------
    def add_style_sample(self, content: str, source: str | None = None) -> bool:
        content = content.strip()
        if not content:
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO style_samples(source, content, created_at) VALUES (?,?,?)",
                    (source, content, datetime.utcnow().isoformat()),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # duplicate

    def get_style_samples(self, limit: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT content FROM style_samples ORDER BY RANDOM() LIMIT ?", (limit,)
            ).fetchall()
        return [r["content"] for r in rows]

    def count_style_samples(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM style_samples").fetchone()
        return row["n"]

    def clear_style_samples(self, source: str | None = None) -> int:
        with self._lock:
            if source:
                cur = self._conn.execute("DELETE FROM style_samples WHERE source=?", (source,))
            else:
                cur = self._conn.execute("DELETE FROM style_samples")
            self._conn.commit()
            return cur.rowcount

    # ---- contacts / whitelist -------------------------------------------
    def get_contact_status(self, user_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM contacts WHERE user_id=?", (user_id,)
            ).fetchone()
        return row["status"] if row else None

    def upsert_contact(self, user_id: int, display: str | None, status: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO contacts(user_id, display, status, created_at, updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       status=excluded.status,
                       display=COALESCE(excluded.display, contacts.display),
                       updated_at=excluded.updated_at""",
                (user_id, display, status, now, now),
            )
            self._conn.commit()

    def set_contact_status(self, user_id: int, status: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE contacts SET status=?, updated_at=? WHERE user_id=?",
                (status, datetime.utcnow().isoformat(), user_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def find_contacts_by_name(self, name: str) -> list[tuple[int, str]]:
        """Approved contacts whose display name contains `name` (case-insensitive)."""
        needle = name.strip().lower()
        if not needle:
            return []
        out = []
        for uid, disp, status in self.list_contacts("approved"):
            # Skip contacts with no real name ("?"), so we never relay blindly.
            if disp and disp != "?" and needle in disp.lower():
                out.append((uid, disp))
        return out

    def set_contact_display(self, user_id: int, display: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE contacts SET display=?, updated_at=? WHERE user_id=?",
                (display, datetime.utcnow().isoformat(), user_id),
            )
            self._conn.commit()

    def list_contacts(self, status: str | None = None) -> list[tuple[int, str, str]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT user_id, display, status FROM contacts WHERE status=? ORDER BY updated_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT user_id, display, status FROM contacts ORDER BY updated_at DESC"
                ).fetchall()
        return [(r["user_id"], r["display"] or "?", r["status"]) for r in rows]

    # ---- lovers (intimate chat mode, without admin powers) --------------
    def add_lover(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO lovers(user_id, created_at) VALUES (?,?)",
                (user_id, datetime.utcnow().isoformat()),
            )
            self._conn.commit()

    def remove_lover(self, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM lovers WHERE user_id=?", (user_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def is_lover(self, user_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM lovers WHERE user_id=?", (user_id,)
            ).fetchone()
        return row is not None

    def list_lovers(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT l.user_id, COALESCE(c.display, '?')
                   FROM lovers l LEFT JOIN contacts c ON c.user_id = l.user_id
                   ORDER BY l.created_at DESC"""
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ---- spouses / marriage (polygamy allowed) --------------------------
    def add_spouse(self, user_id: int, name: str | None = None) -> bool:
        """Marry a user. Returns True if newly married, False if already married."""
        with self._lock:
            existing = self._conn.execute(
                "SELECT 1 FROM spouses WHERE user_id=?", (user_id,)
            ).fetchone()
            self._conn.execute(
                """INSERT INTO spouses(user_id, name, married_at) VALUES (?,?,?)
                   ON CONFLICT(user_id) DO UPDATE SET
                       name=COALESCE(excluded.name, spouses.name)""",
                (user_id, name, datetime.utcnow().isoformat()),
            )
            self._conn.commit()
            return existing is None

    def remove_spouse(self, user_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM spouses WHERE user_id=?", (user_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def is_spouse(self, user_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM spouses WHERE user_id=?", (user_id,)
            ).fetchone()
        return row is not None

    def list_spouses(self) -> list[tuple[int, str]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT s.user_id, COALESCE(s.name, c.display, '?')
                   FROM spouses s LEFT JOIN contacts c ON c.user_id = s.user_id
                   ORDER BY s.married_at"""
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

    # ---- ships (marrying two other users to each other) -----------------
    def contact_display(self, user_id: int) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT display FROM contacts WHERE user_id=?", (user_id,)
            ).fetchone()
        return (row["display"] if row and row["display"] else str(user_id))

    def add_ship(self, a: int, b: int) -> bool:
        lo, hi = (a, b) if a < b else (b, a)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO ships(user_a, user_b, created_at) VALUES (?,?,?)",
                    (lo, hi, datetime.utcnow().isoformat()),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False  # already shipped

    def remove_ship(self, a: int, b: int) -> bool:
        lo, hi = (a, b) if a < b else (b, a)
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM ships WHERE user_a=? AND user_b=?", (lo, hi)
            )
            self._conn.commit()
            return cur.rowcount > 0

    def partners_of(self, user_id: int) -> list[tuple[int, str]]:
        """Who this user is married to (their shipped partners), as (id, name)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_a, user_b FROM ships WHERE user_a=? OR user_b=?",
                (user_id, user_id),
            ).fetchall()
        out = []
        for r in rows:
            other = r["user_b"] if r["user_a"] == user_id else r["user_a"]
            out.append((other, self.contact_display(other)))
        return out

    def list_ships(self) -> list[tuple[int, str, int, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_a, user_b FROM ships ORDER BY created_at"
            ).fetchall()
        return [(r["user_a"], self.contact_display(r["user_a"]),
                 r["user_b"], self.contact_display(r["user_b"])) for r in rows]

    # ---- pending cross-chat questions -----------------------------------
    def add_pending_ask(self, target_id: int, asker_id: int, question: str | None) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO pending_asks(target_id, asker_id, question, created_at)
                   VALUES (?,?,?,?)
                   ON CONFLICT(target_id, asker_id) DO UPDATE SET
                       question=excluded.question, created_at=excluded.created_at""",
                (target_id, asker_id, question, datetime.utcnow().isoformat()),
            )
            self._conn.commit()

    def pop_pending_asks(self, target_id: int) -> list[tuple[int, str]]:
        """Return and clear all pending questions waiting on this target."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT asker_id, question FROM pending_asks WHERE target_id=?",
                (target_id,),
            ).fetchall()
            if rows:
                self._conn.execute(
                    "DELETE FROM pending_asks WHERE target_id=?", (target_id,)
                )
                self._conn.commit()
        return [(r["asker_id"], r["question"] or "") for r in rows]

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
