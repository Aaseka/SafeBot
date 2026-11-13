"""SQLite persistence: per-server config, warnings, flags, and the RAG knowledge base."""
from __future__ import annotations

import json
import time

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS guild_config (
    guild_id        INTEGER PRIMARY KEY,
    log_channel_id  INTEGER,
    max_warnings    INTEGER NOT NULL DEFAULT 3,
    timeout_minutes INTEGER NOT NULL DEFAULT 60,
    strictness      TEXT    NOT NULL DEFAULT 'balanced',
    images_enabled  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS flags (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    content    TEXT    NOT NULL,
    category   TEXT    NOT NULL,
    reason     TEXT    NOT NULL,
    scores     TEXT    NOT NULL DEFAULT '{}',
    status     TEXT    NOT NULL DEFAULT 'open',
    image_hashes TEXT  NOT NULL DEFAULT '[]',
    created_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS image_hashes (
    guild_id INTEGER NOT NULL,
    hash     TEXT    NOT NULL,          -- 64-bit dHash, hex
    label    TEXT    NOT NULL,          -- blocked | allowed
    PRIMARY KEY (guild_id, hash)
);
CREATE TABLE IF NOT EXISTS warnings (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id   INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    flag_id    INTEGER,
    reason     TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS docs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id  INTEGER NOT NULL,
    kind      TEXT    NOT NULL,          -- rule | faq | example
    text      TEXT    NOT NULL,
    meta      TEXT    NOT NULL DEFAULT '{}',
    embedding BLOB    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_warnings ON warnings (guild_id, user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_docs ON docs (guild_id, kind);
"""

CONFIG_KEYS = {"log_channel_id", "max_warnings", "timeout_minutes", "strictness", "images_enabled"}

# Columns added after the first release; applied to older databases on startup.
MIGRATIONS = (
    "ALTER TABLE guild_config ADD COLUMN images_enabled INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE flags ADD COLUMN image_hashes TEXT NOT NULL DEFAULT '[]'",
)


class DB:
    def __init__(self, path: str):
        self.path = path
        self.conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self.conn = await aiosqlite.connect(self.path)
        self.conn.row_factory = aiosqlite.Row
        await self.conn.execute("PRAGMA journal_mode=WAL")
        await self.conn.executescript(SCHEMA)
        for sql in MIGRATIONS:
            try:
                await self.conn.execute(sql)
            except aiosqlite.OperationalError:  # column already exists
                pass
        await self.conn.commit()

    async def close(self) -> None:
        if self.conn:
            await self.conn.close()

    async def _one(self, sql: str, args=()) -> dict | None:
        cur = await self.conn.execute(sql, args)
        row = await cur.fetchone()
        return dict(row) if row else None

    async def _all(self, sql: str, args=()) -> list[dict]:
        cur = await self.conn.execute(sql, args)
        return [dict(r) for r in await cur.fetchall()]

    async def _write(self, sql: str, args=()) -> int:
        cur = await self.conn.execute(sql, args)
        await self.conn.commit()
        return cur.lastrowid

    # ---- config ----
    async def config(self, guild_id: int) -> dict:
        row = await self._one("SELECT * FROM guild_config WHERE guild_id=?", (guild_id,))
        if row is None:
            await self._write("INSERT OR IGNORE INTO guild_config (guild_id) VALUES (?)", (guild_id,))
            row = await self._one("SELECT * FROM guild_config WHERE guild_id=?", (guild_id,))
        return row

    async def set_config(self, guild_id: int, **values) -> None:
        values = {k: v for k, v in values.items() if k in CONFIG_KEYS and v is not None}
        if not values:
            return
        await self.config(guild_id)
        cols = ", ".join(f"{k}=?" for k in values)
        await self._write(f"UPDATE guild_config SET {cols} WHERE guild_id=?", (*values.values(), guild_id))

    # ---- flags & warnings ----
    async def add_flag(self, guild_id, user_id, channel_id, content, category, reason, scores,
                       image_hashes: list[str] | None = None) -> int:
        return await self._write(
            "INSERT INTO flags (guild_id, user_id, channel_id, content, category, reason, scores, image_hashes,"
            " created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (guild_id, user_id, channel_id, content, category, reason, json.dumps(scores),
             json.dumps(image_hashes or []), time.time()),
        )

    async def get_flag(self, flag_id: int) -> dict | None:
        return await self._one("SELECT * FROM flags WHERE id=?", (flag_id,))

    async def set_flag_status(self, flag_id: int, status: str) -> None:
        await self._write("UPDATE flags SET status=? WHERE id=?", (status, flag_id))

    async def add_warning(self, guild_id, user_id, flag_id, reason) -> None:
        await self._write(
            "INSERT INTO warnings (guild_id, user_id, flag_id, reason, created_at) VALUES (?,?,?,?,?)",
            (guild_id, user_id, flag_id, reason, time.time()),
        )

    async def remove_warning_for_flag(self, flag_id: int) -> None:
        await self._write("DELETE FROM warnings WHERE flag_id=?", (flag_id,))

    async def count_warnings(self, guild_id, user_id, window_days: int = 30) -> int:
        since = time.time() - window_days * 86400
        row = await self._one(
            "SELECT COUNT(*) AS n FROM warnings WHERE guild_id=? AND user_id=? AND created_at>=?",
            (guild_id, user_id, since),
        )
        return row["n"]

    async def list_warnings(self, guild_id, user_id, limit: int = 10) -> list[dict]:
        return await self._all(
            "SELECT * FROM warnings WHERE guild_id=? AND user_id=? ORDER BY created_at DESC LIMIT ?",
            (guild_id, user_id, limit),
        )

    async def clear_warnings(self, guild_id, user_id) -> None:
        await self._write("DELETE FROM warnings WHERE guild_id=? AND user_id=?", (guild_id, user_id))

    # ---- image hash memory ----
    async def label_image_hashes(self, guild_id, hashes: list[str], label: str) -> None:
        await self.conn.executemany(
            "INSERT INTO image_hashes (guild_id, hash, label) VALUES (?,?,?)"
            " ON CONFLICT (guild_id, hash) DO UPDATE SET label=excluded.label",
            [(guild_id, h, label) for h in hashes],
        )
        await self.conn.commit()

    async def image_hashes(self, guild_id) -> list[dict]:
        return await self._all("SELECT hash, label FROM image_hashes WHERE guild_id=?", (guild_id,))

    # ---- knowledge base ----
    async def add_doc(self, guild_id, kind, text, meta: dict, embedding: bytes) -> int:
        return await self._write(
            "INSERT INTO docs (guild_id, kind, text, meta, embedding) VALUES (?,?,?,?,?)",
            (guild_id, kind, text, json.dumps(meta), embedding),
        )

    async def docs(self, guild_id, kinds: set[str] | None = None) -> list[dict]:
        rows = await self._all("SELECT * FROM docs WHERE guild_id=? ORDER BY id", (guild_id,))
        if kinds:
            rows = [r for r in rows if r["kind"] in kinds]
        for r in rows:
            r["meta"] = json.loads(r["meta"])
        return rows

    async def delete_doc(self, guild_id, doc_id) -> bool:
        cur = await self.conn.execute("DELETE FROM docs WHERE guild_id=? AND id=?", (guild_id, doc_id))
        await self.conn.commit()
        return cur.rowcount > 0
