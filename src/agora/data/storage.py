"""SQLite storage for application interaction events (POST /log_interaction).

Events written here are actual app events (display, click). They are kept apart
from the IDEA source records and are never used as training or evaluation data.
"""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVENT_TYPES = ("display", "click")

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    event_id       TEXT PRIMARY KEY,
    user_id        TEXT NOT NULL,
    feed_id        TEXT NOT NULL,
    article_id     TEXT NOT NULL,
    event_type     TEXT NOT NULL CHECK (event_type IN ('display', 'click')),
    timestamp      TEXT NOT NULL,
    displayed_rank INTEGER NOT NULL CHECK (displayed_rank >= 1),
    model_version  TEXT NOT NULL,
    received_at    TEXT NOT NULL,
    source         TEXT NOT NULL DEFAULT 'agora-app'
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions (user_id, timestamp);
"""

COLUMNS = ("event_id", "user_id", "feed_id", "article_id", "event_type", "timestamp",
           "displayed_rank", "model_version", "received_at", "source")


class InteractionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # One connection shared by FastAPI's worker threads: every access is serialised.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def insert(self, event: dict[str, Any]) -> bool:
        """Store one event; return False if its event_id was already stored."""
        if event["event_type"] not in EVENT_TYPES:
            raise ValueError(f"event_type must be one of {EVENT_TYPES}")
        row = dict(event)
        row.setdefault("received_at", datetime.now(timezone.utc).isoformat())
        row.setdefault("source", "agora-app")
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT OR IGNORE INTO interactions ({', '.join(COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in COLUMNS)})",
                [row[c] for c in COLUMNS])
            self._conn.commit()
            return cursor.rowcount == 1

    def recent(self, user_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = f"SELECT {', '.join(COLUMNS)} FROM interactions"
        params: list[Any] = []
        if user_id:
            query += " WHERE user_id = ?"
            params.append(user_id)
        query += " ORDER BY received_at DESC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(zip(COLUMNS, r)) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0])
