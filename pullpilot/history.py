"""SQLite-backed review history for the web UI. One row per completed review;
the full response payload is stored as JSON so past reviews re-render exactly."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(_ROOT, "data", "reviews.db")

_SCHEMA = """CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    target TEXT NOT NULL,
    engine TEXT NOT NULL,
    n_issues INTEGER NOT NULL,
    summary TEXT NOT NULL,
    payload TEXT NOT NULL
)"""


def _connect(path: Optional[str] = None) -> sqlite3.Connection:
    path = path or DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(_SCHEMA)
    return con


def save_review(target: str, engine: str, payload: dict,
                path: Optional[str] = None) -> int:
    con = _connect(path)
    with con:
        cur = con.execute(
            "INSERT INTO reviews (created_at, target, engine, n_issues, summary, payload) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"),
             target, engine, len(payload.get("issues", [])),
             payload.get("summary", ""), json.dumps(payload)))
    con.close()
    return cur.lastrowid


def list_reviews(limit: int = 20, path: Optional[str] = None) -> List[dict]:
    con = _connect(path)
    rows = con.execute(
        "SELECT id, created_at, target, engine, n_issues, summary "
        "FROM reviews ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    keys = ("id", "created_at", "target", "engine", "n_issues", "summary")
    return [dict(zip(keys, r)) for r in rows]


def get_review(review_id: int, path: Optional[str] = None) -> Optional[dict]:
    con = _connect(path)
    row = con.execute("SELECT payload FROM reviews WHERE id = ?",
                      (review_id,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else None
