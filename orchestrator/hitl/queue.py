
import json
import time
from functools import lru_cache

from .. import config
from ..db import Backend

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, task_id TEXT, user_id TEXT, level TEXT, trigger_name TEXT,
  status TEXT, payload TEXT, decision TEXT, created_at REAL, decided_at REAL, resolved_at REAL);
CREATE TABLE IF NOT EXISTS chat (
  id INTEGER PRIMARY KEY AUTOINCREMENT, review_id TEXT, role TEXT, content TEXT, ts REAL);
"""
PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
  review_id TEXT PRIMARY KEY, task_id TEXT, user_id TEXT, level TEXT, trigger_name TEXT,
  status TEXT, payload TEXT, decision TEXT, created_at DOUBLE PRECISION, decided_at DOUBLE PRECISION,
  resolved_at DOUBLE PRECISION);
CREATE TABLE IF NOT EXISTS chat (
  id BIGSERIAL PRIMARY KEY, review_id TEXT, role TEXT, content TEXT, ts DOUBLE PRECISION);
"""
# status: pending -> decided -> resolved      (a human must decide)
#         notified -> resolved                (information only)


def _row(r) -> dict:
    d = dict(r)
    d["payload"] = json.loads(d["payload"])
    d["decision"] = json.loads(d["decision"]) if d["decision"] else None
    d["trigger"] = d.pop("trigger_name")
    return d


class ReviewQueue:
    def __init__(self, path=None, url=None) -> None:
        if path is not None:                       # an explicit path always means SQLite (used by tests)
            self.db = Backend(path=path)
        elif url or config.DATABASE_URL:
            self.db = Backend(url=url or config.DATABASE_URL)
        else:
            self.db = Backend(path=config.REVIEW_DB)
        self.db.init(SQLITE_SCHEMA, PG_SCHEMA)
        self.path = self.db.path

    def submit(self, review_id: str, task_id: str, user_id: str, level: str, trigger: str,
               payload: dict, status: str = "pending") -> bool:
        """Idempotent: returns True only the first time this review_id is created."""
        _, n = self.db.run(
            "INSERT INTO reviews (review_id, task_id, user_id, level, trigger_name, status, payload, created_at) "
            "VALUES (?,?,?,?,?,?,?,?) ON CONFLICT (review_id) DO NOTHING",
            (review_id, task_id, user_id, level, trigger, status, json.dumps(payload, default=str), time.time()))
        return n == 1

    def get(self, review_id: str) -> dict | None:
        rows, _ = self.db.run("SELECT * FROM reviews WHERE review_id=?", (review_id,))
        return _row(rows[0]) if rows else None

    def list_reviews(self, status: str | None = None, user_id: str | None = None, limit: int = 100) -> list[dict]:
        sql, params = "SELECT * FROM reviews WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if user_id:
            sql += " AND user_id=?"
            params.append(user_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows, _ = self.db.run(sql, tuple(params))
        return [_row(r) for r in rows]

    def decide(self, review_id: str, decision: dict) -> bool:
        """Record the human's decision. Works once, and only while the review is pending."""
        _, n = self.db.run("UPDATE reviews SET status='decided', decision=?, decided_at=? "
                           "WHERE review_id=? AND status='pending'",
                           (json.dumps(decision), time.time(), review_id))
        return n == 1

    def mark_resolved(self, review_id: str) -> None:
        self.db.run("UPDATE reviews SET status='resolved', resolved_at=? "
                    "WHERE review_id=? AND status IN ('decided','notified')", (time.time(), review_id))

    ack = mark_resolved

    def add_chat(self, review_id: str, role: str, content: str) -> None:
        self.db.run("INSERT INTO chat (review_id, role, content, ts) VALUES (?,?,?,?)",
                    (review_id, role, content, time.time()))

    def chat(self, review_id: str) -> list[dict]:
        rows, _ = self.db.run("SELECT role, content, ts FROM chat WHERE review_id=? ORDER BY id", (review_id,))
        return [dict(r) for r in rows]


@lru_cache(maxsize=1)
def get_queue() -> ReviewQueue:
    return ReviewQueue()