
import sqlite3
from pathlib import Path


def is_postgres(url) -> bool:
    return bool(url) and str(url).startswith(("postgres://", "postgresql://"))


class Backend:
    """One thin layer over SQLite and PostgreSQL. Callers write SQL with ? placeholders and use only
    syntax that both support (for example ON CONFLICT ... DO NOTHING / DO UPDATE)."""

    def __init__(self, path=None, url=None) -> None:
        self.url = url if is_postgres(url) else None
        self.path = Path(path) if (path is not None and not self.url) else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def kind(self) -> str:
        return "postgres" if self.url else "sqlite"

    def _connect(self):
        if self.url:
            import psycopg
            from psycopg.rows import dict_row
            return psycopg.connect(self.url, row_factory=dict_row)
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        return conn

    def run(self, sql: str, params=()):
        if self.url:
            sql = sql.replace("?", "%s")
        conn = self._connect()
        try:
            cur = conn.execute(sql, tuple(params))
            rows = cur.fetchall() if cur.description else []
            conn.commit()
            return rows, cur.rowcount
        finally:
            conn.close()

    def run_many(self, sql: str, rows: list) -> None:
        if self.url:
            sql = sql.replace("?", "%s")
        conn = self._connect()
        try:
            if self.url:
                with conn.cursor() as cur:
                    cur.executemany(sql, rows)
            else:
                conn.executemany(sql, rows)
            conn.commit()
        finally:
            conn.close()

    def init(self, sqlite_script: str, pg_script: str) -> None:
        conn = self._connect()
        try:
            if self.url:
                conn.execute(pg_script)
            else:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(sqlite_script)
            conn.commit()
        finally:
            conn.close()


def main() -> None:
    """python -m orchestrator.db init  -> creates every table (run once before the services start)."""
    import argparse

    from . import config
    from .hitl.checkpoint import make_checkpointer
    from .hitl.queue import ReviewQueue
    from .obs.store import SpanStore

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init"])
    ap.parse_args()
    q, s = ReviewQueue(), SpanStore()
    make_checkpointer()
    print(f"database ready: queue={q.db.kind} traces={s.db.kind} checkpoints="
          f"{'postgres' if is_postgres(config.DATABASE_URL) else 'sqlite'}")


if __name__ == "__main__":
    main()