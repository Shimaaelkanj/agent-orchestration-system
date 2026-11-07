import sqlite3
from pathlib import Path

from .. import config
from ..db import is_postgres


def make_checkpointer(path: Path | str | None = None):
    """Durable graph checkpoints: a paused run survives restarts and can be resumed by another process.
    PostgreSQL when DATABASE_URL is set (and no explicit path is given), otherwise a SQLite file."""
    if path is None and is_postgres(config.DATABASE_URL):
        from langgraph.checkpoint.postgres import PostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool
        pool = ConnectionPool(conninfo=config.DATABASE_URL, max_size=10, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        saver = PostgresSaver(pool)
        saver.setup()
        return saver
    from langgraph.checkpoint.sqlite import SqliteSaver
    path = Path(path or config.CHECKPOINT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    return SqliteSaver(conn)