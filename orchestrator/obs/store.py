import json
import logging

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .. import config
from ..db import Backend

log = logging.getLogger("orchestrator.obs.store")

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, task_id TEXT, name TEXT,
  start_ns INTEGER, end_ns INTEGER, status TEXT, attrs TEXT, events TEXT);
CREATE INDEX IF NOT EXISTS ix_spans_task ON spans(task_id);
CREATE INDEX IF NOT EXISTS ix_spans_name ON spans(name);
"""
PG_SCHEMA = SQLITE_SCHEMA.replace("start_ns INTEGER, end_ns INTEGER", "start_ns BIGINT, end_ns BIGINT")
UPSERT = ("INSERT INTO spans (span_id, trace_id, parent_id, task_id, name, start_ns, end_ns, status, attrs, events) "
          "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT (span_id) DO UPDATE SET trace_id=excluded.trace_id, "
          "parent_id=excluded.parent_id, task_id=excluded.task_id, name=excluded.name, "
          "start_ns=excluded.start_ns, end_ns=excluded.end_ns, status=excluded.status, "
          "attrs=excluded.attrs, events=excluded.events")
END_NODES = {"node.delivery": "done", "node.abort": "rejected", "node.escalate": "needs_human"}


def _loads(text, default):
    try:
        return json.loads(text) if text else default
    except Exception:
        return default


def _to_span(r) -> dict:
    start = r["start_ns"] / 1e9
    end = (r["end_ns"] or r["start_ns"]) / 1e9
    return {"span_id": r["span_id"], "trace_id": r["trace_id"], "parent_id": r["parent_id"],
            "task_id": r["task_id"], "name": r["name"], "start": start, "end": end,
            "duration_s": max(0.0, end - start), "status": r["status"],
            "attrs": _loads(r["attrs"], {}), "events": _loads(r["events"], [])}


class SpanStore:
    def __init__(self, path=None, url=None) -> None:
        if path is not None:                       # an explicit path always means SQLite (used by tests)
            self.db = Backend(path=path)
        elif url or config.DATABASE_URL:
            self.db = Backend(url=url or config.DATABASE_URL)
        else:
            self.db = Backend(path=config.TRACE_DB)
        self.db.init(SQLITE_SCHEMA, PG_SCHEMA)
        self.path = self.db.path

    def add_many(self, rows: list[tuple]) -> None:
        self.db.run_many(UPSERT, rows)

    # ---------- reading ----------
    def spans(self, task_id: str) -> list[dict]:
        rows, _ = self.db.run("SELECT * FROM spans WHERE task_id=? ORDER BY start_ns", (task_id,))
        return [_to_span(r) for r in rows]

    def all_spans(self, limit: int = 200_000) -> list[dict]:
        rows, _ = self.db.run("SELECT * FROM spans WHERE task_id != '' ORDER BY start_ns LIMIT ?", (limit,))
        return [_to_span(r) for r in rows]

    def tasks(self, limit: int = 200) -> list[dict]:
        rows, _ = self.db.run(
            "SELECT task_id, MIN(start_ns) AS s, MAX(end_ns) AS e, COUNT(*) AS n FROM spans "
            "WHERE task_id != '' GROUP BY task_id ORDER BY s DESC LIMIT ?", (limit,))
        ids = [r["task_id"] for r in rows]
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        intake_rows, _ = self.db.run(
            f"SELECT task_id, attrs FROM spans WHERE name='node.intake' AND task_id IN ({ph})", tuple(ids))
        intake = {r["task_id"]: _loads(r["attrs"], {}) for r in intake_rows}
        end_rows, _ = self.db.run("SELECT task_id, name FROM spans WHERE name IN "
                                  f"('node.delivery','node.abort','node.escalate') AND task_id IN ({ph})", tuple(ids))
        ends = {r["task_id"]: END_NODES[r["name"]] for r in end_rows}
        wait_rows, _ = self.db.run(
            f"SELECT DISTINCT task_id FROM spans WHERE status='escalated' AND task_id IN ({ph})", tuple(ids))
        waiting = {r["task_id"] for r in wait_rows}
        out = []
        for r in rows:
            tid = r["task_id"]
            a = intake.get(tid, {})
            out.append({"task_id": tid, "start": r["s"] / 1e9, "end": (r["e"] or r["s"]) / 1e9, "spans": r["n"],
                        "task_text": a.get("task.text", ""), "user_id": a.get("user.id", ""),
                        "replay_of": a.get("replay.of", ""),
                        "outcome": ends.get(tid, "waiting" if tid in waiting else "unknown")})
        return out

    def decisions(self, task_id: str) -> list[dict]:
        """Every recorded agent decision of a task, in execution order (the basis of replay)."""
        out = []
        for s in self.spans(task_id):
            a = s["attrs"]
            if not s["name"].startswith("decision.") or "decision.output" not in a:
                continue
            try:
                inp, outp = json.loads(a.get("decision.input", "null")), json.loads(a["decision.output"])
            except Exception:
                continue            # truncated or unreadable: replay will run this step live
            out.append({"kind": a.get("decision.kind", ""), "key": a.get("decision.key", ""),
                        "index": int(a.get("decision.index", 0)), "input": inp, "output": outp,
                        "status": s["status"], "span_id": s["span_id"], "summary": a.get("decision.summary", "")})
        return out

    def count_decisions(self, task_id: str, kind: str, key: str) -> int:
        return sum(1 for d in self.decisions(task_id) if d["kind"] == kind and d["key"] == key)

    def task_start(self, task_id: str) -> dict | None:
        for s in self.spans(task_id):
            if s["name"] == "node.intake":
                a = s["attrs"]
                return {"task": a.get("task.text", ""), "user_id": a.get("user.id", ""),
                        "review_requested": bool(a.get("review.requested", False)),
                        "approve_final": bool(a.get("approve.final", False))}
        return None

    def task_result(self, task_id: str) -> dict:
        res = {"status": "unknown", "final_answer": "", "reason": ""}
        spans = self.spans(task_id)
        for s in spans:
            if s["name"] in END_NODES:
                a = s["attrs"]
                res = {"status": END_NODES[s["name"]], "final_answer": a.get("result.final_answer", ""),
                       "reason": a.get("result.reason", "")}
        if res["status"] == "unknown" and any(s["status"] == "escalated" for s in spans):
            res["status"] = "waiting"
        return res

    def delete_task(self, task_id: str) -> int:
        _, n = self.db.run("DELETE FROM spans WHERE task_id=?", (task_id,))
        return n


def _row(s) -> tuple:
    ctx = s.get_span_context()
    attrs = dict(s.attributes or {})
    events = [{"name": e.name, "ts": e.timestamp, "attributes": dict(e.attributes or {})} for e in s.events]
    parent = format(s.parent.span_id, "016x") if s.parent else None
    return (format(ctx.span_id, "016x"), format(ctx.trace_id, "032x"), parent, str(attrs.get("task.id", "")),
            s.name, s.start_time, s.end_time, str(attrs.get("obs.status", "success")),
            json.dumps(attrs, default=str), json.dumps(events, default=str))


class SQLiteExporter(SpanExporter):
    """Exports finished spans to the store (SQLite or PostgreSQL, whichever the store uses)."""

    def __init__(self, store: SpanStore) -> None:
        self.store = store

    def export(self, spans):
        try:
            self.store.add_many([_row(s) for s in spans])
            return SpanExportResult.SUCCESS
        except Exception as e:
            log.warning("span export failed: %s", str(e)[:150])
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True