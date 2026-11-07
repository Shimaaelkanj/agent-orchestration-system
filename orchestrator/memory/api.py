import html
import json
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from .. import config
from . import get_ltm
from .long_term import check_user
from .management import consolidate, expire

app = FastAPI(title="Agent memory API")


def require_key(x_api_key: str | None = Header(default=None), key: str | None = None):
    """Placeholder protection. Real authentication arrives with the review UI phase."""
    if config.ADMIN_API_KEY and (x_api_key or key) != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _user(user_id: str) -> str:
    try:
        return check_user(user_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _ts(ts) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(ts))) if ts else ""


def _row(m) -> dict:
    return {"id": m.id, "kind": m.kind, "text": m.text, "importance": round(m.importance, 3),
            "access_count": int(m.meta.get("access_count", 0)),
            "created_at": _ts(m.meta.get("created_at")), "last_accessed": _ts(m.meta.get("last_accessed"))}


@app.get("/memory/{user_id}", dependencies=[Depends(require_key)])
def list_memories(user_id: str):
    """Everything the system remembers about this user (also serves as a data export)."""
    return [_row(m) for m in get_ltm().fetch(_user(user_id))]


@app.get("/memory/{user_id}/search", dependencies=[Depends(require_key)])
def search_memories(user_id: str, q: str):
    return [{**_row(m), "similarity": round(m.similarity or 0, 3)}
            for m in get_ltm().search(_user(user_id), q, k=5, touch=False)]


@app.delete("/memory/{user_id}/{memory_id}", dependencies=[Depends(require_key)])
def delete_one(user_id: str, memory_id: str):
    if not get_ltm().delete(_user(user_id), memory_id):
        raise HTTPException(status_code=404, detail="Memory not found for this user")
    return {"deleted": 1}


@app.delete("/memory/{user_id}", dependencies=[Depends(require_key)])
def delete_all(user_id: str):
    """Erase all data held about a user (data-deletion request)."""
    return {"deleted": get_ltm().delete_user(_user(user_id))}


@app.post("/maintenance/expire", dependencies=[Depends(require_key)])
def run_expire():
    return {"expired": len(expire(get_ltm()))}


@app.post("/maintenance/consolidate/{user_id}", dependencies=[Depends(require_key)])
def run_consolidate(user_id: str):
    return {"merged": consolidate(get_ltm(), _user(user_id))}


@app.get("/dashboard/{user_id}", response_class=HTMLResponse, dependencies=[Depends(require_key)])
def dashboard(user_id: str, key: str | None = None):
    user = _user(user_id)
    rows = [_row(m) for m in get_ltm().fetch(user)]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    body = "".join(
        f"<tr><td>{html.escape(r['kind'])}</td><td>{html.escape(r['text']).replace(chr(10), '<br>')}</td>"
        f"<td>{r['importance']}</td><td>{r['access_count']}</td><td>{r['last_accessed']}</td>"
        f"<td><button onclick=\"del('{r['id']}')\">delete</button></td></tr>" for r in rows)
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) or "nothing stored"
    qs = f"?key={key}" if key else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Memory: {html.escape(user)}</title>
<style>body{{font-family:system-ui;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:14px}}th{{background:#f2f2f2}}</style></head>
<body><h2>What the system remembers about <code>{html.escape(user)}</code></h2>
<p>{len(rows)} memories ({html.escape(summary)}) · embedder: <code>{html.escape(get_ltm().embedder.name)}</code></p>
<button onclick="delAll()">Delete ALL my data</button>
<table><tr><th>Kind</th><th>Memory</th><th>Importance</th><th>Used</th><th>Last used</th><th></th></tr>{body}</table>
<script>
const U={json.dumps(user)}, Q={json.dumps(qs)};
async function del(id){{await fetch('/memory/'+U+'/'+id+Q,{{method:'DELETE'}});location.reload();}}
async function delAll(){{if(confirm('Delete every memory for '+U+'?')){{await fetch('/memory/'+U+Q,{{method:'DELETE'}});location.reload();}}}}
</script></body></html>"""