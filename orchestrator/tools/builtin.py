
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
import json
import time
from pydantic import BaseModel, Field

from ..config import WORKSPACE
from .. import config
from .registry import ToolRegistry, ToolSpec
import logging



WORKSPACE.mkdir(parents=True, exist_ok=True)


def _safe_path(rel: str) -> Path:
    root = WORKSPACE.resolve()
    p = (root / rel).resolve()
    if not p.is_relative_to(root):
        raise ValueError("Path escapes workspace")
    return p

log = logging.getLogger("orchestrator.tools")
# ---------- schemas ----------
class SearchIn(BaseModel):
    query: str
    max_results: int = Field(default=5)

class SearchOut(BaseModel):
    results: list[dict]

class ReadIn(BaseModel):
    path: str = Field(description="Path relative to the workspace")

class ReadOut(BaseModel):
    content: str

class WriteIn(BaseModel):
    path: str
    content: str

class WriteOut(BaseModel):
    bytes_written: int

class CodeIn(BaseModel):
    code: str = Field(description="Python source to run")
    timeout_s: int = Field(default=10)

class CodeOut(BaseModel):
    stdout: str
    stderr: str
    returncode: int

class SQLIn(BaseModel):
    query: str = Field(description="A single read-only SELECT statement")

class SQLOut(BaseModel):
    columns: list[str]
    rows: list[list]

class ApiIn(BaseModel):
    method: str = Field(default="GET")
    url: str
    json_body: dict | None = None

class ApiOut(BaseModel):
    status_code: int
    body: str


# ---------- implementations ----------
def web_search(inp: SearchIn) -> dict:
    if config.DEMO_MODE:                       # offline fixture "web" for the demo
        from ..demo.corpus import search
        return {"results": search(inp.query, min(inp.max_results, 10))}

    n = min(inp.max_results, 10)
    key = os.getenv("TAVILY_API_KEY")

    # 1) Tavily (preferred)
    if key:
        try:
            r = httpx.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {key}"},
                json={"query": inp.query, "max_results": n},
                timeout=20,
            )
            r.raise_for_status()
            items = r.json().get("results", [])
            return {"results": [{"title": i.get("title"), "url": i.get("url"),
                                 "snippet": i.get("content")} for i in items]}
        except Exception as e:
            log.warning("Tavily failed (%s), falling back to DuckDuckGo", str(e)[:120])

    # 2) Fallback: DuckDuckGo, no key needed
    from ddgs import DDGS
    with DDGS() as d:
        items = d.text(inp.query, max_results=n)
    return {"results": [{"title": i.get("title"), "url": i.get("href"),
                         "snippet": i.get("body")} for i in items]}



def read_file(inp: ReadIn) -> dict:
    return {"content": _safe_path(inp.path).read_text(encoding="utf-8")[:20000]}


def write_file(inp: WriteIn) -> dict:
    p = _safe_path(inp.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(inp.content, encoding="utf-8")
    return {"bytes_written": len(inp.content.encode())}


_ENV_KEYS = {"PATH", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT"}


def run_python(inp: CodeIn) -> dict:
    # Basic isolation only: temp dir, isolated interpreter, minimal env (no API keys), timeout.
    # SYSTEMROOT etc. are required for Python to start on Windows.
    safe_env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEYS}
    with tempfile.TemporaryDirectory(dir=WORKSPACE) as d:
        script = Path(d) / "main.py"
        script.write_text(inp.code, encoding="utf-8")
        try:
            r = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=d, capture_output=True, text=True,
                timeout=min(inp.timeout_s, 30), env=safe_env,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Code execution timed out")
    if r.returncode != 0:
        # Raising makes the registry log success=False and return "ERROR: ..." to the agent,
        # so it can see the traceback and fix its code.
        raise RuntimeError(f"Code failed (exit {r.returncode}): {r.stderr[-1500:]}")
    return {"stdout": r.stdout[:5000], "stderr": r.stderr[:2000], "returncode": 0}


def query_database(inp: SQLIn) -> dict:
    q = inp.query.strip().rstrip(";")
    if not q.lower().startswith("select") or ";" in q:
        raise ValueError("Only a single SELECT statement is allowed")
    db = WORKSPACE / "app.db"  # swapped for PostgreSQL in the memory phase
    if not db.exists():
        raise FileNotFoundError("workspace/app.db does not exist")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = con.execute(q)
        cols = [c[0] for c in cur.description or []]
        return {"columns": cols, "rows": [list(r) for r in cur.fetchmany(50)]}
    finally:
        con.close()


def call_api(inp: ApiIn) -> dict:
    allowed = {h.strip() for h in os.getenv("API_ALLOWED_HOSTS", "").split(",") if h.strip()}
    host = urlparse(inp.url).hostname
    if host not in allowed:
        raise PermissionError(f"Host '{host}' is not in API_ALLOWED_HOSTS")
    r = httpx.request(inp.method.upper(), inp.url, json=inp.json_body, timeout=15)
    return {"status_code": r.status_code, "body": r.text[:5000]}
class DeleteIn(BaseModel):
    path: str = Field(description="File path relative to the workspace")

class DeleteOut(BaseModel):
    deleted: bool

class EmailIn(BaseModel):
    to: str
    subject: str
    body: str

class EmailOut(BaseModel):
    queued: bool
    outbox_file: str


def delete_file(inp: DeleteIn) -> dict:
    p = _safe_path(inp.path)
    if not p.is_file():
        raise FileNotFoundError(inp.path)
    p.unlink()
    return {"deleted": True}


def send_email(inp: EmailIn) -> dict:
    """SIMULATED: writes to workspace/outbox instead of sending anything."""
    outbox = WORKSPACE / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)
    path = outbox / f"{int(time.time() * 1000)}.json"
    path.write_text(json.dumps(inp.model_dump(), indent=2), encoding="utf-8")
    return {"queued": True, "outbox_file": path.name}

# ---------- registration ----------
def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolSpec("web_search", "Search the web for current information.", SearchIn, SearchOut,
                          web_search, {"research"}, rate_limit_per_min=20))
    reg.register(ToolSpec("read_file", "Read a text file from the workspace.", ReadIn, ReadOut,
                          read_file, {"research", "data_analysis", "code_execution"}))
    reg.register(ToolSpec("write_file", "Write a text file into the workspace.", WriteIn, WriteOut,
                          write_file, {"code_execution"}))
    reg.register(ToolSpec("run_python", "Run Python code and return stdout/stderr.", CodeIn, CodeOut,
                          run_python, {"data_analysis", "code_execution"}, rate_limit_per_min=10))
    reg.register(ToolSpec("query_database", "Run a read-only SELECT query.", SQLIn, SQLOut,
                          query_database, {"data_analysis"}))
    # Anything except a GET changes the outside world, so it needs human approval.
    reg.register(ToolSpec("call_api", "Call an allow-listed HTTP API.", ApiIn, ApiOut,
                          call_api, {"research"}, rate_limit_per_min=15,
                          sensitive=lambda inp: inp.method.upper() != "GET"))
    reg.register(ToolSpec("delete_file", "Permanently delete a file from the workspace.", DeleteIn, DeleteOut,
                          delete_file, {"code_execution", "data_analysis"}, sensitive=True))
    reg.register(ToolSpec("send_email", "Send an email to a person (external communication).", EmailIn, EmailOut,
                          send_email, {"writing", "research"}, sensitive=True))
    return reg