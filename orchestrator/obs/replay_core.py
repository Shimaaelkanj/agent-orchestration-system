
import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from ..tools import ApprovalRequired
from .tracing import agent_scope, set_status, span


@dataclass
class Kind:
    """How one kind of agent decision is stored, restored, colored and summarized."""
    name: str
    encode: Callable[[Any], Any]
    decode: Callable[[Any], Any]
    status: Callable[[Any], str]
    summary: Callable[[Any], str]


def norm(x):
    return json.loads(json.dumps(x, default=str))

class ReplayedError(RuntimeError):
    """Base class of errors re-created from a recording."""

def encode_exc(e: Exception) -> dict:
    if isinstance(e, ApprovalRequired):
        return {"__exc__": "ApprovalRequired", "tool": e.tool, "agent": e.agent,
                "tool_args": e.tool_args, "reasoning": e.reasoning}
    return {"__exc__": "Error", "type": type(e).__name__, "message": str(e)[:2000]}


def rebuild_exc(d: dict) -> Exception:
    if d.get("__exc__") == "ApprovalRequired":
        a = ApprovalRequired(d["tool"], d["agent"], d["tool_args"])
        a.reasoning = d.get("reasoning", "")
        return a
    # same class NAME as the original, so messages like "RuntimeError: boom" come out identical
    return type(d.get("type") or "Error", (ReplayedError,), {})(d.get("message", ""))


_ACTIVE = None


def current():
    return _ACTIVE


def set_active(controller) -> None:
    global _ACTIVE
    _ACTIVE = controller


class ReplayController:
    """For every agent decision: reuse the recording if the input is identical, apply your override,
    or run live when something upstream changed."""

    def __init__(self, decisions: list[dict], overrides: list[dict] | None = None) -> None:
        self.recorded = {(d["kind"], d["key"], d["index"]): d for d in decisions}
        self.overrides = {(o["kind"], o.get("key", ""), int(o.get("index", 0))): o for o in (overrides or [])}
        self.counters: dict = defaultdict(int)
        self.steps: list[dict] = []
        self._lock = threading.Lock()

    def _next(self, kind: str, key: str) -> int:
        with self._lock:
            i = self.counters[(kind, key)]
            self.counters[(kind, key)] += 1
            return i

    def human_override(self, tag: str):
        o = self.overrides.get(("human", tag, 0))
        return o.get("output") if o else None

    def handle(self, kind: Kind, key: str, inputs: dict, live: Callable, agent: str):
        idx = self._next(kind.name, key)
        rec = self.recorded.get((kind.name, key, idx))
        ov = self.overrides.get((kind.name, key, idx)) or {}
        patch = ov.get("input_patch") or {}
        eff = norm({**inputs, **patch})
        attrs = {"decision.kind": kind.name, "decision.key": key, "decision.index": idx, "replay.active": True}
        with agent_scope(agent), span(f"decision.{kind.name}", **attrs) as sp:
            exc, result, live_run = None, None, False
            if ov.get("output") is not None:
                mode, out = "override-output", norm(ov["output"])
            elif rec is not None and rec["input"] == eff and not ov.get("force_live"):
                mode, out = "replayed", rec["output"]
            else:
                mode, live_run = ("live-edited" if patch else "live"), True
                try:
                    result = live(eff)
                    out = norm(kind.encode(result))
                except Exception as e:
                    exc, out = e, encode_exc(e)
            if not live_run:
                if isinstance(out, dict) and "__exc__" in out:
                    exc = rebuild_exc(out)
                else:
                    try:
                        result = kind.decode(out)
                    except Exception as e:
                        raise ValueError(f"Invalid {kind.name} output for replay: {e}") from e
            summary = kind.summary(result) if exc is None else f"raised {out.get('type') or out.get('__exc__')}"
            self.steps.append({
                "kind": kind.name, "key": key, "index": idx, "mode": mode,
                "changed": rec is None or norm(rec["output"]) != out,
                "summary": summary, "orig_summary": rec.get("summary") if rec else None,
                "input": eff, "output": out,
                "orig_input": rec["input"] if rec else None, "orig_output": rec["output"] if rec else None})
            sp.set_attribute("decision.input", json.dumps(eff, default=str))
            sp.set_attribute("decision.output", json.dumps(out, default=str))
            sp.set_attribute("decision.summary", summary[:200])
            sp.set_attribute("replay.mode", mode)
            if exc is not None:
                raise exc
            set_status(sp, kind.status(result))
            return result