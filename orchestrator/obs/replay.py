
import argparse
import difflib
import json
import logging
import subprocess
import sys
import time
import uuid

from .. import config
from ..hitl.queue import ReviewQueue
from .replay_core import ReplayController, set_active
from .tracing import get_store

log = logging.getLogger("orchestrator.obs.replay")


def _human_decision(ctrl: ReplayController, queue: ReviewQueue, orig_task: str, tag: str, default: str):
    """Who answers a human question during a replay: your override, else the original human, else a default."""
    ov = ctrl.human_override(tag)
    if ov:
        return ov, "override"
    item = queue.get(f"{orig_task}-{tag}")
    if item and item.get("decision"):
        return item["decision"], "original"
    return {"action": default, "note": "replay: no recorded human decision"}, "default"


def run_replay(orig_task: str, overrides: list[dict] | None = None, store=None, queue=None,
               graph=None, human_default: str = "reject", replay_id: str | None = None) -> dict:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from ..graph import build_graph

    store = store or get_store()
    queue = queue or ReviewQueue()
    start = store.task_start(orig_task)
    if not start:
        raise ValueError(f"No recorded trace for task {orig_task}")
    rid = replay_id or f"replay-{orig_task}-{uuid.uuid4().hex[:6]}"
    ctrl = ReplayController(store.decisions(orig_task), overrides)
    graph = graph or build_graph(MemorySaver())
    cfg = {"configurable": {"thread_id": rid}}
    human_steps: list[dict] = []
    set_active(ctrl)
    try:
        graph.invoke({"task": start["task"], "user_id": start["user_id"], "task_id": rid,
                      "review_requested": start["review_requested"], "replay_of": orig_task}, cfg)
        for _ in range(config.MAX_HUMAN_ROUNDS + 4):
            snap = graph.get_state(cfg)
            if not snap.next:
                break
            review_id = next((it.value.get("review_id") for t in snap.tasks for it in getattr(t, "interrupts", ())
                              if isinstance(it.value, dict)), "")
            tag = review_id[len(rid) + 1:] if review_id.startswith(rid + "-") else review_id
            decision, source = _human_decision(ctrl, queue, orig_task, tag, human_default)
            human_steps.append({"tag": tag, "source": source, "action": decision.get("action"),
                                "note": decision.get("note", "")})
            graph.invoke(Command(resume=decision), cfg)
        final = graph.get_state(cfg).values or {}
    finally:
        set_active(None)

    original = store.task_result(orig_task)
    new_answer = final.get("final_answer") or ""
    steps = ctrl.steps
    return {
        "replay_id": rid, "original_task": orig_task, "created": time.time(), "status": "finished",
        "overrides": overrides or [], "human": human_steps, "original": original,
        "replayed": {"status": final.get("status"), "final_answer": new_answer,
                     "reason": final.get("escalation_reason", "")},
        "steps": steps,
        "diverged_at": next((i for i, s in enumerate(steps) if s["mode"] != "replayed"), None),
        "n_live": sum(1 for s in steps if s["mode"].startswith("live")),
        "answer_changed": new_answer != (original.get("final_answer") or ""),
        "diff": list(difflib.unified_diff((original.get("final_answer") or "").splitlines(), new_answer.splitlines(),
                                          "original", "replay", lineterm="", n=2)),
    }


# ---------------- background runs (used by the UI) ----------------
def _path(rid: str):
    return config.REPLAY_DIR / f"{rid}.json"


def launch(orig_task: str, overrides: list[dict], human_default: str = "reject") -> str:
    config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    rid = f"replay-{orig_task}-{uuid.uuid4().hex[:6]}"
    ov_path = config.REPLAY_DIR / f"{rid}.overrides.json"
    ov_path.write_text(json.dumps(overrides), encoding="utf-8")
    _path(rid).write_text(json.dumps({"replay_id": rid, "original_task": orig_task, "status": "running",
                                      "created": time.time(), "overrides": overrides}), encoding="utf-8")
    with open(config.REPLAY_DIR / f"{rid}.log", "ab") as logf:
        subprocess.Popen([sys.executable, "-m", "orchestrator.obs.replay", "run", orig_task,
                          "--overrides", str(ov_path), "--id", rid, "--human-default", human_default],
                         cwd=str(config.BASE_DIR), stdout=logf, stderr=subprocess.STDOUT)
    return rid


def load_reports(orig_task: str | None = None) -> list[dict]:
    out = []
    if config.REPLAY_DIR.exists():
        for p in config.REPLAY_DIR.glob("replay-*.json"):
            if p.name.endswith(".overrides.json"):
                continue
            try:
                rep = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if orig_task is None or rep.get("original_task") == orig_task:
                out.append(rep)
    return sorted(out, key=lambda r: r.get("created", 0), reverse=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="replay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("task_id")
    p.add_argument("--overrides", default="", help="JSON file with a list of overrides")
    p.add_argument("--id", default="")
    p.add_argument("--human-default", default="reject", choices=["approve", "reject"])
    a = ap.parse_args()

    overrides = json.load(open(a.overrides, encoding="utf-8")) if a.overrides else []
    rid = a.id or f"replay-{a.task_id}-{uuid.uuid4().hex[:6]}"
    config.REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    try:
        rep = run_replay(a.task_id, overrides, replay_id=rid, human_default=a.human_default)
    except Exception as e:
        _path(rid).write_text(json.dumps({"replay_id": rid, "original_task": a.task_id, "status": "failed",
                                          "created": time.time(), "error": f"{type(e).__name__}: {e}"}),
                              encoding="utf-8")
        raise
    _path(rid).write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"\nREPLAY {rid}: original={rep['original']['status']} replay={rep['replayed']['status']} "
          f"live steps={rep['n_live']} diverged at step={rep['diverged_at']}")
    for i, s in enumerate(rep["steps"]):
        print(f"  {i:2} {s['kind']}:{s['key']}#{s['index']:<2} {s['mode']:<16} changed={s['changed']}  {s['summary'][:70]}")
    if rep["diff"]:
        print("\n".join(rep["diff"]))


if __name__ == "__main__":
    main()