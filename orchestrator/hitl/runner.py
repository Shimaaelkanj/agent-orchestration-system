
import logging
import uuid
from functools import lru_cache

from langgraph.types import Command

from .. import config
from ..graph import build_graph
from .queue import ReviewQueue

log = logging.getLogger("orchestrator.hitl.runner")


@lru_cache(maxsize=1)
def _graph():
    from .checkpoint import make_checkpointer
    return build_graph(checkpointer=make_checkpointer())


def _cfg(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def _outcome(graph, task_id: str) -> dict:
    """Is the run finished, or paused waiting for a human?"""
    snap = graph.get_state(_cfg(task_id))
    values = snap.values or {}
    if snap.next:      # a node is still pending: the run is paused at an interrupt
        review_id = None
        for t in snap.tasks:
            for it in getattr(t, "interrupts", ()):
                review_id = (it.value or {}).get("review_id")
        return {"state": "waiting", "task_id": task_id, "review_id": review_id, "values": values}
    return {"state": values.get("status", "unknown"), "task_id": task_id, "review_id": None, "values": values}


def start(task: str, user_id: str | None = None, review: bool = False, task_id: str | None = None,
          graph=None, approve_final: bool = False) -> dict:
    graph = graph or _graph()
    task_id = task_id or uuid.uuid4().hex[:12]
    graph.invoke({"task": task, "user_id": user_id or config.DEFAULT_USER_ID, "task_id": task_id,
                  "review_requested": review, "approve_final": approve_final}, _cfg(task_id))
    return _outcome(graph, task_id)


def resume(review_id: str, queue: ReviewQueue | None = None, graph=None) -> dict:
    """Continue a paused run using the decision a human recorded for this review."""
    queue = queue or ReviewQueue()
    graph = graph or _graph()
    item = queue.get(review_id)
    if not item:
        raise ValueError(f"unknown review {review_id}")
    if item["status"] != "decided":
        raise ValueError(f"review is '{item['status']}', not 'decided'")
    current = _outcome(graph, item["task_id"])
    if current["state"] != "waiting" or current["review_id"] != review_id:
        raise ValueError("the run is not waiting on this review")
    graph.invoke(Command(resume=item["decision"]), _cfg(item["task_id"]))
    queue.mark_resolved(review_id)
    return _outcome(graph, item["task_id"])


def print_outcome(out: dict) -> None:
    v = out["values"]
    print("\nSUBTASKS:")
    for sid, r in (v.get("results") or {}).items():
        by = " (by human)" if r.get("by_human") else ""
        print(f"  subtask {sid}: {r['status']} (attempts={r['attempts']}, conf={r['confidence']:.2f}){by}")
    print("\nSTATUS:", out["state"])
    if out["state"] == "waiting":
        item = ReviewQueue().get(out["review_id"]) or {}
        print(f"REVIEW ID: {out['review_id']}  level={item.get('level')}  trigger={item.get('trigger')}")
        print("PAUSED. A human must decide.")
        print("  Review UI:  python -m streamlit run orchestrator/hitl/ui.py")
        print("  Terminal:   python -m orchestrator.hitl.cli show", out["review_id"])
    elif out["state"] in ("needs_human", "rejected"):
        print("REASON:", v.get("escalation_reason"))
    else:
        print("MEMORY:", v.get("memory_report"))
        print("\nFINAL ANSWER:\n", v.get("final_answer"))


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resume").add_argument("review_id")
    a = ap.parse_args()
    print_outcome(resume(a.review_id))