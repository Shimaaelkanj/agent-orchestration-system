
import argparse
import os
import sys
import time

import httpx

from orchestrator import config
from orchestrator.obs.tracing import get_store

API = os.getenv("API_URL", "http://localhost:8080")
HEADERS = {"X-API-Key": config.ADMIN_API_KEY} if config.ADMIN_API_KEY else {}
TASK_A = ("Research the pros and cons of vector databases for a small startup team, "
          "analyze the benchmark data, and write a recommendation.")
TASK_B = TASK_A.replace("startup", "mobile app")
store = get_store()
results: list[tuple[bool, str]] = []


def api(method, path, **kw):
    r = httpx.request(method, API + path, headers=HEADERS, timeout=30, **kw)
    r.raise_for_status()
    return r.json()


def wait_for_api():
    last = ""
    for _ in range(90):
        try:
            api("GET", "/health")
            return
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2)
    raise SystemExit(f"The API at {API} did not become ready. Last error: {last}\nCheck: docker compose logs api")


def wait(task_id, want, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = api("GET", f"/tasks/{task_id}")
        if s["outcome"] in want and (s["outcome"] != "waiting" or s["pending_reviews"]):
            return s
        time.sleep(2)
    raise SystemExit(f"Timed out waiting for task {task_id}. Check: docker compose logs worker specialist-worker")


def run_scenario(label, task, user, manual):
    print(f"\n=== Run {label} ===\n{task}")
    tid = api("POST", "/tasks", json={"task": task, "user_id": user, "approve_final": True})["task_id"]
    s = wait(tid, {"waiting", "done", "rejected", "needs_human"})
    if s["outcome"] == "waiting":
        rv = s["pending_reviews"][0]
        print(f"  paused for a human: {rv['level']} / {rv['trigger']}  (review {rv['review_id']})")
        if manual:
            print("  -> open http://localhost:8501 and approve it there")
        else:
            api("POST", f"/reviews/{rv['review_id']}/decision",
                json={"action": "approve", "note": "approved by the demo script", "by": "demo-script"})
            print("  the demo script approved the final deliverable")
        s = wait(tid, {"done", "rejected", "needs_human"})
    print(f"  outcome: {s['outcome']}")
    return tid, s


def check(ok, text):
    results.append((ok, text))
    print(f"  [{'OK' if ok else 'FAIL'}] {text}")


def decs(tid, kind, key=None):
    return [d for d in store.decisions(tid) if d["kind"] == kind and (key is None or d["key"] == key)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", action="store_true", help="wait for you to approve in the review UI")
    a = ap.parse_args()
    mode = "SCRIPTED agents (deterministic, fictional fixture data)" if config.DEMO_MODE else "LIVE models (results vary)"
    print(f"Demo mode: {mode}")
    wait_for_api()
    user = f"demo-{int(time.time())}"             # a fresh user, so earlier demo runs cannot influence memory

    ta, sa = run_scenario("A (first time)", TASK_A, user, a.manual)
    print("What happened in run A:")
    plan = decs(ta, "plan")[0]["output"]
    check(len(plan["subtasks"]) >= 4, f"the supervisor decomposed the task into {len(plan['subtasks'])} subtasks")
    par = {s["attrs"].get("decision.key"): s for s in store.spans(ta)
           if s["name"] == "decision.specialist" and s["attrs"].get("decision.key") in ("t1", "t2")}
    t1, t2 = par.get("t1"), par.get("t2")
    check(bool(t1 and t2 and t1["start"] < t2["end"] and t2["start"] < t1["end"]),
          "research specialists t1 and t2 ran in parallel")
    rejected = [d for d in decs(ta, "review") if not d["output"].get("approved", True)]
    check(bool(rejected), "the reviewer caught a problem and sent it back"
          + (f": {rejected[0]['output']['feedback']}" if rejected else ""))
    check(sa["outcome"] == "done", "a human approved the final deliverable and the task completed")

    tb, sb = run_scenario("B (similar task, memory available)", TASK_B, user, a.manual)
    print("What happened in run B:")
    used = next(s["attrs"].get("plan.used_memory") for s in store.spans(tb) if s["name"] == "node.planning")
    check(bool(used), "memory from run A was injected into the planning prompt")
    check(not [d for d in decs(tb, "review") if not d["output"].get("approved", True)],
          "no review rejection this time: the plan asked for citations up front")
    check(len(decs(tb, "specialist", "t4")) == 1, "the writing step succeeded on the first attempt")
    check(sb["outcome"] == "done", "a human approved the final deliverable and the task completed")

    print("\nFinal deliverable of run B:\n" + sb["final_answer"])
    print("\nExplore it:  traces http://localhost:8502   reviews http://localhost:8501   API docs "
          + os.getenv("PUBLIC_API_URL", "http://localhost:8081") + "/docs")
    failed = [t for ok, t in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} showcase checks passed")
    sys.exit(1 if failed and config.DEMO_MODE else 0)
    

if __name__ == "__main__":
    main()