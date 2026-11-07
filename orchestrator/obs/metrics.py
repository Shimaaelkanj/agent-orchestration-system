
import time
from collections import defaultdict

END_NODES = {"node.delivery": "done", "node.abort": "rejected", "node.escalate": "needs_human"}


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def task_summary(spans: list[dict], reviews: list[dict] | None = None, now: float | None = None) -> dict:
    now = now or time.time()
    reviews = [r for r in (reviews or []) if r.get("level") != "notify"]
    spans = sorted(spans, key=lambda s: s["start"])
    names = {s["name"] for s in spans}
    intake = next((s["attrs"] for s in spans if s["name"] == "node.intake"), {})

    outcome = next((v for k, v in END_NODES.items() if k in names), None)
    if outcome is None:
        outcome = "waiting" if any(s["status"] == "escalated" for s in spans) else "unknown"
    task_type = next((s["attrs"]["plan.task_type"] for s in reversed(spans)
                      if s["name"] == "node.planning" and "plan.task_type" in s["attrs"]), "unknown")

    by_agent: dict = {}
    tin = tout = unpriced = 0
    cost = 0.0
    for s in (x for x in spans if x["name"] == "llm.call"):
        a = s["attrs"]
        agent = a.get("agent") or a.get("llm.role") or "unknown"
        model = a.get("llm.model", "?")
        d = by_agent.setdefault(f"{agent}|{model}", {"agent": agent, "model": model, "calls": 0,
                                                     "input_tokens": 0, "output_tokens": 0,
                                                     "cost_usd": 0.0, "seconds": 0.0})
        i, o, c = int(_f(a.get("llm.input_tokens"))), int(_f(a.get("llm.output_tokens"))), _f(a.get("llm.cost_usd"))
        d["calls"] += 1
        d["input_tokens"] += i
        d["output_tokens"] += o
        d["cost_usd"] += c
        d["seconds"] += s["duration_s"]
        tin, tout, cost = tin + i, tout + o, cost + c
        if not a.get("llm.priced", True):
            unpriced += 1

    tools: dict = {}
    for s in (x for x in spans if x["name"] == "tool.call"):
        a = s["attrs"]
        t = tools.setdefault(a.get("tool.name", "?"), {"calls": 0, "failures": 0, "latency_ms": 0.0})
        t["calls"] += 1
        t["failures"] += 0 if a.get("tool.success", True) else 1
        t["latency_ms"] += _f(a.get("tool.latency_ms"))

    human = 0.0
    for r in reviews:
        end = r.get("decided_at") or (now if r.get("status") == "pending" else None)
        if end and r.get("created_at"):
            human += max(0.0, end - r["created_at"])

    nodes = [s for s in spans if s["name"].startswith("node.")]
    return {
        "task_id": spans[0]["task_id"] if spans else "", "task_text": intake.get("task.text", ""),
        "user_id": intake.get("user.id", ""), "outcome": outcome, "task_type": task_type,
        "started": spans[0]["start"] if spans else now,
        "wall_s": (max(s["end"] for s in spans) - spans[0]["start"]) if spans else 0.0,   # includes waiting for humans
        "active_s": sum(s["duration_s"] for s in nodes),                                   # time the system really worked
        "human_wait_s": human,
        "llm_calls": sum(d["calls"] for d in by_agent.values()),
        "input_tokens": tin, "output_tokens": tout, "total_tokens": tin + tout,
        "tool_calls": sum(t["calls"] for t in tools.values()),
        "tool_failures": sum(t["failures"] for t in tools.values()),
        "cost_usd": cost, "unpriced_calls": unpriced,
        "escalated": bool(reviews) or any(s["status"] == "escalated" for s in spans),
        "by_agent": list(by_agent.values()), "tools": tools,
    }


def aggregate(summaries: list[dict]) -> dict:
    types: dict = defaultdict(lambda: {"tasks": 0, "cost_usd": 0.0, "tokens": 0, "active_s": 0.0, "escalated": 0})
    agents: dict = {}
    tools: dict = {}
    days: dict = defaultdict(lambda: {"tasks": 0, "escalated": 0})
    for s in summaries:
        t = types[s["task_type"]]
        t["tasks"] += 1
        t["cost_usd"] += s["cost_usd"]
        t["tokens"] += s["total_tokens"]
        t["active_s"] += s["active_s"]
        t["escalated"] += 1 if s["escalated"] else 0
        for a in s["by_agent"]:
            d = agents.setdefault((a["agent"], a["model"]), {"agent": a["agent"], "model": a["model"], "calls": 0,
                                                             "tokens": 0, "cost_usd": 0.0, "seconds": 0.0})
            d["calls"] += a["calls"]
            d["tokens"] += a["input_tokens"] + a["output_tokens"]
            d["cost_usd"] += a["cost_usd"]
            d["seconds"] += a["seconds"]
        for name, tl in s["tools"].items():
            d = tools.setdefault(name, {"tool": name, "calls": 0, "failures": 0, "latency_ms": 0.0})
            d["calls"] += tl["calls"]
            d["failures"] += tl["failures"]
            d["latency_ms"] += tl["latency_ms"]
        day = days[time.strftime("%Y-%m-%d", time.localtime(s["started"]))]
        day["tasks"] += 1
        day["escalated"] += 1 if s["escalated"] else 0

    n = len(summaries)
    return {
        "by_type": [{"task_type": k, "tasks": v["tasks"], "cost_usd": round(v["cost_usd"], 6),
                     "avg_cost_usd": round(v["cost_usd"] / v["tasks"], 6), "avg_tokens": round(v["tokens"] / v["tasks"]),
                     "avg_active_s": round(v["active_s"] / v["tasks"], 2),
                     "escalation_rate": round(v["escalated"] / v["tasks"], 3)} for k, v in sorted(types.items())],
        "by_agent": sorted(({**v, "cost_usd": round(v["cost_usd"], 6), "seconds": round(v["seconds"], 2)}
                            for v in agents.values()), key=lambda r: (-r["cost_usd"], -r["tokens"])),
        "tools": sorted(({**v, "failure_rate": round(v["failures"] / v["calls"], 3) if v["calls"] else 0.0,
                          "avg_latency_ms": round(v["latency_ms"] / v["calls"], 1) if v["calls"] else 0.0}
                         for v in tools.values()), key=lambda r: -r["calls"]),
        "trend": [{"day": d, "tasks": v["tasks"], "escalated": v["escalated"],
                   "escalation_rate": round(v["escalated"] / v["tasks"], 3)} for d, v in sorted(days.items())],
        "totals": {"tasks": n, "cost_usd": round(sum(s["cost_usd"] for s in summaries), 6),
                   "tokens": sum(s["total_tokens"] for s in summaries),
                   "tool_calls": sum(s["tool_calls"] for s in summaries),
                   "escalation_rate": round(sum(1 for s in summaries if s["escalated"]) / n, 3) if n else 0.0,
                   "unpriced_calls": sum(s["unpriced_calls"] for s in summaries)},
    }