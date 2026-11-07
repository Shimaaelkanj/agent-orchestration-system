
from collections import defaultdict

COLORS = {"success": "#c8e6c9", "warning": "#ffe0b2", "failure": "#ffcdd2", "escalated": "#e1bee7"}
ICON = {"success": "✅", "warning": "⚠️", "failure": "❌", "escalated": "🧑"}
LEAF = {"llm.call", "tool.call"}


def build_tree(spans: list[dict]) -> list[tuple[int, dict]]:
    """Depth-first order. Node spans (children of the task's unrecorded root) are the roots."""
    ids = {s["span_id"] for s in spans}
    kids: dict = defaultdict(list)
    roots = []
    for s in sorted(spans, key=lambda x: x["start"]):
        (kids[s["parent_id"]] if s["parent_id"] in ids else roots).append(s)
    out: list = []

    def walk(s: dict, depth: int) -> None:
        out.append((depth, s))
        for c in kids[s["span_id"]]:
            walk(c, depth + 1)

    for r in roots:
        walk(r, 0)
    return out


def describe(s: dict) -> str:
    a, n = s["attrs"], s["name"]
    if n == "llm.call":
        return f"llm {a.get('llm.role', '')} · {a.get('llm.model', '')} · {a.get('llm.total_tokens', 0)} tok"
    if n == "tool.call":
        return f"tool {a.get('tool.name', '?')}" + ("" if a.get("tool.success", True) else " (failed)")
    if n.startswith("decision."):
        return f"{n[9:]} {a.get('decision.key', '')}#{a.get('decision.index', 0)}".replace("  ", " ")
    return n


def label(depth: int, s: dict) -> str:
    prefix = ("│  " * (depth - 1) + "└─ ") if depth else ""
    return f"{prefix}{ICON.get(s['status'], '·')} {describe(s)}"


def rollup(order: list[tuple[int, dict]]) -> dict:
    """Cost, tokens and tool calls of every span INCLUDING its descendants."""
    totals = {}
    for _, s in order:
        a = s["attrs"]
        own_llm = s["name"] == "llm.call"
        totals[s["span_id"]] = {"cost": float(a.get("llm.cost_usd", 0) or 0) if own_llm else 0.0,
                                "tokens": int(a.get("llm.total_tokens", 0) or 0) if own_llm else 0,
                                "tools": 1 if s["name"] == "tool.call" else 0}
    for _, s in reversed(order):                  # children come after parents in depth-first order
        p = s["parent_id"]
        if p in totals:
            for k in ("cost", "tokens", "tools"):
                totals[p][k] += totals[s["span_id"]][k]
    return totals


def _dot_label(s: dict, roll: dict) -> str:
    lines = [describe(s)]
    if s["attrs"].get("agent"):
        lines.append(str(s["attrs"]["agent"]))
    lines.append(f"{s['duration_s']:.2f}s" + (f" · ${roll.get('cost', 0):.4f}" if roll.get("cost") else ""))
    if s["status"] == "failure" and s["attrs"].get("error.message"):
        lines.append(str(s["attrs"]["error.message"])[:50])
    return "\\n".join(x.replace('"', "'") for x in lines)


def to_dot(order: list[tuple[int, dict]], rolled: dict, selected_id: str | None = None,
           show_leaf: bool = True) -> str:
    shown = {s["span_id"] for _, s in order if show_leaf or s["name"] not in LEAF}
    lines = ["digraph trace {", "rankdir=TB;",
             'node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=10, margin="0.12,0.06"];',
             'edge [color="#888888"];', 'task [label="task", shape=ellipse, fillcolor="#eeeeee"];']
    for _, s in order:
        sid = s["span_id"]
        if sid not in shown:
            continue
        sel = sid == selected_id
        lines.append(f'n{sid} [label="{_dot_label(s, rolled.get(sid, {}))}", fillcolor="{COLORS.get(s["status"], "#ffffff")}", '
                     f'color="{"#000000" if sel else "#999999"}", penwidth={3 if sel else 1}];')
        src = f"n{s['parent_id']}" if s["parent_id"] in shown else "task"
        lines.append(f"{src} -> n{sid};")
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    """Text view of a trace:  python -m orchestrator.obs.tree [task_id]"""
    import argparse

    from .tracing import get_store
    ap = argparse.ArgumentParser()
    ap.add_argument("task_id", nargs="?")
    a = ap.parse_args()
    store = get_store()
    tasks = store.tasks(1)
    tid = a.task_id or (tasks[0]["task_id"] if tasks else None)
    if not tid:
        raise SystemExit("No traces yet. Run a task first.")
    order = build_tree(store.spans(tid))
    roll = rollup(order)
    mark = {"success": "OK   ", "warning": "WARN ", "failure": "FAIL ", "escalated": "HUMAN"}
    print(f"trace for task {tid}")
    for depth, s in order:
        r = roll[s["span_id"]]
        print(f"{'  ' * depth}[{mark.get(s['status'], '?    ')}] {describe(s)}  {s['duration_s']:.2f}s  ${r['cost']:.4f}")


if __name__ == "__main__":
    main()