
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `orchestrator` importable

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from orchestrator.hitl.queue import ReviewQueue  # noqa: E402
from orchestrator.obs import metrics, replay, tree  # noqa: E402
from orchestrator.obs.tracing import get_store  # noqa: E402

st.set_page_config(page_title="Agent observability", layout="wide")
store = get_store()
queue = ReviewQueue()
ICON = tree.ICON
BIG_KEYS = {"llm.prompt", "llm.response", "llm.tool_calls", "decision.input", "decision.output",
            "tool.args", "tool.output", "plan.subtasks"}


def show_messages(prompt_json: str) -> None:
    try:
        msgs = json.loads(prompt_json)
    except Exception:
        st.code(str(prompt_json))
        return
    for m in msgs:
        content = (m.get("data", {}) if isinstance(m, dict) else {}).get("content", "")
        if isinstance(content, list):
            content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        st.markdown(f"**{m.get('type', '?') if isinstance(m, dict) else '?'}**")
        st.code(content or "(empty)", language=None)


def show_json_or_text(value) -> None:
    if value is None or value == "":
        st.caption("(nothing recorded)")
        return
    try:
        st.json(json.loads(value) if isinstance(value, str) else value)
    except Exception:
        st.code(str(value), language=None)


def render_detail(s: dict, roll: dict) -> None:
    a, n = s["attrs"], s["name"]
    st.subheader(f"{ICON.get(s['status'], '')} {n}")
    c = st.columns(4)
    c[0].metric("Status", s["status"])
    c[1].metric("Latency", f"{s['duration_s']:.2f}s")
    c[2].metric("Cost (with children)", f"${roll.get('cost', 0):.5f}")
    c[3].metric("Tokens (with children)", roll.get("tokens", 0))
    if a.get("agent"):
        st.write(f"**Agent:** `{a['agent']}`")
    if s["status"] == "failure":
        st.error(a.get("error.message", "failed"))
    if a.get("human.review_id"):
        st.info(f"Human review `{a['human.review_id']}` ({a.get('human.level')} / {a.get('human.trigger')}): "
                f"{a.get('human.reason', '')}" + (f"  Decision: **{a['human.action']}**" if a.get("human.action") else ""))
    if a.get("replay.mode"):
        st.caption(f"Replay mode: {a['replay.mode']}")

    if n == "llm.call":
        names = ["Prompt", "Response", "Attributes", "Events"]
    elif n.startswith("decision."):
        names = ["Input", "Output", "Attributes", "Events"]
    elif n == "tool.call":
        names = ["Arguments", "Result", "Attributes", "Events"]
    else:
        names = ["Attributes", "Events"]
    tabs = dict(zip(names, st.tabs(names)))

    if "Prompt" in tabs:
        with tabs["Prompt"]:
            show_messages(a.get("llm.prompt", "[]"))
        with tabs["Response"]:
            st.code(a.get("llm.response", "") or "(empty)", language=None)
            if a.get("llm.tool_calls"):
                st.write("**Tool calls requested by the model:**")
                show_json_or_text(a["llm.tool_calls"])
            st.caption(f"{a.get('llm.input_tokens', 0)} in / {a.get('llm.output_tokens', 0)} out tokens"
                       + (" (estimated)" if a.get("llm.tokens_estimated") else "")
                       + f" · cost ${float(a.get('llm.cost_usd', 0) or 0):.6f}"
                       + ("" if a.get("llm.priced", True) else " · model not in the price table"))
    if "Input" in tabs:
        with tabs["Input"]:
            show_json_or_text(a.get("decision.input"))
        with tabs["Output"]:
            show_json_or_text(a.get("decision.output"))
    if "Arguments" in tabs:
        with tabs["Arguments"]:
            show_json_or_text(a.get("tool.args"))
        with tabs["Result"]:
            show_json_or_text(a.get("tool.output") or a.get("error.message"))
    with tabs["Attributes"]:
        st.json({k: v for k, v in a.items() if k not in BIG_KEYS})
        if a.get("plan.subtasks"):
            st.write("**Plan subtasks:**")
            show_json_or_text(a["plan.subtasks"])
    with tabs["Events"]:
        if s["events"]:
            st.dataframe(pd.DataFrame([{"event": e["name"], **{k: str(v) for k, v in (e.get("attributes") or {}).items()}}
                                       for e in s["events"]]))
        else:
            st.caption("No events on this span.")


def render_report(rep: dict) -> None:
    if rep.get("status") == "running":
        st.info("This replay is still running. Press Refresh in the sidebar.")
        return
    if rep.get("status") == "failed":
        st.error(rep.get("error", "failed"))
        return
    o, r = rep["original"], rep["replayed"]
    c = st.columns(4)
    c[0].metric("Original outcome", o.get("status"))
    c[1].metric("Replay outcome", r.get("status"))
    c[2].metric("Live steps", rep["n_live"])
    c[3].metric("Diverged at step", "never" if rep["diverged_at"] is None else rep["diverged_at"])
    st.write("**Step by step** (`replayed` = recording reused for free, `live` = the model ran again, "
             "`override-output` / `live-edited` = your change)")
    st.dataframe(pd.DataFrame([{"step": i, "decision": f"{s['kind']}:{s['key']}#{s['index']}", "mode": s["mode"],
                                "changed": s["changed"], "original": s.get("orig_summary") or "",
                                "replay": s.get("summary") or ""} for i, s in enumerate(rep["steps"])]))
    a, b = st.columns(2)
    a.caption("Original final answer")
    a.code(o.get("final_answer") or o.get("reason") or "(none)", language=None)
    b.caption("Replay final answer")
    b.code(r.get("final_answer") or r.get("reason") or "(none)", language=None)
    if rep["diff"]:
        st.code("\n".join(rep["diff"]), language="diff")
    elif not rep["answer_changed"]:
        st.success("The final answer is identical.")
    if rep.get("human"):
        st.write("**Human decisions used during the replay**")
        st.dataframe(pd.DataFrame(rep["human"]))
    st.caption(f"The replay's own trace is available in the sidebar as task `{rep['replay_id']}`.")


# ---------------- sidebar: choose a task ----------------
st.sidebar.title("Observability")
if st.sidebar.button("Refresh"):
    st.rerun()
tasks = store.tasks()
if not tasks:
    st.title("Agent observability")
    st.info('No traces yet. Run a task first:  python -m orchestrator.main "your task"')
    st.stop()
options = {f"{t['task_id']} · {t['outcome']} · {(t['task_text'] or '')[:40]}": t for t in tasks}
task = options[st.sidebar.selectbox("Task", list(options), key="task_pick")]
task_id = task["task_id"]
st.sidebar.caption(f"started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task['start']))}")
if task["replay_of"]:
    st.sidebar.warning(f"This is a replay of {task['replay_of']}")

all_reviews = queue.list_reviews(limit=10000)
reviews_by_task: dict = {}
for rv in all_reviews:
    reviews_by_task.setdefault(rv["task_id"], []).append(rv)
spans_by_task: dict = {}
for sp in store.all_spans():
    spans_by_task.setdefault(sp["task_id"], []).append(sp)
summaries = [metrics.task_summary(sp, reviews_by_task.get(tid, [])) for tid, sp in spans_by_task.items()]

tab_trace, tab_cost, tab_replay = st.tabs(["Trace explorer", "Cost and performance", "Replay"])

# ---------------- tab 1: trace explorer ----------------
with tab_trace:
    spans = spans_by_task.get(task_id, [])
    summ = metrics.task_summary(spans, reviews_by_task.get(task_id, []))
    st.write(f"**Task:** {summ['task_text']}")
    m = st.columns(6)
    m[0].metric("Outcome", summ["outcome"])
    m[1].metric("LLM tokens", summ["total_tokens"])
    m[2].metric("Tool calls", summ["tool_calls"])
    m[3].metric("Cost", f"${summ['cost_usd']:.5f}")
    m[4].metric("Active time", f"{summ['active_s']:.1f}s")
    m[5].metric("Human wait", f"{summ['human_wait_s']:.0f}s")
    st.caption("Colors: green = success · orange = warning (rejection, retry, low score, tool error) · "
               "red = failure · purple = waiting for or resolved by a human")

    order = tree.build_tree(spans)
    rolled = tree.rollup(order)
    show_leaf = st.checkbox("Show LLM and tool calls in the graph", value=True, key="show_leaf")
    idx = st.selectbox("Select a span (the graph highlights it, the panel below shows its full context)",
                       list(range(len(order))), format_func=lambda i: tree.label(*order[i]), key="span_pick")
    selected = order[idx][1]
    st.graphviz_chart(tree.to_dot(order, rolled, selected["span_id"], show_leaf))
    st.divider()
    render_detail(selected, rolled.get(selected["span_id"], {}))

# ---------------- tab 2: cost and performance ----------------
with tab_cost:
    agg = metrics.aggregate(summaries)
    t = agg["totals"]
    c = st.columns(5)
    c[0].metric("Tasks", t["tasks"])
    c[1].metric("Total cost", f"${t['cost_usd']:.4f}")
    c[2].metric("Total tokens", t["tokens"])
    c[3].metric("Tool calls", t["tool_calls"])
    c[4].metric("Escalation rate", f"{t['escalation_rate'] * 100:.0f}%")
    if t["unpriced_calls"]:
        st.warning(f"{t['unpriced_calls']} model calls used a model that is not in the price table, so their cost "
                   f"shows as $0. Add it to {metrics.__name__.split('.')[0]}/data/pricing.json.")

    st.subheader("Tasks")
    st.dataframe(pd.DataFrame([{k: s[k] for k in ("task_id", "outcome", "task_type", "total_tokens", "llm_calls",
                                                  "tool_calls", "cost_usd", "active_s", "human_wait_s", "wall_s")}
                               for s in sorted(summaries, key=lambda s: -s["started"])]))
    for title, rows, x, y in [("Cost per task type", agg["by_type"], "task_type", "cost_usd"),
                              ("Most expensive agents", agg["by_agent"], "agent", "cost_usd")]:
        st.subheader(title)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df)
            st.bar_chart(df.groupby(x)[y].sum())
    st.subheader("Tool usage patterns")
    if agg["tools"]:
        st.dataframe(pd.DataFrame(agg["tools"]))
    st.subheader("Escalation rate by day")
    if agg["trend"]:
        df = pd.DataFrame(agg["trend"])
        st.dataframe(df)
        st.line_chart(df.set_index("day")["escalation_rate"])

# ---------------- tab 3: replay ----------------
with tab_replay:
    st.write("Replay re-runs a task from its **recorded decisions**. A step whose input is unchanged reuses the "
             "recording (free). A step you change, and every later step whose input differs, runs **live**. "
             "Replays never write memory, never create real reviews and never execute sensitive tools.")
    decs = store.decisions(task_id)
    key = f"overrides-{task_id}"
    st.session_state.setdefault(key, [])
    if not decs:
        st.info("No recorded decisions for this task.")
    else:
        st.dataframe(pd.DataFrame([{"step": i, "decision": f"{d['kind']}:{d['key']}#{d['index']}",
                                    "status": d["status"], "summary": d["summary"]} for i, d in enumerate(decs)]))
        pick = st.selectbox("Decision to inspect or change", list(range(len(decs))), key="replay_pick",
                            format_func=lambda i: f"{i}: {decs[i]['kind']}:{decs[i]['key']}#{decs[i]['index']}")
        d = decs[pick]
        l, r = st.columns(2)
        l.write("**Recorded input**")
        l.json(d["input"])
        r.write("**Recorded output**")
        r.json(d["output"])
        patch_txt = st.text_area("Modify the INPUT: a JSON object merged into this step's input", "{}",
                                 key=f"patch-{pick}", height=90)
        out_txt = st.text_area("Replace the OUTPUT: JSON, same shape as the recorded output (skips the model)", "",
                               key=f"out-{pick}", height=120)
        force = st.checkbox("Force a live re-run of this step", key=f"force-{pick}")
        if st.button("Add this change", key=f"add-{pick}"):
            try:
                ov = {"kind": d["kind"], "key": d["key"], "index": d["index"]}
                patch = json.loads(patch_txt or "{}")
                if patch:
                    ov["input_patch"] = patch
                if out_txt.strip():
                    ov["output"] = json.loads(out_txt)
                if force:
                    ov["force_live"] = True
                if len(ov) == 3:
                    st.error("Nothing to add: edit the input, replace the output, or force a live run.")
                else:
                    st.session_state[key].append(ov)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

        st.write("**Answer a human question differently** (tag like `plan-0` or `step-t1-0`)")
        hc = st.columns(3)
        tag = hc[0].text_input("Review tag", key="h_tag")
        action = hc[1].selectbox("Action", ["approve", "reject", "modify"], key="h_action")
        note = hc[2].text_input("Note", key="h_note")
        if st.button("Add human override", key="h_add") and tag.strip():
            st.session_state[key].append({"kind": "human", "key": tag.strip(), "index": 0,
                                          "output": {"action": action, "note": note, "by": "replay"}})

        st.write("**Changes that will be applied**")
        if st.session_state[key]:
            st.json(st.session_state[key])
        else:
            st.caption("None yet: the replay will reproduce the original run (free).")
        b = st.columns(3)
        human_default = b[2].selectbox("If a new human question appears", ["reject", "approve"], key="h_default")
        if b[0].button("Run replay"):
            rid = replay.launch(task_id, st.session_state[key], human_default)
            st.success(f"Replay {rid} started in the background. Press Refresh to follow it.")
        if b[1].button("Clear changes"):
            st.session_state[key] = []
            st.rerun()

    st.divider()
    st.subheader("Replay results for this task")
    reports = replay.load_reports(task_id)
    if not reports:
        st.caption("No replays yet.")
    else:
        pick_r = st.selectbox("Replay", list(range(len(reports))), key="report_pick",
                              format_func=lambda i: f"{reports[i]['replay_id']} · {reports[i].get('status')}")
        render_report(reports[pick_r])