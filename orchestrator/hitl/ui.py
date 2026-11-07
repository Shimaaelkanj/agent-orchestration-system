
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # make `orchestrator` importable

import streamlit as st  # noqa: E402

from orchestrator import config  # noqa: E402
from orchestrator.hitl.chat import ask_agent  # noqa: E402
from orchestrator.hitl.launch import resume_in_background  # noqa: E402
from orchestrator.hitl.queue import ReviewQueue  # noqa: E402

st.set_page_config(page_title="Agent reviews", layout="wide")
q = ReviewQueue()
ICON = {"notify": "[info]", "approve_action": "[action]", "approve_plan": "[plan]", "take_over": "[takeover]"}

# ---------------- sidebar: the queue ----------------
reviewer = st.sidebar.text_input("Your name", value="reviewer")
view = st.sidebar.selectbox("Show", ["pending", "notified", "decided", "resolved", "all"])
if st.sidebar.button("Refresh"):
    st.rerun()
items = q.list_reviews(None if view == "all" else view)
if not items:
    st.title("Human review")
    st.info(f"No reviews with status '{view}'.")
    st.stop()
labels = {f"{ICON.get(i['level'], '')} {i['review_id']}": i for i in items}
item = labels[st.sidebar.radio("Reviews", list(labels))]
p, rid, prop = item["payload"], item["review_id"], item["payload"]["proposed"]

# ---------------- context ----------------
st.title("Human review")
c1, c2, c3 = st.columns(3)
c1.metric("Level", p["level"])
c2.metric("Trigger", p["trigger"])
c3.metric("Status", item["status"])
st.subheader("Task")
st.write(p["task"])
st.warning(p["reason"])

st.subheader("Execution progress")
st.dataframe([{"id": s["id"], "specialist": s["specialist"], "status": s["status"], "attempts": s["attempts"],
               "by human": s["by_human"], "description": s["description"]} for s in p["progress"]],
             use_container_width=True)
with st.expander("Full plan (JSON)"):
    st.json(p["plan"])

# ---------------- decision point and proposed action ----------------
st.subheader("Decision point")
st.json(p["decision_point"])
if prop["type"] == "tool_call":
    st.write("**The agent wants to run this tool:**")
    st.code(f"{prop['tool']}(\n{json.dumps(prop['args'], indent=2)}\n)", language="python")
    st.write("**Agent's reasoning:**")
    st.write(prop.get("reasoning") or "(the agent gave no reasoning)")
elif prop["type"] == "accept_output":
    st.write("**Deliverable awaiting acceptance:**")
    st.code(prop.get("output") or "(no output yet)")
    if prop.get("last_error"):
        st.error(f"Last error: {prop['last_error']}")
    if prop.get("feedback"):
        st.caption(f"Latest feedback to the agent: {prop['feedback']}")
elif prop["type"] == "execute_plan":
    st.write("**Plan awaiting approval:**")
    st.dataframe([{"id": s["id"], "specialist": s["specialist"], "depends_on": ", ".join(s["depends_on"]),
                   "description": s["description"]} for s in prop["plan"]["subtasks"]])
elif prop["type"] == "accepted_output":
    st.code(prop.get("output", ""))

with st.expander(f"Relevant memories and past decisions ({len(p['memories'])})"):
    for m in p["memories"] or []:
        st.write(f"**{m['kind']}** (similarity {m['similarity']}): {m['text']}")
    if not p["memories"]:
        st.caption("Nothing relevant remembered yet.")

# ---------------- chat panel ----------------
st.subheader("Ask the agent")
for m in q.chat(rid):
    with st.chat_message("user" if m["role"] == "human" else "assistant"):
        st.write(m["content"])
if question := st.chat_input("Ask a clarifying question before you decide"):
    with st.spinner("Thinking..."):
        ask_agent(q, rid, question)
    st.rerun()

# ---------------- decision ----------------
if item["status"] == "notified":
    st.info("Information only: the run continued without you.")
    if st.button("Acknowledge"):
        q.ack(rid)
        st.rerun()
elif item["status"] == "pending":
    st.subheader("Your decision")
    with st.form(f"decision-{rid}"):
        note = st.text_area("Note: guidance for the agent, or your reason (optional)")
        if prop["type"] == "tool_call":
            edited = st.text_area("Modify: edited tool arguments (JSON)", value=json.dumps(prop["args"], indent=2), height=140)
        elif prop["type"] == "execute_plan":
            edited = st.text_area("Modify: edited plan (JSON). Leave unchanged and write a note to ask for a re-plan",
                                  value=json.dumps(prop["plan"], indent=2), height=260)
        else:
            edited = st.text_area("Modify: edited output (or leave it and write a note so the agent retries)",
                                  value=prop.get("output", ""), height=140)
        mine = st.text_area("Take over: the output YOU provide (the agents stand down)", height=110)
        b1, b2, b3, b4 = st.columns(4)
        approve = b1.form_submit_button("Approve")
        modify = b2.form_submit_button("Modify")
        reject = b3.form_submit_button("Reject")
        takeover = b4.form_submit_button("Take over")

    decision, problem = None, None
    if approve:
        decision = {"action": "approve", "note": note}
    elif reject:
        decision = {"action": "reject", "note": note}
    elif takeover:
        if mine.strip():
            decision = {"action": "take_over", "output": mine, "note": note}
        else:
            problem = "Write the output you want to provide in the 'Take over' box."
    elif modify:
        decision = {"action": "modify", "note": note}
        try:
            if prop["type"] == "tool_call":
                decision["args"] = json.loads(edited)
            elif prop["type"] == "execute_plan":
                new_plan = json.loads(edited)
                if new_plan != prop["plan"]:
                    decision["plan"] = new_plan
                elif not note.strip():
                    problem = "Nothing was modified. Edit the plan, or write a note asking for a re-plan."
            elif edited.strip() != (prop.get("output") or "").strip():
                decision["output"] = edited
            elif not note.strip():
                problem = "Nothing was modified. Edit the output, or write guidance in the note."
        except json.JSONDecodeError as e:
            problem = f"The JSON is invalid: {e}"
    if problem:
        st.error(problem)
    elif decision:
        decision["by"] = reviewer
        if q.decide(rid, decision):
            resume_in_background(rid)
            st.success("Decision recorded. The agents are continuing in the background. Press Refresh to follow it.")
        else:
            st.error("This review was already decided.")
else:
    st.subheader("Decision")
    st.json(item["decision"])
    lp = config.LOGS_DIR / f"resume_{rid}.log"
    if lp.exists():
        with st.expander("Run log after your decision", expanded=True):
            st.code("\n".join(lp.read_text(errors="ignore").splitlines()[-60:]))