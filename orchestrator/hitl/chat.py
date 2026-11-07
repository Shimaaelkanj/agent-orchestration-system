
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from ..obs.tracing import agent_scope, node_span
from ..llm import get_llm

SYSTEM = ("You are the AI agent system that paused to ask a human reviewer for a decision. Answer the "
          "reviewer's questions using ONLY the context below. If the context does not contain the answer, "
          "say you do not know. Be concise. You cannot take actions from this chat: the human decides with "
          "the buttons. Never reveal secrets or keys.\n\nCONTEXT:\n")


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def build_context(item: dict) -> str:
    p = item["payload"]
    lines = [f"Task: {p['task']}",
             f"Why a human is needed ({p['level']} / {p['trigger']}): {p['reason']}",
             "Plan and progress:"]
    for s in p["progress"]:
        lines.append(f"- {s['id']} [{s['status']}] ({s['specialist']}) {s['description']} -> {s['output'][:300]}")
    lines.append("Decision point: " + json.dumps(p["decision_point"], default=str))
    lines.append("Proposed action: " + json.dumps(p["proposed"], default=str)[:2500])
    if p.get("memories"):
        lines.append("Relevant memories:")
        lines += [f"- [{m['kind']}] {m['text']}" for m in p["memories"]]
    return "\n".join(lines)[:7000]


def ask_agent(queue, review_id: str, question: str) -> str:
    item = queue.get(review_id)
    if not item:
        raise ValueError("unknown review")
    messages = [SystemMessage(SYSTEM + build_context(item))]
    for m in queue.chat(review_id)[-8:]:
        messages.append(HumanMessage(m["content"]) if m["role"] == "human" else AIMessage(m["content"]))
    messages.append(HumanMessage(question))
    with node_span("human.chat", item["task_id"], **{"human.review_id": review_id}), agent_scope("supervisor"):
        answer = _text(get_llm("supervisor").invoke(messages).content).strip() or "(no answer)"
    queue.add_chat(review_id, "human", question)
    queue.add_chat(review_id, "agent", answer)
    return answer