
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .. import config
from ..llm import get_llm, structured
from .long_term import LongTermMemory

log = logging.getLogger("orchestrator.memory.service")

_SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|tvly-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{20,})")

def remember_decisions(ltm: LongTermMemory, user_id: str, state: dict) -> int:
    """Store what humans decided, so future plans and reviews can learn from it."""
    n = 0
    for e in state.get("human_log") or []:
        note = f" Note: {_clip(e['note'], 250)}" if e.get("note") else ""
        text = (f"For a task like '{_clip(state.get('task', ''), 200)}', the human reviewer chose to "
                f"{e['action']} at '{e['trigger']}' (review level {e['level']}).{note}")
        ltm.add(user_id, "decision", text,
                {"task_id": state.get("task_id", ""), "trigger": e["trigger"], "action": e["action"]})
        n += 1
    return n
    
def _clip(text, n: int = 500) -> str:
    t = _SECRET.sub("[redacted]", " ".join(str(text).split()))
    return t if len(t) <= n else t[: n - 1] + "…"


# ---------------- step 2: extraction ----------------
class MemoryExtraction(BaseModel):
    approach: str = Field(description="One or two sentences: the approach that worked and why. Empty string if unclear.")
    facts: list[str] = Field(description="Durable facts discovered in the results, max 3. Empty list if none.")
    preferences: list[str] = Field(description="Preferences the user EXPLICITLY stated in the request (format, tone, language). Empty list if none.")


EXTRACT_PROMPT = """You extract long-term memories from a finished task.
Only use what is explicitly present in the text. Never guess or invent.
- approach: how the task was solved, in general terms that would help a similar future task.
- facts: durable facts stated in the results (not task-specific numbers).
- preferences: ONLY if the user stated a preference in the request itself. Otherwise return an empty list.
Never include secrets, keys, or passwords."""


def build_episode(state: dict, outcome: str) -> str:
    plan = (state.get("plan") or {}).get("subtasks", [])
    steps = "; ".join(f"{s['id']} ({s['specialist']}): {_clip(s['description'], 160)}" for s in plan)
    tools = _tools_used(state)
    reason = f" ({_clip(state.get('escalation_reason', ''), 200)})" if outcome != "done" else ""
    return (f"Task: {_clip(state.get('task', ''), 300)}\nOutcome: {outcome}{reason}\n"
            f"Plan: {steps or 'none'}\nTools used: {', '.join(tools) or 'none'}")


def _tools_used(state: dict) -> list[str]:
    names: list[str] = []
    for r in (state.get("results") or {}).values():
        names += re.findall(r"^TOOL (\w+)", r.get("evidence") or "", re.M)
    return sorted(set(names))


def failure_notes(state: dict) -> list[str]:
    desc = {s["id"]: s for s in (state.get("plan") or {}).get("subtasks", [])}
    notes = []
    for sid, r in (state.get("results") or {}).items():
        trouble = r.get("feedback") or r.get("error")
        if trouble and (r.get("attempts", 0) > 1 or r.get("status") == "escalate"):
            s = desc.get(sid, {})
            notes.append(f"Subtask '{_clip(s.get('description', sid), 160)}' ({s.get('specialist', '?')}) "
                         f"needed retries or failed: {_clip(trouble, 240)}")
    return notes


def llm_extract(state: dict) -> MemoryExtraction | None:
    """Best effort: a weak model may fail here, and that must never fail the task."""
    try:
        body = "\n".join(f"[{k}] {_clip(v.get('output', ''), 400)}"
                         for k, v in (state.get("results") or {}).items())
        msg = (f"User request:\n{_clip(state.get('task', ''), 400)}\n\nResults:\n{body}\n\n"
               f"Final answer:\n{_clip(state.get('final_answer', ''), 800)}")
        llm = structured(get_llm("supervisor"), "supervisor", MemoryExtraction)
        return llm.invoke([SystemMessage(EXTRACT_PROMPT), HumanMessage(msg)])
    except Exception as e:
        log.warning("memory extraction failed: %s", str(e)[:150])
        return None


def remember_task(ltm: LongTermMemory, user_id: str, state: dict,
                  outcome: str = "done", use_llm: bool | None = None) -> dict:
    use_llm = config.USE_LLM_EXTRACTION if use_llm is None else use_llm
    base = {"task_id": state.get("task_id", "")}
    counts = {"episode": 0, "approach": 0, "failure": 0, "fact": 0, "preference": 0}

    ltm.add(user_id, "episode", build_episode(state, outcome), {**base, "outcome": outcome})
    counts["episode"] += 1
    for note in failure_notes(state):
        ltm.add(user_id, "failure", note, base)
        counts["failure"] += 1

    if outcome == "done" and use_llm:
        ex = llm_extract(state)
        if ex:
            if ex.approach.strip():
                ltm.add(user_id, "approach", _clip(ex.approach), base)
                counts["approach"] += 1
            for kind, items in (("fact", ex.facts[:3]), ("preference", ex.preferences[:3])):
                for item in items:
                    if item.strip():
                        ltm.add(user_id, kind, _clip(item, 300), base)
                        counts[kind] += 1
    return counts


# ---------------- step 3: retrieval for planning ----------------
def recall_for_planning(ltm: LongTermMemory, user_id: str, task: str, max_chars: int = 2500) -> str:
    sections = [("Similar past tasks", ["episode", "summary"]),
                ("Approaches that worked", ["approach"]),
                ("Things that went wrong before", ["failure"]),
                ("Relevant facts", ["fact"])]
    lines: list[str] = []
    for title, kinds in sections:
        mems = ltm.search(user_id, task, kinds=kinds)
        if mems:
            lines.append(f"{title}:")
            lines += [f"- {_clip(m.text, 400)}" for m in mems]
    prefs = ltm.top(user_id, ["preference"], k=3)       # preferences apply to every task
    if prefs:
        lines.append("User preferences:")
        lines += [f"- {_clip(m.text, 200)}" for m in prefs]
    return "\n".join(lines)[:max_chars]