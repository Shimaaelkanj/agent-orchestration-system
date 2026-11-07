import copy
import functools
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from . import config
from .hitl import policy
from .hitl.notify import notify
from .hitl.policy import Level, Trigger
from .hitl.queue import get_queue
from .memory import get_ltm, get_wm
from .memory.note_tools import register_memory_tools
from .memory.service import remember_decisions, remember_task
from .memory.working import CURRENT_TASK
from .obs import replay_core
from .obs.decisions import create_plan, recall_for_planning, review_output, run_specialist, synthesize
from .obs.tracing import annotate, event, mark, node_span, propagate
from .schemas import ExecutionPlan, ReviewVerdict, SpecialistReport, Subtask, drop_trailing_writing, validate_plan
from .tools import ApprovalRequired, build_registry
from .agents import fallback as plan_fallback

log = logging.getLogger("orchestrator.graph")
registry = build_registry()
register_memory_tools(registry, get_wm)

RETRYABLE = {"pending", "rejected", "failed"}
HUMAN_STATUSES = {"escalate", "awaiting_approval"}


class State(TypedDict, total=False):
    task: str
    user_id: str
    task_id: str
    replay_of: str
    review_requested: bool
    approve_final: bool
    human_guidance: str
    plan_approved: bool
    human_rounds: int
    human_log: list
    memory_context: str
    memory_report: dict
    plan: dict
    results: dict
    final_answer: str
    status: str            # planning | executing | done | needs_human | rejected | replan | taken_over
    escalation_reason: str


# ---------------- helpers ----------------
def _safe(fn, what: str):
    """Memory, notification and lookup problems must never fail a task."""
    try:
        return fn()
    except Exception as e:
        log.warning("%s failed: %s", what, str(e)[:150])
        return None


def _replaying() -> bool:
    return replay_core.current() is not None


def _clip(text, n: int = 300) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _init_results(plan: ExecutionPlan) -> dict:
    return {s.id: {"status": "pending", "output": "", "confidence": 0.0, "error": None, "attempts": 0,
                   "feedback": None, "evidence": "", "prior_evidence": "", "proposed_action": None,
                   "by_human": False, "reason": "", "escalation_kind": ""} for s in plan.subtasks}


def _review_id(state: State, tag: str) -> str:
    # Deterministic: the node that pauses is re-executed on resume and must find the SAME review.
    return f"{state['task_id']}-{tag}-{state.get('human_rounds', 0)}"


def _memories(state: State, extra: str) -> list[dict]:
    mems = get_ltm().search(state["user_id"], f"{state['task']} {extra}", k=4,
                            kinds=["decision", "failure", "approach", "episode"], touch=False)
    return [{"kind": m.kind, "text": _clip(m.text, 300), "similarity": round(m.similarity or 0, 2)} for m in mems]


def _package(state: State, rid: str, level: Level, trigger: Trigger, reason: str,
             decision_point: dict, proposed: dict) -> dict:
    """Everything a human needs to decide without opening the logs."""
    plan = state.get("plan") or {}
    results = state.get("results") or {}
    progress = []
    for s in plan.get("subtasks", []):
        r = results.get(s["id"], {})
        progress.append({"id": s["id"], "specialist": s["specialist"], "description": s["description"],
                         "depends_on": s["depends_on"], "status": r.get("status", "planned"),
                         "attempts": r.get("attempts", 0), "confidence": r.get("confidence", 0.0),
                         "by_human": r.get("by_human", False), "output": (r.get("output") or "")[:600]})
    extra = f"{decision_point.get('description', '')} {proposed.get('tool', '')}".strip()
    return {"review_id": rid, "task_id": state["task_id"], "user_id": state["user_id"], "task": state["task"],
            "level": level.value, "trigger": trigger.value, "reason": reason, "plan": plan,
            "progress": progress, "decision_point": decision_point, "proposed": proposed,
            "memories": _safe(lambda: _memories(state, extra), "memory lookup") or [],
            "human_log": state.get("human_log", []), "created_at": time.time()}


def _submit(payload: dict, status: str = "pending") -> None:
    if _replaying():                       # a replay must never create real reviews or notifications
        return
    is_new = get_queue().submit(payload["review_id"], payload["task_id"], payload["user_id"],
                                payload["level"], payload["trigger"], payload, status)
    if is_new:
        notify(payload)


def _human_update(state: State, d: dict, trigger: Trigger, level: Level, rid: str) -> dict:
    entry = {"review_id": rid, "level": level.value, "trigger": trigger.value, "action": d.get("action"),
             "note": (d.get("note") or "")[:500], "by": d.get("by", "reviewer"), "ts": time.time()}
    return {"human_rounds": state.get("human_rounds", 0) + 1,
            "human_log": list(state.get("human_log", [])) + [entry]}


def traced(name: str):
    """Wrap a graph node in a span that belongs to this task's trace."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state: State):
            with node_span(f"node.{name}", state.get("task_id") or "unknown"):
                return fn(state)
        return wrapper
    return deco


# ---------------- nodes ----------------
def intake(state: State) -> dict:
    task = (state.get("task") or "").strip()
    if not task:
        raise ValueError("Empty task")
    task_id = state.get("task_id") or uuid.uuid4().hex[:12]
    user_id = state.get("user_id") or config.DEFAULT_USER_ID
    log.info("intake [task=%s user=%s]: %s", task_id, user_id, task[:100])
    annotate(**{"task.text": task, "user.id": user_id, "review.requested": bool(state.get("review_requested")),
                "approve.final": bool(state.get("approve_final")), "replay.of": state.get("replay_of")})
    return {"task": task, "task_id": task_id, "user_id": user_id, "results": {}, "status": "planning",
            "review_requested": bool(state.get("review_requested")),
            "approve_final": bool(state.get("approve_final")), "human_rounds": 0,
            "human_log": [], "plan_approved": False}


def _accept_plan(state: State, plan: ExecutionPlan, ctx: str, attempt: int, fallback: bool = False) -> dict:
    if config.STRIP_FINAL_WRITING:
        plan = drop_trailing_writing(plan)
    plan = plan.model_copy(update={"confidence": max(0.0, min(1.0, plan.confidence))})
    plan_dict = plan.model_dump(mode="json")
    _safe(lambda: get_wm().set_plan(state["task_id"], plan_dict), "working memory: set_plan")
    log.info("plan (confidence %.2f%s): %s", plan.confidence, ", FALLBACK" if fallback else "",
             [(s.id, s.specialist.value, s.depends_on) for s in plan.subtasks])
    annotate(**{"plan.confidence": plan.confidence, "plan.n_subtasks": len(plan.subtasks),
                "plan.task_type": "+".join(sorted({s.specialist.value for s in plan.subtasks})),
                "plan.attempts": attempt, "plan.fallback": fallback,
                "plan.subtasks": [{"id": s.id, "specialist": s.specialist.value, "depends_on": s.depends_on,
                                   "description": s.description[:200]} for s in plan.subtasks]})
    return {"plan": plan_dict, "results": _init_results(plan), "status": "executing",
            "memory_context": ctx, "plan_approved": False}


def planning(state: State) -> dict:
    ctx = _safe(lambda: recall_for_planning(get_ltm(), state["user_id"], state["task"]), "memory recall") or ""
    if ctx:
        log.info("memory: injected %d chars into the planning prompt", len(ctx))
    annotate(**{"memory.chars": len(ctx), "plan.used_memory": bool(ctx)})

    err = None
    for attempt in range(1, 4):
        try:
            plan = create_plan(state["task"], err, ctx, state.get("human_guidance", ""))
            validate_plan(plan)
        except (ValueError, TypeError, KeyError, replay_core.ReplayedError) as e:
            err = _clip(str(e), 400)
            log.warning("plan attempt %d failed: %s", attempt, err)
            event("plan.invalid", error=err, attempt=attempt)
            mark("warning")
            continue
        return _accept_plan(state, plan, ctx, attempt)

    if plan_fallback.enabled():          # degrade gracefully: one subtask, and a human is asked to approve it
        log.warning("planning failed 3 times; using the single-subtask fallback plan")
        event("plan.fallback", error=err or "")
        mark("warning")
        return _accept_plan(state, plan_fallback.fallback_plan(state["task"]), ctx, 4, fallback=True)
    mark("failure")
    return {"status": "needs_human", "escalation_reason": f"Could not produce a valid plan: {err}"}


def plan_gate(state: State) -> dict:
    """Escalation level 'Approve plan': a human reviews the plan BEFORE any work starts."""
    plan = state["plan"]
    triggers = [] if state.get("plan_approved") else policy.plan_triggers(
        plan.get("confidence", 1.0), state.get("review_requested", False))
    if not triggers:
        return {"status": "executing"}
    if state.get("human_rounds", 0) >= config.MAX_HUMAN_ROUNDS:
        return {"status": "needs_human", "escalation_reason": "Too many human review rounds"}

    why = {Trigger.USER_REQUESTED: "You asked for human review of the plan",
           Trigger.LOW_PLAN_CONFIDENCE: (f"The supervisor's confidence in its plan is {plan.get('confidence', 0):.2f}, "
                                         f"below the {config.PLAN_CONFIDENCE_THRESHOLD} threshold")}
    reason = "; ".join(why[t] for t in triggers)
    trigger = triggers[0]
    level = policy.level_for(trigger)
    rid = _review_id(state, "plan")
    _submit(_package(state, rid, level, trigger, reason, {"type": "plan", "confidence": plan.get("confidence")},
                     {"type": "execute_plan", "plan": plan}))
    annotate(**{"human.review_id": rid, "human.level": level.value, "human.trigger": trigger.value,
                "human.reason": reason})
    event("human.escalation", review_id=rid, level=level.value, trigger=trigger.value)

    decision = interrupt({"review_id": rid}) or {}     # PAUSE: the graph is checkpointed here
    annotate(**{"human.resumed": True, "human.action": decision.get("action"), "human.by": decision.get("by"),
                "human.note": decision.get("note")})
    event("human.resolution", review_id=rid, action=decision.get("action"))
    mark("escalated")
    return _apply_plan_decision(state, decision, trigger, level, rid)


def _apply_plan_decision(state: State, d: dict, trigger: Trigger, level: Level, rid: str) -> dict:
    action = d.get("action", "reject")
    note = (d.get("note") or "").strip()
    out = _human_update(state, d, trigger, level, rid)
    if action == "approve":
        out.update(plan_approved=True, status="executing")
    elif action == "modify":
        if d.get("plan"):
            try:
                new_plan = ExecutionPlan.model_validate(d["plan"])
                validate_plan(new_plan)
            except Exception as e:
                out.update(status="replan", human_guidance=f"{note} (your edited plan was invalid: {e})".strip())
                return out
            out.update(plan=new_plan.model_dump(mode="json"), results=_init_results(new_plan),
                       plan_approved=True, status="executing")
        else:
            out.update(status="replan", human_guidance=note or "The human reviewer asked for a different plan.")
    elif action == "take_over":
        text = (d.get("output") or "").strip()
        if text:
            out.update(final_answer=text, status="taken_over")
        else:
            out.update(status="rejected", escalation_reason="Take over requested without an output")
    else:
        out.update(status="rejected", escalation_reason=note or "Rejected by the human reviewer")
    return out


def execute(state: State) -> dict:
    plan = ExecutionPlan.model_validate(state["plan"])
    results = copy.deepcopy(state["results"])
    task_id = state["task_id"]
    ready = [s for s in plan.subtasks
             if results[s.id]["status"] in RETRYABLE
             and all(results[d]["status"] == "approved" for d in s.depends_on)]
    if not ready:
        mark("failure")
        return {"status": "needs_human", "escalation_reason": "No runnable subtasks (deadlock)"}
    annotate(**{"execute.ready": [s.id for s in ready]})

    def work(s: Subtask) -> None:
        CURRENT_TASK.set(task_id)
        r = results[s.id]
        r["attempts"] += 1
        deps = {d: (_safe(lambda d=d: get_wm().get_output(task_id, d), "wm read") or results[d]["output"])
                for d in s.depends_on}
        try:
            rep, ev = run_specialist(s, deps, r["feedback"], registry)
            r.update(status="needs_review", output=rep.answer, confidence=rep.confidence, error=None,
                     evidence=(r.get("prior_evidence", "") + "\n" + ev).strip())
            _safe(lambda: get_wm().set_output(task_id, s.id, rep.answer), "wm write")
        except ApprovalRequired as a:
            # A sensitive tool was requested. Nothing ran. Ask a human.
            r.update(status="awaiting_approval", error=None, reason=f"The agent wants to run the sensitive tool '{a.tool}'",
                     proposed_action={"type": "tool_call", "tool": a.tool, "agent": a.agent,
                                      "args": a.tool_args, "reasoning": a.reasoning or s.description})
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            log.error("specialist failed on %s: %s", s.id, msg)
            r.update(status="failed", error=msg,
                     feedback=f"Previous attempt crashed with: {msg}. Use a different approach.")
            _safe(lambda: get_wm().log_error(task_id, s.id, msg), "wm error log")
            if r["attempts"] >= config.MAX_ATTEMPTS:
                r.update(status="escalate", escalation_kind=Trigger.FAILED_TWICE.value,
                         reason=f"Failed {r['attempts']} times. Last error: {msg[:200]}")

    with ThreadPoolExecutor(max_workers=config.MAX_PARALLEL) as pool:
        list(pool.map(propagate(work), ready))      # propagate keeps the trace context inside worker threads
    statuses = {sid: r["status"] for sid, r in results.items()}
    annotate(**{"execute.statuses": statuses})
    if set(statuses.values()) & {"failed", "escalate", "awaiting_approval"}:
        mark("warning")
    return {"results": results}


def review(state: State) -> dict:
    plan = ExecutionPlan.model_validate(state["plan"])
    results = copy.deepcopy(state["results"])
    task_id = state["task_id"]
    for s in plan.subtasks:
        r = results[s.id]
        if r["status"] != "needs_review":
            continue
        try:
            v = review_output(s, SpecialistReport(answer=r["output"], confidence=r["confidence"]),
                              r.get("evidence", ""))
        except Exception as e:       # a reviewer that cannot answer counts as a rejection, and is retried
            log.warning("reviewer failed on %s: %s", s.id, _clip(str(e), 200))
            v = ReviewVerdict(approved=False, confidence=0.0,
                              feedback=f"The reviewer could not produce a verdict ({type(e).__name__}); retry.")
        spec_c, rev_c = r["confidence"], v.confidence
        conf = min(spec_c, rev_c)

        if not v.approved:
            r["feedback"] = v.feedback[:400]
            _safe(lambda: get_wm().log_error(task_id, s.id, f"rejected: {v.feedback[:200]}"), "wm error log")
            if r["attempts"] >= config.MAX_ATTEMPTS:
                r.update(status="escalate", escalation_kind=Trigger.FAILED_TWICE.value,
                         reason=f"The reviewer rejected it {r['attempts']} times: {v.feedback[:200]}")
            else:
                r["status"] = "rejected"
        else:
            trig = policy.quality_trigger(conf)
            if trig == Trigger.LOW_QUALITY:
                r.update(status="escalate", escalation_kind=Trigger.LOW_QUALITY.value,
                         reason=f"Quality score {conf:.2f} is below the {config.CONFIDENCE_THRESHOLD} threshold")
            else:
                r["status"] = "approved"
                if trig == Trigger.BORDERLINE_QUALITY:      # level NOTIFY: continue, but tell the human
                    payload = _package(state, f"{task_id}-notify-{s.id}-{r['attempts']}", Level.NOTIFY, trig,
                                       f"Subtask {s.id} was accepted with a borderline quality score of {conf:.2f}",
                                       {"type": "info", "subtask_id": s.id, "description": s.description},
                                       {"type": "accepted_output", "output": r["output"][:2000], "confidence": conf})
                    _safe(lambda: _submit(payload, status="notified"), "notification")

        log.info("review %s -> %s (specialist conf %.2f, reviewer conf %.2f)", s.id, r["status"], spec_c, rev_c)
    statuses = {sid: r["status"] for sid, r in results.items()}
    annotate(**{"review.statuses": statuses})
    if set(statuses.values()) & {"rejected", "escalate"}:
        mark("warning")
    return {"results": results}


def human_gate(state: State) -> dict:
    """Levels 'Approve action' and 'Take over': pause on one subtask that needs a human."""
    if state.get("human_rounds", 0) >= config.MAX_HUMAN_ROUNDS:
        return {"status": "needs_human", "escalation_reason": "Too many human review rounds"}
    plan = ExecutionPlan.model_validate(state["plan"])
    results = copy.deepcopy(state["results"])
    s = next(x for x in plan.subtasks if results[x.id]["status"] in HUMAN_STATUSES)
    r = results[s.id]

    if r["status"] == "awaiting_approval":
        trigger, proposed = Trigger.SENSITIVE_ACTION, dict(r["proposed_action"])
    else:
        trigger = Trigger(r.get("escalation_kind") or Trigger.FAILED_TWICE.value)
        proposed = {"type": "accept_output", "output": r["output"][:4000], "confidence": r["confidence"],
                    "last_error": r["error"], "feedback": r["feedback"]}
    level = policy.level_for(trigger)
    rid = _review_id(state, f"step-{s.id}")
    reason = r.get("reason") or "A human decision is needed"
    _submit(_package(state, rid, level, trigger, reason,
                     {"type": "subtask", "subtask_id": s.id, "specialist": s.specialist.value,
                      "description": s.description}, proposed))
    annotate(**{"human.review_id": rid, "human.level": level.value, "human.trigger": trigger.value,
                "human.reason": reason, "human.subtask": s.id})
    event("human.escalation", review_id=rid, level=level.value, trigger=trigger.value)

    decision = interrupt({"review_id": rid}) or {}     # PAUSE
    annotate(**{"human.resumed": True, "human.action": decision.get("action"), "human.by": decision.get("by"),
                "human.note": decision.get("note")})
    event("human.resolution", review_id=rid, action=decision.get("action"))
    mark("escalated")
    return _apply_step_decision(state, results, s, r, decision, trigger, level, rid)


def _run_approved(s: Subtask, r: dict, args: dict, note: str) -> None:
    """Execute the human-approved (and possibly edited) action ourselves, then let the agent continue."""
    pa = r["proposed_action"]
    if _replaying():          # replays never execute real tools
        res = json.dumps({"simulated": True, "note": "replay: tool not executed"})
    else:
        res = registry.invoke(pa["tool"], s.specialist.value, args, approved=True)
    r["prior_evidence"] = (r.get("prior_evidence", "") + "\n" +
                           f"TOOL {pa['tool']} args={json.dumps(args)} -> {res[:1500]} (approved by a human)").strip()
    r.update(status="pending", error=None, proposed_action=None, attempts=max(0, r["attempts"] - 1),
             feedback=(f"A human approved this action and the system ALREADY executed it: "
                       f"{pa['tool']}({json.dumps(args)[:300]}) returned {res[:600]}. "
                       f"Do NOT run it again; continue the task using this result."
                       + (f" Human note: {note}" if note else "")))


def _apply_step_decision(state: State, results: dict, s: Subtask, r: dict, d: dict,
                         trigger: Trigger, level: Level, rid: str) -> dict:
    action = d.get("action", "reject")
    note = (d.get("note") or "").strip()
    out = _human_update(state, d, trigger, level, rid)
    out["results"] = results
    awaiting = r["status"] == "awaiting_approval"
    task_id = state["task_id"]

    def accept(text: str) -> None:
        r.update(status="approved", output=text, confidence=1.0, error=None, feedback=None,
                 by_human=True, proposed_action=None)
        _safe(lambda: get_wm().set_output(task_id, s.id, text), "wm write")

    def retry(guidance: str) -> None:
        r.update(status="pending", attempts=0, error=None, proposed_action=None,
                 feedback=f"Human reviewer guidance: {guidance}")

    if action == "reject":
        out.update(status="rejected", escalation_reason=note or f"Rejected by the human reviewer at subtask {s.id}")
    elif action == "take_over":
        text = (d.get("output") or "").strip()
        if text:
            accept(text)
        else:
            out.update(status="rejected", escalation_reason="Take over requested without an output")
    elif awaiting:                                    # approve / modify a sensitive action
        args = d["args"] if action == "modify" and d.get("args") else r["proposed_action"]["args"]
        _run_approved(s, r, args, note)
    elif action == "approve":
        if r.get("output"):
            accept(r["output"])
        else:
            retry(note or "try again")
    elif action == "modify":
        edited = (d.get("output") or "").strip()
        if edited:
            accept(edited)
        else:
            retry(note or "try again")
    else:
        out.update(status="rejected", escalation_reason=note or "Rejected by the human reviewer")
    return out


def synthesis(state: State) -> dict:
    outputs = {k: v["output"] for k, v in state["results"].items()}
    return {"final_answer": synthesize(state["task"], outputs)}

def final_gate(state: State) -> dict:
    """Optional last gate: a human approves the finished deliverable before it is delivered and memorized."""
    if not state.get("approve_final"):
        return {"status": "executing"}
    if state.get("human_rounds", 0) >= config.MAX_HUMAN_ROUNDS:
        return {"status": "needs_human", "escalation_reason": "Too many human review rounds"}
    trigger = Trigger.FINAL_APPROVAL
    level = policy.level_for(trigger)
    rid = _review_id(state, "final")
    reason = "You asked to approve the final deliverable before it is delivered"
    _submit(_package(state, rid, level, trigger, reason, {"type": "final_answer"},
                     {"type": "accept_output", "output": (state.get("final_answer") or "")[:6000],
                      "confidence": 1.0, "last_error": None, "feedback": None}))
    annotate(**{"human.review_id": rid, "human.level": level.value, "human.trigger": trigger.value,
                "human.reason": reason})
    event("human.escalation", review_id=rid, level=level.value, trigger=trigger.value)

    decision = interrupt({"review_id": rid}) or {}     # PAUSE
    annotate(**{"human.resumed": True, "human.action": decision.get("action"), "human.by": decision.get("by"),
                "human.note": decision.get("note")})
    event("human.resolution", review_id=rid, action=decision.get("action"))
    mark("escalated")
    return _apply_final_decision(state, decision, trigger, level, rid)


def _apply_final_decision(state: State, d: dict, trigger: Trigger, level: Level, rid: str) -> dict:
    action = d.get("action", "reject")
    note = (d.get("note") or "").strip()
    text = (d.get("output") or "").strip()
    out = _human_update(state, d, trigger, level, rid)
    if action == "reject":
        out.update(status="rejected", escalation_reason=note or "The final deliverable was rejected by the human reviewer")
    elif action == "take_over":
        if text:
            out.update(final_answer=text, status="taken_over")
        else:
            out.update(status="rejected", escalation_reason="Take over requested without an output")
    elif action == "modify" and text:
        out.update(final_answer=text, status="executing")
    else:
        out.update(status="executing")           # approve (or modify with only a note)
    return out


def route_after_final(state: State) -> str:
    st = state.get("status")
    if st == "rejected":
        return "abort"
    if st == "needs_human":
        return "escalate"
    return "memorize"







def memorize(state: State) -> dict:
    if _replaying():                  # a replay must never change long-term memory
        return {"memory_report": {"skipped": "replay"}}
    outcome = "done_by_human" if state.get("status") == "taken_over" else "done"
    counts = _safe(lambda: remember_task(get_ltm(), state["user_id"], state, outcome=outcome),
                   "long-term memory write") or {}
    _safe(lambda: remember_decisions(get_ltm(), state["user_id"], state), "decision memory")
    log.info("memorized: %s", counts)
    _safe(lambda: get_wm().clear(state["task_id"]), "working memory clear")
    return {"memory_report": counts}


def _save_run(state: State) -> Path:
    run_dir = config.LOGS_DIR / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"run_{int(time.time())}.json"
    path.write_text(json.dumps(dict(state), indent=2, default=str), encoding="utf-8")
    return path


def delivery(state: State) -> dict:
    annotate(**{"result.status": "done", "result.final_answer": state.get("final_answer", "")})
    if not _replaying():
        path = _save_run(state)
        log.info("delivered, saved to %s", path)
    return {"status": "done"}


def abort(state: State) -> dict:
    """A human rejected the task: stop, remember why, clean up."""
    log.warning("TASK REJECTED BY HUMAN: %s", state.get("escalation_reason"))
    annotate(**{"result.status": "rejected", "result.reason": state.get("escalation_reason", "")})
    counts: dict = {}
    if not _replaying():
        counts = _safe(lambda: remember_task(get_ltm(), state["user_id"], state, outcome="rejected"),
                       "long-term memory write") or {}
        _safe(lambda: remember_decisions(get_ltm(), state["user_id"], state), "decision memory")
        _safe(lambda: get_wm().clear(state["task_id"]), "working memory clear")
        _save_run({**state, "status": "rejected", "memory_report": counts})
    return {"status": "rejected", "memory_report": counts}


def escalate(state: State) -> dict:
    """Unrecoverable (invalid plan, deadlock, too many human rounds): stop and flag."""
    log.warning("ESCALATED: %s", state.get("escalation_reason"))
    annotate(**{"result.status": "needs_human", "result.reason": state.get("escalation_reason", "")})
    mark("escalated")
    counts: dict = {}
    if state.get("user_id") and not _replaying():
        counts = _safe(lambda: remember_task(get_ltm(), state["user_id"], state, outcome="needs_human"),
                       "long-term memory write") or {}
        _safe(lambda: remember_decisions(get_ltm(), state["user_id"], state), "decision memory")
        path = _save_run({**state, "status": "needs_human", "memory_report": counts})
        log.info("escalated run saved to %s", path)
    return {"status": "needs_human", "memory_report": counts}


# ---------------- routing ----------------
def _statuses(state: State) -> set[str]:
    return {r["status"] for r in state["results"].values()}


def route_after_planning(state: State) -> str:
    return "escalate" if state.get("status") == "needs_human" else "plan_gate"


def route_after_plan_gate(state: State) -> str:
    return {"replan": "planning", "rejected": "abort", "needs_human": "escalate",
            "taken_over": "memorize"}.get(state.get("status"), "execute")


def route_after_execute(state: State) -> str:
    if state.get("status") == "needs_human":
        return "escalate"
    s = _statuses(state)
    if s & HUMAN_STATUSES:
        return "human_gate"
    return "review" if "needs_review" in s else "execute"


def route_after_review(state: State) -> str:
    s = _statuses(state)
    if s & HUMAN_STATUSES:
        return "human_gate"
    return "synthesis" if s == {"approved"} else "execute"


def route_after_human(state: State) -> str:
    st = state.get("status")
    if st == "rejected":
        return "abort"
    if st == "needs_human":
        return "escalate"
    s = _statuses(state)
    if s & HUMAN_STATUSES:
        return "human_gate"
    if "needs_review" in s:
        return "review"
    return "synthesis" if s == {"approved"} else "execute"


def build_graph(checkpointer=None):
    g = StateGraph(State)
    for name, fn in [("intake", intake), ("planning", planning), ("plan_gate", plan_gate),
                     ("execute", execute), ("review", review), ("human_gate", human_gate),
                     ("synthesis", synthesis), ("final_gate", final_gate), ("memorize", memorize),
                     ("delivery", delivery), ("abort", abort), ("escalate", escalate)]:
        g.add_node(name, traced(name)(fn))
    g.add_edge(START, "intake")
    g.add_edge("intake", "planning")
    g.add_conditional_edges("planning", route_after_planning, {"plan_gate": "plan_gate", "escalate": "escalate"})
    g.add_conditional_edges("plan_gate", route_after_plan_gate,
                            {"execute": "execute", "planning": "planning", "abort": "abort",
                             "escalate": "escalate", "memorize": "memorize"})
    g.add_conditional_edges("execute", route_after_execute,
                            {"review": "review", "execute": "execute", "human_gate": "human_gate",
                             "escalate": "escalate"})
    g.add_conditional_edges("review", route_after_review,
                            {"synthesis": "synthesis", "execute": "execute", "human_gate": "human_gate"})
    g.add_conditional_edges("human_gate", route_after_human,
                            {"human_gate": "human_gate", "execute": "execute", "review": "review",
                             "synthesis": "synthesis", "abort": "abort", "escalate": "escalate"})
    g.add_edge("synthesis", "final_gate")
    g.add_conditional_edges("final_gate", route_after_final,
                            {"memorize": "memorize", "abort": "abort", "escalate": "escalate"})
    g.add_edge("memorize", "delivery")
    g.add_edge("delivery", END)
    g.add_edge("abort", END)
    g.add_edge("escalate", END)
    return g.compile(checkpointer=checkpointer)
  